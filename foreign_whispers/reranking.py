"""Failure analysis and duration-aware translation re-ranking.

Two public functions:

- ``analyze_failures``: pure heuristic that classifies the dominant failure
  mode in a clip evaluation report.
- ``get_shorter_translations``: hybrid re-ranker that produces shorter
  target-language candidates fitting a duration budget.  Runs three stages
  in order — a persistent learned-cache lookup, a deterministic rule-based
  pre-pass, and an OpenRouter-routed LLM fallback — short-circuiting as
  soon as a candidate fits the budget.  Successful LLM outputs are
  written back to the on-disk cache so future runs serve them for free.
"""

import dataclasses
import datetime
import functools
import hashlib
import json
import logging
import os
import pathlib
import re
import tempfile
import threading

import requests

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


# ── Configuration ─────────────────────────────────────────────────────────
CHARS_PER_SECOND = 15.0  # rough TTS rate for Romance languages

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_TIMEOUT_S = 10.0

DEFAULT_LEARNED_PATH = pathlib.Path("pipeline_data/api/learned_shortenings.json")

# Spanish phrase contraction table — only fires on exact substring match.
_SPANISH_CONTRACTIONS: list[tuple[str, str]] = [
    ("en este momento", "ahora"),
    ("en aquel momento", "entonces"),
    ("de manera que", "así"),
    ("de modo que", "así"),
    ("a pesar de que", "pese a"),
    ("a pesar de", "pese a"),
    ("debido a que", "porque"),
    ("debido a", "por"),
    ("con el fin de", "para"),
    ("con el objeto de", "para"),
    ("por lo tanto", "así"),
    ("sin embargo", "pero"),
    ("además de", "y"),
    ("una gran cantidad de", "muchos"),
    ("un montón de", "muchos"),
]

_SPANISH_HINT_CHARS = set("¿¡ñÑáéíóúÁÉÍÓÚ")
_SPANISH_HINT_WORDS = {"el", "la", "los", "las", "que", "de", "en", "y", "es"}

# ── Learned-cache state ───────────────────────────────────────────────────
_LEARNED_CACHE: dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_LOADED = False
_NO_KEY_WARNED = False


def _learned_cache_path() -> pathlib.Path:
    return pathlib.Path(os.environ.get("LEARNED_SHORTENINGS_PATH", str(DEFAULT_LEARNED_PATH)))


def _learning_enabled() -> bool:
    return os.environ.get("OPENROUTER_LEARN", "1") != "0"


def _cache_key(source_text: str, baseline: str) -> str:
    payload = f"{source_text}\u0000{baseline}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_learned_cache() -> None:
    """Lazy-load the learned cache from disk on first use."""
    global _CACHE_LOADED
    with _CACHE_LOCK:
        if _CACHE_LOADED:
            return
        _CACHE_LOADED = True
        path = _learned_cache_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _LEARNED_CACHE.update(data)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load learned shortenings from %s: %s", path, exc)


def _save_learned_entry(source_text: str, baseline: str, shortest: str) -> None:
    """Update in-memory cache and atomically flush to disk.

    Only overwrites an existing entry when *shortest* is strictly shorter
    than the cached value.
    """
    if not _learning_enabled():
        return
    key = _cache_key(source_text, baseline)
    with _CACHE_LOCK:
        existing = _LEARNED_CACHE.get(key)
        if existing and len(existing.get("shortest", "")) <= len(shortest):
            return
        _LEARNED_CACHE[key] = {
            "source": source_text,
            "baseline": baseline,
            "shortest": shortest,
            "rationale": "LLM",
            "added": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        path = _learned_cache_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(path.parent),
                delete=False,
                suffix=".tmp",
            ) as tmp:
                json.dump(_LEARNED_CACHE, tmp, ensure_ascii=False, indent=2)
                tmp_path = tmp.name
            os.replace(tmp_path, path)
        except OSError as exc:
            logger.warning("Failed to persist learned shortening to %s: %s", path, exc)


def _learned_lookup(source_text: str, baseline: str) -> str | None:
    if not _learning_enabled():
        return None
    _load_learned_cache()
    entry = _LEARNED_CACHE.get(_cache_key(source_text, baseline))
    return entry["shortest"] if entry else None


# ── Stage 1: rule-based pre-pass ──────────────────────────────────────────
def _looks_spanish(text: str) -> bool:
    if any(ch in _SPANISH_HINT_CHARS for ch in text):
        return True
    words = re.findall(r"\b\w+\b", text.lower())
    return any(w in _SPANISH_HINT_WORDS for w in words)


def _normalize_punctuation(text: str) -> str:
    out = text.replace("…", "")
    out = re.sub(r"\.{3,}", "", out)
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s*,\s*\.", ".", out)
    return out.strip()


def _apply_spanish_contractions(text: str) -> str:
    out = text
    for long_form, short_form in _SPANISH_CONTRACTIONS:
        pattern = re.compile(re.escape(long_form), re.IGNORECASE)

        def _sub(match: re.Match[str]) -> str:
            matched = match.group(0)
            if matched and matched[0].isupper():
                return short_form[:1].upper() + short_form[1:]
            return short_form

        out = pattern.sub(_sub, out)
    return out


def _apply_rules(baseline: str) -> list[tuple[str, str]]:
    """Return a list of (text, rationale) intermediates produced by the rules.

    The list contains only stages whose output strictly differs from the
    baseline.  Stages are applied cumulatively.
    """
    stages: list[tuple[str, str]] = []
    current = baseline

    normalized = _normalize_punctuation(current)
    if normalized and normalized != current:
        stages.append((normalized, "normalized whitespace"))
        current = normalized

    if _looks_spanish(current):
        contracted = _apply_spanish_contractions(current)
        if contracted and contracted != current:
            stages.append((contracted, "contracted Spanish phrases"))
            current = contracted

    return stages


# ── Stage 2: OpenRouter LLM fallback ──────────────────────────────────────
_LLM_SYSTEM_PROMPT = (
    "You shorten translations so they fit a tight character budget for "
    "speech dubbing. You preserve the original meaning, tone, and target "
    "language. You return ONLY a JSON array of strings — no prose, no code "
    "fences, no commentary. Each string is a valid alternative translation."
)


def _build_user_prompt(
    source_text: str,
    baseline: str,
    target_chars: int,
    context_prev: str,
    context_next: str,
) -> str:
    return (
        f"Source (English): {source_text}\n"
        f"Current translation: {baseline}\n"
        f"Previous segment: {context_prev or '(none)'}\n"
        f"Next segment: {context_next or '(none)'}\n"
        f"Target length: <= {target_chars} characters.\n\n"
        "Return a JSON array of 3 alternative translations in the same "
        "target language as the current translation, sorted shortest first. "
        "Each must preserve meaning. Output JSON only."
    )


def _parse_llm_json(content: str) -> list[str]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("expected JSON list")
    return [str(item) for item in parsed if isinstance(item, (str, int, float))]


@functools.lru_cache(maxsize=512)
def _call_openrouter_cached(
    source_text: str,
    baseline: str,
    target_chars: int,
    context_prev: str,
    context_next: str,
) -> tuple[str, ...]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        global _NO_KEY_WARNED
        if not _NO_KEY_WARNED:
            logger.warning(
                "OPENROUTER_API_KEY not set — skipping LLM re-ranking stage."
            )
            _NO_KEY_WARNED = True
        return ()

    model = os.environ.get("OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/CS-6613-AI/foreign-whispers",
        "X-Title": "foreign-whispers",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(
                    source_text, baseline, target_chars, context_prev, context_next
                ),
            },
        ],
        "temperature": 0.3,
    }

    try:
        response = requests.post(
            OPENROUTER_URL, headers=headers, json=body, timeout=OPENROUTER_TIMEOUT_S
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        return tuple(_parse_llm_json(content))
    except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("OpenRouter LLM call failed: %s", exc)
        return ()


# ── Public entry point ────────────────────────────────────────────────────
def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    Hybrid re-ranker with three stages:

    1. **Learned-cache lookup**: if this exact ``(source_text, baseline)``
       pair has been shortened before, the cached result is included as a
       candidate and the LLM stage is skipped when it already fits.
    2. **Rule-based pre-pass**: deterministic whitespace normalization and
       a small Spanish phrase-contraction table, applied cumulatively.
    3. **OpenRouter LLM fallback**: only invoked when the rule-based output
       is still over budget *and* ``OPENROUTER_API_KEY`` is set. Successful
       outputs are written back to the learned cache.

    Args:
        source_text: Original source-language segment text.
        baseline_es: Baseline target-language translation (from argostranslate).
            The ``_es`` suffix is historical; the function works for any
            target language.
        target_duration_s: Time budget in seconds for this segment.
        context_prev: Text of the preceding segment (for LLM coherence).
        context_next: Text of the following segment (for LLM coherence).

    Returns:
        List of ``TranslationCandidate`` sorted shortest first.  The
        baseline is always included as the final fallback.  Caller selects
        the candidate whose ``len(text) / 15.0`` is closest to
        ``target_duration_s``.
    """
    target_chars = max(1, int(target_duration_s * CHARS_PER_SECOND))

    rationale_priority = {
        "learned (cached LLM output)": 0,
        "LLM (rank 1)": 1,
        "LLM (rank 2)": 2,
        "LLM (rank 3)": 3,
        "contracted Spanish phrases": 4,
        "normalized whitespace": 5,
        "baseline": 6,
    }
    collected: dict[str, str] = {}

    def _add(text: str, rationale: str) -> None:
        text = text.strip()
        if not text:
            return
        existing = collected.get(text)
        if existing is None or rationale_priority.get(
            rationale, 99
        ) < rationale_priority.get(existing, 99):
            collected[text] = rationale

    learned = _learned_lookup(source_text, baseline_es)
    if learned:
        _add(learned, "learned (cached LLM output)")

    learned_fits = learned is not None and len(learned) <= target_chars
    baseline_fits = len(baseline_es) <= target_chars

    rule_intermediates: list[tuple[str, str]] = []
    rule_fits = False
    if not learned_fits and not baseline_fits:
        rule_intermediates = _apply_rules(baseline_es)
        for text, rationale in rule_intermediates:
            _add(text, rationale)
        if rule_intermediates and len(rule_intermediates[-1][0]) <= target_chars:
            rule_fits = True

    needs_llm = (
        not learned_fits
        and not baseline_fits
        and not rule_fits
    )

    if needs_llm:
        llm_outputs = _call_openrouter_cached(
            source_text, baseline_es, target_chars, context_prev, context_next
        )
        if llm_outputs:
            shortest_llm = min(llm_outputs, key=len)
            _save_learned_entry(source_text, baseline_es, shortest_llm)
            for i, text in enumerate(llm_outputs, start=1):
                _add(text, f"LLM (rank {i})" if i <= 3 else "LLM (rank 3)")

    _add(baseline_es, "baseline")

    candidates = [
        TranslationCandidate(text=text, char_count=len(text), brevity_rationale=rationale)
        for text, rationale in collected.items()
    ]
    candidates.sort(key=lambda c: (c.char_count, c.text))

    logger.info(
        "get_shorter_translations: baseline=%d chars, target=%d chars, "
        "returned %d candidates (shortest=%d)",
        len(baseline_es),
        target_chars,
        len(candidates),
        candidates[0].char_count if candidates else 0,
    )
    return candidates
