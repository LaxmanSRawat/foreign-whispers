"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import math as _math
import statistics as _stats
from typing import Protocol

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


# ─────────────────── multi-dimensional scorecard ────────────────────


class WhisperService(Protocol):
    """STT service used for the intelligibility round-trip dimension."""

    def transcribe(self, wav_path: str, language: str = "es") -> str: ...


class BackTranslator(Protocol):
    """Reverse-direction translator for the semantic-fidelity dimension."""

    def translate(self, text: str, src: str = "es", dst: str = "en") -> str: ...


class Embedder(Protocol):
    """Sentence embedder for the semantic-fidelity dimension."""

    def encode(self, texts: list[str]):  # → sequence of float vectors
        ...


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level Levenshtein distance / reference length, in [0, 1+]."""
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m] / n


def _cosine_similarity(v1, v2) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(v1, v2))
    n1 = _math.sqrt(sum(float(a) * float(a) for a in v1))
    n2 = _math.sqrt(sum(float(b) * float(b) for b in v2))
    return dot / (n1 * n2) if n1 and n2 else 0.0


def dubbing_scorecard(
    metrics:         list[SegmentMetrics],
    aligned:         list[AlignedSegment],
    align_report:    dict,
    *,
    audio_path:      str | None       = None,
    whisper:         WhisperService | None   = None,
    back_translator: BackTranslator | None   = None,
    embedder:        Embedder | None         = None,
) -> dict:
    """Multi-dimensional dubbing quality scorecard.

    Returns a dict with up to four sub-scores plus an ``overall_score``,
    each in ``[0, 1]`` (1 = best).  Heavy services (Whisper STT,
    back-translator, sentence embedder) are injected as Protocols — when
    a service is absent the corresponding dimension is silently dropped
    so the function still returns a useful timing/naturalness baseline.

    Dimensions:

    - **timing_score** — derived from ``align_report``: high MAE or many
      severe stretches push the score down.
    - **naturalness_score** — speaking-rate consistency from the per-segment
      ``stretch_factor`` distribution.  High variance (audio jumping
      between fast and slow) is penalised.
    - **intelligibility_score** *(needs whisper + audio_path)* — TTS → STT
      round-trip word error rate against the translated transcript.
    - **semantic_score** *(needs back_translator + embedder)* — back-translate
      target-language text, embed it alongside the source-language text,
      report cosine similarity.

    The ``overall_score`` is the unweighted mean of every sub-score that
    was actually computed (so a 2-dim run averages two values, a 4-dim
    run averages four).
    """
    scores: dict[str, float] = {}

    # 1. timing — penalise *real* timing problems: severe stretches
    # (audio noticeably sped up / slowed down) and accumulated drift
    # (timeline pushed off original anchors). Don't include
    # ``mean_abs_duration_error_s`` here — it's a per-segment fitness
    # gap (predicted_tts vs source_window) that stays large even when
    # the scheduler does the right thing (e.g. plays the audio at
    # natural speed and pads with silence).
    pct_severe = float(align_report.get("pct_severe_stretch", 0.0))
    drift_s    = abs(float(align_report.get("total_cumulative_drift_s", 0.0)))
    timing = max(0.0, 1.0 - (pct_severe / 100.0) - (drift_s * 0.1))
    scores["timing_score"] = round(timing, 3)

    # 2. naturalness — variance of the realised stretch factors.
    if aligned:
        speeds = [a.stretch_factor for a in aligned]
        stdev = _stats.stdev(speeds) if len(speeds) > 1 else 0.0
        naturalness = max(0.0, 1.0 - stdev / 0.5)
    else:
        naturalness = 0.0
    scores["naturalness_score"] = round(naturalness, 3)

    # 3. intelligibility — TTS → STT round-trip WER.
    if whisper is not None and audio_path is not None:
        target = " ".join(m.translated_text for m in metrics).lower().strip()
        recovered = whisper.transcribe(audio_path, language="es").lower().strip()
        wer = _word_error_rate(target, recovered)
        scores["intelligibility_score"] = round(max(0.0, 1.0 - wer), 3)

    # 4. semantic — back-translate + embed + cosine.
    if back_translator is not None and embedder is not None:
        es_concat = " ".join(m.translated_text for m in metrics).strip()
        en_source = " ".join(m.source_text   for m in metrics).strip()
        en_back   = back_translator.translate(es_concat, src="es", dst="en")
        embeds    = embedder.encode([en_source, en_back])
        cos       = _cosine_similarity(embeds[0], embeds[1])
        scores["semantic_score"] = round(max(0.0, cos), 3)

    # Overall = mean of computed sub-scores.
    sub = [v for k, v in scores.items() if k != "overall_score"]
    scores["overall_score"] = round(sum(sub) / len(sub), 3) if sub else 0.0
    return scores
