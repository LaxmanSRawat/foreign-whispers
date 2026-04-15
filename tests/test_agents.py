# tests/test_agents.py — renamed module is now foreign_whispers.reranking
import json
from unittest.mock import MagicMock

import pytest
import requests

from foreign_whispers import reranking
from foreign_whispers.reranking import (
    FailureAnalysis,
    TranslationCandidate,
    analyze_failures,
    get_shorter_translations,
)


@pytest.fixture(autouse=True)
def _isolate_reranker_state(tmp_path, monkeypatch):
    """Reset module-level cache state and redirect persistence to tmp_path."""
    monkeypatch.setenv("LEARNED_SHORTENINGS_PATH", str(tmp_path / "learned.json"))
    monkeypatch.setenv("OPENROUTER_LEARN", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reranking._LEARNED_CACHE.clear()
    reranking._CACHE_LOADED = False
    reranking._NO_KEY_WARNED = False
    reranking._call_openrouter_cached.cache_clear()
    yield
    reranking._LEARNED_CACHE.clear()
    reranking._CACHE_LOADED = False
    reranking._call_openrouter_cached.cache_clear()


# ── analyze_failures ──────────────────────────────────────────────────────
def test_analyze_failures_returns_dataclass():
    result = analyze_failures({"mean_abs_duration_error_s": 0.5})
    assert isinstance(result, FailureAnalysis)
    assert result.failure_category == "ok"


def test_analyze_failures_detects_overflow():
    result = analyze_failures({"pct_severe_stretch": 30})
    assert result.failure_category == "duration_overflow"


def test_analyze_failures_detects_drift():
    result = analyze_failures({"total_cumulative_drift_s": 5.0})
    assert result.failure_category == "cumulative_drift"


def test_analyze_failures_detects_stretch_quality():
    result = analyze_failures({"mean_abs_duration_error_s": 1.2})
    assert result.failure_category == "stretch_quality"


# ── get_shorter_translations ──────────────────────────────────────────────
def _mock_llm_response(strings: list[str]) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "choices": [{"message": {"content": json.dumps(strings)}}]
    }
    return response


def test_returns_baseline_when_already_fits():
    """Short baseline within budget → returns just the baseline."""
    result = get_shorter_translations("hello", "hola", target_duration_s=10.0)
    assert len(result) == 1
    assert result[0].text == "hola"
    assert result[0].brevity_rationale == "baseline"
    assert result[0].char_count == 4


def test_returns_shortest_first_sorted(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        requests,
        "post",
        MagicMock(return_value=_mock_llm_response(["aa", "bbbb", "cccccc"])),
    )
    result = get_shorter_translations(
        "long source text here",
        "Esta es una traducción mucho más larga de lo que el presupuesto permite",
        target_duration_s=0.5,
    )
    char_counts = [c.char_count for c in result]
    assert char_counts == sorted(char_counts)


def test_phrase_contraction_fires():
    """Spanish baseline with a contractable phrase produces a shorter variant."""
    result = get_shorter_translations(
        "I am working at this moment",
        "En este momento estoy trabajando en el proyecto importante.",
        target_duration_s=1.0,
    )
    rationales = [c.brevity_rationale for c in result]
    assert "contracted Spanish phrases" in rationales
    contracted = next(
        c for c in result if c.brevity_rationale == "contracted Spanish phrases"
    )
    assert "ahora" in contracted.text.lower()
    assert "en este momento" not in contracted.text.lower()


def test_llm_skipped_without_api_key(monkeypatch):
    """No OPENROUTER_API_KEY → LLM stage is skipped, no exception."""
    post_mock = MagicMock()
    monkeypatch.setattr(requests, "post", post_mock)
    result = get_shorter_translations(
        "source",
        "Una traducción que sin embargo es demasiado larga para el presupuesto.",
        target_duration_s=0.5,
    )
    post_mock.assert_not_called()
    assert any(c.brevity_rationale == "baseline" for c in result)


def test_llm_called_when_rules_insufficient(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    post_mock = MagicMock(
        return_value=_mock_llm_response(["corto", "más corto", "muy muy corto"])
    )
    monkeypatch.setattr(requests, "post", post_mock)
    result = get_shorter_translations(
        "source text",
        "Una traducción extremadamente larga que necesita ser acortada.",
        target_duration_s=0.4,
    )
    post_mock.assert_called_once()
    texts = {c.text for c in result}
    assert "corto" in texts
    assert "más corto" in texts


def test_llm_failure_falls_back(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        requests, "post", MagicMock(side_effect=requests.Timeout("boom"))
    )
    result = get_shorter_translations(
        "source",
        "Una traducción demasiado larga para el presupuesto disponible.",
        target_duration_s=0.4,
    )
    assert any(c.brevity_rationale == "baseline" for c in result)
    assert all(isinstance(c, TranslationCandidate) for c in result)


def test_llm_malformed_json_falls_back(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    bad_response = MagicMock()
    bad_response.raise_for_status = MagicMock()
    bad_response.json.return_value = {
        "choices": [{"message": {"content": "this is not json at all"}}]
    }
    monkeypatch.setattr(requests, "post", MagicMock(return_value=bad_response))
    result = get_shorter_translations(
        "source",
        "Una traducción demasiado larga para el presupuesto disponible.",
        target_duration_s=0.4,
    )
    assert any(c.brevity_rationale == "baseline" for c in result)


def test_learned_cache_persists_llm_output(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    cache_path = tmp_path / "learned.json"

    post_mock = MagicMock(
        return_value=_mock_llm_response(["versión corta", "una versión más corta"])
    )
    monkeypatch.setattr(requests, "post", post_mock)

    src = "I need this translated"
    baseline = "Necesito que esto sea traducido en una forma muy extensa y larga."
    # Budget = 1.0s * 15 = 15 chars. Baseline (66 chars) won't fit → LLM runs.
    # The shortest LLM output "versión corta" (13 chars) fits the same budget,
    # so the second call should hit the cache and skip the LLM.
    result1 = get_shorter_translations(src, baseline, target_duration_s=1.0)
    assert post_mock.call_count == 1
    assert cache_path.exists()
    on_disk = json.loads(cache_path.read_text())
    assert any(entry["shortest"] == "versión corta" for entry in on_disk.values())

    # Reset module state to simulate a fresh process; cache should still hit.
    reranking._LEARNED_CACHE.clear()
    reranking._CACHE_LOADED = False
    reranking._call_openrouter_cached.cache_clear()
    post_mock.reset_mock()

    result2 = get_shorter_translations(src, baseline, target_duration_s=1.0)
    post_mock.assert_not_called()
    rationales = [c.brevity_rationale for c in result2]
    assert "learned (cached LLM output)" in rationales
    cached = next(
        c for c in result2 if c.brevity_rationale == "learned (cached LLM output)"
    )
    assert cached.text == "versión corta"


def test_learned_cache_disabled_via_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_LEARN", "0")
    cache_path = tmp_path / "learned.json"

    post_mock = MagicMock(return_value=_mock_llm_response(["corto"]))
    monkeypatch.setattr(requests, "post", post_mock)

    get_shorter_translations(
        "src",
        "Una traducción demasiado larga para el presupuesto.",
        target_duration_s=0.3,
    )
    assert not cache_path.exists()


def test_learned_cache_only_overwrites_when_strictly_shorter(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    cache_path = tmp_path / "learned.json"

    src = "source"
    baseline = "Una traducción larga que necesita acortarse mucho de verdad."

    # First call: LLM returns "abc" (length 3) — this becomes the cached value.
    monkeypatch.setattr(
        requests, "post", MagicMock(return_value=_mock_llm_response(["abc"]))
    )
    get_shorter_translations(src, baseline, target_duration_s=0.2)
    first_disk = json.loads(cache_path.read_text())
    cached_entries = [e for e in first_disk.values() if e["baseline"] == baseline]
    assert len(cached_entries) == 1
    assert cached_entries[0]["shortest"] == "abc"

    # Reset in-process caches so the LLM is called again with a longer answer.
    reranking._LEARNED_CACHE.clear()
    reranking._CACHE_LOADED = False
    reranking._call_openrouter_cached.cache_clear()

    monkeypatch.setattr(
        requests, "post", MagicMock(return_value=_mock_llm_response(["abcdef"]))
    )
    # Use a smaller budget so the cached "abc" doesn't fit and the LLM is called.
    get_shorter_translations(src, baseline, target_duration_s=0.05)
    second_disk = json.loads(cache_path.read_text())
    cached_entries = [e for e in second_disk.values() if e["baseline"] == baseline]
    assert len(cached_entries) == 1
    assert cached_entries[0]["shortest"] == "abc"  # unchanged
