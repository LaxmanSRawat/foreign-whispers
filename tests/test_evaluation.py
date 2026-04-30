# tests/test_evaluation.py
import pytest

from foreign_whispers.alignment import compute_segment_metrics, global_align
from foreign_whispers.evaluation import (
    _word_error_rate,
    clip_evaluation_report,
    dubbing_scorecard,
)


def _make_transcripts(src_dur=3.0, tgt_chars=30):
    en = {"segments": [{"start": 0.0, "end": src_dur, "text": "Hello world"}]}
    es = {"segments": [{"start": 0.0, "end": src_dur, "text": "x" * tgt_chars}]}
    return en, es


def test_report_keys():
    en, es = _make_transcripts()
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    report = clip_evaluation_report(metrics, aligned)
    assert set(report.keys()) == {
        "mean_abs_duration_error_s",
        "pct_severe_stretch",
        "n_gap_shifts",
        "n_translation_retries",
        "total_cumulative_drift_s",
    }


def test_report_no_issues_for_easy_segment():
    en, es = _make_transcripts(src_dur=3.0, tgt_chars=15)  # 1s predicted, 3s budget
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    report = clip_evaluation_report(metrics, aligned)
    assert report["n_gap_shifts"] == 0
    assert report["n_translation_retries"] == 0
    assert report["total_cumulative_drift_s"] == 0.0


def test_report_counts_retries_for_hard_segment():
    # tgt_chars=16 ("ba" * 16) → predicted ≈ 2.13s in 1.0s window → stretch ≈ 2.13 → REQUEST_SHORTER
    en = {"segments": [{"start": 0.0, "end": 1.0, "text": "Hello world"}]}
    es = {"segments": [{"start": 0.0, "end": 1.0, "text": "ba" * 16}]}
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    report = clip_evaluation_report(metrics, aligned)
    assert report["n_translation_retries"] == 1


def test_report_empty_inputs():
    report = clip_evaluation_report([], [])
    assert report["mean_abs_duration_error_s"] == 0.0
    assert report["n_gap_shifts"] == 0


# ───────────────────────── dubbing_scorecard ─────────────────────────

def _scorecard_inputs(tgt_chars: int = 15, src_dur: float = 3.0):
    en, es = _make_transcripts(src_dur=src_dur, tgt_chars=tgt_chars)
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    report = clip_evaluation_report(metrics, aligned)
    return metrics, aligned, report


def test_dubbing_scorecard_timing_only_when_services_absent():
    """With no services injected, only timing + naturalness are reported."""
    metrics, aligned, report = _scorecard_inputs()
    sc = dubbing_scorecard(metrics, aligned, report)
    assert "timing_score" in sc
    assert "naturalness_score" in sc
    assert "intelligibility_score" not in sc
    assert "semantic_score" not in sc
    assert 0.0 <= sc["overall_score"] <= 1.0


def test_dubbing_scorecard_full_with_mocks():
    """All four dimensions populate when services are injected."""
    metrics, aligned, report = _scorecard_inputs()

    class MockWhisper:
        def transcribe(self, wav_path: str, language: str = "es") -> str:
            # Recover the translated text exactly → WER = 0 → score 1.
            return " ".join(m.translated_text for m in metrics)

    class MockBackTranslator:
        def translate(self, text: str, src: str = "es", dst: str = "en") -> str:
            return "hello world"  # matches the source EN we constructed

    class MockEmbedder:
        def encode(self, texts):
            # Return identical vectors so cosine = 1.
            return [[1.0, 0.0, 0.0] for _ in texts]

    sc = dubbing_scorecard(
        metrics, aligned, report,
        audio_path="/tmp/fake.wav",
        whisper=MockWhisper(),
        back_translator=MockBackTranslator(),
        embedder=MockEmbedder(),
    )
    assert set(sc.keys()) == {
        "timing_score", "naturalness_score",
        "intelligibility_score", "semantic_score",
        "overall_score",
    }
    assert sc["intelligibility_score"] == pytest.approx(1.0, abs=0.01)
    assert sc["semantic_score"]        == pytest.approx(1.0, abs=0.01)


def test_dubbing_scorecard_partial_services():
    """Whisper alone provides 3 dimensions; embedder alone needs back_translator too."""
    metrics, aligned, report = _scorecard_inputs()

    class MockWhisper:
        def transcribe(self, wav_path, language="es"):
            return ""  # nothing recovered → WER = 1 → intelligibility 0

    sc = dubbing_scorecard(
        metrics, aligned, report,
        audio_path="/tmp/fake.wav",
        whisper=MockWhisper(),
    )
    assert "intelligibility_score" in sc
    assert sc["intelligibility_score"] == 0.0
    assert "semantic_score" not in sc  # back_translator+embedder both required


def test_word_error_rate_basics():
    assert _word_error_rate("a b c", "a b c") == 0.0
    assert _word_error_rate("a b c", "a b d") == pytest.approx(1 / 3)
    assert _word_error_rate("", "")           == 0.0
    assert _word_error_rate("",  "x")          == 1.0
    assert _word_error_rate("a", "")           == 1.0
