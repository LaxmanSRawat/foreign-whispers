import json
import pathlib
import pytest
from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    _estimate_duration,
    _estimate_duration_baseline,
    compute_segment_metrics,
    decide_action,
    global_align,
    global_align_dp,
)
from foreign_whispers.evaluation import clip_evaluation_report


def _make_metrics(src_dur: float, tgt_chars: int) -> SegmentMetrics:
    return SegmentMetrics(
        index=0,
        source_start=0.0,
        source_end=src_dur,
        source_duration_s=src_dur,
        source_text="x" * 10,
        translated_text="ba" * tgt_chars,  # tgt_chars vowel clusters → tgt_chars syllables
        src_char_count=10,
        tgt_char_count=tgt_chars,
    )


def test_syllable_count_simple():
    # "hola mundo" → ho-la-mun-do = 4 syllables
    from foreign_whispers.alignment import _count_syllables
    assert _count_syllables("hola mundo") == 4


def test_syllable_count_accents():
    # "cómo están" → có-mo-es-tán = 4 syllables
    from foreign_whispers.alignment import _count_syllables
    assert _count_syllables("cómo están") == 4


def test_syllable_count_empty_string():
    from foreign_whispers.alignment import _count_syllables
    assert _count_syllables("") == 1  # floor prevents zero-division in predicted_tts_s


def test_syllable_count_punctuation_only():
    from foreign_whispers.alignment import _count_syllables
    assert _count_syllables("...") == 1  # no vowels → floor returns 1


def test_syllable_count_consonants_only():
    from foreign_whispers.alignment import _count_syllables
    assert _count_syllables("grr") == 1  # no vowels → floor returns 1


def test_segment_metrics_predicted_tts_matches_estimate():
    # predicted_tts_s should equal whatever _estimate_duration returns,
    # whichever heuristic is currently active.
    m = SegmentMetrics(
        index=0, source_start=0.0, source_end=2.0, source_duration_s=2.0,
        source_text="hello world", translated_text="hola mundo",
        src_char_count=11, tgt_char_count=10,
    )
    assert m.predicted_tts_s == pytest.approx(_estimate_duration("hola mundo"), rel=0.01)


def test_segment_metrics_predicted_stretch():
    m = _make_metrics(src_dur=2.0, tgt_chars=30)
    expected = _estimate_duration("ba" * 30) / 2.0
    assert m.predicted_stretch == pytest.approx(expected, rel=0.01)


def test_segment_metrics_overflow():
    m = _make_metrics(src_dur=2.0, tgt_chars=60)
    expected = max(0.0, _estimate_duration("ba" * 60) - 2.0)
    assert m.overflow_s == pytest.approx(expected, rel=0.01)


# Action thresholds in decide_action are stretch_factor based:
#   <=1.1 ACCEPT, <=1.4 MILD_STRETCH, <=1.8 GAP_SHIFT, <=2.5 REQUEST_SHORTER, else FAIL.
# These tests pick tgt_chars so the *current* predictor lands the segment
# in the intended band. If you change _estimate_duration, re-derive these
# from the band edges using ``_estimate_duration("ba" * N)``.

def test_decide_action_accept():
    # tgt_chars=14 → predicted ≈ 1.89s in 3.0s window → stretch ≈ 0.63 → ACCEPT
    assert decide_action(_make_metrics(3.0, 14)) == AlignAction.ACCEPT


def test_decide_action_mild_stretch():
    # tgt_chars=30 → predicted ≈ 3.74s in 3.0s window → stretch ≈ 1.25 → MILD_STRETCH
    assert decide_action(_make_metrics(3.0, 30)) == AlignAction.MILD_STRETCH


def test_decide_action_gap_shift():
    # tgt_chars=40 → predicted ≈ 4.89s in 3.0s window → stretch ≈ 1.63 → GAP_SHIFT (with gap)
    m = _make_metrics(3.0, 40)
    assert decide_action(m, available_gap_s=3.0) == AlignAction.GAP_SHIFT


def test_decide_action_request_shorter():
    # tgt_chars=55 → predicted ≈ 6.62s in 3.0s window → stretch ≈ 2.21 → REQUEST_SHORTER
    assert decide_action(_make_metrics(3.0, 55)) == AlignAction.REQUEST_SHORTER


def test_decide_action_fail():
    # tgt_chars=70 → predicted ≈ 8.34s in 3.0s window → stretch ≈ 2.78 → FAIL
    assert decide_action(_make_metrics(3.0, 70)) == AlignAction.FAIL


def test_compute_segment_metrics_length():
    en = {"segments": [
        {"start": 0.0, "end": 3.0, "text": " Hello world"},
        {"start": 3.0, "end": 6.0, "text": " How are you"},
    ]}
    es = {"segments": [
        {"start": 0.0, "end": 3.0, "text": " Hola mundo"},
        {"start": 3.0, "end": 6.0, "text": " Como estas"},
    ]}
    metrics = compute_segment_metrics(en, es)
    assert len(metrics) == 2
    assert metrics[0].index == 0
    assert metrics[1].index == 1


def test_compute_segment_metrics_text_stripped():
    en = {"segments": [{"start": 0.0, "end": 2.0, "text": "  hi  "}]}
    es = {"segments": [{"start": 0.0, "end": 2.0, "text": "  hola  "}]}
    m = compute_segment_metrics(en, es)[0]
    assert m.source_text == "hi"
    assert m.translated_text == "hola"


def test_global_align_accept_no_drift():
    en = {"segments": [{"start": 0.0, "end": 3.0, "text": "Hello"}]}
    es = {"segments": [{"start": 0.0, "end": 3.0, "text": "Hola"}]}
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=[])
    assert aligned[0].scheduled_start == pytest.approx(0.0)
    assert aligned[0].action == AlignAction.ACCEPT


def test_estimate_duration_beats_syllable_rate_baseline():
    """The fitted predictor should be at least as accurate as the legacy
    ``syllables / 4.5`` heuristic on real Chatterbox-CPU output.

    Fixture: 8 (text, observed_duration_s) pairs sampled from
    ``pipeline_data/api/tts_audio/chatterbox/c-fb1074a/...align.json``.
    Mean absolute error must drop relative to the syllable-rate baseline.
    """
    pairs = [
        ("60 minutos horas extra.",                           1.78),
        ("¿Cuál es el peor escenario?",                       1.46),
        ("Te preocupa que sea",                               1.30),
        ("cerrado durante semanas y semanas y semanas",       2.38),
        ("la economía realmente se ve afectada porque",       2.42),
        ("es la sangre de la vida a cierto",                  1.82),
        ("extensión. Así que la realidad es más larga",       2.58),
        ("que esto continúa, el mayor impacto",               2.26),
    ]
    baseline_err = sum(abs(_estimate_duration_baseline(t) - obs) for t, obs in pairs) / len(pairs)
    fitted_err   = sum(abs(_estimate_duration(t)          - obs) for t, obs in pairs) / len(pairs)

    assert fitted_err < baseline_err, (
        f"fitted predictor MAE ({fitted_err:.3f}s) should be lower than "
        f"syllable-rate MAE ({baseline_err:.3f}s)"
    )
    # Also assert a meaningful absolute bound — the regression on this dataset
    # achieves ~0.19s; allow some slack for future re-fits.
    assert fitted_err < 0.30, f"fitted MAE {fitted_err:.3f}s exceeds 0.30s budget"


def test_global_align_gap_shift_accumulates_drift():
    en = {"segments": [
        {"start": 0.0, "end": 1.0, "text": "x"},
        {"start": 2.0, "end": 4.0, "text": "x"},
    ]}
    es = {"segments": [
        # tgt_chars=12 → predicted ≈ 1.67s in 1.0s window → stretch ≈ 1.67 → GAP_SHIFT
        {"start": 0.0, "end": 1.0, "text": "ba" * 12},
        # tgt_chars=4 → predicted ≈ 0.75s in 2.0s window → stretch ≈ 0.38 → ACCEPT
        {"start": 2.0, "end": 4.0, "text": "ba" * 4},
    ]}
    silence = [{"start_s": 1.0, "end_s": 3.0, "label": "silence"}]
    metrics = compute_segment_metrics(en, es)
    aligned = global_align(metrics, silence_regions=silence)
    assert aligned[0].action == AlignAction.GAP_SHIFT
    assert aligned[1].scheduled_start > aligned[1].original_start


# ───────────────────────── global_align_dp ──────────────────────────

def test_global_align_dp_matches_greedy_when_all_segments_accept():
    """When every segment fits its window comfortably, both schedulers
    should pick ACCEPT for every segment with zero drift."""
    en = {"segments": [
        {"start": 0.0, "end": 5.0, "text": "x"},
        {"start": 5.0, "end": 10.0, "text": "x"},
    ]}
    es = {"segments": [
        # Both well under 1.1× stretch.
        {"start": 0.0, "end": 5.0, "text": "ba" * 5},
        {"start": 5.0, "end": 10.0, "text": "ba" * 5},
    ]}
    metrics = compute_segment_metrics(en, es)
    greedy = global_align(metrics, silence_regions=[])
    dp = global_align_dp(metrics, silence_regions=[])
    assert [a.action for a in greedy] == [AlignAction.ACCEPT, AlignAction.ACCEPT]
    assert [a.action for a in dp]     == [AlignAction.ACCEPT, AlignAction.ACCEPT]
    # Schedules identical.
    for g, d in zip(greedy, dp):
        assert g.scheduled_start == pytest.approx(d.scheduled_start)
        assert g.scheduled_end   == pytest.approx(d.scheduled_end)


def test_global_align_dp_dominates_greedy_total_cost():
    """DP enumerates a superset of greedy's per-segment choices, so its
    chosen schedule's cost (under DP's own cost function) must be
    ≤ greedy's cost on the same input.  This is a structural property
    of the optimisation, not a per-axis claim."""
    # Over-budget scenario where greedy's GAP_SHIFT is feasible
    # but accepting the small overflow may be cheaper.
    en = {"segments": [
        {"start": 0.0, "end": 1.0, "text": "x"},  # tight window
        {"start": 2.0, "end": 5.0, "text": "x"},
    ]}
    es = {"segments": [
        {"start": 0.0, "end": 1.0, "text": "ba" * 12},  # GAP_SHIFT-able
        {"start": 2.0, "end": 5.0, "text": "ba" * 4},   # ACCEPT
    ]}
    silence = [{"start_s": 1.0, "end_s": 2.0, "label": "silence"}]
    metrics = compute_segment_metrics(en, es)

    greedy = global_align(metrics, silence_regions=silence)
    dp = global_align_dp(metrics, silence_regions=silence)

    def _total_overflow(aligned):
        return sum(max(0.0, m.predicted_tts_s - (a.scheduled_end - a.scheduled_start))
                   for m, a in zip(metrics, aligned))

    # DP must not produce a schedule whose total overflow is worse than greedy.
    assert _total_overflow(dp) <= _total_overflow(greedy) + 1e-6


def test_global_align_dp_real_clip_no_regression():
    """End-to-end smoke test on the cached translation JSON.

    Loads the project's actual Spanish translation, runs both schedulers,
    and asserts DP's clip_evaluation_report doesn't regress on the
    bottom-line drift number.  Skipped when the data isn't on disk so
    CI on a fresh checkout still passes.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    en_p = repo_root / "pipeline_data/api/transcriptions/whisper" / \
        "Strait of Hormuz disruption threatens to shake global economy.json"
    es_p = repo_root / "pipeline_data/api/translations/argos" / \
        "Strait of Hormuz disruption threatens to shake global economy.json"
    if not (en_p.exists() and es_p.exists()):
        pytest.skip("translation/transcription artifacts not on disk")

    en = json.loads(en_p.read_text())
    es = json.loads(es_p.read_text())
    metrics = compute_segment_metrics(en, es)

    greedy_aligned = global_align(metrics, silence_regions=[])
    dp_aligned     = global_align_dp(metrics, silence_regions=[])

    greedy_report = clip_evaluation_report(metrics, greedy_aligned)
    dp_report     = clip_evaluation_report(metrics, dp_aligned)

    # DP's drift must be ≤ greedy's. Other axes (severe stretches,
    # gap shifts) we don't pin individually because the cost weights
    # may trade them against each other.
    assert dp_report["total_cumulative_drift_s"] <= \
        greedy_report["total_cumulative_drift_s"] + 1e-6, \
        f"DP regressed on drift: {dp_report} vs greedy {greedy_report}"
