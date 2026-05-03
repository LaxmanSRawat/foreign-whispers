"""Voice resolution for Chatterbox speaker cloning.

Resolves which reference WAV to use for a given target language
and optional speaker ID. The Chatterbox container expects a filename
relative to its /app/voices/ mount point.
"""

import re
from pathlib import Path


def _list_speaker_wavs(lang_dir: Path) -> list[str]:
    """Return sorted non-default WAV filenames in a language directory.

    Excludes ``default.wav`` — those are fallback voices, not speaker-specific
    reference files available for round-robin assignment.
    """
    if not lang_dir.is_dir():
        return []
    return sorted(
        f.name for f in lang_dir.glob("*.wav") if f.name != "default.wav"
    )


def _extract_speaker_index(speaker_id: str) -> int:
    """Extract the numeric index from a pyannote speaker label.

    ``SPEAKER_00`` → 0, ``SPEAKER_12`` → 12.  Returns 0 if the label
    doesn't end with a number (defensive fallback).
    """
    match = re.search(r"(\d+)$", speaker_id)
    return int(match.group(1)) if match else 0


def assign_speaker_voices(
    speakers_dir: Path,
    target_language: str,
    speaker_ids: list[str | None],
) -> dict[str | None, str]:
    """Return {speaker_id: wav_path} with no two speakers sharing the same WAV.

    Speakers are sorted alphabetically, then assigned round-robin over the
    sorted voice file list by *position* (not by numeric ID), so the first
    distinct speaker always gets voice[0], the second voice[1], etc.
    Wrapping only occurs when #speakers > #voices — guaranteed unique otherwise.
    Falls back per-speaker to resolve_speaker_wav() when no voice files exist.
    """
    lang_dir = Path(speakers_dir) / target_language
    voice_files = _list_speaker_wavs(lang_dir)

    result: dict[str | None, str] = {}
    named = sorted(s for s in speaker_ids if s is not None)
    for i, spk in enumerate(named):
        if voice_files:
            result[spk] = f"{target_language}/{voice_files[i % len(voice_files)]}"
        else:
            result[spk] = resolve_speaker_wav(Path(speakers_dir), target_language, spk)

    if None in speaker_ids:
        result[None] = resolve_speaker_wav(Path(speakers_dir), target_language, None)

    return result


def resolve_speaker_wav(
    speakers_dir: Path,
    target_language: str,
    speaker_id: str | None = None,
) -> str:
    """Resolve the reference WAV path for voice cloning.

    **Strategy — round-robin with fallback:**

    1. If ``speakers/{lang}/`` contains multiple non-default WAV files
       **and** a ``speaker_id`` is provided, assign voices via round-robin
       over those files (sorted alphabetically, indexed by the speaker
       number extracted from the label).
    2. Otherwise fall back to ``speakers/{lang}/default.wav`` if it exists.
    3. Otherwise fall back to ``speakers/default.wav`` (global default).

    Args:
        speakers_dir: Absolute path to the speakers directory.
        target_language: Language code (e.g. "es", "fr").
        speaker_id: Optional speaker identifier (e.g. "SPEAKER_00").

    Returns:
        Relative path string for the Chatterbox container (e.g. "es/clf_09697_02130564644.wav").
    """
    # ---- YOUR CODE HERE ----
    speakers_dir = Path(speakers_dir)
    lang_dir = speakers_dir / target_language

    # Round-robin: when the language directory has speaker WAV files
    # (anything besides default.wav), cycle through them.
    if speaker_id:
        voice_files = _list_speaker_wavs(lang_dir)
        if voice_files:
            idx = _extract_speaker_index(speaker_id) % len(voice_files)
            return f"{target_language}/{voice_files[idx]}"

    # Fallback 1: language-specific default
    lang_default = lang_dir / "default.wav"
    if lang_default.is_file():
        return f"{target_language}/default.wav"

    # Fallback 2: global default
    return "default.wav"
    # ---- END YOUR CODE ----
