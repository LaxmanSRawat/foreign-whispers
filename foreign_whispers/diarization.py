"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import dataclasses
import logging

logger = logging.getLogger(__name__)


def _shim_torchaudio_for_pyannote() -> None:
    """Restore torchaudio APIs that pyannote.audio 3.x expects.

    torchaudio >=2.10 removed several top-level helpers that pyannote.audio
    3.4.0 still references in ``pyannote/audio/core/io.py``:

    - ``AudioMetaData`` (type hint)
    - ``list_audio_backends()`` (called in ``Audio.__init__``)
    - ``info(path, backend=...)`` (called to read file metadata)

    This shim restores all three using ``soundfile`` (already a dependency)
    as the underlying backend.  ``torchaudio.load`` and
    ``torchaudio.functional.resample`` still exist in 2.11 and need no shim.
    """
    try:
        import torchaudio
    except ImportError:
        return

    @dataclasses.dataclass
    class AudioMetaData:
        sample_rate: int = 0
        num_frames: int = 0
        num_channels: int = 0
        bits_per_sample: int = 0
        encoding: str = ""

    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]

    if not hasattr(torchaudio, "info"):
        def _info(path, backend=None):  # noqa: ARG001 — backend kept for signature parity
            import soundfile as sf
            sf_info = sf.info(str(path))
            return AudioMetaData(
                sample_rate=int(sf_info.samplerate),
                num_frames=int(sf_info.frames),
                num_channels=int(sf_info.channels),
                bits_per_sample=0,
                encoding=str(sf_info.subtype or ""),
            )
        torchaudio.info = _info  # type: ignore[attr-defined]

    # torchaudio 2.10 retired the legacy sox/soundfile dispatcher and routes
    # ``torchaudio.load`` through ``load_with_torchcodec``, which raises at call
    # time unless the ``torchcodec`` package is installed.  pyannote.audio 3.4
    # just calls ``torchaudio.load(path)``, so we replace it with a soundfile
    # reader that returns the same ``(waveform, sample_rate)`` contract
    # pyannote expects (channels-first float32 tensor).
    _needs_load_shim = not getattr(
        getattr(torchaudio, "load", None), "_fw_soundfile_patched", False
    )
    if _needs_load_shim:
        import numpy as np
        import soundfile as sf
        import torch as _torch

        def _load(
            path,
            frame_offset: int = 0,
            num_frames: int = -1,
            normalize: bool = True,  # noqa: ARG001 — soundfile always returns float
            channels_first: bool = True,
            format=None,  # noqa: ARG001 — soundfile sniffs from header
            buffer_size=4096,  # noqa: ARG001
            backend=None,  # noqa: ARG001
        ):
            sf_frames = -1 if num_frames in (-1, None) else int(num_frames)
            data, sample_rate = sf.read(
                str(path),
                start=int(frame_offset),
                frames=sf_frames,
                dtype="float32",
                always_2d=True,
            )  # shape: (frames, channels)
            waveform = _torch.from_numpy(np.ascontiguousarray(data))
            if channels_first:
                waveform = waveform.transpose(0, 1).contiguous()
            return waveform, int(sample_rate)

        _load._fw_soundfile_patched = True  # type: ignore[attr-defined]
        torchaudio.load = _load  # type: ignore[assignment]
        torchaudio.load_with_torchcodec = _load  # type: ignore[attr-defined]

    # PyTorch 2.6 flipped torch.load(weights_only=True) by default.  pyannote
    # checkpoints contain custom pickled classes (TorchVersion, Specifications,
    # and more) that aren't in the default safe-globals allowlist.  Allowlisting
    # each one is whack-a-mole, so we patch torch.load to force
    # weights_only=False.  lightning_fabric (a pyannote dep) passes
    # weights_only=True *explicitly*, so setdefault is not enough — we must
    # overwrite the kwarg unconditionally.  The checkpoint comes from an
    # HTTPS-pinned HuggingFace model that the operator already trusts via their
    # HF token, so the extra trust surface is limited to the pyannote model.
    try:
        import torch
        if not getattr(torch.load, "_fw_weights_only_patched", False):
            _orig_load = torch.load

            def _load_with_legacy_default(*args, **kwargs):
                kwargs["weights_only"] = False
                return _orig_load(*args, **kwargs)

            _load_with_legacy_default._fw_weights_only_patched = True  # type: ignore[attr-defined]
            torch.load = _load_with_legacy_default  # type: ignore[assignment]
    except ImportError:
        pass


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    _shim_torchaudio_for_pyannote()
    try:
        from pyannote.audio import Pipeline
    except (ImportError, TypeError):
        logger.warning("pyannote.audio not installed — returning empty diarization.")
        return []

    try:
        pipeline    = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(audio_path)
        return [
            {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as exc:
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
        return []

def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
) -> list[dict]:
    """Assign a speaker label to each transcription segment.

    For each segment, finds the diarization interval with the greatest
    temporal overlap and copies its speaker label. If diarization is
    empty, all segments default to ``SPEAKER_00``.

    Args:
        segments: Whisper-style ``[{id, start, end, text, ...}]``.
        diarization: pyannote-style ``[{start_s, end_s, speaker}]``.

    Returns:
        New list of segment dicts, each with an added ``speaker`` key.
        Original list is not mutated.
    """
    if not diarization:
        logger.warning("Empty diarization — assigning SPEAKER_00 to all segments.")
        return [
            {**segment, "speaker": "SPEAKER_00"}
            for segment in segments
        ]

    assigned = []
    for segment in segments:
        best_overlap = 0.0
        best_speaker = "SPEAKER_00"
        for turn in diarization:
            overlap = max(0, min(segment["end"], turn["end_s"]) - max(segment["start"], turn["start_s"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        assigned.append({**segment, "speaker": best_speaker})
    return assigned
    