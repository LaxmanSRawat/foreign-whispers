"""POST /api/tts/{video_id} — TTS with audio-sync endpoint (issue 381)."""

import asyncio
import functools
import json
import pathlib

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.services.tts_service import TTSService
from foreign_whispers.voice_resolution import resolve_speaker_wav

router = APIRouter(prefix="/api")


async def _run_in_threadpool(executor, fn, *args, **kwargs):
    """Run a sync function in the default thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))


def _build_voice_map(trans_path: pathlib.Path, target_language: str) -> dict[str, str]:
    """Build a speaker → voice mapping from a diarized transcript.

    Reads the translated JSON, extracts unique ``speaker`` labels, and
    resolves each to a Chatterbox-relative WAV path via
    :func:`resolve_speaker_wav`.  Returns an empty dict when the
    transcript has no speaker labels (diarization didn't run).
    """
    if not trans_path.exists():
        return {}
    try:
        translated = json.loads(trans_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    segments = translated.get("segments", [])
    unique_speakers = sorted(
        {seg.get("speaker") for seg in segments if seg.get("speaker")}
    )
    if not unique_speakers:
        return {}
    return {
        spk: resolve_speaker_wav(settings.speakers_dir, target_language, spk)
        for spk in unique_speakers
    }


@router.post("/tts/{video_id}")
async def tts_endpoint(
    video_id: str,
    request: Request,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
    alignment: bool = Query(False),
    target_language: str = Query("es", pattern=r"^[a-z]{2}(-[A-Z]{2})?$"),
    speaker_wav: str | None = Query(
        None,
        description=(
            "Reference voice WAV path relative to the speakers directory "
            "(e.g. 'es/default.wav'). When omitted, voices are resolved "
            "automatically from diarization labels or the language default."
        ),
    ),
):
    """Generate TTS audio for a translated transcript.

    *config* is an opaque directory name for caching.
    *alignment* enables temporal alignment (clamped stretch).
    *target_language* picks the speaker-voice directory under
    ``pipeline_data/speakers/{target_language}/``.
    *speaker_wav* overrides automatic voice resolution with an explicit
    reference WAV path (relative to ``pipeline_data/speakers/``).
    """
    trans_dir = settings.translations_dir
    audio_dir = settings.tts_audio_dir / config
    audio_dir.mkdir(parents=True, exist_ok=True)

    svc = TTSService(
        ui_dir=settings.data_dir,
        tts_engine=None,
    )

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    wav_path = audio_dir / f"{title}.wav"

    if wav_path.exists():
        return {
            "video_id": video_id,
            "audio_path": str(wav_path),
            "config": config,
        }

    source_path = str(trans_dir / f"{title}.json")

    # ── Voice resolution ────────────────────────────────────────────
    # If the caller provided an explicit speaker_wav, use it as-is.
    # Otherwise, build a per-speaker voice map from diarization labels
    # (Task 4).  If no diarization ran, resolve the language default.
    if speaker_wav is None:
        voice_map = _build_voice_map(
            trans_dir / f"{title}.json", target_language
        )
        if not voice_map:
            # No diarization labels — fall back to language default
            speaker_wav = resolve_speaker_wav(
                settings.speakers_dir, target_language
            )

    await _run_in_threadpool(
        None,
        svc.text_file_to_speech,
        source_path,
        str(audio_dir),
        alignment=alignment,
        target_language=target_language,
        speaker_wav=speaker_wav,
    )

    return {
        "video_id": video_id,
        "audio_path": str(wav_path),
        "config": config,
    }


@router.get("/audio/{video_id}")
async def get_audio(
    video_id: str,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
):
    """Stream the TTS-synthesized WAV audio."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    audio_path = settings.tts_audio_dir / config / f"{title}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(str(audio_path), media_type="audio/wav")
