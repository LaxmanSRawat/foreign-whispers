"""POST /api/diarize/{video_id} — speaker diarization (issue fw-lua)."""

import json
import subprocess

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.schemas.diarize import DiarizeResponse
from api.src.services.alignment_service import AlignmentService
from foreign_whispers.diarization import assign_speakers

router = APIRouter(prefix="/api")

_alignment_service = AlignmentService(settings=settings)


def _merge_labels_into_transcript(title: str, diar_segments: list[dict]) -> None:
    """Idempotently merge speaker labels into the cached transcription and translation JSONs.

    Runs whether the diarization is fresh or cached so that re-transcribing
    a video (or hitting the diarize cache after a code change that adds the
    merge) still produces a labelled transcript on disk.

    Both the transcription and translation JSONs share the same segment
    timestamps, so ``assign_speakers`` (overlap-based matching) works for both.
    The TTS engine reads from the translation JSON, so labels must be present
    there for per-speaker voice selection to work.
    """
    for json_path in [
        settings.transcriptions_dir / f"{title}.json",
        settings.translations_dir / f"{title}.json",
    ]:
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text())
        data["segments"] = assign_speakers(data.get("segments", []), diar_segments)
        json_path.write_text(json.dumps(data))


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str):
    """Run speaker diarization on a video's audio track.

    Steps:
    1. Extract audio from video via ffmpeg
    2. Run pyannote diarization
    3. Cache and merge speaker labels into the transcription
    """
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    diar_dir = settings.diarizations_dir
    diar_dir.mkdir(parents=True, exist_ok=True)
    diar_path = diar_dir / f"{title}.json"

    if diar_path.exists():
        data = json.loads(diar_path.read_text())
        speakers = data.get("speakers", [])
        diar_segments = data.get("segments", [])
        skipped = True
    else:
        video_path = settings.videos_dir / f"{title}.mp4"
        audio_path = diar_dir / f"{title}.wav"
        subprocess.run(
            ["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-y", str(audio_path)],
            check=True,
        )
        diar_segments = _alignment_service.diarize(str(audio_path))
        speakers = sorted({s["speaker"] for s in diar_segments})
        diar_path.write_text(json.dumps({"speakers": speakers, "segments": diar_segments}))
        skipped = False

    # Merge happens on every call — even cache hits — so labels reach the
    # transcript whenever it has been regenerated since the last diarize.
    _merge_labels_into_transcript(title, diar_segments)

    return DiarizeResponse(
        video_id=video_id,
        speakers=speakers,
        segments=diar_segments,
        skipped=skipped,
    )

