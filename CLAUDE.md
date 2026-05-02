# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**Foreign Whispers** — a pipeline that accepts YouTube videos and outputs the video with spoken and written subtitles in a target language. The pipeline covers:

1. Download video + closed captions from YouTube
2. Speech-to-text via Whisper
3. Source → target language translation (offline, via `argostranslate`)
4. Translated text → speech via open-source TTS (Chatterbox)
5. Next.js frontend + FastAPI backend

```text
foreign-whispers/
├── api/src/                     # Layered FastAPI backend
│   ├── main.py                  # App factory (create_app), lifespan
│   ├── core/config.py           # Pydantic settings (env-driven)
│   ├── core/dependencies.py     # FastAPI Depends providers
│   ├── routers/                 # Route modules (videos, pipeline, align, …)
│   ├── schemas/                 # Pydantic request/response models
│   ├── services/                # Business logic — *_service.py (facade) + *_engine.py (heavy)
│   └── inference/               # Whisper/TTS backend abstraction (local|remote)
├── foreign_whispers/            # Alignment/evaluation library
├── frontend/                    # Next.js UI (port 8501 in Docker)
├── tests/                       # ~30 pytest modules (unit + integration)
├── notebooks/                   # 8 integration suites + end-to-end pipeline
├── docs/                        # tts-temporal-alignment-research.md
├── Makefile                     # Docker + notebook helpers
├── video_registry.yml           # Single source of truth for pipeline videos
├── pipeline_data/api/           # Runtime artifacts — model-namespaced dirs
└── docker-compose.yml           # All services (nvidia / cpu profiles)
```

## Running the App

**Always use Docker Compose — never launch with `uvicorn` or `next dev` directly.**

This host has an NVIDIA GPU; default to the `nvidia` profile. A `cpu` profile exists for hosts without a GPU.

```bash
docker compose --profile nvidia up -d        # GPU
docker compose --profile cpu up -d           # CPU-only fallback
```

Service URLs:

| Service                | URL                   |
| ---------------------- | --------------------- |
| Frontend (Next.js)     | http://localhost:8501 |
| API (FastAPI)          | http://localhost:8080 |
| STT (Whisper/speaches) | http://localhost:8000 |
| TTS (Chatterbox)       | http://localhost:8020 |

After changing Python source or `video_registry.yml`, rebuild the API image:

```bash
docker compose --profile nvidia build api
docker compose --profile nvidia up -d api
```

### Pre-pull the Whisper model (one-time)

The speaches container starts with `WHISPER__MODEL` set but does **not** auto-download — `/v1/audio/transcriptions` returns 422 (`Field required: model`) or `Model '…' is not installed locally` until the model is fetched. Pull it once after first `up`; the named volume `whisper-cache` keeps it across rebuilds and restarts:

```bash
# nvidia profile
curl -X POST "http://localhost:8000/v1/models/Systran/faster-whisper-medium"
# cpu profile
curl -X POST "http://localhost:8000/v1/models/Systran/faster-whisper-small"
```

Verify with `curl http://localhost:8000/v1/models`. Re-pull is only needed after `docker compose down -v` or `docker volume rm foreign-whispers_whisper-cache`. Any HTTP client that hits `/v1/audio/transcriptions` directly (e.g. the Task 4 `WhisperHTTPAdapter` in [notebooks/alignment_integration/alignment_integration.ipynb](notebooks/alignment_integration/alignment_integration.ipynb)) must include the `model` form field — speaches' OpenAPI schema lists it as required.

### Makefile shortcuts

`make help` lists everything. The most useful targets (defined in [Makefile](Makefile)):

```bash
make build           # Rebuild API image (NVIDIA profile)
make up              # Start all services (NVIDIA)
make cpu             # Start all services (CPU profile)
make down            # Stop all services
make logs            # Tail all service logs
make ps              # Show running containers
make rebuild         # build + up
make notebook        # Run the end-to-end pipeline notebook inside the API container
make notebook-quick  # Same, but skip heavy cells (44, 48)
make notebook-check  # Per-cell pass/fail report (allow_errors=True)
```

## Tests

Pytest, no mypy/pyright, no coverage gate. Run from the repo root with `uv` (Python 3.11):

```bash
uv run pytest                                          # full suite
uv run pytest tests/test_alignment.py -v               # one file
uv run pytest tests/test_alignment.py::test_name -v    # one test
uv run pytest -m "not requires_pyannote"               # skip diarization tests
```

Custom markers (declared in [pyproject.toml](pyproject.toml)):

- `requires_silero` — needs `silero-vad` and `torch` installed.
- `requires_pyannote` — needs `pyannote.audio` and `FW_HF_TOKEN` set.

For notebook smoke checks: `make notebook-check` runs every cell with `allow_errors=True` and prints a per-cell pass/fail report.

## Frontend Dev

From [frontend/](frontend/):

```bash
npm run dev      # next dev
npm run lint     # eslint
npm run build
npm run start
```

[frontend/next.config.ts](frontend/next.config.ts) rewrites `/api/:path*` to `${API_URL || "http://localhost:8080"}/api/:path*` with a 10-minute proxy timeout (TTS can be slow). In Docker the frontend reaches the API via the compose network; `API_URL` is set there.

## Environment Variables

### Documented in [.env.example](.env.example)

- `FW_HF_TOKEN` — **required** for pyannote diarization; pipeline still runs without it but diarization steps will fail.
- `FW_LOGFIRE_WRITE_TOKEN` — optional Logfire tracing.
- `UID` / `GID` — Docker-created file ownership; run `id -u && id -g` to find values.
- `FW_WHISPER_BACKEND` / `FW_TTS_BACKEND` — `"local"` (in-process) or `"remote"` (sidecar containers). Compose sets `"remote"` automatically.
- `FW_WHISPER_API_URL` / `FW_CHATTERBOX_API_URL` — sidecar URLs; only override for non-default hosts.

### Read at runtime but not in `.env.example`

These are real, code-read env vars worth knowing. Grep the source if in doubt.

- `CHATTERBOX_SPEAKER_WAV` — path under `pipeline_data/speakers/` to the reference WAV used for Chatterbox voice cloning. Empty disables cloning. Read in [api/src/services/tts_engine.py:20](api/src/services/tts_engine.py#L20). Resolution rules live in [foreign_whispers/voice_resolution.py](foreign_whispers/voice_resolution.py): `{lang}/{speaker_id}.wav` → `{lang}/default.wav` → `default.wav`.
- `CHATTERBOX_API_URL` — direct URL the TTS engine module reads (parallels `FW_CHATTERBOX_API_URL` in `core/config.py`); default `http://localhost:8020`.
- `FW_ALIGNMENT` — `"on"` (default) or `"off"` to bypass duration-aware alignment in `text_file_to_speech`.
- `FW_TTS_WORKERS` — TTS synthesis parallelism (default `3`).
- `FW_TTS_HTTP_TIMEOUT` — per-request read timeout (seconds) for Chatterbox calls. Default `600`. CPU synthesis takes ~80 s for a short sentence; the previous hard-coded 60 s timeout silently dropped most segments. Lower to `60` on GPU if you want faster failure.
- `FW_TTS_WARMUP` — `"on"` (default) or `"off"`. When on, synthesizes one short warmup phrase per distinct speaker voice before the main synthesis loop and uses the synthesized output as the cloning reference for all segments of that speaker. Eliminates inter-segment voice drift from Chatterbox's stochastic sampling. Disable for fast CPU dev runs or A/B comparisons.
- `FW_TTS_TEMPERATURE` — float (default `0.05`). Sampling temperature passed to Chatterbox on every synthesis call. API range 0.05–5.0; lower values reduce per-call variation. There is no seed parameter in the Chatterbox API, so this is the only direct lever for narrowing variance.
- `FW_DOWNLOAD_DURATION_SECONDS` — **dev affordance**. When set to a positive int, the download stage fetches only the first N seconds of every YouTube video (via yt-dlp `download_ranges`). Lets the slow CPU TTS stage finish in ~10 min instead of ~75 for a full clip. Leave unset/empty/0 for normal full-length downloads.
- `FW_USE_GPU_ENCODE` — when set, video stitching uses GPU encoding ([api/src/services/stitch_engine.py:135](api/src/services/stitch_engine.py#L135)).
- `IMAGEMAGICK_BINARY` — ImageMagick path for moviepy text rendering.
- `YT_COOKIES_FILE` — yt-dlp cookies (default `/app/cookies.txt`).
- `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` / `OPENROUTER_LEARN` / `LEARNED_SHORTENINGS_PATH` — OpenRouter reranking for the `REQUEST_SHORTER` alignment action. Optional; reranking falls back to local heuristics when absent.

## Architecture

### Pipeline flow

```text
YouTube URL → yt-dlp download → Whisper STT → argostranslate → Chatterbox TTS → moviepy/ffmpeg stitch → output video
```

### Layered backend

```text
routers/*.py                  HTTP I/O, request validation
   ↓ (FastAPI Depends)
services/*_service.py         Thin facades — orchestrate engine + I/O
   ↓
services/*_engine.py          Heavy lifting (e.g. tts_engine.text_file_to_speech)
   ↓
inference/                    WhisperBackend / TTSBackend ABCs
                              Local vs remote chosen by env vars
                              (factories in inference/__init__.py)
```

Key conventions:

- **Lazy model loading.** Models are *not* loaded on startup. The lifespan in [api/src/main.py](api/src/main.py) reserves `app.state._whisper_model` / `app.state._tts_model`; backends populate them on first request. This keeps the API container fast to start even when speaches/Chatterbox sidecars are still warming.
- **Settings & paths.** All directory names are `@property` methods on `Settings` in [api/src/core/config.py](api/src/core/config.py) — `settings.videos_dir`, `settings.transcriptions_dir`, etc. **Never hardcode** these in router/service code.
- **Graceful degradation.** `silero-vad`, `pyannote.audio`, and `logfire` are optional; absent imports are caught and logged, not raised. The pytest markers above mirror this.

### `foreign_whispers/` library surface

The alignment/evaluation library imported by the API:

- **[foreign_whispers/alignment.py](foreign_whispers/alignment.py)** — `compute_segment_metrics(en, es) → list[SegmentMetrics]` (using linear regression for baseline duration estimation), `global_align(metrics, silence_regions, max_stretch=1.4) → list[AlignedSegment]` (O(n) greedy), `global_align_dp(metrics, silence_regions) → list[AlignedSegment]` (forward DP scheduler), and the `AlignAction` enum with five outcomes:
  - `ACCEPT` — ≤10% over the source window.
  - `MILD_STRETCH` — 10–40% over; safe to time-stretch with pyrubberband.
  - `GAP_SHIFT` — 40–80% over; consume adjacent silence.
  - `REQUEST_SHORTER` — 80–150% over; ask for a reranked shorter translation.
  - `FAIL` — >150% over.
  - Includes both a single-pass O(n) greedy scheduler and a DP scheduler with lookahead drift tracking.
- **[foreign_whispers/backends.py](foreign_whispers/backends.py)** — `DurationAwareTTSBackend` ABC for future duration-controlled TTS; current implementations use the simpler `TTSBackend` in [api/src/inference/base.py](api/src/inference/base.py).
- **[foreign_whispers/voice_resolution.py](foreign_whispers/voice_resolution.py)** — `resolve_speaker_wav(speakers_dir, target_language, speaker_id=None)` → relative WAV path for Chatterbox voice cloning.
- **[foreign_whispers/vad.py](foreign_whispers/vad.py)** — `detect_speech_activity()` wraps Silero VAD.
- **[foreign_whispers/diarization.py](foreign_whispers/diarization.py)** — `diarize_audio()` wraps pyannote.audio (needs `FW_HF_TOKEN`).
- **[foreign_whispers/reranking.py](foreign_whispers/reranking.py)** — `get_shorter_translations()` produces shorter alternatives when alignment returns `REQUEST_SHORTER`.
- **[foreign_whispers/evaluation.py](foreign_whispers/evaluation.py)** — `clip_evaluation_report(metrics, aligned)` writes the `.align.json` sidecar. Also features `dubbing_scorecard(...)` for complete dubbing evaluation (timing, naturalness, intelligibility, semantics) using injected `typing.Protocol` dependencies.

**Integration trace.** [api/src/services/tts_engine.py](api/src/services/tts_engine.py) `text_file_to_speech` calls `_build_alignment()` → for each segment reads `aligned_seg.action` and routes accordingly: `REQUEST_SHORTER` triggers `get_shorter_translations()`; `MILD_STRETCH` is applied post-synthesis via `pyrubberband` in `_postprocess_segment()`; finally `clip_evaluation_report()` writes the `.align.json` sidecar.

## Video Registry

[video_registry.yml](video_registry.yml) is the single source of truth for all videos in the pipeline. Add entries there; the API reads it at startup to populate `/api/videos`. After adding a video, **rebuild and restart the API container** (see "Running the App").

## Python Version

`pyproject.toml` pins `requires-python = ">=3.11,<3.12"`. Always invoke via `uv run …` so the locked interpreter is selected. Local virtualenvs outside uv may break on 3.12+.

## Personal Progress

Per-task checklist mirroring the official project plan lives in [PROGRESS.md](PROGRESS.md). Tick boxes as you go; this is for personal session continuity, not team issue tracking.

## Issue Tracking

This repo uses **bd (beads)** for all issue tracking. Full workflow lives in [AGENTS.md](AGENTS.md).

```bash
bd ready --json     # unblocked work
bd show <id>        # issue details
bd update <id> --claim --json
bd close <id> --reason "Done" --json
```

Active issues mentioned for context:

- `fw-tov` — TTS temporal alignment implementation. Design context: [docs/tts-temporal-alignment-research.md](docs/tts-temporal-alignment-research.md).
- `jhg` — Hugging Face Spaces deployment.
