# Foreign Whispers — Progress Tracker

Personal checklist mirroring the official project plan at <https://aegean.ai/aiml-common/projects/nlp/foreign-whispers>.
Tick boxes as you go and add notes inline. Each task lists the **file(s) to modify** and the **acceptance criterion** so you can pick up cold from any point.

> The repo's official issue tracker is **bd (beads)** (see [AGENTS.md](AGENTS.md)). Use this file for *personal* progress; promote anything that needs follow-up by the team into a bd issue.

---

## Phase 1 — Environment setup and end-to-end run

- [x] **Step 1 · Clone the repository.** `git clone https://github.com/aegean-ai/foreign-whispers.git && cd foreign-whispers`. Read the [README.md](README.md).
- [x] **Step 2 · Configure environment variables.** `.env` exists with `FW_HF_TOKEN` set.
- [x] **Step 3 · Start the Docker stack.** `docker compose --profile nvidia up -d`. (Verified: containers up; `whisper-cpu` healthy; `frontend` and `chatterbox-cpu` show *unhealthy* — investigate health probes.)
- [x] **Step 4 · Install the local Python library.** `uv.lock` present (4225 lines).
- [ ] **Step 5 · (Optional) Set up Logfire observability.** Skipped (optional).
- [x] **Step 6 · Run the end-to-end pipeline notebook.** Verified artifacts in `pipeline_data/api/{videos, transcriptions/whisper, translations/argos, tts_audio/chatterbox, dubbed_videos, dubbed_captions}`.

---

## Phase 2 — Integration notebooks

Work through these **in order**. Each contains tasks marked `YOUR CODE HERE`.

### Notebook 1 — Download integration *(no coding)*

Notebook: [notebooks/download_integration/](notebooks/download_integration/)

- [x] **Read & understand** the segment-dict format (`{start, end, text}`). Every downstream stage consumes this shape.

---

### Notebook 2 — Transcription integration *(no coding)*

Notebook: [notebooks/transcription_integration/transcription_integration.ipynb](notebooks/transcription_integration/transcription_integration.ipynb)

- [x] **Compare** YouTube-caption segment durations vs Whisper-STT durations.

---

### Notebook 3 — Translation integration

Notebook: [notebooks/translation_integration/translation_integration.ipynb](notebooks/translation_integration/translation_integration.ipynb)

- [x] **Task · Duration-aware re-ranking** — implemented at [foreign_whispers/reranking.py:377](foreign_whispers/reranking.py#L377). Three-stage hybrid: learned cache → rule-based contractions → OpenRouter LLM fallback. Goes well beyond the requested stub.

---

### Notebook 4 — Diarization integration *(5 tasks — largest notebook)*

Notebook: [notebooks/diarization_integration/diarization_integration.ipynb](notebooks/diarization_integration/diarization_integration.ipynb)

- [x] **Task 1 · `assign_speakers` merge function** — [foreign_whispers/diarization.py:160](foreign_whispers/diarization.py#L160). All 4 unit tests pass (re-verified).
- [x] **Task 2 · Diarize API endpoint** — [api/src/routers/diarize.py](api/src/routers/diarize.py), [api/src/schemas/diarize.py](api/src/schemas/diarize.py). Registered in [api/src/main.py:90](api/src/main.py#L90); `diarizations_dir` added in [api/src/core/config.py:58](api/src/core/config.py#L58).
- [x] **Task 3 · Merge speaker labels into transcription** — `_merge_labels_into_transcript` in [api/src/routers/diarize.py:19](api/src/routers/diarize.py#L19). Nice touch: runs on every call (including cache hits) so labels survive transcript regeneration.
- [x] **Task 4 · Frontend pipeline integration** — diarize stage correctly placed between Transcribe and Translate in [frontend/src/hooks/use-pipeline.ts:25](frontend/src/hooks/use-pipeline.ts#L25) and [frontend/src/components/pipeline-table.tsx:35](frontend/src/components/pipeline-table.tsx#L35); types and API client updated. *(Cosmetic cleanup pending — see Findings below.)*
- [x] **Task 5 · Per-speaker TTS voice selection** — `_build_speaker_voice_map` + per-segment voice switching in [api/src/services/tts_engine.py:204](api/src/services/tts_engine.py#L204) and [api/src/services/tts_engine.py:519](api/src/services/tts_engine.py#L519). Uses the "round-robin with fallback" strategy documented in notebook cell 41. Note: `resolve_speaker_wav()` (officially Notebook 6 Task 2) was implemented as part of this — so N6 T2 is **already done**.

---

### Notebook 5 — Alignment integration *(4 tasks — most analytically demanding)*

Notebook: [notebooks/alignment_integration/alignment_integration.ipynb](notebooks/alignment_integration/alignment_integration.ipynb)

- [x] **Task 1 · Improve TTS duration prediction** — replaced the syllables/4.5 heuristic with a closed-form linear regression on `(chars, syllables, words)` in [foreign_whispers/alignment.py:48](foreign_whispers/alignment.py#L48). MAE on the Hormuz training pairs dropped **0.608 s → 0.186 s (–69%)**. Legacy formula kept as `_estimate_duration_baseline()` for A/B comparison in the notebook.
- [x] **Task 2 · Duration-aware translation re-ranking** — wired `get_shorter_translations()` into the notebook (function was already implemented as the 3-stage hybrid in earlier project work; no source change needed). Notebook cell shows action-distribution before vs after re-ranking.
- [x] **Task 3 · Beat the greedy optimizer** — `global_align_dp()` added at [foreign_whispers/alignment.py:303](foreign_whispers/alignment.py#L303). Forward DP over `(segment_index, drift_quantum)` state with cost = `overflow + severe_stretch + drift²`; the drift² term is the lookahead lever. Three new tests cover greedy parity, total-cost dominance, and real-clip no-regression.
- [x] **Task 4 · Dubbing quality scorecard** — `dubbing_scorecard()` in [foreign_whispers/evaluation.py:55](foreign_whispers/evaluation.py#L55) returns timing, naturalness, intelligibility, and semantic sub-scores in [0, 1] plus an overall mean. Heavy services (Whisper STT, back-translator, embedder) injected as `typing.Protocol` classes so unit tests can mock them.

---

### Notebook 6 — TTS integration *(4 tasks — baseline vs aligned modes + voice cloning)*

Notebook: [notebooks/tts_integration/tts_integration.ipynb](notebooks/tts_integration/tts_integration.ipynb)

- [ ] **Task 1 · Understand the existing Chatterbox client** *(no coding, exploration)*
  - Read through the baseline-vs-aligned cells; understand what `synthesize` returns and how alignment is applied post-hoc.
- [x] **Task 2 · Voice resolution function** — implemented during N4 T5. Lives at [foreign_whispers/voice_resolution.py](foreign_whispers/voice_resolution.py). *Verify the 5 provided tests still pass when you reach this notebook.*
- [ ] **Task 3 · Add `speaker_wav` to the TTS API**
  - **Modify:** [api/src/core/config.py](api/src/core/config.py) (add `speakers_dir`), [api/src/routers/tts.py](api/src/routers/tts.py), [api/src/services/tts_service.py](api/src/services/tts_service.py)
  - **Acceptance:** speaker selection exposed as a query parameter; manual `curl` test produces audio in the chosen voice.
- [ ] **Task 4 · Per-speaker voice assignment**
  - **Modify:** [api/src/routers/tts.py](api/src/routers/tts.py)
  - **Goal:** when diarized segments exist, build a speaker → voice mapping and switch voices per segment.
  - **Acceptance:** test with a multi-speaker video; each speaker uses a distinct reference voice.

---

### Notebook 7 — Stitch integration *(no coding)*

Notebook: [notebooks/stitch_integration/stitch_integration.ipynb](notebooks/stitch_integration/stitch_integration.ipynb)

- [ ] **Verify** the dubbed output plays correctly with rolling two-line VTT captions. Video track is copied as-is via ffmpeg remux (no re-encoding); only audio is replaced.

---

## Audit findings (2026-04-28) — through Phase 2 Notebook 4

### Working correctly

- All 4 `assign_speakers` unit tests pass (`uv run pytest tests/test_diarization.py -v` → 4 passed).
- `POST /api/diarize/{video_id}` is wired end-to-end: ffmpeg audio extraction → pyannote → cache → merge into transcript.
- Diarize stage appears in the frontend pipeline tracker in the correct slot (between Transcribe and Translate).
- Per-speaker voice switching is fully wired through `tts_engine.text_file_to_speech` — `_build_speaker_voice_map` resolves one WAV per distinct speaker (efficient: O(speakers), not O(segments)).
- N3 re-ranker (`get_shorter_translations`) is implemented well beyond the requested stub: 3-stage hybrid (learned cache → rules → LLM fallback) with a learned-output cache.

### Cosmetic cleanup ✅ fixed

- [frontend/src/components/pipeline-table.tsx:34](frontend/src/components/pipeline-table.tsx#L34) — removed the `// Add after "transcribe" entry:` scaffolding comment.
- [frontend/src/hooks/use-pipeline.ts:197-198](frontend/src/hooks/use-pipeline.ts#L197-L198) — removed the `// After:` / `// Before:` scaffolding comments.

### Docker healthchecks ✅ fixed

Both unhealthy containers had real bugs in the probe config — pipeline functioned but `docker compose ps` lied. Root causes:

- **frontend** — the container's `/etc/hosts` maps `localhost` to `::1` only, while `next-server` binds IPv4. `wget --spider http://localhost:8501` resolved to `[::1]:8501` and got "connection refused" forever. Switched the healthcheck URL to `http://127.0.0.1:8501` ([docker-compose.yml:148](docker-compose.yml#L148)).
- **chatterbox (cpu + gpu)** — the upstream `travisvn/chatterbox-tts-api` image bakes a `HEALTHCHECK` of `curl -f http://localhost:5123/health` into its Dockerfile, but our compose env sets `PORT=8020`. The probe hit the wrong port forever. Added a compose-level `healthcheck:` block to both services (which overrides the image-level directive) probing the correct `http://localhost:8020/health` ([docker-compose.yml:66](docker-compose.yml#L66) and [docker-compose.yml:92](docker-compose.yml#L92)).

After `docker compose --profile cpu up -d`, all four services report `healthy`.

### Bonus already done (early)

- **N6 Task 2** (`resolve_speaker_wav`) was implemented as part of N4 T5 — so when you reach Notebook 6 you can skip straight to Task 3 after a test re-run.
- The `_estimate_duration` helper extraction (commit `aa3b9c3`) is partial credit for **N5 Task 1** — the *replacement* of the ~15 chars/sec heuristic is still pending.

### Nothing skipped

Every task that was *required* through Notebook 4 has corresponding code. No gaps found.

---

## Quick reference — completed today

Use this scratch area as a session log so you can pick up where you left off.

```
YYYY-MM-DD  – e.g. "Notebook 4 Task 1 done; tests passing locally"
```
