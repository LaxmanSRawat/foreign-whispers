"""Tests for the speaker-warmup voice consistency strategy and temperature wiring."""
import json
import pathlib
import tempfile
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_minimal_transcripts(tmp_path, title="vid", speakers=None):
    """Write minimal ES + EN transcript JSON files under tmp_path."""
    es_dir = tmp_path / "translations" / "argos"
    en_dir = tmp_path / "transcriptions" / "whisper"
    es_dir.mkdir(parents=True, exist_ok=True)
    en_dir.mkdir(parents=True, exist_ok=True)

    if speakers is None:
        speakers = [None, None]

    segs = [
        {"id": i, "start": float(i * 3), "end": float(i * 3 + 3), "text": f"seg {i}", "speaker": spk}
        for i, spk in enumerate(speakers)
    ]
    en_segs = [
        {"id": i, "start": float(i * 3), "end": float(i * 3 + 3), "text": f"en seg {i}"}
        for i in range(len(speakers))
    ]

    (es_dir / f"{title}.json").write_text(
        json.dumps({"segments": segs, "text": " ".join(s["text"] for s in segs), "language": "es"})
    )
    (en_dir / f"{title}.json").write_text(
        json.dumps({"segments": en_segs, "text": " ".join(s["text"] for s in en_segs)})
    )
    return es_dir / f"{title}.json"


# ---------------------------------------------------------------------------
# _build_warmup_voice_map
# ---------------------------------------------------------------------------

class TestBuildWarmupVoiceMap:
    def test_synthesizes_once_per_distinct_wav(self, tmp_path):
        """Two speakers sharing the same reference WAV → one warmup synthesis, not two."""
        from api.src.services.tts_engine import ChatterboxClient, _build_warmup_voice_map

        engine = MagicMock(spec=ChatterboxClient)
        engine.synthesize_warmup_reference.return_value = b"RIFF....fake-wav-bytes"

        voice_map = {
            "SPEAKER_00": "es/voice_a.wav",
            "SPEAKER_01": "es/voice_a.wav",  # same reference
        }

        result = _build_warmup_voice_map(engine, voice_map, str(tmp_path))

        # Only one synthesis call despite two speakers
        assert engine.synthesize_warmup_reference.call_count == 1
        # Both speakers must point to the same warmup file
        assert result["SPEAKER_00"] == result["SPEAKER_01"]
        # The warmup WAV must actually exist on disk
        assert pathlib.Path(result["SPEAKER_00"]).exists()

    def test_different_wavs_each_get_their_own_warmup(self, tmp_path):
        """Speakers with different reference WAVs each get their own warmup synthesis."""
        from api.src.services.tts_engine import ChatterboxClient, _build_warmup_voice_map

        engine = MagicMock(spec=ChatterboxClient)
        engine.synthesize_warmup_reference.return_value = b"RIFF....fake-wav-bytes"

        voice_map = {
            "SPEAKER_00": "es/voice_a.wav",
            "SPEAKER_01": "es/voice_b.wav",
        }

        result = _build_warmup_voice_map(engine, voice_map, str(tmp_path))

        assert engine.synthesize_warmup_reference.call_count == 2
        assert result["SPEAKER_00"] != result["SPEAKER_01"]
        assert pathlib.Path(result["SPEAKER_00"]).exists()
        assert pathlib.Path(result["SPEAKER_01"]).exists()

    def test_fallback_on_synthesis_failure(self, tmp_path):
        """If warmup synthesis raises, the original reference WAV path is retained."""
        from api.src.services.tts_engine import ChatterboxClient, _build_warmup_voice_map

        engine = MagicMock(spec=ChatterboxClient)
        engine.synthesize_warmup_reference.side_effect = RuntimeError("TTS server down")

        voice_map = {"SPEAKER_00": "es/voice_a.wav"}

        result = _build_warmup_voice_map(engine, voice_map, str(tmp_path))

        # Must not raise, and original path retained
        assert result["SPEAKER_00"] == "es/voice_a.wav"

    def test_noop_for_non_chatterbox_engine(self, tmp_path):
        """Non-ChatterboxClient engines return the original voice_map unchanged."""
        from api.src.services.tts_engine import _build_warmup_voice_map

        engine = MagicMock()  # not a ChatterboxClient instance
        voice_map = {"SPEAKER_00": "es/voice_a.wav"}

        result = _build_warmup_voice_map(engine, voice_map, str(tmp_path))

        assert result is voice_map  # exact same object returned

    def test_warmup_wav_written_to_work_dir(self, tmp_path):
        """Warmup WAV files are written inside work_dir, not elsewhere."""
        from api.src.services.tts_engine import ChatterboxClient, _build_warmup_voice_map

        engine = MagicMock(spec=ChatterboxClient)
        engine.synthesize_warmup_reference.return_value = b"RIFF....fake-wav-bytes"

        voice_map = {"SPEAKER_00": "es/voice_a.wav"}

        result = _build_warmup_voice_map(engine, voice_map, str(tmp_path))

        warmup_path = pathlib.Path(result["SPEAKER_00"])
        assert warmup_path.parent == tmp_path


# ---------------------------------------------------------------------------
# Temperature wiring
# ---------------------------------------------------------------------------

class TestTemperatureWiring:
    def test_temperature_in_default_synthesis(self):
        """_synthesize_default includes temperature in the JSON body."""
        import requests as _requests
        from api.src.services.tts_engine import ChatterboxClient, _CHATTERBOX_TEMPERATURE

        client = ChatterboxClient(base_url="http://fake:8020", speaker_wav="")
        mock_resp = MagicMock()
        mock_resp.content = b"wav-bytes"
        mock_resp.raise_for_status = MagicMock()

        with patch.object(_requests, "post", return_value=mock_resp) as mock_post:
            client._synthesize_default("Hola")

        _, kwargs = mock_post.call_args
        body = kwargs.get("json", {})
        assert "temperature" in body
        assert body["temperature"] == pytest.approx(_CHATTERBOX_TEMPERATURE)

    def test_temperature_in_voice_synthesis(self, tmp_path):
        """_synthesize_with_voice includes temperature in the multipart form data."""
        import requests as _requests
        from api.src.services.tts_engine import ChatterboxClient, _CHATTERBOX_TEMPERATURE

        # Create a dummy speaker WAV so the path resolution succeeds
        fake_wav = tmp_path / "voice.wav"
        fake_wav.write_bytes(b"RIFF....fake")

        client = ChatterboxClient(base_url="http://fake:8020", speaker_wav="")
        mock_resp = MagicMock()
        mock_resp.content = b"wav-bytes"
        mock_resp.raise_for_status = MagicMock()

        with patch.object(_requests, "post", return_value=mock_resp) as mock_post:
            client._synthesize_with_voice("Hola", str(fake_wav))

        _, kwargs = mock_post.call_args
        form_data = kwargs.get("data", {})
        assert "temperature" in form_data
        assert float(form_data["temperature"]) == pytest.approx(_CHATTERBOX_TEMPERATURE)


# ---------------------------------------------------------------------------
# Integration: text_file_to_speech uses warmup voice map
# ---------------------------------------------------------------------------

class TestTextFileToSpeechWarmup:
    def _fake_synthesize_raw(self, engine, text, wav_path, speaker_wav=None):
        """Write a minimal WAV-like file so postprocessing doesn't crash."""
        import numpy as np
        import soundfile as sf
        sr = 22050
        sf.write(wav_path, [0.0] * sr, sr)
        return pathlib.Path(wav_path).read_bytes()

    def test_warmup_map_called_when_enabled(self, tmp_path, monkeypatch):
        """_build_warmup_voice_map is called during text_file_to_speech when FW_TTS_WARMUP=on."""
        monkeypatch.setenv("FW_TTS_WARMUP", "on")

        from api.src.services import tts_engine
        from api.src.services.tts_engine import ChatterboxClient

        es_path = _write_minimal_transcripts(tmp_path, speakers=["SPEAKER_00", "SPEAKER_01"])
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        engine = MagicMock(spec=ChatterboxClient)

        with patch.object(tts_engine, "_build_warmup_voice_map", wraps=lambda e, vm, wd: vm) as mock_warmup, \
             patch.object(tts_engine, "_synthesize_raw", side_effect=self._fake_synthesize_raw):
            tts_engine.text_file_to_speech(str(es_path), str(out_dir), tts_engine=engine)

        mock_warmup.assert_called_once()

    def test_warmup_map_skipped_when_disabled(self, tmp_path, monkeypatch):
        """_build_warmup_voice_map is NOT called when FW_TTS_WARMUP=off."""
        monkeypatch.setenv("FW_TTS_WARMUP", "off")

        from api.src.services import tts_engine
        from api.src.services.tts_engine import ChatterboxClient

        es_path = _write_minimal_transcripts(tmp_path, speakers=["SPEAKER_00"])
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        engine = MagicMock(spec=ChatterboxClient)

        with patch.object(tts_engine, "_build_warmup_voice_map") as mock_warmup, \
             patch.object(tts_engine, "_synthesize_raw", side_effect=self._fake_synthesize_raw):
            tts_engine.text_file_to_speech(str(es_path), str(out_dir), tts_engine=engine)

        mock_warmup.assert_not_called()
