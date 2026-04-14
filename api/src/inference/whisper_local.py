"""Local Whisper backend — runs the model in-process."""

from __future__ import annotations

import logging

import torch
import whisper

from api.src.inference.base import WhisperBackend

logger = logging.getLogger(__name__)


class LocalWhisperBackend(WhisperBackend):
    """Wraps ``whisper.load_model()`` + ``model.transcribe()``.

    Automatically selects CUDA if a GPU is available, otherwise falls back
    to CPU so the backend works on any machine.
    """

    def __init__(self, model_name: str = "base", device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading local Whisper model (%s) on device=%s...", model_name, device)
        self._model = whisper.load_model(model_name, device=device)
        self._model_name = model_name
        self._device = device
        logger.info("Whisper model loaded on %s.", device)

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe *audio_path* using the local Whisper model."""
        logger.info(
            "Transcribing %s with local Whisper (%s) on %s",
            audio_path, self._model_name, self._device,
        )
        return self._model.transcribe(audio_path)

    def __repr__(self) -> str:
        return f"<LocalWhisperBackend model={self._model_name!r} device={self._device!r}>"
