"""Local TTS backend — runs the model in-process."""

from __future__ import annotations

import logging

import torch
from TTS.api import TTS

from api.src.inference.base import TTSBackend

logger = logging.getLogger(__name__)


class LocalTTSBackend(TTSBackend):
    """Wraps ``TTS.api.TTS()`` + ``tts.tts_to_file()``.

    Automatically selects CUDA if a GPU is available, otherwise falls back
    to CPU so the backend works on any machine.
    """

    def __init__(self, model_name: str = "tts_models/es/css10/vits", gpu: bool | None = None) -> None:
        if gpu is None:
            gpu = torch.cuda.is_available()
        device_label = "cuda" if gpu else "cpu"
        logger.info("Loading local TTS model (%s) on device=%s...", model_name, device_label)
        self._tts = TTS(model_name=model_name, progress_bar=False, gpu=gpu)
        self._model_name = model_name
        self._gpu = gpu
        logger.info("TTS model loaded on %s.", device_label)

    def synthesize(self, text: str, output_path: str) -> str:
        """Synthesize *text* to a WAV file at *output_path*."""
        logger.info("Synthesizing TTS to %s (gpu=%s)", output_path, self._gpu)
        self._tts.tts_to_file(text=text, file_path=output_path)
        return output_path

    def __repr__(self) -> str:
        return f"<LocalTTSBackend model={self._model_name!r} gpu={self._gpu!r}>"
