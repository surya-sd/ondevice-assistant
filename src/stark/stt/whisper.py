"""Whisper STT via mlx-whisper (Apple Silicon optimised)."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class WhisperSTT:
    """
    Transcribes a numpy float32 audio array using mlx-whisper.

    Models are downloaded on first use and cached by mlx-whisper.
    """

    def __init__(self, model: str = "mlx-community/whisper-small-mlx-q4", language: str = "en") -> None:
        self.model_name = model
        self.language = language
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            import mlx_whisper  # noqa: F401 — triggers model download on first import
            self._loaded = True
            log.info("mlx-whisper ready (model will download on first transcription if needed).")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: float32 numpy array, mono, at sample_rate Hz.
            sample_rate: must be 16000 for Whisper.

        Returns:
            Transcribed text string (stripped).
        """
        self._ensure_loaded()
        import mlx_whisper

        if sample_rate != 16000:
            raise ValueError(f"Whisper requires 16kHz audio, got {sample_rate}Hz")

        # mlx_whisper.transcribe accepts a numpy array or file path
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_name,
            language=self.language,
            word_timestamps=False,
            verbose=False,
        )
        text: str = result.get("text", "").strip()
        log.debug("STT: %r", text)
        return text
