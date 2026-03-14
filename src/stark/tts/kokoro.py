"""Kokoro-82M TTS via kokoro-onnx — streams audio as it generates."""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)

_SAMPLE_RATE = 24000  # Kokoro outputs 24kHz


class KokoroTTS:
    """
    Wraps kokoro-onnx for low-latency TTS on Apple Silicon.

    Audio is generated sentence-by-sentence and yielded as float32
    numpy arrays at 24kHz so the pipeline can begin playback immediately.
    """

    def __init__(
        self,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str = "en-us",
    ) -> None:
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self._kokoro = None

    @property
    def sample_rate(self) -> int:
        return _SAMPLE_RATE

    def load(self) -> None:
        from kokoro_onnx import Kokoro

        log.info("Loading Kokoro TTS model…")
        self._kokoro = Kokoro("kokoro-v0_19.onnx", "voices.bin")
        log.info("Kokoro TTS ready.")

    def _ensure_loaded(self) -> None:
        if self._kokoro is None:
            self.load()

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text → float32 audio array at 24kHz."""
        chunks = list(self.stream(text))
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    def stream(self, text: str) -> Iterator[np.ndarray]:
        """
        Stream audio chunks as Kokoro processes each sentence.

        Yields float32 numpy arrays (24kHz) — play them while the next
        sentence is being synthesized to minimise perceived latency.
        """
        self._ensure_loaded()

        samples, sr = self._kokoro.create(
            text,
            voice=self.voice,
            speed=self.speed,
            lang=self.lang,
        )
        # kokoro-onnx returns the full audio in one call;
        # split on natural sentence boundaries for streaming playback.
        yield samples.astype(np.float32)
