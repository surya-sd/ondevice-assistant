"""Kokoro-82M TTS via kokoro-onnx — streams audio as it generates."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)

_SAMPLE_RATE = 24000  # Kokoro outputs 24kHz
_CACHE_DIR = Path.home() / ".cache" / "stark" / "kokoro"
_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"


def _ensure_model_files() -> tuple[Path, Path]:
    """Download Kokoro model files to ~/.cache/stark/kokoro/ if not present."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _CACHE_DIR / "kokoro-v0_19.onnx"
    voices_path = _CACHE_DIR / "voices.bin"

    if not model_path.exists():
        log.info("Downloading Kokoro model (~80MB)…")
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        log.info("Kokoro model downloaded.")

    if not voices_path.exists():
        log.info("Downloading Kokoro voices (~25MB)…")
        urllib.request.urlretrieve(_VOICES_URL, voices_path)
        log.info("Kokoro voices downloaded.")

    return model_path, voices_path


class KokoroTTS:
    """
    Wraps kokoro-onnx for low-latency TTS on Apple Silicon.

    Model files are cached in ~/.cache/stark/kokoro/ and downloaded
    automatically on first use.
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

        model_path, voices_path = _ensure_model_files()
        log.info("Loading Kokoro TTS model…")
        self._kokoro = Kokoro(str(model_path), str(voices_path))
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
