"""Silero VAD — chunked, streaming voice activity detection."""

from __future__ import annotations

import logging
from collections import deque
from typing import Callable, Iterator

import numpy as np

log = logging.getLogger(__name__)


class SileroVAD:
    """
    Wraps the Silero VAD model for real-time, chunk-based detection.

    Usage:
        vad = SileroVAD(threshold=0.5, sample_rate=16000)
        for speech_audio in vad.iter_speech(audio_chunks):
            ...  # numpy float32 array of one complete speech segment
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        max_silence_ms: int = 800,
        sample_rate: int = 16000,
    ) -> None:
        self.threshold = threshold
        self.min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self.max_silence_samples = int(sample_rate * max_silence_ms / 1000)
        self.sample_rate = sample_rate
        self._model, self._utils = self._load_model()

    def _load_model(self):
        import torch

        log.info("Loading Silero VAD model…")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        model.eval()
        log.info("Silero VAD ready.")
        return model, utils

    def _score(self, chunk: np.ndarray) -> float:
        import torch

        tensor = torch.from_numpy(chunk.astype(np.float32))
        with torch.no_grad():
            prob = self._model(tensor, self.sample_rate).item()
        return float(prob)

    def iter_speech(
        self,
        chunks: Iterator[np.ndarray],
        on_speech_start: Callable[[], None] | None = None,
        on_speech_end: Callable[[], None] | None = None,
    ) -> Iterator[np.ndarray]:
        """
        Yields complete speech segments (numpy float32 arrays) from a chunk iterator.

        Each yielded array is one utterance — silence-padded on both sides is stripped.
        """
        buffer: list[np.ndarray] = []
        silence_samples = 0
        in_speech = False

        for chunk in chunks:
            score = self._score(chunk)
            is_speech = score >= self.threshold

            if is_speech:
                if not in_speech:
                    in_speech = True
                    if on_speech_start:
                        on_speech_start()
                silence_samples = 0
                buffer.append(chunk)
            else:
                if in_speech:
                    buffer.append(chunk)
                    silence_samples += len(chunk)

                    if silence_samples >= self.max_silence_samples:
                        audio = np.concatenate(buffer)
                        if len(audio) >= self.min_speech_samples:
                            if on_speech_end:
                                on_speech_end()
                            yield audio
                        buffer = []
                        silence_samples = 0
                        in_speech = False

        # flush any remaining speech at end of stream
        if in_speech and buffer:
            audio = np.concatenate(buffer)
            if len(audio) >= self.min_speech_samples:
                if on_speech_end:
                    on_speech_end()
                yield audio
