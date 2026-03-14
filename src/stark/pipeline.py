"""
Main pipeline orchestrator: VAD → STT → LLM → TTS → playback.

Each stage is timed and logged. Pass --dev to the CLI for verbose output.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, Optional

import numpy as np
import sounddevice as sd

from .config import Settings
from .context.macos import MacOSContext
from .llm.engine import LLMEngine
from .stt.whisper import WhisperSTT
from .tts.kokoro import KokoroTTS
from .vad.silero import SileroVAD

log = logging.getLogger(__name__)


class Timer:
    def __init__(self, name: str, dev: bool) -> None:
        self.name = name
        self.dev = dev
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = (time.perf_counter() - self._start) * 1000
        if self.dev:
            log.info("[timing] %s: %.0f ms", self.name, elapsed)
        self.elapsed_ms = elapsed


class Pipeline:
    """
    Wires all components together and runs the listen → respond loop.
    """

    def __init__(self, settings: Settings, dev: bool = False) -> None:
        self.cfg = settings
        self.dev = dev

        self.vad = SileroVAD(
            threshold=settings.vad.threshold,
            min_speech_ms=settings.vad.min_speech_ms,
            max_silence_ms=settings.vad.max_silence_ms,
            sample_rate=settings.audio.sample_rate,
        )
        self.stt = WhisperSTT(
            model=settings.stt.model,
            language=settings.stt.language,
        )
        self.llm = LLMEngine(
            model=settings.llm.model,
            draft_model=settings.llm.draft_model,
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
            system_prompt=settings.llm.system_prompt,
        )
        self.tts = KokoroTTS(
            voice=settings.tts.voice,
            speed=settings.tts.speed,
            lang=settings.tts.lang,
        )
        self.context = MacOSContext() if settings.context.enabled else None

    def load(self) -> None:
        """Pre-load all models so first response isn't slow."""
        log.info("Loading models…")
        self.llm.load()
        self.tts.load()
        log.info("All models loaded. Listening…")

    def run(self) -> None:
        """Start the continuous listen-respond loop (blocking)."""
        audio_cfg = self.cfg.audio

        with sd.InputStream(
            samplerate=audio_cfg.sample_rate,
            channels=audio_cfg.channels,
            dtype="float32",
            blocksize=audio_cfg.chunk_size,
            device=audio_cfg.device,
        ) as stream:
            log.info("Microphone open. Say something!")
            chunks = self._mic_chunks(stream)

            for speech_audio in self.vad.iter_speech(
                chunks,
                on_speech_start=lambda: log.info("▶ Listening…"),
                on_speech_end=lambda: log.info("■ Processing…"),
            ):
                self._handle_utterance(speech_audio)

    def _mic_chunks(self, stream: sd.InputStream) -> Iterator[np.ndarray]:
        chunk_size = self.cfg.audio.chunk_size
        while True:
            data, _ = stream.read(chunk_size)
            yield data[:, 0] if data.ndim > 1 else data  # ensure mono

    def _handle_utterance(self, audio: np.ndarray) -> None:
        active_context: Optional[str] = None
        if self.context and self.cfg.context.include_in_prompt:
            active_context = self.context.get()

        # ── STT ──────────────────────────────────────────────────────────────
        with Timer("STT", self.dev):
            text = self.stt.transcribe(audio, sample_rate=self.cfg.audio.sample_rate)

        if not text:
            log.debug("Empty transcription — skipping.")
            return

        log.info("You: %s", text)

        # ── LLM ──────────────────────────────────────────────────────────────
        with Timer("LLM", self.dev):
            response = self.llm.generate(text, context=active_context)

        if not response:
            log.debug("Empty LLM response — skipping.")
            return

        log.info("Stark: %s", response)

        # ── TTS + playback ────────────────────────────────────────────────────
        with Timer("TTS", self.dev):
            for audio_chunk in self.tts.stream(response):
                sd.play(audio_chunk, samplerate=self.tts.sample_rate, blocking=True)
