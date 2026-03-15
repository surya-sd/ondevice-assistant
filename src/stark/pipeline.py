"""
Main pipeline orchestrator: VAD → STT → LLM → TTS → playback.

Each stage is timed and logged. Pass --dev to the CLI for verbose output.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
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


_SYMBOL_MAP = [
    (r"\$(\d+(?:\.\d+)?)", r"\1 dollars"),
    (r"(\d+)%", r"\1 percent"),
    (r"(\d+)x\b", r"\1 times"),
    (r"\b(\d+)\s*:\s*(\d+)\b", r"\1 \2"),   # e.g. 3:15 -> "3 15" (TTS reads naturally)
    (r"&", " and "),
    (r"[*_`#]", ""),                          # markdown symbols
    (r"\[([^\]]+)\]\([^)]+\)", r"\1"),        # [text](url) -> text
    (r"  +", " "),
]


def _is_sentence_end(text: str) -> bool:
    """Return True when text ends with a natural spoken sentence boundary."""
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in ".!?…"


def _prep_for_tts(text: str) -> str:
    """Strip markdown and expand symbols so TTS reads naturally."""
    for pattern, replacement in _SYMBOL_MAP:
        text = re.sub(pattern, replacement, text)
    return text.strip()


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
            top_p=settings.llm.top_p,
            system_prompt=settings.llm.system_prompt,
            history_turns=settings.llm.history_turns,
        )
        self.tts = KokoroTTS(
            voice=settings.tts.voice,
            speed=settings.tts.speed,
            lang=settings.tts.lang,
        )
        self.context = MacOSContext() if settings.context.enabled else None

    _HISTORY_PATH = Path.home() / ".cache" / "stark" / "history.json"

    def load(self) -> None:
        """Pre-load all models so first response isn't slow."""
        log.info("Loading models…")
        self.stt.load()
        self.llm.load()
        self.tts.load()
        if self.cfg.llm.history_persist:
            self.llm.load_history(
                self._HISTORY_PATH,
                max_age_hours=self.cfg.llm.history_max_age_hours,
            )
        log.info("All models loaded. Listening…")

    def run(self) -> None:
        """Start the continuous listen-respond loop (blocking)."""
        audio_cfg = self.cfg.audio

        try:
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
        finally:
            if self.cfg.llm.history_persist:
                self.llm.save_history(self._HISTORY_PATH)

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

        # ── LLM → TTS pipeline (streamed, sentence by sentence) ──────────────
        # As soon as a sentence boundary is detected in the LLM stream, send it
        # to TTS immediately — don't wait for the full response to finish.
        full_response: list[str] = []
        sentence_buf = ""
        first_audio = True

        llm_timer = Timer("LLM first sentence", self.dev)
        llm_timer.__enter__()

        for token in self.llm.stream(text, context=active_context):
            sentence_buf += token
            full_response.append(token)

            if _is_sentence_end(sentence_buf):
                if first_audio:
                    llm_timer.__exit__(None, None, None)
                    first_audio = False

                sentence = _prep_for_tts(sentence_buf.strip())
                sentence_buf = ""

                if sentence:
                    with Timer("TTS sentence", self.dev):
                        for audio_chunk in self.tts.stream(sentence):
                            sd.play(audio_chunk, samplerate=self.tts.sample_rate, blocking=True)

        # flush any remaining text after the last sentence boundary
        remainder = _prep_for_tts(sentence_buf.strip())
        if remainder:
            with Timer("TTS remainder", self.dev):
                for audio_chunk in self.tts.stream(remainder):
                    sd.play(audio_chunk, samplerate=self.tts.sample_rate, blocking=True)

        response = "".join(full_response)
        if not response.strip():
            log.debug("Empty LLM response — skipping.")
            return

        log.info("Stark: %s", response.strip())
