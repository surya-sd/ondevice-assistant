"""Tone and energy detection from raw voice audio."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

_SPEED_MIN = 0.5
_SPEED_MAX = 2.0

# (tts_speed_multiplier, llm_hint)
_STATE_MAP: dict[str, tuple[float, str]] = {
    "excited":  (1.07, "[User sounds energetic — match their energy, be warm and engaged]"),
    "stressed": (0.90, "[User sounds stressed — be calm and grounding, don't rush]"),
    "anxious":  (0.88, "[User sounds anxious — be reassuring, keep it simple]"),
    "low":      (0.82, "[User sounds tired or low-energy — be gentle, slow your pace, don't push]"),
    "neutral":  (1.00, ""),
}


@dataclass
class ToneReading:
    state: str          # excited | stressed | anxious | low | neutral
    tts_speed: float    # absolute speed value to pass to TTS
    llm_hint: str       # injected text; empty string for neutral
    rms: float          # raw RMS value (for debug logging)
    speech_rate: float  # raw words-per-second value (for debug logging)


class ToneAnalyzer:
    """
    Classifies user emotional energy from audio features (no ML model).

    Uses two signals:
    - RMS energy: how loud/intense the voice is
    - Speech rate: words per second (adjusted for VAD silence tail)

    Combined into a 2x2 grid: loud×fast=excited, loud×slow=stressed,
    quiet×fast=anxious, quiet×slow=low.
    """

    def __init__(
        self,
        base_speed: float = 0.95,
        rms_loud_threshold: float = 0.05,
        speech_rate_fast_threshold: float = 2.5,
        min_words: int = 3,
    ) -> None:
        self.base_speed = base_speed
        self.rms_loud_threshold = rms_loud_threshold
        self.speech_rate_fast_threshold = speech_rate_fast_threshold
        self.min_words = min_words

    def analyze(
        self,
        audio: np.ndarray,
        text: str,
        sample_rate: int,
    ) -> ToneReading:
        if audio is None or len(audio) == 0 or not text.strip():
            return ToneReading("neutral", self.base_speed, "", 0.0, 0.0)

        rms = self._compute_rms(audio)
        speech_rate = self._compute_speech_rate(text, audio, sample_rate)
        word_count = len(text.split())
        state = self._classify(rms, speech_rate, word_count)

        multiplier, llm_hint = _STATE_MAP[state]
        if state == "neutral":
            tts_speed = self.base_speed
        else:
            raw = self.base_speed * multiplier
            tts_speed = max(_SPEED_MIN, min(_SPEED_MAX, raw))

        log.debug(
            "[tone] state=%s rms=%.4f wps=%.2f speed=%.2f",
            state, rms, speech_rate, tts_speed,
        )
        return ToneReading(state, tts_speed, llm_hint, rms, speech_rate)

    def _compute_rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

    def _compute_speech_rate(
        self, text: str, audio: np.ndarray, sample_rate: int
    ) -> float:
        duration_s = len(audio) / sample_rate
        # Subtract estimated VAD silence tail (up to 1.5s, at most 40% of total)
        silence_tail_s = min(1.5, duration_s * 0.4)
        effective_duration = max(0.3, duration_s - silence_tail_s)
        words = len(text.split())
        return words / effective_duration

    def _classify(self, rms: float, speech_rate: float, word_count: int) -> str:
        if word_count < self.min_words:
            return "neutral"

        loud = rms >= self.rms_loud_threshold
        fast = speech_rate >= self.speech_rate_fast_threshold

        if loud and fast:
            return "excited"
        if loud and not fast:
            return "stressed"
        if not loud and fast:
            return "anxious"
        return "low"
