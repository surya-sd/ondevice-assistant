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
    state: str           # excited | stressed | anxious | low | neutral
    tts_speed: float     # absolute speed value to pass to TTS
    llm_hint: str        # injected text; empty string for neutral
    rms: float           # raw RMS value (for debug logging)
    speech_rate: float   # raw words-per-second value (for debug logging)
    smoothed_rate: float # EMA-smoothed speech rate (for debug logging)


class ToneAnalyzer:
    """
    Classifies user emotional energy from audio features (no ML model).

    Uses two signals:
    - RMS energy: how loud/intense the voice is
    - Speech rate: words per second (adjusted for VAD silence tail)

    Combined into a 2x2 grid: loud×fast=excited, loud×slow=stressed,
    quiet×fast=anxious, quiet×slow=low.

    TTS speed adapts over time via an EMA of the user's speech rate, so Stark
    gradually mirrors the user's natural speaking pace across turns.
    """

    def __init__(
        self,
        base_speed: float = 0.95,
        rms_loud_threshold: float = 0.05,
        speech_rate_fast_threshold: float = 2.5,
        min_words: int = 3,
        ema_alpha: float = 0.35,
        pace_speed_min: float = 0.82,
        pace_speed_max: float = 1.15,
        pace_rate_slow: float = 1.0,
        pace_rate_fast: float = 3.5,
    ) -> None:
        self.base_speed = base_speed
        self.rms_loud_threshold = rms_loud_threshold
        self.speech_rate_fast_threshold = speech_rate_fast_threshold
        self.min_words = min_words
        self.ema_alpha = ema_alpha
        self.pace_speed_min = pace_speed_min
        self.pace_speed_max = pace_speed_max
        self.pace_rate_slow = pace_rate_slow
        self.pace_rate_fast = pace_rate_fast
        # EMA initialised to the midpoint of the pace range (neutral baseline)
        self._smoothed_rate: float = (pace_rate_slow + pace_rate_fast) / 2.0

    def analyze(
        self,
        audio: np.ndarray,
        text: str,
        sample_rate: int,
    ) -> ToneReading:
        if audio is None or len(audio) == 0 or not text.strip():
            return ToneReading("neutral", self.base_speed, "", 0.0, 0.0, self._smoothed_rate)

        rms = self._compute_rms(audio)
        speech_rate = self._compute_speech_rate(text, audio, sample_rate)
        word_count = len(text.split())
        state = self._classify(rms, speech_rate, word_count)

        # Update EMA only for utterances long enough to be meaningful
        if word_count >= self.min_words:
            self._smoothed_rate = (
                self.ema_alpha * speech_rate
                + (1.0 - self.ema_alpha) * self._smoothed_rate
            )

        multiplier, llm_hint = _STATE_MAP[state]
        pace_speed = self._pace_speed()
        raw = pace_speed * multiplier
        tts_speed = max(_SPEED_MIN, min(_SPEED_MAX, raw))

        log.debug(
            "[tone] state=%s rms=%.4f wps=%.2f smoothed=%.2f pace_speed=%.2f speed=%.2f",
            state, rms, speech_rate, self._smoothed_rate, pace_speed, tts_speed,
        )
        return ToneReading(state, tts_speed, llm_hint, rms, speech_rate, self._smoothed_rate)

    def _pace_speed(self) -> float:
        """Map the EMA-smoothed speech rate to a TTS speed via linear interpolation."""
        rate_range = self.pace_rate_fast - self.pace_rate_slow
        t = (self._smoothed_rate - self.pace_rate_slow) / rate_range
        t = max(0.0, min(1.0, t))
        return self.pace_speed_min + t * (self.pace_speed_max - self.pace_speed_min)

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
