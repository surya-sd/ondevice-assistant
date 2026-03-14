"""Configuration loading via pydantic-settings + YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 512
    device: Optional[int | str] = None


class VADConfig(BaseModel):
    threshold: float = 0.5
    min_speech_ms: int = 250
    max_silence_ms: int = 800


class STTConfig(BaseModel):
    model: str = "mlx-community/whisper-small-mlx-q4"
    language: str = "en"


class LLMConfig(BaseModel):
    model: str = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    draft_model: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    max_tokens: int = 60
    temperature: float = 0.7
    system_prompt: str = (
        "You are Stark, a sharp and direct voice assistant. "
        "Keep responses concise — 1-3 sentences max. No filler words."
    )

    @field_validator("system_prompt")
    @classmethod
    def strip_prompt(cls, v: str) -> str:
        return v.strip()


class TTSConfig(BaseModel):
    model: str = "kokoro"
    voice: str = "af_heart"
    speed: float = 1.0
    lang: str = "en-us"


class ContextConfig(BaseModel):
    enabled: bool = True
    include_in_prompt: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STARK_", env_nested_delimiter="__")

    audio: AudioConfig = AudioConfig()
    vad: VADConfig = VADConfig()
    stt: STTConfig = STTConfig()
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    context: ContextConfig = ContextConfig()

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Settings":
        """Load from default.yaml, then optional local.yaml, then env vars."""
        data: dict = {}

        default = _CONFIG_DIR / "default.yaml"
        if default.exists():
            data = yaml.safe_load(default.read_text()) or {}

        local = config_path or (_CONFIG_DIR / "local.yaml")
        if local.exists():
            local_data = yaml.safe_load(local.read_text()) or {}
            data = _deep_merge(data, local_data)

        return cls.model_validate(data)


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
