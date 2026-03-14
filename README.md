# Stark

On-device voice assistant for macOS (Apple Silicon). Fully offline, sub-second responses.

## Stack

| Stage | Component |
|-------|-----------|
| VAD | Silero VAD |
| STT | Whisper Small (mlx-whisper, 4-bit) |
| LLM | Qwen 2.5 3B + 0.5B speculative decoding (mlx-lm, 4-bit) |
| TTS | Kokoro-82M (kokoro-onnx) |
| Context | macOS AXUIElement (active window) |

## Requirements

- macOS (Apple Silicon — M1/M2/M3/M4)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Microphone access + Accessibility permission (for window context)

## Setup

```bash
uv sync
```

Models are downloaded automatically on first run from HuggingFace.

## Run

```bash
# Normal mode
uv run stark

# Dev mode (verbose timing + debug logs)
uv run stark --dev

# Custom config
uv run stark --config path/to/config.yaml
```

## Configuration

Edit `config/default.yaml` or create `config/local.yaml` to override specific keys.
`local.yaml` is git-ignored — safe for personal settings.

## Architecture

```
Microphone → VAD → STT → LLM (speculative) → TTS → Speaker
                              ↑
                         Active Window Context (AXUIElement)
```

Each stage is modular with a clean interface. Swap any component by editing the relevant module under `src/stark/{vad,stt,llm,tts}/`.
