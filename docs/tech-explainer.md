# Stark — Tech Explainer

A plain-English reference for the jargon used in this project.

---

## How it works end-to-end

```
You speak → VAD → STT → LLM → TTS → You hear
                         ↑
                  active window context
```

---

## The pipeline stages

### VAD — Voice Activity Detection
Software that listens to raw audio and answers one question: *is a human speaking right now, yes or no?* Without it, the pipeline would try to transcribe silence, background noise, your keyboard — everything. Silero VAD is a tiny neural network trained specifically to detect human speech in real-time.

### STT — Speech To Text
Converts audio waveform (raw sound data) into words. Whisper is OpenAI's model for this — it was trained on 680,000 hours of audio so it understands accents, mumbling, background noise well. The "large-v3-turbo" version is a distilled (compressed, faster) version of their best model.

### LLM — Large Language Model
The "brain." A neural network trained on massive amounts of text that can understand and generate human language. Qwen 2.5 3B means it has 3 billion internal parameters (knobs that were tuned during training). More parameters generally = smarter, but slower and heavier.

### TTS — Text To Speech
The reverse of STT — converts text back into audio. Kokoro is a neural TTS model (82 million parameters) that learned to mimic human speech patterns from recordings. "ONNX" is a standard format for running neural networks efficiently without needing the original training framework.

---

## Key techniques

### Speculative Decoding
LLMs generate text one token at a time — each word requires a full pass through the whole model. Speculative decoding cheats this by running a tiny fast model (0.5B) to *guess* the next several tokens ahead, then the big model verifies all the guesses in one shot. If the guesses are right (they usually are for common phrases), you get multiple tokens for the cost of one. Net result: 2-3x faster with zero quality loss.

### Quantisation (the Q4 in model names)
Neural network weights are normally stored as 32-bit or 16-bit floating point numbers. Quantisation rounds them down to 4-bit integers (Q4). This makes the model ~4x smaller and faster to load, with only a small accuracy penalty. A 3B model at Q4 is ~1.8GB instead of ~6GB — fits comfortably in RAM alongside everything else.

### Chunked / Streaming
Instead of waiting for the full audio or full response before doing anything, we process in pieces. VAD reads 32ms audio chunks continuously. TTS starts playing audio as soon as the first sentence is ready rather than waiting for the full response. This is why the latency feels low — stages overlap instead of waiting in a strict queue.

---

## Hardware & platform concepts

### MLX
Apple's own machine learning framework built specifically for Apple Silicon (M1/M2/M3/M4). Normal AI frameworks like PyTorch were designed for NVIDIA GPUs — they work on Mac but don't use the hardware well. MLX talks directly to the unified memory architecture of Apple chips, which is why you can run a 3B model fast on a MacBook.

### Unified Memory (why M1 is good for this)
Normal computers have separate CPU RAM and GPU VRAM. Your GPU can only use what's in its own VRAM — so a GPU with 8GB VRAM can only run models up to 8GB. Apple Silicon has one shared pool of memory that both CPU and GPU read from. Your 16GB M1 means all 16GB is available to the model — equivalent to having a 16GB GPU, which is a $1000+ discrete card.

---

## LLM concepts

### Context Window
How much text the LLM can "see" at once. Think of it as working memory. Qwen 2.5 3B has a 32K token context window — roughly 24,000 words. We keep 20 conversation turns in it, which is well within limits. When the window fills up, oldest messages drop off — the model forgets the beginning of very long conversations.

### System Prompt
Hidden instructions sent to the LLM before the conversation starts that define its personality and rules. The user never sees it. Ours tells Stark to be casual, use contractions, avoid "Sure! Absolutely!", cap responses to 3 sentences. It's the main lever for controlling how the model behaves.

---

## macOS specific

### AXUIElement / Accessibility API
macOS has a built-in system where apps can expose their UI structure to other programs — originally designed for screen readers for visually impaired users. We exploit the same API to read the active window title. No screen capture, no special permissions beyond the Accessibility toggle in System Settings.

---

## Tuning reference

| Setting | Value | Why |
|---------|-------|-----|
| VAD silence window | 1500ms | Comfortable pause for "hmm", thinking, mid-sentence gaps |
| LLM temperature | 0.8 | More natural variation, less robotic repetition |
| top_p | 0.8 | Nucleus sampling — keeps token choices plausible |
| Response budget | 80 tokens | Short enough for voice, not so tight it clips sentences |
| TTS speed | 0.95x | Slightly slower = warmer, less rushed |
| History window | 20 turns | ~40 messages, fits comfortably in Qwen's 32K context |
