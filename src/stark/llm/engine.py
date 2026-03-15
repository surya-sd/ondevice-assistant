"""MLX-LM inference with speculative decoding (main + draft model)."""

from __future__ import annotations

import logging
from collections import deque
from typing import Iterator, List, Optional

log = logging.getLogger(__name__)


class LLMEngine:
    """
    Wraps mlx-lm for text generation with optional speculative decoding.

    Maintains a rolling conversation history so context persists across turns.
    Speculative decoding requires main and draft models to share the same
    tokenizer vocabulary (e.g. both Qwen 2.5 family).
    """

    def __init__(
        self,
        model: str,
        draft_model: Optional[str] = None,
        max_tokens: int = 80,
        temperature: float = 0.8,
        top_p: float = 0.8,
        system_prompt: str = "",
        history_turns: int = 20,
    ) -> None:
        self.model_name = model
        self.draft_model_name = draft_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt

        # Rolling history: stores alternating user/assistant dicts.
        # maxlen = history_turns * 2 (each turn = 1 user + 1 assistant message).
        self._history: deque[dict] = deque(maxlen=history_turns * 2)

        self._model = None
        self._tokenizer = None
        self._draft_model = None
        self._draft_tokenizer = None

        # KV cache — persists across turns so only new tokens are processed
        self._kv_cache = None
        self._cached_tokens: int = 0

    def load(self) -> None:
        """Explicitly load models into memory (call once at startup)."""
        from mlx_lm import load

        log.info("Loading LLM: %s", self.model_name)
        self._model, self._tokenizer = load(self.model_name)

        if self.draft_model_name and self.draft_model_name.strip():
            log.info("Loading draft model: %s", self.draft_model_name)
            self._draft_model, self._draft_tokenizer = load(self.draft_model_name)

        from mlx_lm.models.cache import make_prompt_cache
        self._kv_cache = make_prompt_cache(self._model)
        self._cached_tokens = 0
        log.info("LLM ready.")

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def generate(self, user_text: str, context: Optional[str] = None) -> str:
        """Generate a response, returning the full string and updating history."""
        response = "".join(self.stream(user_text, context=context))
        return response

    def stream(self, user_text: str, context: Optional[str] = None) -> Iterator[str]:
        """
        Stream generated tokens for user_text.

        Uses a persistent KV cache — only the new tokens (current user message)
        are processed each turn instead of re-processing the full history.
        Yields string fragments as they are produced.
        """
        self._ensure_loaded()
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        self._history.append({"role": "user", "content": user_text})

        system = self._build_system_prompt(context)
        messages = [{"role": "system", "content": system}, *self._history]

        full_prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_tokens: List[int] = self._tokenizer.encode(full_prompt)

        # Only pass tokens the model hasn't seen yet in this cache
        new_tokens = full_tokens[self._cached_tokens:]

        log.debug(
            "LLM: %d cached + %d new tokens (%d turns)",
            self._cached_tokens, len(new_tokens), len(messages),
        )

        kwargs = dict(
            model=self._model,
            tokenizer=self._tokenizer,
            prompt=new_tokens,
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=self.temperature, top_p=self.top_p),
            prompt_cache=self._kv_cache,
        )

        if self._draft_model is not None:
            kwargs["draft_model"] = self._draft_model

        collected = []
        n_generated = 0
        for chunk in stream_generate(**kwargs):
            collected.append(chunk.text)
            n_generated += 1
            yield chunk.text

        response = "".join(collected)
        self._history.append({"role": "assistant", "content": response})
        # Cache now holds: previously cached tokens + new prompt tokens + generated tokens
        self._cached_tokens = len(full_tokens) + n_generated

    def clear_history(self) -> None:
        """Wipe conversation history and reset KV cache — starts a fresh session."""
        self._history.clear()
        if self._model is not None:
            from mlx_lm.models.cache import make_prompt_cache
            self._kv_cache = make_prompt_cache(self._model)
        self._cached_tokens = 0
        log.info("Conversation history cleared.")

    def _build_system_prompt(self, context: Optional[str]) -> str:
        if context and context.strip():
            return (
                f"{self.system_prompt}\n\n"
                f"[Background: user is currently in {context.strip()}. "
                f"Use this silently for relevance only — never mention it unless asked.]"
            )
        return self.system_prompt
