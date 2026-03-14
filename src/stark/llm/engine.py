"""MLX-LM inference with speculative decoding (main + draft model)."""

from __future__ import annotations

import logging
from typing import Iterator, Optional

log = logging.getLogger(__name__)


class LLMEngine:
    """
    Wraps mlx-lm for text generation with optional speculative decoding.

    Speculative decoding requires main and draft models to share the same
    tokenizer vocabulary (e.g. both Qwen 2.5 family).
    """

    def __init__(
        self,
        model: str,
        draft_model: Optional[str] = None,
        max_tokens: int = 60,
        temperature: float = 0.7,
        system_prompt: str = "",
    ) -> None:
        self.model_name = model
        self.draft_model_name = draft_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt

        self._model = None
        self._tokenizer = None
        self._draft_model = None
        self._draft_tokenizer = None

    def load(self) -> None:
        """Explicitly load models into memory (call once at startup)."""
        from mlx_lm import load

        log.info("Loading LLM: %s", self.model_name)
        self._model, self._tokenizer = load(self.model_name)

        if self.draft_model_name:
            log.info("Loading draft model: %s", self.draft_model_name)
            self._draft_model, self._draft_tokenizer = load(self.draft_model_name)

        log.info("LLM ready.")

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def generate(self, user_text: str, context: Optional[str] = None) -> str:
        """
        Generate a response for user_text, returning the full string.

        Args:
            user_text: The transcribed user utterance.
            context:   Optional active-window context string to prepend.

        Returns:
            Generated response text.
        """
        return "".join(self.stream(user_text, context=context))

    def stream(self, user_text: str, context: Optional[str] = None) -> Iterator[str]:
        """
        Stream generated tokens for user_text.

        Yields string fragments as they are produced.
        """
        self._ensure_loaded()
        from mlx_lm import stream_generate

        system = self._build_system_prompt(context)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        kwargs = dict(
            model=self._model,
            tokenizer=self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temp=self.temperature,
        )

        if self._draft_model is not None:
            kwargs["draft_model"] = self._draft_model

        log.debug("LLM prompt: %r", prompt[:120])

        for response in stream_generate(**kwargs):
            yield response.text

    def _build_system_prompt(self, context: Optional[str]) -> str:
        if context and context.strip():
            return f"{self.system_prompt}\n\nActive context: {context.strip()}"
        return self.system_prompt
