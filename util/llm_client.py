"""
OllamaClient – thin wrapper around openai.AsyncOpenAI pointing at a local
Ollama instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion


@dataclass
class LLMResponse:
    content: str | None
    thinking: str | None = None


class OllamaClient:
    """Async LLM client backed by a local Ollama instance."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        enable_thinking: bool = False,
        num_ctx: int = 16384,
        temperature: float = 0.4,
    ) -> None:
        self._model = model
        self._enable_thinking = enable_thinking
        self._num_ctx = num_ctx
        self._temperature = temperature
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key="ollama",
        )

    async def chat(self, messages: list[dict[str, Any]]) -> LLMResponse:
        """Send a chat request and return a structured LLMResponse."""
        # Ollama reads context size from options.num_ctx (default is often 4096).
        # Without this, long research briefs are truncated and JSON schemas are lost.
        extra_body: dict[str, Any] = {
            "think": self._enable_thinking,
            "options": {
                "num_ctx": self._num_ctx,
                "temperature": self._temperature,
            },
        }
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "extra_body": extra_body,
        }

        completion: ChatCompletion = await self._client.chat.completions.create(**kwargs)
        msg = completion.choices[0].message

        thinking: str | None = None
        if hasattr(msg, "thinking") and msg.thinking:  # type: ignore[attr-defined]
            thinking = msg.thinking  # type: ignore[attr-defined]

        return LLMResponse(
            content=msg.content,
            thinking=thinking,
        )
