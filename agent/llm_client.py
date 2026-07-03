"""
OllamaClient – thin wrapper around openai.AsyncOpenAI pointing at a local
Ollama instance for committee deliberation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str | None
    thinking: str | None = None
    raw_message: ChatCompletionMessage | None = None


class OllamaClient:
    """Async LLM client backed by a local Ollama instance."""

    def __init__(self, base_url: str, model: str, enable_thinking: bool = False) -> None:
        self._model = model
        self._enable_thinking = enable_thinking
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key="ollama",
        )

    async def chat(self, messages: list[dict[str, Any]]) -> LLMResponse:
        """Send a chat request and return a structured LLMResponse."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "extra_body": {"think": self._enable_thinking},
        }

        logger.debug(
            "LLM request: %d messages, think=%s",
            len(messages),
            self._enable_thinking,
        )

        completion: ChatCompletion = await self._client.chat.completions.create(**kwargs)
        msg = completion.choices[0].message

        thinking: str | None = None
        if hasattr(msg, "thinking") and msg.thinking:  # type: ignore[attr-defined]
            thinking = msg.thinking  # type: ignore[attr-defined]

        response = LLMResponse(
            content=msg.content,
            thinking=thinking,
            raw_message=msg,
        )

        logger.debug(
            "LLM response: has_content=%s, has_thinking=%s",
            bool(response.content),
            bool(response.thinking),
        )
        return response
