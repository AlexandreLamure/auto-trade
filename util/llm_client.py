"""
OllamaClient – async wrapper around the native Ollama /api/chat endpoint.

The OpenAI-compatible /v1/chat/completions path in Ollama 0.30+ does not
forward num_ctx, so the runner loads at the VRAM default (4096). Native
/api/chat applies options.num_ctx and reloads the model if it differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class LLMResponse:
    content: str | None
    thinking: str | None = None


def _native_base_url(base_url: str) -> str:
    """Strip a trailing /v1 so OpenAI-style URLs still hit native /api/*."""
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url


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
        # Chat completions can run tens of seconds; this is HTTP read timeout,
        # not a news/trade cycle cap.
        self._http = httpx.AsyncClient(
            base_url=_native_base_url(base_url),
            timeout=httpx.Timeout(600.0),
        )

    async def chat(self, messages: list[dict[str, Any]]) -> LLMResponse:
        """Send a chat request and return a structured LLMResponse."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "think": self._enable_thinking,
            "options": {
                "num_ctx": self._num_ctx,
                "temperature": self._temperature,
            },
        }
        response = await self._http.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return LLMResponse(content=None)

        error = data.get("error")
        if error:
            raise RuntimeError(f"Ollama chat failed: {error}")

        msg = data.get("message") or {}
        if not isinstance(msg, dict):
            return LLMResponse(content=None)

        thinking = msg.get("thinking")
        return LLMResponse(
            content=msg.get("content"),
            thinking=str(thinking) if thinking else None,
        )
