"""Shared LLM JSON block extraction."""

from __future__ import annotations

import re

_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_block(text: str) -> str | None:
    match = _JSON_BLOCK_RE.search(text)
    return match.group(1).strip() if match else None
