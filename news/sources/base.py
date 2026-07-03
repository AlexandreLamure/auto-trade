"""News source plugin protocol and helpers."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Protocol
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from news.models import RawArticle

_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b")

_TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
    }
)


class NewsSource(Protocol):
    name: str

    async def fetch(self) -> list[RawArticle]: ...


NewsSourceFn = Callable[[], Awaitable[list[RawArticle]]]


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    query = parse_qs(parsed.query, keep_blank_values=False)
    filtered = {k: v for k, v in query.items() if k.lower() not in _TRACKING_PARAMS}
    clean_query = urlencode(filtered, doseq=True)
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            parsed.params,
            clean_query,
            "",
        )
    )


def url_hash(url: str) -> str:
    normalized = normalize_url(url)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def parse_published(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(value).astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def extract_tickers(text: str, *, known: list[str] | None = None) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    for match in _TICKER_RE.finditer(text):
        sym = match.group(1).upper()
        if sym not in seen:
            seen.add(sym)
            found.append(sym)

    if known:
        upper_text = text.upper()
        for sym in known:
            s = sym.upper()
            if s in seen:
                continue
            if re.search(rf"\b{re.escape(s)}\b", upper_text):
                seen.add(s)
                found.append(s)

    return found


def normalize_title(title: str) -> str:
    text = title.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def title_keywords(title: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "at",
        "is", "are", "was", "were", "with", "as", "by", "from", "its", "it",
        "that", "this", "after", "over", "about", "into", "says", "say",
    }
    words = normalize_title(title).split()
    return {w for w in words if len(w) > 2 and w not in stop}


def title_similarity(a: str, b: str) -> float:
    ka = title_keywords(a)
    kb = title_keywords(b)
    if not ka or not kb:
        return 0.0
    intersection = ka & kb
    union = ka | kb
    return len(intersection) / len(union)
