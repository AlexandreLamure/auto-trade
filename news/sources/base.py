"""News source helpers."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from news.models import Signal

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


# Collapse common finance headline variants so Jaccard matching clusters better.
_TITLE_SYNONYMS = {
    "buyback": "repurchase",
    "buybacks": "repurchase",
    "repurchases": "repurchase",
    "repurchasing": "repurchase",
    "expanding": "expand",
    "expands": "expand",
    "expanded": "expand",
    "increases": "increase",
    "increased": "increase",
    "increasing": "increase",
    "boosts": "boost",
    "boosted": "boost",
    "boosting": "boost",
    "raises": "raise",
    "raised": "raise",
    "raising": "raise",
    "announces": "announce",
    "announced": "announce",
    "announcing": "announce",
    "reports": "report",
    "reported": "report",
    "reporting": "report",
    "earnings": "earning",
    "dividend": "distribution",
    "dividends": "distribution",
    "distributions": "distribution",
}


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
        "why", "how", "what", "here", "heres", "just", "now", "new",
    }
    words = normalize_title(title).split()
    out: set[str] = set()
    for w in words:
        if len(w) <= 2 or w in stop:
            continue
        out.add(_TITLE_SYNONYMS.get(w, w))
    return out


def title_similarity(a: str, b: str) -> float:
    """Headline similarity in [0, 1].

    Uses the max of Jaccard and overlap coefficient so a short headline can
    still match a longer variant that adds secondary context.
    """
    ka = title_keywords(a)
    kb = title_keywords(b)
    if not ka or not kb:
        return 0.0
    inter = len(ka & kb)
    if inter == 0:
        return 0.0
    jaccard = inter / len(ka | kb)
    overlap = inter / min(len(ka), len(kb))
    return max(jaccard, overlap)
