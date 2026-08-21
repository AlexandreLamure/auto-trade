"""Reddit source via OAuth (preferred) or RSS fallback.

Unauthenticated www.reddit.com/*.json returns 403 as of 2026. A script app
(client_credentials) against oauth.reddit.com is the supported path. Without
credentials, public RSS is tried instead.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from news.models import Signal
from news.sources.base import extract_tickers
from news.sources.http_helpers import valid_api_key
from news.sources.rss import parse_atom_text

logger = logging.getLogger(__name__)

SUBREDDITS = ("stocks", "investing", "wallstreetbets")
DEFAULT_USER_AGENT = "python:auto-trade:1.0 (local news agent)"
TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
OAUTH_BASE = "https://oauth.reddit.com"
RSS_URLS = (
    "https://www.reddit.com/r/{sub}/.rss",
    "https://old.reddit.com/r/{sub}/.rss",
)


def _headers(user_agent: str, *, token: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _listing_to_signals(
    data: dict,
    *,
    sub: str,
    watchlist: list[str],
) -> list[Signal]:
    signals: list[Signal] = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data") or {}
        permalink = post.get("permalink") or ""
        if not permalink:
            continue
        full_url = f"https://www.reddit.com{permalink}"
        title = (post.get("title") or "").strip()
        if not title:
            continue
        selftext = (post.get("selftext") or "")[:500]
        snippet = selftext or title[:500]
        created = post.get("created_utc")
        published = (
            datetime.fromtimestamp(created, tz=timezone.utc) if created else None
        )
        tickers = extract_tickers(f"{title} {snippet}", known=watchlist)
        signals.append(
            Signal(
                url=full_url,
                title=title,
                source=f"reddit:{sub}",
                published_at=published,
                snippet=snippet,
                tickers_hint=tickers,
            )
        )
    return signals


async def _access_token(
    client: httpx.AsyncClient,
    *,
    client_id: str,
    client_secret: str,
    user_agent: str,
) -> str:
    resp = await client.post(
        TOKEN_URL,
        auth=(client_id, client_secret),
        data={"grant_type": "client_credentials"},
        headers=_headers(user_agent),
    )
    resp.raise_for_status()
    payload = resp.json()
    token = payload.get("access_token") if isinstance(payload, dict) else None
    if not token:
        raise RuntimeError("Reddit token response missing access_token")
    return str(token)


async def _fetch_via_oauth(
    *,
    watchlist: list[str],
    client_id: str,
    client_secret: str,
    user_agent: str,
) -> list[Signal]:
    signals: list[Signal] = []
    async with httpx.AsyncClient(timeout=20) as client:
        token = await _access_token(
            client,
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        for sub in SUBREDDITS:
            try:
                resp = await client.get(
                    f"{OAUTH_BASE}/r/{sub}/hot",
                    params={"limit": 25},
                    headers=_headers(user_agent, token=token),
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                logger.warning("Reddit OAuth fetch failed for r/%s: %s", sub, exc)
                continue
            if isinstance(data, dict):
                signals.extend(
                    _listing_to_signals(data, sub=sub, watchlist=watchlist)
                )
    return signals


async def _fetch_via_rss(
    *,
    watchlist: list[str],
    user_agent: str,
) -> list[Signal]:
    """Public RSS still works when unauthenticated JSON is 403."""
    signals: list[Signal] = []
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml",
    }
    async with httpx.AsyncClient(timeout=20, headers=headers) as client:
        for sub in SUBREDDITS:
            got_sub = False
            last_exc: Exception | None = None
            for template in RSS_URLS:
                url = template.format(sub=sub)
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                except httpx.HTTPError as exc:
                    last_exc = exc
                    continue
                parsed = parse_atom_text(
                    resp.text,
                    source_label=f"reddit:{sub}",
                    watchlist=watchlist,
                )
                if parsed:
                    signals.extend(parsed)
                    got_sub = True
                    break
            if not got_sub:
                logger.warning(
                    "Reddit RSS fetch failed for r/%s: %s",
                    sub,
                    last_exc or "empty feed",
                )
    return signals


async def fetch_reddit(
    *,
    watchlist_tickers: list[str] | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    client_id: str = "",
    client_secret: str = "",
) -> list[Signal]:
    watchlist = watchlist_tickers or []
    ua = (user_agent or "").strip() or DEFAULT_USER_AGENT

    if valid_api_key(client_id) and valid_api_key(client_secret):
        try:
            return await _fetch_via_oauth(
                watchlist=watchlist,
                client_id=client_id,
                client_secret=client_secret,
                user_agent=ua,
            )
        except (httpx.HTTPError, RuntimeError) as exc:
            logger.warning("Reddit OAuth failed (%s); trying RSS", exc)

    return await _fetch_via_rss(watchlist=watchlist, user_agent=ua)
