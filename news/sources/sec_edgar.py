"""SEC EDGAR filings via RSS and company submissions API."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from news.models import Signal
from news.sources.base import parse_published
from news.sources.rss import parse_atom_text

logger = logging.getLogger(__name__)

CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

# Latest filings RSS feeds (form-type filtered)
EDGAR_RSS_FEEDS = (
    (
        "8-K",
        "https://www.sec.gov/cgi-bin/browse-edgar?"
        "action=getcurrent&type=8-k&company=&dateb=&owner=include&count=40&output=atom",
    ),
    (
        "10-Q",
        "https://www.sec.gov/cgi-bin/browse-edgar?"
        "action=getcurrent&type=10-q&company=&dateb=&owner=include&count=40&output=atom",
    ),
    (
        "10-K",
        "https://www.sec.gov/cgi-bin/browse-edgar?"
        "action=getcurrent&type=10-k&company=&dateb=&owner=include&count=40&output=atom",
    ),
)

IMPORTANT_FORMS = frozenset({"8-K", "10-Q", "10-K", "10-K/A", "8-K/A", "10-Q/A"})


async def _load_cik_map(client: httpx.AsyncClient, user_agent: str) -> dict[str, str]:
    """Return ticker -> zero-padded CIK mapping."""
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    try:
        resp = await client.get(CIK_MAP_URL, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        logger.warning("SEC CIK map fetch failed: %s", exc)
        return {}

    mapping: dict[str, str] = {}
    if isinstance(data, dict):
        for entry in data.values():
            if not isinstance(entry, dict):
                continue
            ticker = str(entry.get("ticker", "")).upper()
            cik = str(entry.get("cik_str", ""))
            if ticker and cik:
                mapping[ticker] = cik.zfill(10)
    return mapping


def _parse_atom_feed(xml_text: str, form_type: str, watchlist: list[str]) -> list[Signal]:
    return parse_atom_text(xml_text, source_label=f"sec_edgar:{form_type}", watchlist=watchlist)


async def fetch_sec_edgar(
    *,
    watchlist_tickers: list[str] | None = None,
    user_agent: str = "auto-trade/1.0 (contact@example.com)",
) -> list[Signal]:
    watchlist = watchlist_tickers or []
    signals: list[Signal] = []
    headers = {"User-Agent": user_agent, "Accept": "application/json, application/atom+xml"}

    async with httpx.AsyncClient(timeout=20, headers=headers) as client:
        # Latest filings RSS
        for form_type, feed_url in EDGAR_RSS_FEEDS:
            try:
                resp = await client.get(feed_url)
                resp.raise_for_status()
                signals.extend(_parse_atom_feed(resp.text, form_type, watchlist))
            except httpx.HTTPError as exc:
                logger.warning("SEC EDGAR RSS failed for %s: %s", form_type, exc)

        # Company submissions for watchlist tickers
        cik_map = await _load_cik_map(client, user_agent)
        seen_accessions: set[str] = {s.url for s in signals}

        for sym in watchlist[:10]:
            cik = cik_map.get(sym.upper())
            if not cik:
                continue
            try:
                resp = await client.get(
                    f"https://data.sec.gov/submissions/CIK{cik}.json"
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                logger.warning("SEC submissions failed for %s: %s", sym, exc)
                continue

            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form") or []
            accessions = recent.get("accessionNumber") or []
            filing_dates = recent.get("filingDate") or []
            primary_docs = recent.get("primaryDocument") or []
            descriptions = recent.get("primaryDocDescription") or []

            for i, form in enumerate(forms[:10]):
                if form not in IMPORTANT_FORMS:
                    continue
                accession = accessions[i] if i < len(accessions) else ""
                if not accession:
                    continue
                accession_clean = accession.replace("-", "")
                cik_int = str(int(cik))  # strip leading zeros for URL path
                doc = primary_docs[i] if i < len(primary_docs) else ""
                url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                    f"{accession_clean}/{doc}"
                )
                if url in seen_accessions:
                    continue
                seen_accessions.add(url)

                desc = descriptions[i] if i < len(descriptions) else form
                title = f"{sym} {form}: {desc}".strip()
                filing_date = filing_dates[i] if i < len(filing_dates) else None
                published = parse_published(filing_date) if filing_date else None
                if published is None:
                    published = datetime.now(timezone.utc)

                signals.append(
                    Signal(
                        url=url,
                        title=title,
                        source=f"sec_edgar:{form}",
                        published_at=published,
                        snippet=title[:500],
                        tickers_hint=[sym.upper()],
                    )
                )

    return signals
