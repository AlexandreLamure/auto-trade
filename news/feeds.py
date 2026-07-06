"""RSS feed configuration – first-class news sources."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeedConfig:
    url: str
    label: str


STATIC_FEEDS: list[FeedConfig] = [
    FeedConfig(url="https://finance.yahoo.com/news/rssindex", label="yahoo_market"),
    FeedConfig(url="https://www.investing.com/rss/news_25.rss", label="investing_macro"),
    FeedConfig(url="https://www.investing.com/rss/news_301.rss", label="investing_earnings"),
    FeedConfig(url="https://www.federalreserve.gov/feeds/press_all.xml", label="fed_press"),
    FeedConfig(url="https://home.treasury.gov/system/files/136/treasury-rss.xml", label="treasury"),
    FeedConfig(
        url="https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml",
        label="fda_press",
    ),
]


def ticker_feeds(tickers: list[str]) -> list[FeedConfig]:
    """Yahoo Finance per-ticker RSS feeds."""
    feeds: list[FeedConfig] = []
    for symbol in tickers:
        sym = symbol.strip().upper()
        if not sym:
            continue
        feeds.append(
            FeedConfig(
                url=f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US",
                label=f"yahoo_{sym}",
            )
        )
    return feeds


def build_feed_list(watchlist_tickers: list[str]) -> list[FeedConfig]:
    return STATIC_FEEDS + ticker_feeds(watchlist_tickers)
