"""Append-only human-readable cycle logs for news and trading agents."""

from __future__ import annotations

from pathlib import Path

_SEPARATOR = "=" * 72


class CycleLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def start_cycle(self, header: str) -> None:
        self.line("")
        self.line(_SEPARATOR)
        self.line(header)
        self.line(_SEPARATOR)

    def section(self, title: str) -> None:
        self.line("")
        self.line(f"--- {title} ---")

    def line(self, text: str = "") -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(text + "\n")


news_log: CycleLog | None = None
trading_log: CycleLog | None = None


def init_news_log(path: Path) -> CycleLog:
    global news_log
    news_log = CycleLog(path)
    return news_log


def init_trading_log(path: Path) -> CycleLog:
    global trading_log
    trading_log = CycleLog(path)
    return trading_log


def get_news_log() -> CycleLog:
    if news_log is None:
        raise RuntimeError("news_log not initialized – call init_news_log() first")
    return news_log


def get_trading_log() -> CycleLog:
    if trading_log is None:
        raise RuntimeError("trading_log not initialized – call init_trading_log() first")
    return trading_log
