import json
from functools import cached_property

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_SOURCE_WEIGHTS_DEFAULT = (
    '{"sec_edgar":1.0,"company_ir":1.0,"rss":0.9,"api":0.8,'
    '"polymarket":0.3,"reddit":0.5,"stocktwits":0.4,"google_trends":0.3}'
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Alpaca
    alpaca_api_key: str = Field(..., description="Alpaca API key")
    alpaca_secret_key: str = Field(..., description="Alpaca secret key")
    alpaca_paper_trade: bool = Field(True, description="Use paper trading")

    # Ollama / LLM
    ollama_base_url: str = Field(
        "http://localhost:11434/v1", description="Ollama OpenAI-compatible base URL"
    )
    ollama_model: str = Field("qwen2.5:7b", description="Ollama model name")

    # Event store (shared by news service and trading agent)
    event_store_path: str = Field("data/events.db", description="SQLite event store path")

    # News API keys (optional – source skipped if unset)
    newsapi_key: str = Field("", description="NewsAPI.org API key")
    finnhub_api_key: str = Field("", description="Finnhub API key")
    alpha_vantage_api_key: str = Field("", description="Alpha Vantage API key")
    marketaux_api_key: str = Field("", description="Marketaux API key")

    # News service
    news_loop_interval_hours: int = Field(
        6, description="How often the news service runs (hours)"
    )

    # Signal source weights (JSON map of source type to weight)
    source_weights: str = Field(
        _SOURCE_WEIGHTS_DEFAULT,
        description="JSON map of source type to weight",
    )
    ir_rss_feeds: str = Field(
        "{}", description='JSON map of ticker to IR RSS URL, e.g. {"AAPL":"https://..."}'
    )
    sec_user_agent: str = Field(
        "auto-trade/1.0 (contact@example.com)",
        description="User-Agent for SEC EDGAR API requests (required by SEC)",
    )
    google_trends_spike_threshold: float = Field(
        1.5, description="Emit Google Trends signal when index exceeds this × trailing avg"
    )

    # Optional signal sources
    enable_reddit: bool = Field(True, description="Fetch Reddit JSON feeds")
    enable_polymarket: bool = Field(True, description="Fetch Polymarket prediction markets")
    enable_google_trends: bool = Field(True, description="Fetch Google Trends spikes")
    enable_sec_edgar: bool = Field(True, description="Fetch SEC EDGAR filings")
    enable_stocktwits: bool = Field(False, description="Fetch Stocktwits (often rate-limited)")

    # Trading scheduler
    loop_interval_hours: int = Field(6, description="Fallback interval when session schedule is off")
    session_schedule: bool = Field(
        True, description="Run on US cash-session cron times instead of a raw interval"
    )
    market_timezone: str = Field(
        "America/New_York", description="Timezone for session cron schedules"
    )
    news_session_times: str = Field(
        "09:20,12:50",
        description="Weekday ET times to run the news service (HH:MM,HH:MM)",
    )
    trade_session_times: str = Field(
        "09:40,13:00",
        description="Weekday ET times to run the trading agent (HH:MM,HH:MM)",
    )
    news_stale_minutes: int = Field(
        90, description="Refresh news before a trade cycle if the store is older than this"
    )

    # Committee
    research_symbol_count: int = Field(
        15, description="Max candidate symbols after dedup (excludes holdings)"
    )
    mover_candidate_slots: int = Field(
        6, description="Candidate slots reserved for gainers and losers"
    )
    event_candidate_slots: int = Field(
        6, description="Candidate slots reserved for event-discovered tickers"
    )
    min_candidate_price: float = Field(
        5.0, description="Drop candidate names below this last price"
    )
    min_adv_shares: float = Field(
        500_000, description="Drop candidate names with 20d avg volume below this"
    )
    enable_debate_round: bool = Field(
        True, description="Run Round 2 trader debate before chair consensus"
    )
    max_cycle_seconds: int = Field(600, description="Hard cap on committee cycle duration (seconds)")
    max_orders_per_cycle: int = Field(3, description="Max Alpaca orders per committee cycle")
    min_agreeing_personas: int = Field(
        2, description="Minimum traders who must agree on side before an order survives"
    )
    min_order_confidence: float = Field(
        0.55, description="Minimum mean persona confidence for a surviving order"
    )
    require_catalyst_or_setup: bool = Field(
        True, description="New buys need a matching event unless extra traders agree"
    )
    earnings_blackout_days: int = Field(
        5, description="Treat earnings/guidance events this recent as an earnings window"
    )

    # Event query (trading agent reads from store)
    events_since_hours: int = Field(72, description="How far back to query market events")
    events_min_importance: int = Field(2, description="Minimum event importance (1-5)")
    events_limit: int = Field(30, description="Max symbol-specific events per cycle")
    macro_events_min_importance: int = Field(4, description="Min importance for macro events")
    macro_events_limit: int = Field(10, description="Max macro events per cycle")
    event_ttl_days: int = Field(30, description="Delete events older than this many days")
    event_discovery_limit: int = Field(
        12, description="Max symbols to add from high-importance events"
    )

    # Risk guard
    max_position_pct: float = Field(
        0.20, description="Max fraction of portfolio equity per name (0–1)"
    )
    block_earnings_buys: bool = Field(
        True, description="Reject new buys while a name is in the earnings window"
    )
    stop_atr_multiple: float = Field(
        2.0, description="Stop distance as a multiple of 14-day ATR"
    )
    stop_pct: float = Field(0.08, description="Minimum stop distance as a fraction of price")
    risk_per_name_pct: float = Field(
        0.005, description="Equity fraction risked per name (vol-aware size)"
    )
    time_stop_days: int = Field(
        30, description="Sell held losers at or past this many days (0 disables)"
    )
    enable_stops: bool = Field(True, description="Attach a stop after each buy fill")

    # LLM thinking mode (Qwen3 etc.)
    enable_thinking: bool = Field(True, description="Enable LLM thinking/reasoning mode")

    @cached_property
    def source_weights_map(self) -> dict[str, float]:
        return _parse_source_weights(self.source_weights)


def _parse_source_weights(raw: str) -> dict[str, float]:
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in data.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


settings = Settings()
