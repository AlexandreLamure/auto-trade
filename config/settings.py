from functools import cached_property

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from news.weights import parse_source_weights

_SOURCE_WEIGHTS_DEFAULT = (
    '{"sec_edgar":1.0,"company_ir":1.0,"rss":0.9,"api":0.8,'
    '"polymarket":0.6,"reddit":0.5,"stocktwits":0.4,"google_trends":0.3}'
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
    loop_interval_hours: int = Field(6, description="How often the agent runs (hours)")

    # Committee
    research_symbol_count: int = Field(6, description="Max candidate symbols after dedup")
    enable_debate_round: bool = Field(
        True, description="Run Round 2 trader debate before chair consensus"
    )
    max_cycle_seconds: int = Field(600, description="Hard cap on committee cycle duration (seconds)")
    max_orders_per_cycle: int = Field(3, description="Max Alpaca orders per committee cycle")

    # Event query (trading agent reads from store)
    events_since_hours: int = Field(72, description="How far back to query market events")
    events_min_importance: int = Field(2, description="Minimum event importance (1-5)")
    events_limit: int = Field(30, description="Max symbol-specific events per cycle")
    macro_events_min_importance: int = Field(4, description="Min importance for macro events")
    macro_events_limit: int = Field(10, description="Max macro events per cycle")

    # Risk guard
    max_position_pct: float = Field(
        0.20, description="Max fraction of cash to allocate per buy (0–1)"
    )

    # LLM thinking mode (Qwen3 etc.)
    enable_thinking: bool = Field(True, description="Enable LLM thinking/reasoning mode")

    @cached_property
    def source_weights_map(self) -> dict[str, float]:
        return parse_source_weights(self.source_weights)


settings = Settings()
