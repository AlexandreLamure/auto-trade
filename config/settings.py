from pydantic_settings import BaseSettings, SettingsConfigDict

from pydantic import Field


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
    watchlist_tickers: str = Field(
        "AAPL,NVDA,MSFT,GOOGL,AMZN",
        description="Comma-separated tickers for RSS/API watchlist",
    )

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

    # Print LLM thinking and decision JSON to stdout each cycle
    verbose: bool = Field(False, description="Verbose output: print LLM internals to stdout")


settings = Settings()
