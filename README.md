# Auto-Trade – Local Autonomous Trading Agent

A fully local autonomous stock trading agent that uses a local LLM (via Ollama),
Alpaca paper trading, and MCP tool servers. A multi-agent investment committee
performs deep research, debates the best move, and rebalances the paper portfolio
toward 30-day PnL — up to 10 minutes per cycle, on the US cash session by default
(news before the open, trade at/after the open, optional afternoon review).

Market intelligence is provided by a **separate news analysis service** that
continuously fetches news, groups articles into market events, and stores
structured summaries in a shared SQLite event store.

---

## Architecture

```
news_main.py  (session cron: 09:20 and 12:50 ET, weekdays)
  └── news/pipeline.py
       ├── news/collector.py       (APIs + RSS feeds)
       ├── news/analyzer.py        (dedupe, group, LLM enrich)
       └── store/events.py         → data/events.db

trade_main.py  (session cron: 09:40 and 13:00 ET, weekdays)
  └── agent/orchestrator.py
       ├── agent/research.py        (Phase 1: Alpaca data + event store query)
       ├── agent/deliberation.py    (Phase 2: 4 traders + chair debate)
       ├── agent/personas.py        (trader personalities)
       ├── agent/risk.py            (pre-trade validation)
       ├── agent/decision.py        (PortfolioDecision parser)
       ├── agent/llm_client.py      (Ollama via OpenAI SDK)
       └── servers/manager.py       (Alpaca MCP only)
            └── servers/alpaca.py   → alpaca-mcp-server (stdio)
```

**News service** (09:20 and 12:50 ET weekdays): resolves a dynamic watchlist from Alpaca (held
positions + gainers/losers + event tickers, same as the trading agent); fetches from RSS
(first-class), NewsAPI, Finnhub, Alpha Vantage, and Marketaux; deduplicates
articles; groups into market events; LLM-enriches each event with summary,
sentiment, importance, and tickers.

**Trading agent** (09:40 and 13:00 ET weekdays): refreshes news if the event store is stale,
then reads portfolio data from Alpaca and market events from the store. Cycles still
run if the cash session opens within 20 minutes so the open is not skipped.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | |
| [Ollama](https://ollama.com/download) | Must be running: `ollama serve` |
| An Ollama model | `ollama pull qwen2.5:7b` (minimum); Qwen3+ recommended for committee |
| Alpaca paper account | [app.alpaca.markets](https://app.alpaca.markets/paper/dashboard/overview) |
| News API keys (optional) | RSS runs without keys; add API keys for broader coverage |

Paper API keys must start with `PK`; live keys start with `AK`. The trading agent validates the prefix against `ALPACA_PAPER_TRADE` at startup.

---

## Setup

### 1. Create a virtual environment and install dependencies

```bash
cd auto-trade
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pre-commit install
```

This installs a **gitleaks** pre-commit hook that scans every commit for secrets (API keys, tokens, etc.) before it lands in git.

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials. See `.env.example` for all options.

### 3. Pull an Ollama model and verify it is running

```bash
ollama pull qwen2.5:7b
ollama serve
```

---

## Running

### Two-process model (recommended)

Run both services in separate terminals:

```bash
# Terminal 1 — news intelligence (populates event store)
python news_main.py

# Terminal 2 — trading agent (reads event store)
python trade_main.py
```

### News service only (populate event store)

```bash
python news_main.py --once
```

### Trading agent — single cycle (testing)

```bash
python trade_main.py --once
```

---

## Logging

Each agent writes human-readable cycle logs to a single file. Cycles are separated
by banner lines. Warnings and errors go to stderr only (nothing printed for normal
operation).

| File | Service |
|---|---|
| `logs/news.log` | Sources fetched, article counts, LLM enrichment |
| `logs/trading.log` | Alpaca calls, event store query, committee deliberation, orders |

---

## Project structure

```
auto-trade/
├── agent/
│   ├── decision.py         # PortfolioDecision + persona parsers
│   ├── deliberation.py     # Multi-round committee loop
│   ├── llm_client.py       # Ollama / OpenAI-compatible LLM client
│   ├── orchestrator.py     # Cycle coordinator, execution, logging
│   ├── personas.py         # Trader + chair prompts
│   ├── research.py         # Alpaca research + event store query
│   ├── risk.py             # Pre-trade validation
│   └── workflow.py         # Shared MCP helpers
├── news/
│   ├── analyzer.py         # Dedupe, group, LLM enrich
│   ├── collector.py        # Multi-source fetch orchestration
│   ├── feeds.py            # RSS feed configuration
│   ├── pipeline.py         # One news cycle
│   └── sources/            # NewsAPI, Finnhub, RSS, etc.
├── store/
│   ├── db.py               # SQLite + WAL mode
│   └── events.py           # Event store CRUD and queries
├── servers/
│   ├── alpaca.py           # Alpaca MCP subprocess launcher
│   └── manager.py          # MCPManager: Alpaca session only
├── config/
│   └── settings.py         # Pydantic settings (reads .env)
├── data/
│   └── events.db           # Shared event store (gitignored)
├── logs/
│   ├── news.log            # News cycle log (human-readable)
│   └── trading.log         # Trading cycle log (human-readable)
├── news_main.py            # News service entry point
├── trade_main.py           # Trading agent entry point
├── .env.example
├── requirements.txt
└── README.md
```

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `ALPACA_API_KEY` | – | Alpaca API key (`PK…` for paper, `AK…` for live) |
| `ALPACA_SECRET_KEY` | – | Alpaca secret key |
| `ALPACA_PAPER_TRADE` | `true` | Use paper trading endpoint |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model name |
| `EVENT_STORE_PATH` | `data/events.db` | Shared SQLite event store |
| `NEWS_LOOP_INTERVAL_HOURS` | `6` | Fallback news interval if `SESSION_SCHEDULE=false` |
| `LOOP_INTERVAL_HOURS` | `6` | Fallback trading interval if `SESSION_SCHEDULE=false` |
| `SESSION_SCHEDULE` | `true` | Use US cash-session cron times instead of a raw interval |
| `MARKET_TIMEZONE` | `America/New_York` | Timezone for session cron |
| `NEWS_SESSION_TIMES` | `09:20,12:50` | Weekday ET news runs |
| `TRADE_SESSION_TIMES` | `09:40,13:00` | Weekday ET trading runs |
| `NEWS_STALE_MINUTES` | `90` | Refresh news before a trade if the store is older |
| `NEWSAPI_KEY` | – | NewsAPI.org key (optional) |
| `FINNHUB_API_KEY` | – | Finnhub key (optional) |
| `ALPHA_VANTAGE_API_KEY` | – | Alpha Vantage key (optional) |
| `MARKETAUX_API_KEY` | – | Marketaux key (optional) |
| `RESEARCH_SYMBOL_COUNT` | `15` | Max candidate symbols after dedup |
| `MOVER_CANDIDATE_SLOTS` | `6` | Slots reserved for gainers and losers |
| `EVENT_CANDIDATE_SLOTS` | `6` | Slots reserved for event-discovered tickers |
| `MIN_CANDIDATE_PRICE` | `5.0` | Drop candidates below this last price |
| `MIN_ADV_SHARES` | `500000` | Drop candidates below this 20-day avg volume |
| `EVENTS_SINCE_HOURS` | `72` | How far back to query events |
| `EVENTS_MIN_IMPORTANCE` | `2` | Min importance for symbol events |
| `ENABLE_DEBATE_ROUND` | `true` | Run Round 2 trader debate before chair |
| `MAX_CYCLE_SECONDS` | `600` | Max cycle duration (10 minutes) |
| `MAX_ORDERS_PER_CYCLE` | `3` | Max orders per rebalancing cycle |
| `MAX_POSITION_PCT` | `0.20` | Max fraction of portfolio equity per name (20%) |
| `ENABLE_THINKING` | `true` | LLM reasoning mode (Qwen3+ models) |

The optimization horizon is fixed at **30 days** in code (`HORIZON_DAYS` in `agent/personas.py`).

---

## Notes and limitations

- **Paper trading only by default.** Set `ALPACA_PAPER_TRADE=false` and use live keys to trade with real money — do so at your own risk.
- The agent places **market orders** with `time_in_force=day`. Orders placed outside market hours queue until the next session open.
- Run `python news_main.py --once` before the first trading cycle to populate the event store.
- RSS feeds run without API keys; optional API keys extend coverage.
- Model quality matters for JSON-only persona output. `qwen2.5:7b` is the documented minimum; larger models (e.g. Qwen3) work better with `ENABLE_THINKING=true`.
- Buy orders are capped by `max_position_pct` using latest bar close prices from research.
