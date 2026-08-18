# Auto-Trade – Local Autonomous Trading Agent

A fully local autonomous stock trading agent that uses a local LLM (via Ollama),
Alpaca paper trading, and MCP tool servers. A multi-agent investment committee
performs deep research, debates the best move, and rebalances the paper portfolio
toward 30-day PnL — up to 10 minutes per cycle.

Market intelligence is provided by a news analysis pipeline that fetches news,
groups articles into market events, and stores structured summaries in a shared
SQLite event store. One process runs news and/or trading: a single shot, or a
daily loop at 12:00 America/New_York. Cycles skip immediately when the US cash
session is closed (weekdays 09:30–16:00 ET).

---

## Architecture

```
main.py  (--news-once | --trade-once | --once | --loop)
  ├── news/pipeline.py
  │    ├── news/collector.py       (APIs + RSS feeds)
  │    ├── news/analyzer.py        (dedupe, group, LLM enrich)
  │    └── store/events.py         → data/events.db
  └── agent/orchestrator.py
       ├── agent/research.py        (Phase 1: Alpaca data + event store query)
       ├── agent/deliberation.py    (Phase 2: 4 traders + chair debate)
       ├── agent/personas.py        (trader personalities)
       ├── agent/risk.py            (pre-trade validation)
       ├── agent/decision.py        (PortfolioDecision parser)
       ├── util/llm_client.py       (Ollama via OpenAI SDK)
       └── servers/manager.py       (Alpaca MCP only)
            └── servers/alpaca.py   → alpaca-mcp-server (stdio)
```

**News cycle**: resolves a dynamic watchlist from Alpaca (held positions +
gainers/losers + event tickers, same as the trading agent); fetches from RSS
(first-class), NewsAPI, Finnhub, Alpha Vantage, and Marketaux; deduplicates
articles; groups into market events; LLM-enriches each event with summary,
sentiment, importance, and tickers. Skips immediately if the US cash session
is closed.

**Trading cycle**: reads portfolio data from Alpaca and market events from the
store. Each cycle writes a decision journal (proposals, fills, later 1d/5d/30d
marks) into the event store. Skips immediately if the US cash session is closed.

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

With no flags, `python main.py` prints help and exits. Pick one mode:

```bash
python main.py --news-once     # one news cycle, then exit
python main.py --trade-once    # one trade cycle, then exit
python main.py --once          # news, then trade, then exit
python main.py --loop          # news then trade every day at 12:00 America/New_York
```

`--once` and `--loop` always run news then trading sequentially (the next step
starts when the previous one finishes). If the US cash session is closed when a
cycle starts, that cycle logs `SKIP – market closed` and returns immediately.

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
├── main.py                 # Single entry point
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
| `MAX_ORDERS_PER_CYCLE` | `3` | Max Alpaca orders per committee cycle |
| `MIN_AGREEING_PERSONAS` | `2` | Traders who must agree on side for an order to survive |
| `MIN_ORDER_CONFIDENCE` | `0.55` | Minimum mean persona confidence for a surviving order |
| `REQUIRE_CATALYST_OR_SETUP` | `true` | New buys need a matching event unless extra traders agree |
| `EARNINGS_BLACKOUT_DAYS` | `5` | Recent earnings/guidance events count as an earnings window |
| `BLOCK_EARNINGS_BUYS` | `true` | Reject new buys in the earnings window |
| `STOP_ATR_MULTIPLE` | `2.0` | Stop distance as a multiple of 14-day ATR |
| `STOP_PCT` | `0.08` | Minimum stop distance as a fraction of price |
| `RISK_PER_NAME_PCT` | `0.005` | Equity fraction risked per name (vol-aware size) |
| `TIME_STOP_DAYS` | `30` | Sell held losers at or past this many days |
| `ENABLE_STOPS` | `true` | Attach a stop after each buy |
| `BEARISH_MIN_IMPORTANCE` | `4` | Bearish events at this importance block new names / sell held |
| `MAX_NEW_NAMES_PER_CYCLE` | `1` | Max new names to buy per cycle |
| `MAX_POSITION_PCT` | `0.20` | Max fraction of portfolio equity per name (20%) |
| `ENABLE_THINKING` | `true` | LLM reasoning mode (Qwen3+ models) |

The optimization horizon is fixed at **30 days** in code (`HORIZON_DAYS` in `agent/personas.py`).

---

## Notes and limitations

- **Paper trading only by default.** Set `ALPACA_PAPER_TRADE=false` and use live keys to trade with real money — do so at your own risk.
- The agent places **limit orders** from the latest NBBO when a quote is available, with `time_in_force=day`, and keeps fractional qty on fractionable names.
- Run `python main.py --news-once` (while the cash session is open) to populate the event store before the first `--trade-once`.
- RSS feeds run without API keys; optional API keys extend coverage.
- Model quality matters for JSON-only persona output. `qwen2.5:7b` is the documented minimum; larger models (e.g. Qwen3) work better with `ENABLE_THINKING=true`.
- Buy orders are capped by `max_position_pct` using latest bar close prices from research.
