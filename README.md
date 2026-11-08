# Auto-Trade – Local Autonomous Trading Agent

A local autonomous stock trading agent: a local LLM (Ollama) runs an investment
committee that trades an Alpaca paper account toward 30-day PnL, using news
grouped into market events by a separate news pipeline.

## How it works

```
main.py  (--news-once | --trade-once | --once | --loop | --clean)
  ├── news/pipeline.py         news cycle → data/events.db
  │    ├── collector.py        RSS, Alpaca news (Benzinga), NewsAPI, Finnhub,
  │    │                       Alpha Vantage, Marketaux, Reddit, SEC EDGAR
  │    └── analyzer.py         dedupe, LLM-cluster into events, LLM-enrich
  └── agent/orchestrator.py    trading cycle
       ├── research.py         Alpaca portfolio, prices, candidates + events → brief
       ├── deliberation.py     4 traders (+ debate if they disagree) → chair
       ├── risk.py             cash, no shorts, max position, max orders
       └── servers/            alpaca-mcp-server (stdio) for data and orders
```

**News cycle**: builds a watchlist (holdings, market movers, tickers from recent
events), fetches news, groups articles into events, and has the LLM summarize
each event with sentiment, importance, and tickers.

**Trading cycle** (skipped when the US cash session is closed): builds a ~5k-token
brief (portfolio, price table, movers, recent orders, events, past decisions and
their outcomes, the thesis and exit plan of each holding). Four trader personas
propose orders, debate if they disagree, and a chair decides the orders, stops,
and plans. Only holdings and filtered candidates can be traded. Every cycle is
journaled in `data/events.db`, with 1d/5d/30d outcome marks from daily closes.

## Setup

Requires Python 3.11+, [Ollama](https://ollama.com/download), and an
[Alpaca paper account](https://app.alpaca.markets/paper/dashboard/overview).

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
pre-commit install              # gitleaks secret scan on every commit
cp .env.example .env            # then fill in your keys (see .env.example)
ollama pull qwen3:14b
```

Paper keys start with `PK`, live keys with `AK`; the prefix is checked against
`ALPACA_PAPER_TRADE` at startup. News API keys are optional.

## Running

```bash
python main.py --news-once     # one news cycle
python main.py --trade-once    # one trade cycle
python main.py --once          # news, then trade
python main.py --loop          # news then trade, every day at 12:00 America/New_York
python main.py --clean         # delete data/events.db and logs/*.log
```

Run a news cycle before the first trade cycle. Human-readable cycle logs go to
`logs/news.log` and `logs/trading.log`; warnings and errors go to stderr.

## Execution notes

- **Paper trading by default.** Live trading (`ALPACA_PAPER_TRADE=false`) is at your own risk.
- Orders are day limit orders from the latest IEX quote (market if no quote); fractional
  qty is kept on fractionable names. Buys are capped by `MAX_POSITION_PCT` of equity.
- The committee picks each buy's stop price (or none). A stopped buy is an OTO `gtc`
  order floored to whole shares (Alpaca rejects fractional OTO). Open orders on a symbol
  are cancelled before trading it; a partial sell re-places the stop on the remainder.
- The horizon is fixed at 30 days (`HORIZON_DAYS` in `agent/personas.py`).
