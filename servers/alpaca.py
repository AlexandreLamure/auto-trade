"""
Alpaca MCP server parameters.

`alpaca-mcp-server` is installed as a regular pip dependency and launched
via the same Python interpreter that runs the agent, so no `uv`/`uvx`
installation is needed on the host system.

Actual v2 tool names exposed by alpaca-mcp-server:
  get_account_info     – account balances, margin, status
  get_all_positions    – all open positions
  get_open_position    – single position by symbol
  get_orders           – order history / open orders
  get_stock_bars       – OHLCV bars for one or more symbols
  place_stock_order    – submit a market/limit buy or sell order
  ... (tools across account, trading, and stock-data toolsets)
"""

import os
import sys
import shutil
from pathlib import Path
from mcp import StdioServerParameters


def _find_alpaca_script() -> str:
    """Resolve the alpaca-mcp-server console script.

    Looks in the Scripts / bin directory next to sys.executable first
    (works inside a venv on both Windows and Unix), then falls back to
    PATH.  Raises RuntimeError if not found.
    """
    scripts_dir = Path(sys.executable).parent
    for name in ("alpaca-mcp-server.exe", "alpaca-mcp-server"):
        candidate = scripts_dir / name
        if candidate.is_file():
            return str(candidate)

    # Fallback: search PATH
    found = shutil.which("alpaca-mcp-server")
    if found:
        return found

    raise RuntimeError(
        "alpaca-mcp-server executable not found. "
        "Run: pip install alpaca-mcp-server"
    )


def get_alpaca_server_params(
    api_key: str,
    secret_key: str,
    paper_trade: bool = True,
) -> StdioServerParameters:
    """Return StdioServerParameters for the Alpaca MCP server.

    The console-script executable installed alongside sys.executable is
    used so the server always runs inside the active virtual environment.
    """
    env = {
        **os.environ,  # inherit PATH and system env
        "ALPACA_API_KEY": api_key,
        "ALPACA_SECRET_KEY": secret_key,
        "ALPACA_PAPER_TRADE": "true" if paper_trade else "false",
        "ALPACA_TOOLSETS": "account,trading,stock-data,assets",
    }

    return StdioServerParameters(
        command=_find_alpaca_script(),
        args=[],
        env=env,
    )
