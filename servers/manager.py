"""
MCPManager – unified tool registry for the Alpaca MCP server.

Opens an Alpaca MCP session and exposes its tools for portfolio data
and order execution.  News is handled by the separate news analysis service.

Usage (async context manager):

    async with MCPManager(settings) as manager:
        result = await manager.call_tool("get_account_info", {})
"""

import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp import types as mcp_types

from config.settings import Settings
from servers.alpaca import get_alpaca_server_params

logger = logging.getLogger(__name__)


class MCPManager:
    """Async context manager that owns the Alpaca MCP client session."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._exit_stack = AsyncExitStack()
        self._tool_sessions: dict[str, ClientSession] = {}

    async def __aenter__(self) -> "MCPManager":
        await self._exit_stack.__aenter__()

        alpaca_params = get_alpaca_server_params(
            api_key=self._settings.alpaca_api_key,
            secret_key=self._settings.alpaca_secret_key,
            paper_trade=self._settings.alpaca_paper_trade,
        )

        alpaca_session = await self._open_session(alpaca_params, "alpaca")
        await self._register_tools(alpaca_session, "alpaca")

        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._exit_stack.__aexit__(*exc_info)

    async def _open_session(
        self, params: StdioServerParameters, label: str
    ) -> ClientSession:
        read, write = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        session: ClientSession = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        return session

    async def _register_tools(self, session: ClientSession, label: str) -> None:
        response = await session.list_tools()
        for tool in response.tools:
            if tool.name in self._tool_sessions:
                logger.warning(
                    "Tool name collision '%s' – '%s' overrides existing registration",
                    tool.name,
                    label,
                )
            self._tool_sessions[tool.name] = session

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> mcp_types.CallToolResult:
        """Execute a tool by name and return its raw CallToolResult."""
        session = self._tool_sessions.get(name)
        if session is None:
            raise ValueError(f"Unknown tool: '{name}'")
        result = await session.call_tool(name, arguments or {})
        return result

    def result_to_text(self, result: mcp_types.CallToolResult) -> str:
        """Flatten a CallToolResult into a plain string for the LLM."""
        parts: list[str] = []
        for block in result.content:
            if isinstance(block, mcp_types.TextContent):
                parts.append(block.text)
            elif isinstance(block, mcp_types.EmbeddedResource):
                parts.append(str(block.resource))
        if result.isError:
            joined = "\n".join(parts) or "(no detail)"
            return f"ERROR: {joined}"
        return "\n".join(parts)
