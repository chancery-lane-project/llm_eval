"""MCP tool executor: connects to the TCLP KG MCP server and runs its tools."""
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


def convert_mcp_tools_to_openai(mcp_tools: list) -> List[Dict[str, Any]]:
    """Convert MCP tool definitions into OpenAI function-calling tool schemas."""
    converted = []
    for tool in mcp_tools:
        converted.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {"type": "object"},
            },
        })
    return converted


class MCPToolExecutor:
    """Per-call MCP session over streamable-HTTP against the TCLP KG server.

    Use as an async context manager so the session is created lazily and torn
    down in the same task (the streamable-HTTP client and ClientSession are
    anyio context managers and must not cross tasks).
    """

    def __init__(self, url: str, token: Optional[str] = None):
        self.url = url
        self.token = token
        self._session: Optional[ClientSession] = None
        self._stack: Optional[AsyncExitStack] = None

    def _headers(self) -> Optional[Dict[str, str]]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else None

    async def __aenter__(self) -> "MCPToolExecutor":
        self._stack = AsyncExitStack()
        read, write, _ = await self._stack.enter_async_context(
            streamablehttp_client(self.url, headers=self._headers())
        )
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc):
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def list_openai_tools(self) -> List[Dict[str, Any]]:
        result = await self._session.list_tools()
        return convert_mcp_tools_to_openai(result.tools)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        result = await self._session.call_tool(name, arguments)
        text = "".join(
            block.text for block in result.content if getattr(block, "text", None)
        )
        if getattr(result, "isError", False):
            return f"ERROR from tool '{name}': {text}"
        return text
