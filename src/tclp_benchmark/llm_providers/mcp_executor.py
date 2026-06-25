"""MCP tool executor: connects to the TCLP KG MCP server and runs its tools."""
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

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
