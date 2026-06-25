"""Unit tests for the MCP tool executor."""
from types import SimpleNamespace

from tclp_benchmark.llm_providers.mcp_executor import convert_mcp_tools_to_openai


def test_convert_single_tool():
    mcp_tool = SimpleNamespace(
        name="search",
        description="Search the KG",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}},
                     "required": ["query"]},
    )
    out = convert_mcp_tools_to_openai([mcp_tool])
    assert out == [{
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the KG",
            "parameters": {"type": "object",
                           "properties": {"query": {"type": "string"}},
                           "required": ["query"]},
        },
    }]


def test_convert_missing_description_defaults_to_empty():
    mcp_tool = SimpleNamespace(name="t", description=None, inputSchema={"type": "object"})
    out = convert_mcp_tools_to_openai([mcp_tool])
    assert out[0]["function"]["description"] == ""
    assert out[0]["function"]["parameters"] == {"type": "object"}
