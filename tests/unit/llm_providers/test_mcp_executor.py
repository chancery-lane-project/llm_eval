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


import pytest
from tclp_benchmark.llm_providers.mcp_executor import MCPToolExecutor


def _content(text):
    return SimpleNamespace(type="text", text=text)


@pytest.mark.asyncio
async def test_call_tool_returns_joined_text(mocker):
    ex = MCPToolExecutor("https://kg.example/mcp", token=None)
    fake_session = mocker.AsyncMock()
    fake_session.call_tool.return_value = SimpleNamespace(
        isError=False, content=[_content('{"results": []}')]
    )
    ex._session = fake_session  # inject a connected session

    out = await ex.call_tool("search", {"query": "net zero"})

    assert out == '{"results": []}'
    fake_session.call_tool.assert_awaited_once_with("search", {"query": "net zero"})


@pytest.mark.asyncio
async def test_call_tool_error_result_is_returned_not_raised(mocker):
    ex = MCPToolExecutor("https://kg.example/mcp", token=None)
    fake_session = mocker.AsyncMock()
    fake_session.call_tool.return_value = SimpleNamespace(
        isError=True, content=[_content("bad facet name")]
    )
    ex._session = fake_session

    out = await ex.call_tool("taxonomy_content", {"filters": {"x": "y"}})

    assert "ERROR" in out and "bad facet name" in out


def test_build_headers_with_token():
    assert MCPToolExecutor("u", token="abc")._headers() == {"Authorization": "Bearer abc"}


def test_build_headers_without_token():
    assert MCPToolExecutor("u", token=None)._headers() is None
