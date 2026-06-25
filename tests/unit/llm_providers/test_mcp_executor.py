"""Unit tests for the MCP tool executor."""
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from tclp_benchmark.llm_providers import mcp_executor as mcp_executor_module
from tclp_benchmark.llm_providers.mcp_executor import (
    MCPToolExecutor,
    convert_mcp_tools_to_openai,
)


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


def _content(text):
    return SimpleNamespace(type="text", text=text)


@pytest.mark.asyncio
async def test_call_tool_returns_joined_text(mocker):
    ex = MCPToolExecutor("https://kg.example/mcp", token=None)
    fake_session = mocker.AsyncMock()
    fake_session.call_tool.return_value = SimpleNamespace(
        isError=False,
        content=[
            _content('{"results": '),
            SimpleNamespace(type="image"),  # no .text → skipped
            _content("[]}"),
        ],
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


@pytest.mark.asyncio
async def test_aenter_closes_stack_on_initialize_failure(monkeypatch):
    closed = {"client": False}

    @asynccontextmanager
    async def fake_client(url, headers=None):
        try:
            yield (object(), object(), None)
        finally:
            closed["client"] = True

    class FakeSession:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def initialize(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(mcp_executor_module, "streamablehttp_client", fake_client)
    monkeypatch.setattr(mcp_executor_module, "ClientSession", FakeSession)

    ex = MCPToolExecutor("https://kg.example/mcp", token=None)
    with pytest.raises(RuntimeError, match="boom"):
        async with ex:
            pass

    assert closed["client"] is True  # entered context manager was torn down
    assert ex._stack is None
    assert ex._session is None
