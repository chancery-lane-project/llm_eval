"""Unit tests for the MCP tool-loop provider."""
import json
import pytest
from types import SimpleNamespace

from tclp_benchmark.llm_providers.base import LLMRequest, LLMResponse
from tclp_benchmark.llm_providers.mcp_provider import MCPToolProvider


def test_llmresponse_has_tool_calls_default_empty():
    resp = LLMResponse(
        text="hi",
        model_name="m",
        provider_name="mcp",
        search_query="q",
        search_results=[],
        metadata={},
        execution_time_ms=1,
    )
    assert resp.tool_calls == []


def _msg(content=None, tool_calls=None):
    m = SimpleNamespace(content=content, tool_calls=tool_calls)
    m.model_dump = lambda: {"role": "assistant", "content": content,
                            "tool_calls": bool(tool_calls)}
    return m


def _completion(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                           usage=SimpleNamespace(total_tokens=1, prompt_tokens=1,
                                                 completion_tokens=0))


def _tool_call(call_id, name, args):
    return SimpleNamespace(id=call_id,
                           function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class _FakeExecutor:
    """Async-context fake standing in for MCPToolExecutor."""
    def __init__(self, *a, **k):
        self.calls = []
    async def __aenter__(self): return self
    async def __aexit__(self, *e): return False
    async def list_openai_tools(self):
        return [{"type": "function", "function": {"name": "search"}}]
    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return '{"results": [{"title": "Evies Clause"}]}'


def _make_provider(mocker, completions):
    """Provider whose AsyncOpenAI returns the queued completions in order."""
    provider = MCPToolProvider(
        config={"max_tokens": 100, "temperature": 0.0, "max_tool_iterations": 8,
                "mcp": {"url": "https://kg.example/mcp"}},
        api_key="sk-test",
        executor_factory=_FakeExecutor,
    )
    provider.client.chat.completions.create = mocker.AsyncMock(side_effect=completions)
    return provider


def _request():
    return LLMRequest(prompt="Status of Evie's Clause?", search_context=[],
                      max_tokens=100, temperature=0.0, model_name="openai/gpt-4o")


@pytest.mark.asyncio
async def test_loop_executes_tool_then_returns_answer(mocker):
    first = _completion(_msg(tool_calls=[_tool_call("c1", "search", {"query": "Evie"})]))
    second = _completion(_msg(content="It is Not Maintained."))
    provider = _make_provider(mocker, [first, second])

    resp = await provider.generate_response(_request())

    assert resp.text == "It is Not Maintained."
    assert resp.error is None
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["name"] == "search"
    assert resp.metadata["tool_calls"] == resp.tool_calls
    assert provider.client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_no_tool_calls_returns_immediately(mocker):
    only = _completion(_msg(content="I don't know."))
    provider = _make_provider(mocker, [only])

    resp = await provider.generate_response(_request())

    assert resp.text == "I don't know."
    assert resp.tool_calls == []
    assert provider.client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_iteration_cap_forces_final_answer(mocker):
    # Always returns a tool call -> cap reached -> one final toolless call.
    looping = [_completion(_msg(tool_calls=[_tool_call(f"c{i}", "search", {"query": "x"})]))
               for i in range(8)]
    final = _completion(_msg(content="Best effort answer."))
    provider = _make_provider(mocker, looping + [final])  # config already sets max_tool_iterations=8

    resp = await provider.generate_response(_request())

    assert resp.text == "Best effort answer."
    assert resp.metadata["hit_tool_limit"] is True
    assert provider.client.chat.completions.create.await_count == 9


@pytest.mark.asyncio
async def test_executor_connection_failure_returns_error_response(mocker):
    class _BoomExecutor(_FakeExecutor):
        async def __aenter__(self): raise ConnectionError("no endpoint")
    provider = MCPToolProvider(
        config={"mcp": {"url": "https://bad"}}, api_key="sk-test",
        executor_factory=_BoomExecutor,
    )
    resp = await provider.generate_response(_request())
    assert resp.error is not None
    assert resp.text == ""
