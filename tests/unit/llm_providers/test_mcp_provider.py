"""Unit tests for the MCP tool-loop provider."""
from tclp_benchmark.llm_providers.base import LLMResponse


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
