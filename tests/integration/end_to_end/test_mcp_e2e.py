"""Live end-to-end test for the MCP tool-loop provider. Skipped without creds."""
import os
import pytest

from tclp_benchmark.llm_providers.base import LLMRequest
from tclp_benchmark.llm_providers.mcp_provider import MCPToolProvider

pytestmark = pytest.mark.skipif(
    not (os.getenv("TCLP_MCP_URL") and os.getenv("OPENROUTER_API_KEY")),
    reason="requires TCLP_MCP_URL and OPENROUTER_API_KEY",
)


@pytest.mark.asyncio
async def test_live_mcp_answers_with_kg():
    provider = MCPToolProvider(
        config={
            "max_tokens": 1500, "temperature": 0.0, "max_tool_iterations": 8,
            "mcp": {"url": os.getenv("TCLP_MCP_URL"), "token": os.getenv("TCLP_MCP_TOKEN")},
        },
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    request = LLMRequest(
        prompt="Search the TCLP knowledge graph: how many clauses are 'Not maintained'?",
        search_context=[], max_tokens=1500, temperature=0.0,
        model_name="openai/gpt-4o",
    )
    resp = await provider.generate_response(request)
    assert resp.error is None
    assert resp.text
    # A KG-backed answer should have called at least one tool.
    assert len(resp.tool_calls) >= 1
