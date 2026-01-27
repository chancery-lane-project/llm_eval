"""Tests for LLM provider abstraction."""
import pytest
from unittest.mock import AsyncMock, patch
from tclp_benchmark.llm_providers.base import (
    BaseLLMProvider,
    LLMRequest,
    LLMResponse,
    SearchResult,
)


def test_search_result_creation():
    """Test SearchResult dataclass."""
    result = SearchResult(
        title="Test Title",
        url="https://example.com",
        snippet="Test snippet",
        rank=1,
    )
    assert result.title == "Test Title"
    assert result.rank == 1


def test_llm_request_creation():
    """Test LLMRequest dataclass."""
    search_results = [
        SearchResult(title="Test", url="https://example.com", snippet="Snippet", rank=1)
    ]
    request = LLMRequest(
        prompt="Test prompt",
        search_context=search_results,
        max_tokens=1000,
        temperature=0.1,
        model_name="gpt-4o",
    )
    assert request.prompt == "Test prompt"
    assert len(request.search_context) == 1


def test_llm_response_creation():
    """Test LLMResponse dataclass."""
    response = LLMResponse(
        text="Test response",
        model_name="gpt-4o",
        provider_name="openai",
        search_query="test query",
        search_results=[],
        metadata={"tokens": 100},
        execution_time_ms=500,
    )
    assert response.text == "Test response"
    assert response.provider_name == "openai"
    assert response.execution_time_ms == 500


def test_base_provider_abstract():
    """Test BaseLLMProvider cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseLLMProvider(config={}, api_key="test")


from tclp_benchmark.llm_providers.openai_provider import OpenAIProvider


@pytest.mark.asyncio
async def test_openai_provider_name():
    """Test OpenAI provider returns correct name."""
    provider = OpenAIProvider(config={}, api_key="test-key")
    assert provider.get_provider_name() == "openai"


@pytest.mark.asyncio
async def test_openai_provider_generates_response():
    """Test OpenAI provider generates response."""
    provider = OpenAIProvider(config={}, api_key="test-key")

    # Mock OpenAI API response
    with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = AsyncMock(
            choices=[AsyncMock(message=AsyncMock(content="Test response"))],
            usage=AsyncMock(completion_tokens=10, prompt_tokens=20, total_tokens=30),
        )

        request = LLMRequest(
            prompt="Test prompt",
            search_context=[],
            max_tokens=1000,
            temperature=0.1,
            model_name="gpt-4o",
        )

        response = await provider.generate_response(request)

        assert response.text == "Test response"
        assert response.provider_name == "openai"
        assert response.model_name == "gpt-4o"
        assert response.error is None


@pytest.mark.asyncio
async def test_openai_provider_handles_errors():
    """Test OpenAI provider handles API errors gracefully."""
    provider = OpenAIProvider(config={}, api_key="test-key")

    # Mock OpenAI API to raise exception
    with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = Exception("API Error")

        request = LLMRequest(
            prompt="Test prompt",
            search_context=[],
            max_tokens=1000,
            temperature=0.1,
            model_name="gpt-4o",
        )

        response = await provider.generate_response(request)

        assert response.text == ""
        assert response.error == "API Error"
