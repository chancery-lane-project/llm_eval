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


from tclp_benchmark.llm_providers.anthropic_provider import AnthropicProvider


@pytest.mark.asyncio
async def test_anthropic_provider_name():
    """Test Anthropic provider returns correct name."""
    provider = AnthropicProvider(config={}, api_key="test-key")
    assert provider.get_provider_name() == "anthropic"


@pytest.mark.asyncio
async def test_anthropic_provider_generates_response():
    """Test Anthropic provider generates response."""
    provider = AnthropicProvider(config={}, api_key="test-key")

    # Mock Anthropic API response
    with patch.object(provider.client.messages, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = AsyncMock(
            content=[AsyncMock(text="Test response")],
            usage=AsyncMock(input_tokens=20, output_tokens=10),
        )

        request = LLMRequest(
            prompt="Test prompt",
            search_context=[],
            max_tokens=1000,
            temperature=0.1,
            model_name="claude-3-7-sonnet-20250219",
        )

        response = await provider.generate_response(request)

        assert response.text == "Test response"
        assert response.provider_name == "anthropic"
        assert response.model_name == "claude-3-7-sonnet-20250219"
        assert response.error is None


@pytest.mark.asyncio
async def test_anthropic_provider_handles_errors():
    """Test Anthropic provider handles API errors gracefully."""
    provider = AnthropicProvider(config={}, api_key="test-key")

    # Mock Anthropic API to raise exception
    with patch.object(provider.client.messages, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = Exception("API Error")

        request = LLMRequest(
            prompt="Test prompt",
            search_context=[],
            max_tokens=1000,
            temperature=0.1,
            model_name="claude-3-7-sonnet-20250219",
        )

        response = await provider.generate_response(request)

        assert response.text == ""
        assert response.error == "API Error"


from tclp_benchmark.llm_providers.gemini_provider import GeminiProvider


@pytest.mark.asyncio
async def test_gemini_provider_name():
    """Test Gemini provider returns correct name."""
    provider = GeminiProvider(config={}, api_key="test-key")
    assert provider.get_provider_name() == "gemini"


@pytest.mark.asyncio
async def test_gemini_provider_generates_response():
    """Test Gemini provider generates response."""
    import google.generativeai as genai

    provider = GeminiProvider(config={}, api_key="test-key")

    # Mock Gemini API response
    mock_response = AsyncMock()
    mock_response.text = "Test response"
    mock_response.usage_metadata = AsyncMock(
        prompt_token_count=20,
        candidates_token_count=10,
        total_token_count=30,
    )

    # Mock the GenerativeModel class
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(return_value=mock_response)

    with patch.object(genai, "GenerativeModel", return_value=mock_model):
        request = LLMRequest(
            prompt="Test prompt",
            search_context=[],
            max_tokens=1000,
            temperature=0.1,
            model_name="gemini-2.0-flash",
        )

        response = await provider.generate_response(request)

        assert response.text == "Test response"
        assert response.provider_name == "gemini"
        assert response.model_name == "gemini-2.0-flash"
        assert response.error is None


@pytest.mark.asyncio
async def test_gemini_provider_handles_errors():
    """Test Gemini provider handles API errors gracefully."""
    import google.generativeai as genai

    provider = GeminiProvider(config={}, api_key="test-key")

    # Mock the GenerativeModel class to raise exception
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(side_effect=Exception("API Error"))

    with patch.object(genai, "GenerativeModel", return_value=mock_model):
        request = LLMRequest(
            prompt="Test prompt",
            search_context=[],
            max_tokens=1000,
            temperature=0.1,
            model_name="gemini-2.0-flash",
        )

        response = await provider.generate_response(request)

        assert response.text == ""
        assert response.error == "API Error"
