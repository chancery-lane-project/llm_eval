"""Tests for OpenRouter LLM provider."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tclp_benchmark.llm_providers.openrouter_provider import OpenRouterProvider
from tclp_benchmark.llm_providers.base import LLMRequest, LLMResponse, SearchResult


@pytest.fixture
def provider_config():
    """OpenRouter provider configuration fixture"""
    return {
        "api_endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "models": ["anthropic/claude-opus-4-5-20251101:online"],
        "max_tokens": 1000,
        "temperature": 0.1,
        "web_search": {
            "enabled": True,
            "max_results": 5,
            "custom_search_prompt": "Search chancerylaneproject.org"
        },
        "rate_limit": {
            "requests_per_minute": 50,
            "tokens_per_minute": 100000
        },
        "retry": {
            "max_attempts": 3,
            "exponential_base": 2,
            "max_wait_seconds": 60
        }
    }


@pytest.fixture
def api_key():
    return "sk-or-test-key-123"


def test_openrouter_provider_init(provider_config, api_key):
    """Test OpenRouter provider initialisation"""
    provider = OpenRouterProvider(config=provider_config, api_key=api_key)

    assert provider.api_key == api_key
    assert provider.api_endpoint == "https://openrouter.ai/api/v1/chat/completions"
    assert provider.max_tokens == 1000
    assert provider.temperature == 0.1
    assert provider.web_search_enabled is True
    assert provider.web_search_max_results == 5


def test_openrouter_provider_name(provider_config, api_key):
    """Test provider name property"""
    provider = OpenRouterProvider(config=provider_config, api_key=api_key)
    assert provider.name == "openrouter"
    # Test backward compatibility
    assert provider.get_provider_name() == "openrouter"


@pytest.mark.asyncio
async def test_generate_response_with_online_suffix(provider_config, api_key):
    """Test response generation with :online suffix (built-in web search)"""
    provider = OpenRouterProvider(config=provider_config, api_key=api_key)

    request = LLMRequest(
        prompt="Find the net zero carbon commitment clause",
        search_context=[],
        model_name="anthropic/claude-opus-4-5-20251101:online",
        max_tokens=1000,
        temperature=0.1,
    )

    # Mock the OpenAI client response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Found Evie's Clause at chancerylaneproject.org"
    mock_response.usage.completion_tokens = 50
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.total_tokens = 150

    with patch.object(provider.client.chat.completions, 'create', new=AsyncMock(return_value=mock_response)) as mock_create:
        response = await provider.generate_response(request)

        # Verify tools were NOT added (using :online suffix instead)
        call_args = mock_create.call_args
        assert "tools" not in call_args[1]

    assert response.text == "Found Evie's Clause at chancerylaneproject.org"
    assert response.model_name == "anthropic/claude-opus-4-5-20251101:online"
    assert response.error is None
    assert response.metadata["completion_tokens"] == 50
    assert response.metadata["prompt_tokens"] == 100
    assert response.metadata["web_search_enabled"] is True


@pytest.mark.asyncio
async def test_generate_response_with_web_plugin(provider_config, api_key):
    """Test response generation with web plugin (not :online suffix)"""
    provider = OpenRouterProvider(config=provider_config, api_key=api_key)

    request = LLMRequest(
        prompt="Find carbon clauses",
        model_name="anthropic/claude-opus-4-5-20251101",  # No :online suffix
        max_tokens=1000,
        temperature=0.1,
        search_context=[],  # Empty search context
    )

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Found clauses..."
    mock_response.usage.completion_tokens = 40
    mock_response.usage.prompt_tokens = 80
    mock_response.usage.total_tokens = 120

    with patch.object(provider.client.chat.completions, 'create', new=AsyncMock(return_value=mock_response)) as mock_create:
        response = await provider.generate_response(request)

        # Verify tools were added to API call
        call_args = mock_create.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tools"][0]["type"] == "web"
        assert call_args[1]["tools"][0]["max_results"] == 5

    assert response.text == "Found clauses..."
    assert response.metadata["web_search_enabled"] is True


@pytest.mark.asyncio
async def test_generate_response_without_web_search(provider_config, api_key):
    """Test response generation with web search disabled"""
    # Disable web search
    provider_config["web_search"]["enabled"] = False
    provider = OpenRouterProvider(config=provider_config, api_key=api_key)

    request = LLMRequest(
        prompt="What is a carbon commitment clause?",
        search_context=[],
        model_name="anthropic/claude-opus-4-5-20251101",  # No :online suffix
        max_tokens=1000,
        temperature=0.1,
    )

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "A carbon commitment clause is..."
    mock_response.usage.completion_tokens = 30
    mock_response.usage.prompt_tokens = 20
    mock_response.usage.total_tokens = 50

    with patch.object(provider.client.chat.completions, 'create', new=AsyncMock(return_value=mock_response)):
        response = await provider.generate_response(request)

    assert response.text == "A carbon commitment clause is..."
    assert response.error is None
    assert response.metadata["web_search_enabled"] is False


@pytest.mark.asyncio
async def test_generate_response_handles_api_error(provider_config, api_key):
    """Test error handling when API call fails"""
    provider = OpenRouterProvider(config=provider_config, api_key=api_key)

    request = LLMRequest(
        prompt="Test prompt",
        search_context=[],
        model_name="anthropic/claude-opus-4-5-20251101:online",
        max_tokens=1000,
        temperature=0.1,
    )

    with patch.object(
        provider.client.chat.completions,
        'create',
        new=AsyncMock(side_effect=Exception("API Error"))
    ):
        response = await provider.generate_response(request)

    assert response.text == ""
    assert "API Error" in response.error


@pytest.mark.asyncio
async def test_generate_response_with_search_context(provider_config, api_key):
    """Test response generation with populated search context"""
    provider = OpenRouterProvider(config=provider_config, api_key=api_key)

    search_results = [
        SearchResult(
            title="Evie's Clause - Net Zero Commitment",
            url="https://chancerylaneproject.org/clause/evies-clause",
            snippet="A net zero carbon commitment clause...",
            rank=1
        )
    ]

    request = LLMRequest(
        prompt="Tell me about net zero clauses",
        model_name="anthropic/claude-opus-4-5-20251101:online",
        max_tokens=1000,
        temperature=0.1,
        search_context=search_results,
    )

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Based on search results: Evie's Clause is..."
    mock_response.usage.completion_tokens = 60
    mock_response.usage.prompt_tokens = 120
    mock_response.usage.total_tokens = 180

    with patch.object(provider.client.chat.completions, 'create', new=AsyncMock(return_value=mock_response)):
        response = await provider.generate_response(request)

    assert "Evie's Clause" in response.text
    assert response.search_results == search_results
    assert response.error is None


@pytest.mark.asyncio
async def test_custom_search_prompt_in_tools(provider_config, api_key):
    """Test that custom_search_prompt is included in tools parameter"""
    provider = OpenRouterProvider(config=provider_config, api_key=api_key)

    request = LLMRequest(
        prompt="Find carbon clauses",
        model_name="anthropic/claude-opus-4-5-20251101",  # No :online suffix
        max_tokens=1000,
        temperature=0.1,
        search_context=[],
    )

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Found clauses..."
    mock_response.usage.completion_tokens = 40
    mock_response.usage.prompt_tokens = 80
    mock_response.usage.total_tokens = 120

    with patch.object(provider.client.chat.completions, 'create', new=AsyncMock(return_value=mock_response)) as mock_create:
        response = await provider.generate_response(request)

        # Verify custom_search_prompt was passed
        call_args = mock_create.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tools"][0]["custom_search_prompt"] == "Search chancerylaneproject.org"

    assert response.text == "Found clauses..."
