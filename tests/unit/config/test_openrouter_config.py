import pytest
from pydantic import ValidationError
from tclp_benchmark.config.models import OpenRouterConfig, WebSearchConfig, RateLimitConfig, RetryConfig


def test_openrouter_config_valid():
    """Test valid OpenRouter configuration"""
    config = OpenRouterConfig(
        enabled=True,
        api_endpoint="https://openrouter.ai/api/v1/chat/completions",
        models=["anthropic/claude-opus-4-5-20251101:online"],
        max_tokens=1000,
        temperature=0.1,
        web_search=WebSearchConfig(
            enabled=True,
            max_results=5,
            custom_search_prompt="Search chancerylaneproject.org for clauses."
        ),
        rate_limit=RateLimitConfig(
            requests_per_minute=50,
            tokens_per_minute=100000
        ),
        retry=RetryConfig(
            max_attempts=3,
            exponential_base=2,
            max_wait_seconds=60
        )
    )

    assert config.enabled is True
    assert config.api_endpoint == "https://openrouter.ai/api/v1/chat/completions"
    assert len(config.models) == 1
    assert config.web_search.enabled is True
    assert config.web_search.max_results == 5
    assert config.rate_limit.requests_per_minute == 50
    assert config.retry.max_attempts == 3


def test_openrouter_config_web_search_optional():
    """Test OpenRouter config with web search disabled"""
    config = OpenRouterConfig(
        enabled=True,
        api_endpoint="https://openrouter.ai/api/v1/chat/completions",
        models=["anthropic/claude-opus-4-5-20251101"],
        max_tokens=1000,
        temperature=0.1,
        web_search=WebSearchConfig(enabled=False)
    )

    assert config.web_search.enabled is False


def test_openrouter_config_invalid_temperature():
    """Test validation for invalid temperature"""
    with pytest.raises(ValidationError):
        OpenRouterConfig(
            enabled=True,
            api_endpoint="https://openrouter.ai/api/v1/chat/completions",
            models=["anthropic/claude-opus-4-5-20251101:online"],
            max_tokens=1000,
            temperature=3.0,  # Invalid: > 2.0
            web_search=WebSearchConfig(enabled=True)
        )


def test_web_search_config_default_max_results():
    """Test WebSearchConfig defaults"""
    config = WebSearchConfig(enabled=True)
    assert config.max_results == 5
    assert config.custom_search_prompt is None
