"""Pydantic models for configuration validation."""
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, List, Any, Optional


class TestPrompt(BaseModel):
    """Individual test prompt configuration."""

    id: str = Field(..., description="Unique prompt identifier")
    text: str = Field(..., description="The prompt text to send to LLMs")
    expected_tclp_mention: bool = Field(
        default=False, description="Whether TCLP should be mentioned"
    )
    expected_clause_name: Optional[str] = Field(
        default=None, description="Expected clause name"
    )
    expected_filters: Optional[Dict[str, str]] = Field(
        default=None, description="Expected taxonomy filters"
    )
    expected_concepts: Optional[List[str]] = Field(
        default=None, description="Expected concepts to be mentioned"
    )
    expected_metadata_type: Optional[str] = Field(
        default=None, description="Expected metadata type"
    )
    expected_standard: Optional[str] = Field(
        default=None, description="Expected standard reference"
    )


class PromptConfig(BaseModel):
    """Configuration for test prompts organised by category."""

    prompts: Dict[str, List[TestPrompt]] = Field(
        ..., description="Prompts organised by category"
    )


class ModelConfig(BaseModel):
    """LLM model configuration."""

    name: str
    max_tokens: int = Field(default=1000, ge=1)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    requests_per_minute: int = Field(ge=1)
    tokens_per_minute: int = Field(ge=1)
    delay_between_requests: Optional[int] = Field(default=None, ge=0, description="Delay in seconds between requests")


class RetryConfig(BaseModel):
    """Retry configuration."""

    max_attempts: int = Field(ge=1)
    exponential_base: int = Field(ge=1)
    max_wait_seconds: int = Field(ge=1)


class ProviderConfig(BaseModel):
    """LLM provider configuration."""

    enabled: bool = Field(default=True)
    models: List[ModelConfig]
    rate_limit: RateLimitConfig
    retry: RetryConfig


class MetricConfig(BaseModel):
    """Evaluation metric configuration."""

    enabled: bool = Field(default=True)
    threshold: float = Field(ge=0.0, le=1.0)
    model: Optional[str] = Field(default=None, description="Model for LLM-as-judge")
    description: Optional[str] = None
    criteria: Optional[str] = None


class WebSearchConfig(BaseModel):
    """Configuration for OpenRouter web search plugin"""
    enabled: bool
    max_results: int = Field(default=5, ge=1, le=20)
    custom_search_prompt: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class OpenRouterConfig(BaseModel):
    """Configuration for OpenRouter LLM provider"""
    enabled: bool
    api_endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    models: List[str]
    max_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0.0, le=2.0)
    web_search: WebSearchConfig
    rate_limit: Optional[RateLimitConfig] = None
    retry: Optional[RetryConfig] = None

    model_config = ConfigDict(extra="forbid")


class ProvidersConfig(BaseModel):
    """Configuration for all LLM providers."""

    openai: Optional[ProviderConfig] = None
    anthropic: Optional[ProviderConfig] = None
    gemini: Optional[ProviderConfig] = None
    openrouter: Optional[OpenRouterConfig] = None


class MetricsConfig(BaseModel):
    """Configuration for all metrics."""

    deepeval_standard: Optional[Dict[str, MetricConfig]] = None
    tclp_custom: Optional[Dict[str, MetricConfig]] = None
