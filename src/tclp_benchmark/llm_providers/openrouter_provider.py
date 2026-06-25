"""OpenRouter LLM provider with web search integration."""

import logging
import time
from typing import Dict, Any
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import openai

from .base import BaseLLMProvider, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter LLM provider with integrated web search."""

    def __init__(self, config: Dict[str, Any], api_key: str):
        """
        Initialise OpenRouter provider.

        Args:
            config: Provider configuration from providers.yaml
            api_key: OpenRouter API key
        """
        super().__init__(config, api_key)

        self.api_endpoint = config.get(
            "api_endpoint",
            "https://openrouter.ai/api/v1/chat/completions"
        )
        self.max_tokens = config.get("max_tokens", 1000)
        self.temperature = config.get("temperature", 0.1)

        # Web search configuration
        web_search_config = config.get("web_search", {})
        self.web_search_enabled = web_search_config.get("enabled", False)
        self.web_search_max_results = web_search_config.get("max_results", 5)
        self.custom_search_prompt = web_search_config.get("custom_search_prompt")

        # Retry configuration
        retry_config = config.get("retry", {})
        self.max_attempts = retry_config.get("max_attempts", 3)
        self.backoff_base = retry_config.get("exponential_base", 2)
        self.max_wait = retry_config.get("max_wait_seconds", 60)

        # Initialise OpenAI-compatible client pointing to OpenRouter
        base_url = self.api_endpoint.replace("/chat/completions", "")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        logger.info(
            f"Initialised OpenRouter provider (endpoint: {self.api_endpoint}, "
            f"web_search: {self.web_search_enabled})"
        )

    @property
    def name(self) -> str:
        """Provider name identifier"""
        return "openrouter"

    def get_provider_name(self) -> str:
        """Return provider name identifier (deprecated, use .name property)."""
        return self.name

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response using OpenRouter API.

        Args:
            request: LLM request with prompt and configuration

        Returns:
            LLMResponse with text and metadata
        """
        start_time = time.time()

        @retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=self.backoff_base, min=1, max=self.max_wait),
            retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
            reraise=True,
        )
        async def _make_request():
            """Inner function with retry logic using instance config."""
            # Construct search-augmented prompt (if search_context provided)
            context = self._format_search_context(request.search_context)
            full_prompt = f"{context}\n\nUser question: {request.prompt}" if context else request.prompt

            # Detect if model uses :online suffix (built-in web search)
            use_online_suffix = ":online" in request.model_name

            # Build API call parameters
            api_params = {
                "model": request.model_name,
                "messages": [
                    {"role": "user", "content": full_prompt}
                ],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }

            # Add web plugin if enabled and NOT using :online suffix
            if self.web_search_enabled and not use_online_suffix:
                tools = [
                    {
                        "type": "web",
                        "max_results": self.web_search_max_results,
                    }
                ]
                if self.custom_search_prompt:
                    tools[0]["custom_search_prompt"] = self.custom_search_prompt

                api_params["tools"] = tools

            logger.debug(
                f"OpenRouter API call: model={request.model_name}, "
                f"web_search={self.web_search_enabled or use_online_suffix}, prompt_length={len(full_prompt)}"
            )

            # Make API call
            response = await self.client.chat.completions.create(**api_params)

            execution_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"OpenRouter response received: model={request.model_name}, "
                f"tokens={response.usage.total_tokens}, time={execution_time}ms"
            )

            return LLMResponse(
                text=response.choices[0].message.content,
                model_name=request.model_name,
                provider_name=self.name,
                search_query=request.prompt,
                search_results=request.search_context,
                metadata={
                    "completion_tokens": response.usage.completion_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "web_search_enabled": self.web_search_enabled or use_online_suffix,
                },
                execution_time_ms=execution_time,
            )

        try:
            return await _make_request()
        except Exception as e:
            logger.error(f"OpenRouter API error: {str(e)}")
            return LLMResponse(
                text="",
                model_name=request.model_name,
                provider_name=self.name,
                search_query=request.prompt,
                search_results=request.search_context,
                metadata={},
                execution_time_ms=int((time.time() - start_time) * 1000),
                error=str(e),
            )
