"""Anthropic LLM provider implementation."""
import time
from typing import Dict, Any
from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import anthropic
from .base import BaseLLMProvider, LLMRequest, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider with web search integration."""

    def __init__(self, config: Dict[str, Any], api_key: str):
        """
        Initialise Anthropic provider.

        Args:
            config: Provider configuration
            api_key: Anthropic API key
        """
        super().__init__(config, api_key)
        self.client = AsyncAnthropic(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=60),
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
        reraise=True,
    )
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response using Anthropic API.

        Args:
            request: LLM request with prompt and search context

        Returns:
            LLM response
        """
        start_time = time.time()

        # Construct search-augmented prompt
        context = self._format_search_context(request.search_context)
        full_prompt = f"{context}\n\nUser question: {request.prompt}" if context else request.prompt

        try:
            response = await self.client.messages.create(
                model=request.model_name,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt,
                    }
                ],
                system="You are a helpful assistant answering questions about legal clauses and climate-aligned contracting.",
            )

            execution_time = int((time.time() - start_time) * 1000)

            return LLMResponse(
                text=response.content[0].text,
                model_name=request.model_name,
                provider_name=self.get_provider_name(),
                search_query=request.prompt,
                search_results=request.search_context,
                metadata={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                execution_time_ms=execution_time,
            )
        except Exception as e:
            return LLMResponse(
                text="",
                model_name=request.model_name,
                provider_name=self.get_provider_name(),
                search_query=request.prompt,
                search_results=request.search_context,
                metadata={},
                execution_time_ms=int((time.time() - start_time) * 1000),
                error=str(e),
            )

    def get_provider_name(self) -> str:
        """Return provider name."""
        return "anthropic"
