"""OpenRouter LLM provider implementation."""
import time
from typing import Dict, Any
from openrouter import OpenRouter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .base import BaseLLMProvider, LLMRequest, LLMResponse


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter API provider with web search integration."""

    def __init__(self, config: Dict[str, Any], api_key: str):
        """
        Initialise OpenRouter provider.

        Args:
            config: Provider configuration
            api_key: OpenRouter API key
        """
        super().__init__(config, api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=60),
        reraise=True,
    )
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response using OpenRouter API.

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
            # OpenRouter uses async context manager
            async with OpenRouter(api_key=self.api_key) as client:
                response = await client.chat.send_async(
                    model=request.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant answering questions about legal clauses and climate-aligned contracting.",
                        },
                        {"role": "user", "content": full_prompt},
                    ],
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )

            execution_time = int((time.time() - start_time) * 1000)

            # Extract token usage if available
            metadata = {}
            if hasattr(response, 'usage') and response.usage:
                metadata = {
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0),
                }

            # Extract text from response
            response_text = ""
            if hasattr(response, 'choices') and response.choices:
                if hasattr(response.choices[0], 'message'):
                    response_text = response.choices[0].message.content
                elif hasattr(response.choices[0], 'content'):
                    response_text = response.choices[0].content
            elif hasattr(response, 'text'):
                response_text = response.text

            return LLMResponse(
                text=response_text,
                model_name=request.model_name,
                provider_name=self.get_provider_name(),
                search_query=request.prompt,
                search_results=request.search_context,
                metadata=metadata,
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
        return "openrouter"
