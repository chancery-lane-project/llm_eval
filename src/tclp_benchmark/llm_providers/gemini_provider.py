"""Google Gemini LLM provider implementation."""
import time
from typing import Dict, Any
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .base import BaseLLMProvider, LLMRequest, LLMResponse


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider with web search integration."""

    def __init__(self, config: Dict[str, Any], api_key: str):
        """
        Initialise Gemini provider.

        Args:
            config: Provider configuration
            api_key: Google API key
        """
        super().__init__(config, api_key)
        self.client = genai.Client(api_key=api_key)
        # Initialize model as None, will be created per request with specific model name
        self.model = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=60),
        retry=retry_if_exception_type((google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable)),
        reraise=True,
    )
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response using Gemini API.

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
            # Create generation config
            config = types.GenerateContentConfig(
                system_instruction="You are a helpful assistant answering questions about legal clauses and climate-aligned contracting.",
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
            )

            # Generate content using new API
            response = await self.client.aio.models.generate_content(
                model=request.model_name,
                contents=full_prompt,
                config=config,
            )

            execution_time = int((time.time() - start_time) * 1000)

            # Extract token counts if available
            metadata = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                metadata = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }

            # Extract text from response
            response_text = response.text if hasattr(response, 'text') else ""

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
        return "gemini"
