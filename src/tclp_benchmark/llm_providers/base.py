"""Base abstraction for LLM providers."""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """Represents a web search result."""

    title: str
    url: str
    snippet: str
    rank: int


@dataclass
class LLMRequest:
    """Standardised LLM request."""

    prompt: str
    search_context: List[SearchResult]
    max_tokens: int
    temperature: float
    model_name: str


@dataclass
class LLMResponse:
    """Standardised LLM response."""

    text: str
    model_name: str
    provider_name: str
    search_query: str
    search_results: List[SearchResult]
    metadata: Dict[str, Any]
    execution_time_ms: int
    error: Optional[str] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: Dict[str, Any], api_key: str):
        """
        Initialise LLM provider.

        Args:
            config: Provider configuration dict
            api_key: API key for authentication
        """
        self.config = config
        self.api_key = api_key

    @abstractmethod
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            request: LLM request with prompt and search context

        Returns:
            LLM response with text and metadata
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Return the provider name.

        Returns:
            Provider name (e.g., 'openai', 'anthropic', 'gemini')
        """
        pass

    def _format_search_context(self, results: List[SearchResult]) -> str:
        """
        Format search results into context string.

        Args:
            results: List of search results

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        context_parts = ["Here are relevant search results:\n"]
        for result in results:
            context_parts.append(f"[{result.rank}] {result.title}")
            context_parts.append(f"URL: {result.url}")
            context_parts.append(f"{result.snippet}\n")

        return "\n".join(context_parts)
