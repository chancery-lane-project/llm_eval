"""MCP tool-loop provider: OpenRouter gateway + client-side KG tool execution."""
import json
import logging
import os
import time
from typing import Any, Dict

import openai
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseLLMProvider, LLMRequest, LLMResponse
from .mcp_executor import MCPToolExecutor

logger = logging.getLogger(__name__)

_EXCERPT_LEN = 500


class MCPToolProvider(BaseLLMProvider):
    """Runs a model (via OpenRouter) with live access to the TCLP KG MCP tools."""

    def __init__(self, config: Dict[str, Any], api_key: str, executor_factory=None):
        super().__init__(config, api_key)
        self.max_tokens = config.get("max_tokens", 1000)
        self.temperature = config.get("temperature", 0.1)
        self.config_max_iterations = config.get("max_tool_iterations", 8)

        retry_config = config.get("retry", {})
        self.max_attempts = retry_config.get("max_attempts", 3)
        self.backoff_base = retry_config.get("exponential_base", 2)
        self.max_wait = retry_config.get("max_wait_seconds", 60)

        # Secrets follow the codebase convention: sourced from env. The yaml
        # carries no URL/token; tests may pass them explicitly via config.
        mcp_config = config.get("mcp") or {}
        self.mcp_url = mcp_config.get("url") or os.getenv("TCLP_MCP_URL")
        self.mcp_token = mcp_config.get("token") or os.getenv("TCLP_MCP_TOKEN")

        api_endpoint = config.get(
            "api_endpoint", "https://openrouter.ai/api/v1/chat/completions"
        )
        base_url = api_endpoint.replace("/chat/completions", "")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._executor_factory = executor_factory or MCPToolExecutor

    @property
    def name(self) -> str:
        return "mcp"

    def get_provider_name(self) -> str:
        return self.name

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        start_time = time.time()
        trace: list = []

        @retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=self.backoff_base, min=1, max=self.max_wait),
            retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
            reraise=True,
        )
        async def _complete(**params):
            return await self.client.chat.completions.create(**params)

        try:
            async with self._executor_factory(self.mcp_url, self.mcp_token) as ex:
                tools = await ex.list_openai_tools()
                messages = [{"role": "user", "content": request.prompt}]

                for _ in range(self.config_max_iterations):
                    completion = await _complete(
                        model=request.model_name,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                    )
                    msg = completion.choices[0].message
                    messages.append(msg.model_dump())

                    if not msg.tool_calls:
                        return self._response(request, msg.content, trace,
                                              start_time, hit_limit=False)

                    for tc in msg.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError as e:
                            err = f"ERROR: invalid JSON arguments: {e}"
                            trace.append({"name": tc.function.name,
                                          "args": tc.function.arguments,
                                          "result_excerpt": err})
                            messages.append({"role": "tool", "tool_call_id": tc.id,
                                             "content": err})
                            continue
                        result = await ex.call_tool(tc.function.name, args)
                        trace.append({
                            "name": tc.function.name,
                            "args": args,
                            "result_excerpt": result[:_EXCERPT_LEN],
                        })
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": result})

                # Iteration cap hit: force a final answer with no tools.
                final = await _complete(
                    model=request.model_name, messages=messages,
                    max_tokens=request.max_tokens, temperature=request.temperature,
                )
                return self._response(request, final.choices[0].message.content,
                                      trace, start_time, hit_limit=True)

        except Exception as e:  # graceful: matches other providers
            logger.error(f"MCP provider error: {e}")
            return LLMResponse(
                text="", model_name=request.model_name, provider_name=self.name,
                search_query=request.prompt, search_results=[], metadata={},
                execution_time_ms=int((time.time() - start_time) * 1000),
                error=str(e), tool_calls=trace,
            )

    def _response(self, request, text, trace, start_time, hit_limit):
        return LLMResponse(
            text=text or "",
            model_name=request.model_name,
            provider_name=self.name,
            search_query=request.prompt,
            search_results=[],
            metadata={"tool_calls": trace, "hit_tool_limit": hit_limit,
                      "tool_call_count": len(trace)},
            execution_time_ms=int((time.time() - start_time) * 1000),
            tool_calls=trace,
        )
