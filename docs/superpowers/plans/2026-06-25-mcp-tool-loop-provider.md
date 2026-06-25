# MCP Tool-Loop Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `MCPToolProvider` that runs LLMs against the TCLP Knowledge Graph via a client-side MCP tool-call loop, so the benchmark can compare KG-backed answers against web-search and training-memory baselines.

**Architecture:** A new single-purpose provider composes the existing OpenRouter `AsyncOpenAI` gateway with a new `MCPToolExecutor`. The executor opens a per-call streamable-HTTP MCP session (lazy, task-safe), exposes the four KG tools as OpenAI function-calling schemas, and executes tool calls. `OpenRouterProvider` is left untouched so the `:online` web-search baseline remains a separate arm. The tool-call trace is written into the response's `metadata` dict, which already persists to the `response_metadata` JSON column — no DB migration, no metric changes.

**Tech Stack:** Python ≥3.9, `mcp` SDK (streamable-HTTP client), `openai` AsyncOpenAI (against OpenRouter), `tenacity`, `pytest` + `pytest-asyncio` + `pytest-mock`.

**Spec:** `docs/superpowers/specs/2026-06-25-mcp-tool-loop-provider-design.md`

---

## File Structure

- **Create** `src/tclp_benchmark/llm_providers/mcp_executor.py` — `MCPToolExecutor` (MCP session lifecycle, `call_tool`, `list_openai_tools`) + pure `convert_mcp_tools_to_openai()` helper.
- **Create** `src/tclp_benchmark/llm_providers/mcp_provider.py` — `MCPToolProvider(BaseLLMProvider)` (the tool loop).
- **Modify** `src/tclp_benchmark/llm_providers/base.py` — add `tool_calls` field to `LLMResponse`.
- **Modify** `src/tclp_benchmark/evaluators/orchestrator.py` — register `"mcp"` in `provider_map` and `api_key_map`.
- **Modify** `config/providers.yaml` — add `mcp` block.
- **Modify** `.env.example` — add `TCLP_MCP_URL`, `TCLP_MCP_TOKEN`.
- **Modify** `pyproject.toml` — add `mcp` dependency.
- **Create** `tests/unit/llm_providers/test_mcp_executor.py`
- **Create** `tests/unit/llm_providers/test_mcp_provider.py`
- **Create** `tests/integration/end_to_end/test_mcp_e2e.py` — env-gated live test.

### Key seam for testability

`MCPToolProvider.generate_response` builds its executor via an injectable factory:

```python
def __init__(self, config, api_key, executor_factory=None):
    ...
    self._executor_factory = executor_factory or MCPToolExecutor
```

The executor is an async context manager (`async with self._executor_factory(...) as ex:`). Tests inject a fake async-context executor, so the loop is testable with **no network and no live model**. The `AsyncOpenAI` client is likewise patched/mocked in tests.

---

## Task 1: Add the `mcp` dependency

**Files:**
- Modify: `pyproject.toml` (dependencies list)

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add to the `dependencies` array (alphabetical-ish, near `google-genai`):

```toml
    "mcp>=1.0.0",
```

- [ ] **Step 2: Install it into the venv**

Run: `.venv/bin/python -m pip install -e ".[dev]"`
Expected: installs `mcp` and its deps; ends with `Successfully installed ... mcp-...`

- [ ] **Step 3: Verify import works**

Run: `.venv/bin/python -c "from mcp import ClientSession; from mcp.client.streamable_http import streamablehttp_client; print('ok')"`
Expected: prints `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add mcp SDK dependency for KG tool-loop provider"
```

---

## Task 2: Add `tool_calls` to `LLMResponse`

**Files:**
- Modify: `src/tclp_benchmark/llm_providers/base.py:28-39`
- Test: `tests/unit/llm_providers/test_mcp_provider.py` (new file; first test lives here)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/llm_providers/test_mcp_provider.py`:

```python
"""Unit tests for the MCP tool-loop provider."""
from tclp_benchmark.llm_providers.base import LLMResponse


def test_llmresponse_has_tool_calls_default_empty():
    resp = LLMResponse(
        text="hi",
        model_name="m",
        provider_name="mcp",
        search_query="q",
        search_results=[],
        metadata={},
        execution_time_ms=1,
    )
    assert resp.tool_calls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/llm_providers/test_mcp_provider.py -q --no-cov`
Expected: FAIL — `TypeError`/`AttributeError` (no `tool_calls`).

- [ ] **Step 3: Add the field**

In `base.py`, the `LLMResponse` dataclass — add after `error`:

```python
    error: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
```

Add `field` to the dataclass import: `from dataclasses import dataclass, field`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/llm_providers/test_mcp_provider.py -q --no-cov`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tclp_benchmark/llm_providers/base.py tests/unit/llm_providers/test_mcp_provider.py
git commit -m "feat: add tool_calls trace field to LLMResponse"
```

---

## Task 3: `convert_mcp_tools_to_openai` (pure conversion)

The MCP `list_tools()` returns objects with `.name`, `.description`, `.inputSchema` (a JSON-schema dict). Convert each to OpenAI function-calling format.

**Files:**
- Create: `src/tclp_benchmark/llm_providers/mcp_executor.py`
- Test: `tests/unit/llm_providers/test_mcp_executor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/llm_providers/test_mcp_executor.py`:

```python
"""Unit tests for the MCP tool executor."""
from types import SimpleNamespace

from tclp_benchmark.llm_providers.mcp_executor import convert_mcp_tools_to_openai


def test_convert_single_tool():
    mcp_tool = SimpleNamespace(
        name="search",
        description="Search the KG",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}},
                     "required": ["query"]},
    )
    out = convert_mcp_tools_to_openai([mcp_tool])
    assert out == [{
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the KG",
            "parameters": {"type": "object",
                           "properties": {"query": {"type": "string"}},
                           "required": ["query"]},
        },
    }]


def test_convert_missing_description_defaults_to_empty():
    mcp_tool = SimpleNamespace(name="t", description=None, inputSchema={"type": "object"})
    out = convert_mcp_tools_to_openai([mcp_tool])
    assert out[0]["function"]["description"] == ""
    assert out[0]["function"]["parameters"] == {"type": "object"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/llm_providers/test_mcp_executor.py -q --no-cov`
Expected: FAIL — module/function does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `src/tclp_benchmark/llm_providers/mcp_executor.py`:

```python
"""MCP tool executor: connects to the TCLP KG MCP server and runs its tools."""
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def convert_mcp_tools_to_openai(mcp_tools: list) -> List[Dict[str, Any]]:
    """Convert MCP tool definitions into OpenAI function-calling tool schemas."""
    converted = []
    for tool in mcp_tools:
        converted.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {"type": "object"},
            },
        })
    return converted
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/llm_providers/test_mcp_executor.py -q --no-cov`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tclp_benchmark/llm_providers/mcp_executor.py tests/unit/llm_providers/test_mcp_executor.py
git commit -m "feat: add MCP->OpenAI tool schema conversion"
```

---

## Task 4: `MCPToolExecutor` session + `call_tool`

The executor is an async context manager. On enter it opens `streamablehttp_client` then `ClientSession`, calls `initialize()`, all within an `AsyncExitStack` so teardown happens in the same task on exit. `call_tool` handles `CallToolResult.isError` (errors arrive as a result flag, **not** an exception).

**Files:**
- Modify: `src/tclp_benchmark/llm_providers/mcp_executor.py`
- Test: `tests/unit/llm_providers/test_mcp_executor.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/llm_providers/test_mcp_executor.py`:

```python
import pytest
from types import SimpleNamespace
from tclp_benchmark.llm_providers.mcp_executor import MCPToolExecutor


def _content(text):
    return SimpleNamespace(type="text", text=text)


@pytest.mark.asyncio
async def test_call_tool_returns_joined_text(mocker):
    ex = MCPToolExecutor("https://kg.example/mcp", token=None)
    fake_session = mocker.AsyncMock()
    fake_session.call_tool.return_value = SimpleNamespace(
        isError=False, content=[_content('{"results": []}')]
    )
    ex._session = fake_session  # inject a connected session

    out = await ex.call_tool("search", {"query": "net zero"})

    assert out == '{"results": []}'
    fake_session.call_tool.assert_awaited_once_with("search", {"query": "net zero"})


@pytest.mark.asyncio
async def test_call_tool_error_result_is_returned_not_raised(mocker):
    ex = MCPToolExecutor("https://kg.example/mcp", token=None)
    fake_session = mocker.AsyncMock()
    fake_session.call_tool.return_value = SimpleNamespace(
        isError=True, content=[_content("bad facet name")]
    )
    ex._session = fake_session

    out = await ex.call_tool("taxonomy_content", {"filters": {"x": "y"}})

    assert "ERROR" in out and "bad facet name" in out


def test_build_headers_with_token():
    assert MCPToolExecutor("u", token="abc")._headers() == {"Authorization": "Bearer abc"}


def test_build_headers_without_token():
    assert MCPToolExecutor("u", token=None)._headers() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/llm_providers/test_mcp_executor.py -q --no-cov`
Expected: FAIL — `MCPToolExecutor` lacks `__init__(url, token)`, `_headers`, `call_tool`.

- [ ] **Step 3: Implement the executor**

Add to `mcp_executor.py` (imports at top, class below the helper):

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


class MCPToolExecutor:
    """Per-call MCP session over streamable-HTTP against the TCLP KG server.

    Use as an async context manager so the session is created lazily and torn
    down in the same task (the streamable-HTTP client and ClientSession are
    anyio context managers and must not cross tasks).
    """

    def __init__(self, url: str, token: Optional[str] = None):
        self.url = url
        self.token = token
        self._session: Optional[ClientSession] = None
        self._stack: Optional[AsyncExitStack] = None

    def _headers(self) -> Optional[Dict[str, str]]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else None

    async def __aenter__(self) -> "MCPToolExecutor":
        self._stack = AsyncExitStack()
        read, write, _ = await self._stack.enter_async_context(
            streamablehttp_client(self.url, headers=self._headers())
        )
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc):
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def list_openai_tools(self) -> List[Dict[str, Any]]:
        result = await self._session.list_tools()
        return convert_mcp_tools_to_openai(result.tools)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        result = await self._session.call_tool(name, arguments)
        text = "".join(
            block.text for block in result.content if getattr(block, "text", None)
        )
        if getattr(result, "isError", False):
            return f"ERROR from tool '{name}': {text}"
        return text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/llm_providers/test_mcp_executor.py -q --no-cov`
Expected: PASS (all four)

- [ ] **Step 5: Commit**

```bash
git add src/tclp_benchmark/llm_providers/mcp_executor.py tests/unit/llm_providers/test_mcp_executor.py
git commit -m "feat: add MCPToolExecutor session lifecycle and call_tool"
```

---

## Task 5: `MCPToolProvider` tool loop

Implements `BaseLLMProvider`. Builds an OpenRouter `AsyncOpenAI` client (same base-url pattern as `OpenRouterProvider`). `generate_response` opens the executor, lists tools, and loops: model call with `tools=` → execute any tool calls → feed results back → repeat until the model stops calling tools or the iteration cap is hit (forced final answer). Trace goes into both `tool_calls` and `metadata["tool_calls"]`.

**Files:**
- Create: `src/tclp_benchmark/llm_providers/mcp_provider.py`
- Test: `tests/unit/llm_providers/test_mcp_provider.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/llm_providers/test_mcp_provider.py`:

```python
import json
import pytest
from types import SimpleNamespace
from contextlib import asynccontextmanager

from tclp_benchmark.llm_providers.base import LLMRequest
from tclp_benchmark.llm_providers.mcp_provider import MCPToolProvider


def _msg(content=None, tool_calls=None):
    m = SimpleNamespace(content=content, tool_calls=tool_calls)
    m.model_dump = lambda: {"role": "assistant", "content": content,
                            "tool_calls": bool(tool_calls)}
    return m


def _completion(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                           usage=SimpleNamespace(total_tokens=1, prompt_tokens=1,
                                                 completion_tokens=0))


def _tool_call(call_id, name, args):
    return SimpleNamespace(id=call_id,
                           function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class _FakeExecutor:
    """Async-context fake standing in for MCPToolExecutor."""
    def __init__(self, *a, **k):
        self.calls = []
    async def __aenter__(self): return self
    async def __aexit__(self, *e): return False
    async def list_openai_tools(self):
        return [{"type": "function", "function": {"name": "search"}}]
    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return '{"results": [{"title": "Evies Clause"}]}'


def _make_provider(mocker, completions):
    """Provider whose AsyncOpenAI returns the queued completions in order."""
    provider = MCPToolProvider(
        config={"max_tokens": 100, "temperature": 0.0, "max_tool_iterations": 8,
                "mcp": {"url": "https://kg.example/mcp"}},
        api_key="sk-test",
        executor_factory=_FakeExecutor,
    )
    provider.client.chat.completions.create = mocker.AsyncMock(side_effect=completions)
    return provider


def _request():
    return LLMRequest(prompt="Status of Evie's Clause?", search_context=[],
                      max_tokens=100, temperature=0.0, model_name="openai/gpt-4o")


@pytest.mark.asyncio
async def test_loop_executes_tool_then_returns_answer(mocker):
    first = _completion(_msg(tool_calls=[_tool_call("c1", "search", {"query": "Evie"})]))
    second = _completion(_msg(content="It is Not Maintained."))
    provider = _make_provider(mocker, [first, second])

    resp = await provider.generate_response(_request())

    assert resp.text == "It is Not Maintained."
    assert resp.error is None
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["name"] == "search"
    assert resp.metadata["tool_calls"] == resp.tool_calls
    assert provider.client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_no_tool_calls_returns_immediately(mocker):
    only = _completion(_msg(content="I don't know."))
    provider = _make_provider(mocker, [only])

    resp = await provider.generate_response(_request())

    assert resp.text == "I don't know."
    assert resp.tool_calls == []
    assert provider.client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_iteration_cap_forces_final_answer(mocker):
    # Always returns a tool call -> cap reached -> one final toolless call.
    looping = [_completion(_msg(tool_calls=[_tool_call(f"c{i}", "search", {"query": "x"})]))
               for i in range(8)]
    final = _completion(_msg(content="Best effort answer."))
    provider = _make_provider(mocker, looping + [final])
    provider.config_max_iterations = 8

    resp = await provider.generate_response(_request())

    assert resp.text == "Best effort answer."
    assert resp.metadata["hit_tool_limit"] is True
    assert provider.client.chat.completions.create.await_count == 9


@pytest.mark.asyncio
async def test_executor_connection_failure_returns_error_response(mocker):
    class _BoomExecutor(_FakeExecutor):
        async def __aenter__(self): raise ConnectionError("no endpoint")
    provider = MCPToolProvider(
        config={"mcp": {"url": "https://bad"}}, api_key="sk-test",
        executor_factory=_BoomExecutor,
    )
    resp = await provider.generate_response(_request())
    assert resp.error is not None
    assert resp.text == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/llm_providers/test_mcp_provider.py -q --no-cov`
Expected: FAIL — `mcp_provider` module / `MCPToolProvider` does not exist.

- [ ] **Step 3: Implement the provider**

Create `src/tclp_benchmark/llm_providers/mcp_provider.py`:

```python
"""MCP tool-loop provider: OpenRouter gateway + client-side KG tool execution."""
import json
import logging
import time
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

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

        mcp_config = config.get("mcp", {})
        self.mcp_url = mcp_config.get("url")
        self.mcp_token = mcp_config.get("token")  # optional; may be None

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
        try:
            async with self._executor_factory(self.mcp_url, self.mcp_token) as ex:
                tools = await ex.list_openai_tools()
                messages = [{"role": "user", "content": request.prompt}]

                for _ in range(self.config_max_iterations):
                    completion = await self.client.chat.completions.create(
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
                        args = json.loads(tc.function.arguments or "{}")
                        result = await ex.call_tool(tc.function.name, args)
                        trace.append({
                            "name": tc.function.name,
                            "args": args,
                            "result_excerpt": result[:_EXCERPT_LEN],
                        })
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": result})

                # Iteration cap hit: force a final answer with no tools.
                final = await self.client.chat.completions.create(
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/llm_providers/test_mcp_provider.py -q --no-cov`
Expected: PASS (all five, incl. the Task-2 field test)

- [ ] **Step 5: Commit**

```bash
git add src/tclp_benchmark/llm_providers/mcp_provider.py tests/unit/llm_providers/test_mcp_provider.py
git commit -m "feat: add MCPToolProvider tool-call loop"
```

---

## Task 6: Register `mcp` in the orchestrator

**Files:**
- Modify: `src/tclp_benchmark/evaluators/orchestrator.py` (imports near line 13-16; `_create_provider` maps at lines 477-494)
- Test: `tests/unit/test_orchestrator.py` (add a focused test)

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_orchestrator.py` (follow the file's existing fixture/style; this is the shape):

```python
def test_create_provider_supports_mcp(monkeypatch):
    from tclp_benchmark.evaluators.orchestrator import EvaluationOrchestrator
    from tclp_benchmark.llm_providers.mcp_provider import MCPToolProvider

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    orch = EvaluationOrchestrator(db_manager=None, verbose=False)

    class _Cfg:
        def model_dump(self):
            return {"mcp": {"url": "https://kg.example/mcp"}}

    provider = orch._create_provider("mcp", _Cfg())
    assert isinstance(provider, MCPToolProvider)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_orchestrator.py::test_create_provider_supports_mcp -q --no-cov`
Expected: FAIL — `KeyError: 'mcp'` in `api_key_map` (or `provider_map`).

- [ ] **Step 3: Register the provider**

In `orchestrator.py`, add the import alongside the other providers (after line 16):

```python
from ..llm_providers.mcp_provider import MCPToolProvider
```

In `_create_provider`, add to **both** maps:

```python
        api_key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "mcp": "OPENROUTER_API_KEY",
        }
```

```python
        provider_map = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "gemini": GeminiProvider,
            "openrouter": OpenRouterProvider,
            "mcp": MCPToolProvider,
        }
```

> Note: the trace already persists via `response_metadata` (the provider writes `metadata["tool_calls"]`), so `_execute_model`'s storage block needs **no change**.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_orchestrator.py::test_create_provider_supports_mcp -q --no-cov`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tclp_benchmark/evaluators/orchestrator.py tests/unit/test_orchestrator.py
git commit -m "feat: register mcp provider in orchestrator"
```

---

## Task 7: Config + env example

**Files:**
- Modify: `config/providers.yaml`
- Modify: `.env.example`

- [ ] **Step 1: Add the `mcp` provider block**

Append to `config/providers.yaml`:

```yaml
mcp:
  enabled: true
  api_endpoint: "https://openrouter.ai/api/v1/chat/completions"
  # MCP server connection (URL required; token optional — auth may be in the URL query string)
  mcp:
    url: "${TCLP_MCP_URL}"
    token: "${TCLP_MCP_TOKEN}"
  max_tool_iterations: 8
  models:
    - name: "anthropic/claude-opus-4.5"
      max_tokens: 1500
      temperature: 0.1
    - name: "openai/gpt-4o"
      max_tokens: 1500
      temperature: 0.1
    - name: "google/gemini-3-flash-preview"
      max_tokens: 1500
      temperature: 0.1
  rate_limit:
    requests_per_minute: 50
    tokens_per_minute: 100000
  retry:
    max_attempts: 3
    exponential_base: 2
    max_wait_seconds: 60
```

> **Check during execution:** confirm how this project's config loader expands `${ENV_VAR}` placeholders (inspect `src/tclp_benchmark/config/loader.py` and `config/models.py`). If it does **not** expand env vars, instead have `MCPToolProvider.__init__` read `TCLP_MCP_URL`/`TCLP_MCP_TOKEN` from `os.getenv` as a fallback when `config["mcp"]["url"]` is empty/contains a literal `${...}`. Adjust the task here to match the loader's actual behaviour before committing.

- [ ] **Step 2: Add env vars to `.env.example`**

Append to `.env.example`:

```bash
# TCLP Knowledge Graph MCP server (for the kg-mcp eval arm)
TCLP_MCP_URL=
# Optional — leave blank if the URL already carries auth as a query string
TCLP_MCP_TOKEN=
```

- [ ] **Step 3: Verify config still loads**

Run: `.venv/bin/python -c "from tclp_benchmark.config.loader import ConfigLoader; print('ok')"`
Expected: prints `ok` (import-level check; full load may need a real config path)

- [ ] **Step 4: Commit**

```bash
git add config/providers.yaml .env.example
git commit -m "feat: add mcp provider config and env example"
```

---

## Task 8: Env-gated integration test (live KG)

**Files:**
- Create: `tests/integration/end_to_end/test_mcp_e2e.py`

- [ ] **Step 1: Write the env-gated test**

```python
"""Live end-to-end test for the MCP tool-loop provider. Skipped without creds."""
import os
import pytest

from tclp_benchmark.llm_providers.base import LLMRequest
from tclp_benchmark.llm_providers.mcp_provider import MCPToolProvider

pytestmark = pytest.mark.skipif(
    not (os.getenv("TCLP_MCP_URL") and os.getenv("OPENROUTER_API_KEY")),
    reason="requires TCLP_MCP_URL and OPENROUTER_API_KEY",
)


@pytest.mark.asyncio
async def test_live_mcp_answers_with_kg():
    provider = MCPToolProvider(
        config={
            "max_tokens": 1500, "temperature": 0.0, "max_tool_iterations": 8,
            "mcp": {"url": os.getenv("TCLP_MCP_URL"), "token": os.getenv("TCLP_MCP_TOKEN")},
        },
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    request = LLMRequest(
        prompt="Search the TCLP knowledge graph: how many clauses are 'Not maintained'?",
        search_context=[], max_tokens=1500, temperature=0.0,
        model_name="openai/gpt-4o",
    )
    resp = await provider.generate_response(request)
    assert resp.error is None
    assert resp.text
    # A KG-backed answer should have called at least one tool.
    assert len(resp.tool_calls) >= 1
```

- [ ] **Step 2: Run (skips without creds)**

Run: `.venv/bin/python -m pytest tests/integration/end_to_end/test_mcp_e2e.py -q --no-cov`
Expected: `1 skipped` (no creds in CI) — or PASS locally with creds set.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/end_to_end/test_mcp_e2e.py
git commit -m "test: add env-gated live MCP integration test"
```

---

## Task 9: Full suite + lint gate

**Files:** none (verification)

- [ ] **Step 1: Run the whole unit suite**

Run: `.venv/bin/python -m pytest tests/ -q --no-cov`
Expected: all pass (prior 103 + the new tests), integration test skipped.

- [ ] **Step 2: Lint the new code**

Run: `.venv/bin/ruff check src/tclp_benchmark/llm_providers/mcp_executor.py src/tclp_benchmark/llm_providers/mcp_provider.py src/tclp_benchmark/llm_providers/base.py src/tclp_benchmark/evaluators/orchestrator.py`
Expected: `All checks passed!`

- [ ] **Step 3: Commit any lint fixes (if needed)**

```bash
git add -A
git commit -m "style: ruff cleanups for mcp provider"
```

---

## Done — how to run the three comparison arms

After implementation, run the prompt suite once per arm (each tagged):

```bash
# Floor: no retrieval
.venv/bin/tclp-benchmark run --version-tag baseline-memory --providers openrouter ...

# Generic web search (already configured :online models)
.venv/bin/tclp-benchmark run --version-tag baseline-websearch --providers openrouter ...

# Structured KG via MCP (this feature)
.venv/bin/tclp-benchmark run --version-tag kg-mcp-v3 --providers mcp ...

# Headline comparison report
.venv/bin/tclp-benchmark report --version-a baseline-websearch --version-b kg-mcp-v3 --format html
```

> Exact `run` flags depend on `cli/commands.py` (`--prompts-config`, `--providers`, `--category`). Confirm against `--help` before the live run.
