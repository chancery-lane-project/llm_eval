# Design: MCP Tool-Loop Provider for the TCLP Semantic Benchmark

**Date:** 2026-06-25
**Status:** Approved (design); implementation plan pending
**Author:** Felix Cohen (with Claude)

## 1. Purpose

Enable the benchmark to run the full test-prompt suite (`config/prompts.yaml`)
against LLMs that have **live access to the TCLP Knowledge Graph via its MCP
server** (`TCLP_KG_v3`), so we can measure whether structured KG access improves
answer accuracy versus generic web search and versus training memory alone.

This is the third retrieval arm. The first two already exist:

- training-memory only (current orchestrator path — no retrieval), and
- OpenRouter `:online` generic web search (`OpenRouterProvider`, already wired).

## 2. Background / current state

- The evaluation pipeline is fully wired end-to-end:
  `cli run` → `EvaluationOrchestrator.run_evaluation` → per prompt × provider ×
  model: call provider → store `LLMResponse` → DeepEval `_evaluate_metrics` →
  store `MetricScore`; then `cli report --version-a --version-b` diffs two
  version-tagged runs into Markdown/HTML.
- Storage (`storage/models.py`, `storage/database.py`) and reporting
  (`reporting/report_generator.py`) are ready and version-tag aware. Tagged runs
  are comparable.
- **Retrieval is stubbed in the orchestrator**: `_execute_model` sends the bare
  prompt with `search_context=[]` and stores `search_results=None`
  ("Will add search later"); `search/__init__.py` is empty.
- The custom metrics (`evaluators/custom_metrics.py`) are **content-based** and
  provider-agnostic: `TCLPCitationMetric`, `TaxonomyAwarenessMetric`,
  `EntityRetrievalAccuracyMetric`, `MetadataPrecisionMetric` (checks maintenance
  status), `RelationshipTraversalScoreMetric` all score the final answer text
  against `expected_concepts`. They require no changes for this work.
- No MCP client code exists and the `mcp` Python SDK is not installed.

### MCP server reality (verified live)

The `TCLP_KG_v3` server exposes four tools, each returning JSON with `meta` +
`results`:

- `search(query, node_type, limit, rerank, include_full_text)` — fusion
  (semantic + BM25) search; hits carry `title`, `url`, `content_type`, scores,
  optional `relationships_text` and `full_text`.
- `entity_lookup(name, limit, include_full_text)` — deterministic name match
  against Entity nodes.
- `taxonomy_facets(scope)` — lists facets and value slugs (includes
  `maintenance_status`: "Maintained" / "Not maintained", `jurisdiction`,
  `sector`, `practice_area`, etc.).
- `taxonomy_content(filters, scope, limit, offset, sort)` — content filtered by
  facet=value (AND-combined).

The semantic data the prompts probe (maintenance status, jurisdiction, sector)
is present in the KG, so the prompts are answerable with correct retrieval.

### Transport decision

OpenRouter does **not** broker MCP servers server-side; its MCP support is a
client-side pattern (convert MCP tools to OpenAI function-calling definitions,
client executes tool calls). Therefore:

- **OpenRouter stays as the model gateway** (one client for GPT/Claude/Gemini).
- **The harness owns the tool-execution loop** and connects to the KG server
  itself, over a **hosted HTTP endpoint** (streamable-HTTP) using the `mcp`
  Python SDK, authenticated with a token.

## 3. Architecture (approach A: composed provider)

A new single-purpose provider composes the OpenRouter chat client with an MCP
tool executor. `OpenRouterProvider` is left untouched so the web-search baseline
remains an independent arm.

| Unit | File | Responsibility |
|---|---|---|
| `MCPToolExecutor` | `src/tclp_benchmark/llm_providers/mcp_executor.py` (new) | Owns the MCP SDK connection (streamable-HTTP → hosted KG endpoint + token). `list_tools()`, `call_tool(name, args) -> str`, `openai_tool_schemas()` (MCP tool list → OpenAI function-calling JSON). **Session lifecycle:** see §3.1. |
| `MCPToolProvider` | `src/tclp_benchmark/llm_providers/mcp_provider.py` (new) | `BaseLLMProvider` implementation. Holds an OpenRouter `AsyncOpenAI` client + an `MCPToolExecutor`. Implements `generate_response()` — runs the tool loop. |
| `LLMResponse` | `src/tclp_benchmark/llm_providers/base.py` (edit) | Add `tool_calls: List[dict] = field(default_factory=list)` after the `error` field to carry the tool-call trace. |
| Orchestrator | `src/tclp_benchmark/evaluators/orchestrator.py` (edit) | Register `"mcp"` in `_create_provider`'s `provider_map` **and add a matching entry to `api_key_map`** — that function raises `ValueError` for any provider absent there. The MCP arm's API key is `OPENROUTER_API_KEY` (gateway); `TCLP_MCP_URL` and the **optional** `TCLP_MCP_TOKEN` are read separately from env (see §3.2). Persist `response.tool_calls` into the existing `response_metadata` JSON column. |
| Config | `config/providers.yaml`, `.env.example` (edit) | New `mcp` provider block (models, `max_tool_iterations`, endpoint ref); env var `TCLP_MCP_URL` (required) and `TCLP_MCP_TOKEN` (**optional** — see §3.2). |

**No DB migration** — the trace rides in `response_metadata` (existing JSON
column). **No metric changes.**

### 3.1 MCP session lifecycle (explicit)

The `mcp` SDK's `streamablehttp_client(...)` and `ClientSession(...)` are
anyio-based **async context managers**. Entering one in one task and exiting in
another raises `RuntimeError: Attempted to exit cancel scope in a different
task`, so lifecycle must be deliberate. Two facts constrain the choice:

- The orchestrator runs **strictly sequentially in a single task** (nested
  `for` loops with bare `await`, no `asyncio.gather`/`create_task`), so a session
  used across calls would stay task-affine.
- `_create_provider` is invoked **fresh per model** (`orchestrator.py`), so a
  session cached on the provider instance would be built and torn down per model
  anyway.

**Decision (correctness over speed, consistent with §10):** scope the MCP
session **per `generate_response` call** using `async with` inside the executor.
The executor exposes an `async with MCPToolExecutor(url, token) as ex:` (or an
internal `_session()` async-context helper) that opens the streamable-HTTP
client + `ClientSession`, calls `initialize()`, yields, and tears down in the
same task. `openai_tool_schemas()` is fetched once per call inside that scope.
The session is created **lazily** (never in `__init__`, which has no running
event loop). If profiling later shows the per-call connect cost matters, a
persistent task-affine session can be introduced behind the same interface — but
that is explicitly out of scope now.

### 3.2 Authentication (two modes, token optional)

The hosted KG endpoint accepts auth **either** embedded in the URL query string
**or** as an HTTP header token. So:

- `TCLP_MCP_URL` — **required**. May already carry credentials as a query string
  (e.g. `https://kg.example/mcp?key=…`), in which case no separate token is
  needed.
- `TCLP_MCP_TOKEN` — **optional**. When set, the executor passes it as an HTTP
  header (`Authorization: Bearer <token>`) to `streamablehttp_client(...)` via
  its `headers=` argument. When unset, the executor connects with the URL alone.

The executor builds the header dict conditionally: `headers = {"Authorization":
f"Bearer {token}"} if token else None`. No failure if the token is absent — that
is a supported configuration, not an error. (A connection that genuinely
requires auth and gets none will fail at connect time and degrade gracefully per
§6.)

## 4. Data flow — the tool loop

Inside `MCPToolProvider.generate_response(request)`:

All of the following runs inside the executor's per-call session scope (§3.1):

```
tools = executor.openai_tool_schemas()          # fetched once in-scope
messages = [{"role": "user", "content": request.prompt}]
trace = []
for i in range(max_tool_iterations):            # default 8
    completion = await openrouter.chat.completions.create(
        model=request.model_name, messages=messages,
        tools=tools, tool_choice="auto",
        max_tokens=request.max_tokens, temperature=request.temperature,
    )
    msg = completion.choices[0].message
    messages.append(msg.model_dump())           # serialise — don't append raw SDK object
    if not msg.tool_calls:
        return LLMResponse(text=msg.content, tool_calls=trace, ...)   # done
    for tc in msg.tool_calls:
        result = await executor.call_tool(tc.function.name,
                                          json.loads(tc.function.arguments))
        trace.append({"name": tc.function.name, "args": ..., "result_excerpt": ...})
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

# iteration cap reached: one final call WITHOUT tools to force an answer,
# then RETURN it with metadata["hit_tool_limit"] = True
final = await openrouter.chat.completions.create(model=..., messages=messages)
return LLMResponse(text=final.choices[0].message.content, tool_calls=trace, ...)
```

Tool execution is sequential (correctness over speed). The assistant message is
serialised with `model_dump()` before being appended (raw SDK message objects
mixed with dicts across turns are fragile). The trace records every tool call
(name, arguments, a result excerpt) for transparency.

## 5. Comparison design — three arms

Same model set and same prompts across all arms; one version tag each.

| version_tag | provider | retrieval |
|---|---|---|
| `baseline-memory` | direct model (existing path) | none — the floor |
| `baseline-websearch` | OpenRouter `:online` (exists) | generic web search |
| `kg-mcp-v3` | new `MCPToolProvider` | structured TCLP KG |

The headline funder report diffs `baseline-websearch` vs `kg-mcp-v3` (generic web
vs our semantic KG). The memory arm establishes the no-retrieval floor.

## 6. Error handling

Matches existing graceful patterns (per `CLAUDE.md`: graceful rate-limit/API
handling).

- **MCP connection failure** → return `LLMResponse(error=…)`; the run continues
  to the next model (mirrors current provider behaviour).
- **Tool returns an error result** → MCP tool errors arrive as
  `CallToolResult.isError == True`, **not** as a raised exception. `call_tool`
  must inspect `result.isError` and extract `result.content[].text`; an error
  result is fed back to the model as the tool message (so it can recover) and
  recorded in the trace.
- **Tool raises / malformed JSON arguments** → feed the error string back as the
  tool result so the model can recover; record it in the trace.
- **Iteration cap reached** → one final answer call without tools; flag
  `hit_tool_limit` in `response_metadata`.
- **Rate limits / transient API errors** → reuse the `tenacity` retry decorator
  pattern already used by the other providers.

## 7. Testing (TDD)

- **Unit:** MCP→OpenAI tool-schema conversion.
- **Unit:** tool loop with a **mocked** OpenRouter client + mocked executor —
  asserts it executes returned tool calls, appends results as `tool` messages,
  terminates when the model returns no tool calls, and respects the iteration
  cap (forced final answer).
- **Unit:** error paths — tool raises, connection fails, malformed tool args.
- **Integration:** one live prompt against the real KG endpoint, **gated by env
  vars** (skipped in CI when creds absent). No live model calls in unit tests.

## 8. Dependencies

- Add `mcp` (official Model Context Protocol Python SDK) to `pyproject.toml`
  `dependencies`. This is the only new dependency and it is genuinely used.

## 9. Risks

- **Variable tool-calling support across models on OpenRouter.** Not every model
  or routed endpoint honours `tools=` consistently (Gemini and some endpoints are
  unreliable). The loop is safe — a model that ignores tools simply answers from
  memory — but a low/zero tool-call count for such a model is a *finding about
  that model*, not a harness bug. The trace makes this visible. Document this in
  the report narrative so stakeholders read the numbers correctly.
- **KG endpoint availability/auth.** Requires a hosted endpoint + token. If
  absent at run time the `kg-mcp-v3` arm cannot run; the connection-failure path
  degrades gracefully per §6.

## 10. Out of scope (YAGNI)

- Generic multi-tool-source provider abstraction (only one tool source exists).
- New or modified metrics.
- Database schema changes.
- Parallel tool execution.
- Rewriting `world_knowledge_eval.py` (separate, already-shipped script).
