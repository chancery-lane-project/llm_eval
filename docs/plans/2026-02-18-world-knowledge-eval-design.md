# World Knowledge Eval — Design Doc

**Date:** 2026-02-18
**Status:** Approved

## Problem

The existing benchmark tests how well LLMs retrieve TCLP content via web search. This eval asks a different question: which models already know about The Chancery Lane Project and its clauses from training data alone, and how many clauses can they name?

## Scope

Standalone script — no changes to the existing `tclp_benchmark` package.

## Files

```
scripts/world_knowledge_eval.py      # standalone script
scripts/world_knowledge_models.yaml  # curated model list
reports/world-knowledge-YYYY-MM-DD.md  # output (generated)
```

## Models

~30 models via OpenRouter, configured in `scripts/world_knowledge_models.yaml`. Initial list:

**Frontier models**
- `openai/gpt-4o`
- `openai/gpt-4o-mini`
- `openai/gpt-4-turbo`
- `openai/gpt-3.5-turbo`
- `openai/o1-mini`
- `anthropic/claude-3.5-sonnet`
- `anthropic/claude-3-opus`
- `anthropic/claude-3-haiku`
- `anthropic/claude-3.5-haiku`
- `google/gemini-2.0-flash-001`
- `google/gemini-pro-1.5`
- `google/gemini-flash-1.5`
- `x-ai/grok-2`
- `amazon/nova-pro-v1`

**Open-weight models**
- `meta-llama/llama-3.3-70b-instruct`
- `meta-llama/llama-3.1-405b-instruct`
- `meta-llama/llama-3.1-70b-instruct`
- `mistralai/mistral-large`
- `mistralai/mixtral-8x22b-instruct`
- `mistralai/mixtral-8x7b-instruct`
- `mistralai/mistral-7b-instruct`
- `qwen/qwen-2.5-72b-instruct`
- `qwen/qwen-2.5-7b-instruct`
- `cohere/command-r-plus`
- `cohere/command-r`
- `databricks/dbrx-instruct`
- `nvidia/llama-3.1-nemotron-70b-instruct`
- `deepseek/deepseek-chat`
- `deepseek/deepseek-r1`
- `microsoft/phi-4`
- `allenai/olmo-2-0325-32b-instruct`
- `01-ai/yi-large`

All models called **without** web search (no `:online` suffix, no tools parameter).

## Ground Truth

Parsed at runtime from `clauses/` folder. Each `.md` file yields two name forms from frontmatter:
- `title` (e.g. "Interest Ratchet Clause for Carbon Saving")
- `eleventyComputed.clause_child_name` (e.g. "Laith and Irsa's Clause")

Both forms are added to a flat set of known names for matching.

## Prompts

**Prompt 1 — Awareness**
> "What is The Chancery Lane Project? Please describe what it does, what it publishes, and who it is aimed at."

Scored: boolean — does response contain "The Chancery Lane Project", "TCLP", or "chancerylaneproject.org"?

**Prompt 2 — Enumeration**
> "List every clause, guide, questionnaire, handbook, or other resource published by The Chancery Lane Project that you know about. For each one, give its full name. Do not search the web — use only your training knowledge."

Scored: real clause count (matched against ground truth), hallucination count (looks like a clause name but matches nothing).

**Hallucination detection heuristic:** any extracted name containing "Clause", "DDQ", "Handbook", "Annex", "Minutes", or "Protocol" that does not match a known ground truth name.

## Scoring

- **Awareness**: boolean (yes/no)
- **Clauses found**: integer count of ground truth matches
- **Hallucinations**: integer count of unmatched clause-like names
- **Accuracy**: `real / (real + hallucinated)` — omitted if both are 0

## Report Format (`reports/world-knowledge-YYYY-MM-DD.md`)

1. **Summary table** — one row per model, sorted by clauses found descending
   Columns: Model | Knows TCLP | Clauses Found | Real | Hallucinated | Accuracy

2. **Statistical overview**
   - Total models tested, % that know TCLP
   - Mean and median clauses found
   - Best performer (most real clauses)
   - Most accurate (highest real:total ratio, min 3 clauses found)
   - Breakdown by model family

3. **Per-model detail** — raw responses to both prompts for qualitative review

## Configuration

`scripts/world_knowledge_models.yaml`:
```yaml
models:
  - openai/gpt-4o
  - anthropic/claude-3.5-sonnet
  # ...

settings:
  max_tokens: 1000
  temperature: 0
  requests_per_minute: 20
```

Uses `OPENROUTER_API_KEY` from `.env` (same as existing project).
