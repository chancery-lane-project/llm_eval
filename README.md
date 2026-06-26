# TCLP Semantic Benchmark

Measure the impact of semantic markup (JSON-LD) on how Large Language Models discover and interact with legal content from The Chancery Lane Project.

## Installation

```bash
pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and add your API keys:

```bash
# Required API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=sk-or-v1-...

# Optional: Google Custom Search (for direct web search)
GOOGLE_SEARCH_API_KEY=...
GOOGLE_SEARCH_ENGINE_ID=...

# Knowledge Graph MCP server (for the kg-mcp eval arm)
TCLP_MCP_URL=https://...            # required for the mcp provider
TCLP_MCP_TOKEN=                     # optional — leave blank if the URL carries auth as a query string
```

The CLI and the test suite load `.env` automatically, so once these are set
`tclp-benchmark` and `pytest` pick them up without exporting anything.

## Usage

### Basic Evaluation

```bash
# Run evaluation with default providers
tclp-benchmark run --version-tag "pre-jsonld"

# Run with specific providers (comma-separated)
tclp-benchmark run --version-tag "test" --providers openai,anthropic

# Run with OpenRouter (unified API with web search)
tclp-benchmark run --version-tag "post-jsonld-v1" --providers openrouter
```

### OpenRouter Integration

OpenRouter provides unified access to multiple LLM providers with built-in web search capabilities. This allows the benchmark to test how LLMs discover TCLP content via web search with semantic markup.

```bash
# Run evaluation using OpenRouter with web search enabled
tclp-benchmark run --version-tag "openrouter-test" --providers openrouter

# Use specific OpenRouter models
# Configure in config/providers.yaml under openrouter.models
```

Key features:
- **Unified API**: Access multiple LLM providers through a single API
- **Web Search**: Built-in web search with customisable prompts targeting chancerylaneproject.org
- **Online Models**: Use `:online` suffix for models with web search enabled
- **Cost Effective**: Compare multiple providers without managing separate API keys

See [docs/openrouter-integration.md](docs/openrouter-integration.md) for comprehensive setup and usage guide.

### Knowledge Graph (MCP) Integration

The `mcp` provider gives models live access to The Chancery Lane Project's
Knowledge Graph through its MCP server, rather than relying on web search or
training memory. It keeps OpenRouter as the model gateway and runs a
client-side tool-call loop: the model calls the KG tools (`search`,
`entity_lookup`, `taxonomy_facets`, `taxonomy_content`), the benchmark executes
them against the KG, and feeds the results back until the model produces an
answer. The KG tools each model calls are recorded in the run's stored
`response_metadata` for transparency.

```bash
# Run the prompt suite with live KG access (requires TCLP_MCP_URL)
tclp-benchmark run --version-tag "kg-mcp-v3" --providers mcp
```

This is designed as the third arm of a three-way comparison. Run the same
prompts under each retrieval mode, tagging each run, then diff them:

```bash
tclp-benchmark run --version-tag "baseline-memory"     --providers openai,anthropic,gemini
tclp-benchmark run --version-tag "baseline-websearch"  --providers openrouter
tclp-benchmark run --version-tag "kg-mcp-v3"           --providers mcp

# Headline comparison: generic web search vs structured Knowledge Graph
tclp-benchmark report --version-a "baseline-websearch" --version-b "kg-mcp-v3" --format html
```

Configure the models and `max_tool_iterations` for this arm in
`config/providers.yaml` under `mcp`. The MCP server URL and optional token are
read from the environment (`TCLP_MCP_URL` / `TCLP_MCP_TOKEN`), never stored in
the config.

A live end-to-end test against the real KG runs only when the credentials are
present (it is skipped otherwise):

```bash
pytest tests/integration/end_to_end/test_mcp_e2e.py -v
```

> **Note:** tool-calling support varies by model on OpenRouter. A model that
> ignores the tools will answer from memory instead — a low or zero tool-call
> count for a given model is a finding about that model, not a fault in the
> benchmark. The recorded tool-call trace makes this visible.

### Generating Reports

```bash
# Generate comparison report between two evaluation runs
tclp-benchmark report --version-a "pre-jsonld" --version-b "post-jsonld-v1"

# Generate detailed single-run report
tclp-benchmark report --version-tag "post-jsonld-v1" --single

# Specify output format and location
tclp-benchmark report --version-a "baseline" --version-b "test" \
  --format html --output reports/comparison.html
```

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src/tclp_benchmark

# Format code
black src/ tests/

# Lint
ruff check src/ tests/
```

## Project Structure

```
tclp_benchmark/
├── cli/              # Command-line interface
├── config/           # Configuration models and loaders
├── evaluators/       # Evaluation orchestration
├── llm_providers/    # LLM provider implementations
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   ├── gemini_provider.py
│   ├── openrouter_provider.py # OpenRouter unified API (gateway + web search)
│   ├── mcp_provider.py        # Knowledge Graph arm: OpenRouter gateway + KG tool-call loop
│   └── mcp_executor.py        # MCP session + KG tool execution
├── reporting/        # Report generation
├── search/           # Web search functionality
└── storage/          # Database models and management
```

## Documentation

- [OpenRouter Integration Guide](docs/openrouter-integration.md) - Comprehensive guide to using OpenRouter
- [MCP Tool-Loop Provider Design](docs/superpowers/specs/2026-06-25-mcp-tool-loop-provider-design.md) - Design spec for the Knowledge Graph (MCP) eval arm
- [Project Context (CLAUDE.md)](CLAUDE.md) - Detailed project background and goals

## License

Copyright © 2025 The Chancery Lane Project
