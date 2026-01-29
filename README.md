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
```

## Usage

### Basic Evaluation

```bash
# Run evaluation with default providers
tclp-benchmark run --version-tag "pre-jsonld"

# Run with specific providers
tclp-benchmark run --version-tag "test" --providers openai anthropic

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
│   ├── openai.py
│   ├── anthropic.py
│   ├── gemini.py
│   └── openrouter.py # OpenRouter unified API
├── reporting/        # Report generation
├── search/           # Web search functionality
└── storage/          # Database models and management
```

## Documentation

- [OpenRouter Integration Guide](docs/openrouter-integration.md) - Comprehensive guide to using OpenRouter
- [Project Context (CLAUDE.md)](CLAUDE.md) - Detailed project background and goals

## License

Copyright © 2025 The Chancery Lane Project
