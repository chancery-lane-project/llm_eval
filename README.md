# TCLP Semantic Benchmark

Measure the impact of semantic markup (JSON-LD) on how Large Language Models discover and interact with legal content from The Chancery Lane Project.

## Installation

```bash
pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and add your API keys.

## Usage

```bash
# Run evaluation
tclp-benchmark run --version-tag "pre-jsonld"

# Generate comparison report
tclp-benchmark report --version-a "pre-jsonld" --version-b "post-jsonld-v1"
```

## Development

```bash
# Run tests
pytest

# Format code
black src/ tests/

# Lint
ruff check src/ tests/
```
