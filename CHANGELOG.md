# Changelog

All notable changes to the TCLP Semantic Benchmark project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **OpenRouter Integration**: Unified API access to multiple LLM providers with built-in web search
  - New `OpenRouterProvider` implementing `BaseLLMProvider` interface
  - Configuration support for OpenRouter in `config/providers.yaml`
  - Web search configuration with customisable search prompts
  - Support for `:online` model suffix for web search-enabled models
  - CLI support: `--providers openrouter`
  - Comprehensive documentation in `docs/openrouter-integration.md`
  - Migration guide in `docs/migration-guides/openrouter-migration.md`
  - End-to-end integration tests in `tests/integration/end_to_end/test_openrouter_e2e.py`
  - Validation script: `scripts/validate_openrouter.sh`

- **Report Enhancements**:
  - OpenRouter provider recognition in reports
  - Web search indicators (🌐 badge) for `:online` models
  - Provider comparison highlighting in HTML and Markdown reports
  - Improved provider identification in detailed reports

- **Environment Configuration**:
  - `OPENROUTER_API_KEY` environment variable support
  - Updated `.env.example` with OpenRouter configuration

### Changed
- Updated `README.md` with OpenRouter usage examples
- Updated `CLAUDE.md` with OpenRouter API key documentation
- Enhanced report generation to distinguish OpenRouter web search from direct API calls

### Documentation
- New comprehensive guide: `docs/openrouter-integration.md`
- New migration guide: `docs/migration-guides/openrouter-migration.md`
- Updated README with OpenRouter integration section
- Added OpenRouter examples to CLI help text

## [0.1.0] - 2025-01-27

### Added
- Initial release of TCLP Semantic Benchmark
- Support for OpenAI, Anthropic, and Google Gemini providers
- Custom TCLP metrics: citation accuracy, taxonomy awareness
- SQLite database for evaluation storage
- Version tagging for semantic markup iterations
- HTML and Markdown report generation
- CLI with `run` and `report` commands
- Configurable prompts, providers, and metrics via YAML
- Rate limiting and retry logic for API providers
- Verbose logging and error handling

### Features
- Multi-LLM testing framework
- DeepEval integration for standard metrics
- Custom domain-specific metrics for legal content
- Before/after comparison reports
- Detailed single-run reports with prompt/response view
- Google Custom Search integration for web search baseline

### Documentation
- Project context in CLAUDE.md
- Basic usage in README.md
- Configuration examples
- Development setup instructions
