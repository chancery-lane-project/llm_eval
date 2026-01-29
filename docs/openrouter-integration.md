# OpenRouter Integration Guide

## Overview

OpenRouter (openrouter.ai) provides a unified API to access multiple LLM providers (OpenAI, Anthropic, Google, etc.) through a single endpoint. This integration enables the TCLP Semantic Benchmark to:

1. **Test multiple providers efficiently** - Access various models without managing separate API keys
2. **Leverage web search** - Use OpenRouter's built-in web search to test content discovery
3. **Compare semantic markup impact** - Measure how JSON-LD affects LLM web search results
4. **Simplify configuration** - One API key, multiple models

## Why OpenRouter?

### Key Benefits

1. **Unified Web Search**: OpenRouter's `:online` models or web search plugin automatically fetch web content and inject it into the LLM context. This is ideal for testing whether LLMs can discover and cite TCLP clauses from chancerylaneproject.org.

2. **Multiple Providers**: Test OpenAI, Anthropic, Google, and other providers through a single integration point.

3. **Cost Transparency**: OpenRouter shows per-request costs and usage statistics.

4. **Fallback Support**: Configure fallback models if primary models are unavailable.

## Setup

### 1. Get API Key

1. Visit [openrouter.ai](https://openrouter.ai)
2. Sign up or log in
3. Navigate to "Keys" section
4. Generate a new API key

### 2. Configure Environment

Add your OpenRouter API key to `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

### 3. Configure Providers

OpenRouter configuration is in `config/providers.yaml`:

```yaml
openrouter:
  enabled: true
  api_endpoint: "https://openrouter.ai/api/v1/chat/completions"

  # Models to test - use :online suffix for web search
  models:
    - "anthropic/claude-opus-4-5-20251101:online"
    - "openai/gpt-4o:online"
    - "google/gemini-2.0-flash-exp:online"

  # Model parameters
  max_tokens: 1000
  temperature: 0.1

  # Web search configuration
  web_search:
    enabled: true
    max_results: 5
    custom_search_prompt: "Search chancerylaneproject.org for legal clauses matching this query. Include exact clause names, maintenance status (schema:creativeWorkStatus), jurisdiction, practice area, and any semantic relationships (parent clauses via schema:hasPart, variants via schema:isBasedOn, related standards like SBTi or GHG Protocol)."

  # Rate limiting
  rate_limit:
    requests_per_minute: 50
    tokens_per_minute: 100000

  # Retry configuration
  retry:
    max_attempts: 3
    exponential_base: 2
    max_wait_seconds: 60
```

## Usage

### Basic Commands

```bash
# Run evaluation with OpenRouter
tclp-benchmark run --version-tag "openrouter-test" --providers openrouter

# Run specific category with OpenRouter
tclp-benchmark run --version-tag "test" --providers openrouter --category direct_lookup

# Run OpenRouter alongside other providers for comparison
tclp-benchmark run --version-tag "comparison" --providers openai anthropic openrouter
```

### Model Selection

OpenRouter supports hundreds of models. Common patterns:

**Web Search Enabled (`:online` suffix)**
```yaml
models:
  - "anthropic/claude-opus-4-5-20251101:online"
  - "openai/gpt-4o:online"
  - "google/gemini-2.0-flash-exp:online"
```

**Standard Models (no web search)**
```yaml
models:
  - "anthropic/claude-opus-4-5-20251101"
  - "openai/gpt-4o"
  - "google/gemini-2.0-flash-exp"
```

**Cost-Optimised Models**
```yaml
models:
  - "anthropic/claude-3-5-haiku-20241022:online"
  - "openai/gpt-4o-mini:online"
```

See [OpenRouter models page](https://openrouter.ai/models) for the complete list.

## Web Search Configuration

### How It Works

When web search is enabled:

1. Test prompt is sent to OpenRouter
2. OpenRouter's web search plugin queries the web using `custom_search_prompt` + user prompt
3. Search results are injected into the LLM context
4. LLM responds with access to web content
5. Benchmark evaluates whether TCLP content was discovered and cited correctly

### Custom Search Prompt

The `custom_search_prompt` guides the web search to focus on TCLP content and semantic markup:

```yaml
web_search:
  enabled: true
  max_results: 5
  custom_search_prompt: "Search chancerylaneproject.org for legal clauses matching this query. Include exact clause names, maintenance status (schema:creativeWorkStatus), jurisdiction, practice area, and any semantic relationships (parent clauses via schema:hasPart, variants via schema:isBasedOn, related standards like SBTi or GHG Protocol)."
```

This prompt:
- Targets `chancerylaneproject.org` domain
- Requests semantic markup fields (schema:creativeWorkStatus, schema:hasPart, etc.)
- Asks for structured metadata (jurisdiction, practice area, standards)

### Adjusting Search Behaviour

**Increase search depth:**
```yaml
web_search:
  max_results: 10  # Default: 5
```

**Disable web search for baseline testing:**
```yaml
web_search:
  enabled: false
```

**Focus on specific content types:**
```yaml
custom_search_prompt: "Search chancerylaneproject.org specifically for maintained clauses (schema:creativeWorkStatus = 'Published') related to carbon reduction. Include jurisdiction and applicable standards."
```

## Testing Semantic Markup Impact

### Recommended Workflow

**1. Baseline Test (Pre-JSON-LD)**
```bash
# Test before semantic markup is added
tclp-benchmark run --version-tag "pre-jsonld" --providers openrouter
```

**2. Post-Implementation Test**
```bash
# Test after adding JSON-LD to chancerylaneproject.org
tclp-benchmark run --version-tag "post-jsonld-v1" --providers openrouter
```

**3. Generate Comparison Report**
```bash
# Compare results
tclp-benchmark report --version-a "pre-jsonld" --version-b "post-jsonld-v1"
```

### What to Look For

The report will show improvements in:

1. **Discovery**: Did the LLM find relevant TCLP clauses more often?
2. **Citation Accuracy**: Are clause names cited correctly?
3. **Taxonomy Awareness**: Does the response mention jurisdiction, practice area, climate impact?
4. **Metadata Richness**: Are maintenance status, standards, and relationships surfaced?

Example improvement:
```
Metric: tclp_citation
Baseline: 0.450
Post-JSON-LD: 0.820
Improvement: +82.2%
```

## Advanced Configuration

### Rate Limiting

OpenRouter has different rate limits per model. Configure conservatively:

```yaml
rate_limit:
  requests_per_minute: 50      # Conservative default
  tokens_per_minute: 100000    # Adjust based on plan
```

Monitor usage on OpenRouter dashboard and adjust if hitting limits.

### Retry Strategy

Configure retry behaviour for transient failures:

```yaml
retry:
  max_attempts: 3              # Retry up to 3 times
  exponential_base: 2          # 2^n backoff (2s, 4s, 8s)
  max_wait_seconds: 60         # Maximum wait between retries
```

### Multiple OpenRouter Configurations

To test different OpenRouter configurations, create custom provider configs:

```yaml
# config/providers-openrouter-aggressive.yaml
openrouter:
  enabled: true
  models:
    - "anthropic/claude-opus-4-5-20251101:online"
  web_search:
    enabled: true
    max_results: 10  # More search results
    custom_search_prompt: "Exhaustively search chancerylaneproject.org..."
```

```bash
# Use custom config
tclp-benchmark run --version-tag "aggressive-search" \
  --providers openrouter \
  --providers-config config/providers-openrouter-aggressive.yaml
```

## Comparison with Direct API Providers

### OpenRouter vs Direct APIs

| Feature | OpenRouter | Direct APIs (OpenAI/Anthropic/Google) |
|---------|------------|---------------------------------------|
| **Setup** | Single API key | Separate key per provider |
| **Web Search** | Built-in with `:online` models | Requires separate implementation |
| **Cost** | Small markup (~5-10%) | Direct pricing |
| **Rate Limits** | Unified limits | Per-provider limits |
| **Model Access** | 200+ models | Provider-specific only |
| **Fallback** | Automatic fallback support | Manual implementation needed |

### When to Use OpenRouter

**Use OpenRouter when:**
- Testing web search-based content discovery
- Comparing multiple providers efficiently
- Semantic markup testing requires web crawling
- Budget allows for slight cost premium

**Use Direct APIs when:**
- Absolute minimum cost is priority
- Web search not needed (testing against known content)
- Provider-specific features required (e.g., Anthropic's system prompts)

## Troubleshooting

### Common Issues

#### 1. API Key Not Found

```
Error: OpenRouter API key not found
```

**Solution:** Ensure `OPENROUTER_API_KEY` is set in `.env`:
```bash
echo "OPENROUTER_API_KEY=sk-or-v1-..." >> .env
```

#### 2. Model Not Available

```
Error: Model anthropic/claude-opus-4-5-20251101:online not available
```

**Solution:** Check [OpenRouter models page](https://openrouter.ai/models) for current availability. Some models require whitelisting or have regional restrictions.

#### 3. Rate Limiting

```
Error: 429 Too Many Requests
```

**Solution:** Reduce `requests_per_minute` in config:
```yaml
rate_limit:
  requests_per_minute: 20  # Reduce from 50
```

#### 4. Web Search Not Working

If results don't show web search content:

**Check model suffix:**
```yaml
# Wrong - no web search
models:
  - "anthropic/claude-opus-4-5-20251101"

# Correct - web search enabled
models:
  - "anthropic/claude-opus-4-5-20251101:online"
```

**Verify web search enabled:**
```yaml
web_search:
  enabled: true  # Must be true
```

#### 5. High Costs

OpenRouter charges per request. Monitor usage:

1. Check OpenRouter dashboard for usage
2. Use smaller models for testing:
   ```yaml
   models:
     - "anthropic/claude-3-5-haiku-20241022:online"
   ```
3. Reduce test case count during development

## Reporting

### OpenRouter-Specific Metrics

Reports automatically highlight OpenRouter results:

**Provider Identification**
- Provider: `openrouter`
- Model: `anthropic/claude-opus-4-5-20251101:online`
- Web Search: `enabled` (if `:online` suffix present)

**Comparison Reports**

When comparing OpenRouter vs direct APIs, reports show:
- Performance differences with/without web search
- Citation accuracy improvements
- Discovery rate changes

Example report section:
```
## Metrics by Provider

### openrouter/anthropic/claude-opus-4-5-20251101:online
- Web Search: Enabled
- TCLP Citation: 0.850
- Taxonomy Awareness: 0.780
```

### Interpreting Results

**High scores with OpenRouter + web search:**
- Semantic markup is working
- LLMs can discover and cite TCLP content
- Metadata is being surfaced correctly

**Low scores with OpenRouter + web search:**
- Semantic markup may not be visible/correct
- Search indexing issues
- Custom search prompt may need refinement

**OpenRouter scores match/exceed direct API:**
- Web search successfully provides relevant context
- Semantic markup benefits are being realised

## Best Practices

### 1. Staged Rollout

```bash
# Start with one model
models:
  - "openai/gpt-4o-mini:online"

# Expand after validating setup
models:
  - "openai/gpt-4o-mini:online"
  - "anthropic/claude-3-5-haiku-20241022:online"

# Full comparison
models:
  - "openai/gpt-4o:online"
  - "anthropic/claude-opus-4-5-20251101:online"
  - "google/gemini-2.0-flash-exp:online"
```

### 2. Version Tagging

Use descriptive version tags:
```bash
# Good
--version-tag "pre-jsonld-baseline"
--version-tag "post-jsonld-v1-with-sbti"
--version-tag "openrouter-gpt4o-online-20250129"

# Less clear
--version-tag "test1"
--version-tag "v2"
```

### 3. Custom Search Prompts

Iterate on search prompts:

```yaml
# v1: Basic
custom_search_prompt: "Search chancerylaneproject.org for clauses."

# v2: Include metadata
custom_search_prompt: "Search chancerylaneproject.org for clauses. Include jurisdiction and practice area."

# v3: Semantic-aware
custom_search_prompt: "Search chancerylaneproject.org for clauses with JSON-LD metadata including schema:creativeWorkStatus, jurisdiction, practice area, and standards."
```

### 4. Cost Management

- Use `-mini` or `-haiku` models for development
- Run full suite only for formal benchmarks
- Archive old evaluation runs to manage database size

### 5. Documentation

Document each evaluation run:
```bash
tclp-benchmark run \
  --version-tag "post-jsonld-v1" \
  --description "Testing JSON-LD with Schema.org and TCLP namespace. Added schema:hasPart relationships." \
  --providers openrouter
```

## Integration with CI/CD

### Automated Benchmarking

Run benchmarks on semantic markup changes:

```yaml
# .github/workflows/benchmark.yml
name: TCLP Benchmark
on:
  push:
    branches: [main]
    paths:
      - 'semantic-markup/**'

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmark
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: |
          pip install -e .
          tclp-benchmark run --version-tag "auto-${{ github.sha }}" --providers openrouter
      - name: Generate report
        run: |
          tclp-benchmark report --version-a "baseline" --version-b "auto-${{ github.sha }}"
```

## Support and Resources

- **OpenRouter Documentation**: [openrouter.ai/docs](https://openrouter.ai/docs)
- **Model List**: [openrouter.ai/models](https://openrouter.ai/models)
- **Discord Support**: [discord.gg/openrouter](https://discord.gg/openrouter)
- **TCLP Project**: [chancerylaneproject.org](https://chancerylaneproject.org)

## Next Steps

1. **Set up API key** and verify configuration
2. **Run baseline test** with OpenRouter
3. **Add semantic markup** to chancerylaneproject.org
4. **Run post-implementation test**
5. **Generate comparison report**
6. **Iterate on semantic markup** based on results

See also:
- [Migration Guide](migration-guides/openrouter-migration.md) - Migrating from direct APIs
- [CLAUDE.md](../CLAUDE.md) - Project context and goals
- [README.md](../README.md) - Quick start guide
