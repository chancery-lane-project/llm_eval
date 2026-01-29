# OpenRouter Migration Guide

This guide helps existing TCLP Benchmark users migrate from direct LLM provider APIs (OpenAI, Anthropic, Google) to OpenRouter, or to add OpenRouter alongside existing providers for comparison.

## Overview

OpenRouter provides a unified API to access multiple LLM providers with built-in web search capabilities. This is particularly useful for testing semantic markup discovery, as OpenRouter's `:online` models can search the web and discover TCLP content with JSON-LD metadata.

## Migration Paths

Choose the migration path that fits your needs:

1. **Add OpenRouter Alongside Existing Providers** (Recommended) - Compare web search vs direct API
2. **Migrate Completely to OpenRouter** - Simplify API key management
3. **Hybrid Approach** - Use OpenRouter for web search tests, direct APIs for baseline

## Prerequisites

- Existing TCLP Benchmark installation
- OpenRouter account and API key
- Previous evaluation runs to compare against (optional)

## Step 1: Add OpenRouter API Key

### 1.1 Get OpenRouter API Key

1. Visit [openrouter.ai](https://openrouter.ai)
2. Sign up or log in
3. Navigate to "Keys" section
4. Generate a new API key

### 1.2 Update `.env` File

Add your OpenRouter API key to `.env`:

```bash
# Existing keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Add OpenRouter key
OPENROUTER_API_KEY=sk-or-v1-...

# Optional: Google Custom Search (if using direct web search)
GOOGLE_SEARCH_API_KEY=...
GOOGLE_SEARCH_ENGINE_ID=...
```

## Step 2: Update Provider Configuration

### Migration Path A: Add OpenRouter Alongside Existing Providers

This is the recommended approach for testing semantic markup impact. Keep your existing provider configs and add OpenRouter for comparison.

**Edit `config/providers.yaml`:**

```yaml
# Existing providers - keep as is
openai:
  enabled: true
  models:
    - name: gpt-4o-mini
      max_tokens: 1000
      temperature: 0.1
  rate_limit:
    requests_per_minute: 50
  retry:
    max_attempts: 3

anthropic:
  enabled: true
  models:
    - name: claude-3-5-haiku-20241022
      max_tokens: 1000
      temperature: 0.1
  rate_limit:
    requests_per_minute: 50
  retry:
    max_attempts: 3

# Add OpenRouter configuration
openrouter:
  enabled: true
  api_endpoint: "https://openrouter.ai/api/v1/chat/completions"

  # Use :online suffix for web search
  models:
    - "openai/gpt-4o-mini:online"
    - "anthropic/claude-3-5-haiku-20241022:online"

  max_tokens: 1000
  temperature: 0.1

  # Enable web search for content discovery
  web_search:
    enabled: true
    max_results: 5
    custom_search_prompt: "Search chancerylaneproject.org for legal clauses matching this query. Include exact clause names, maintenance status (schema:creativeWorkStatus), jurisdiction, practice area, and any semantic relationships (parent clauses via schema:hasPart, variants via schema:isBasedOn, related standards like SBTi or GHG Protocol)."

  rate_limit:
    requests_per_minute: 50
    tokens_per_minute: 100000

  retry:
    max_attempts: 3
    exponential_base: 2
    max_wait_seconds: 60
```

### Migration Path B: Migrate Completely to OpenRouter

If you want to simplify API key management and use only OpenRouter:

**Edit `config/providers.yaml`:**

```yaml
# Disable direct providers
openai:
  enabled: false

anthropic:
  enabled: false

gemini:
  enabled: false

# Configure OpenRouter with multiple models
openrouter:
  enabled: true
  api_endpoint: "https://openrouter.ai/api/v1/chat/completions"

  # Access multiple providers through OpenRouter
  models:
    - "openai/gpt-4o:online"
    - "anthropic/claude-opus-4-5-20251101:online"
    - "google/gemini-2.0-flash-exp:online"

  max_tokens: 1000
  temperature: 0.1

  web_search:
    enabled: true
    max_results: 5
    custom_search_prompt: "Search chancerylaneproject.org for legal clauses..."

  rate_limit:
    requests_per_minute: 50
    tokens_per_minute: 100000

  retry:
    max_attempts: 3
    exponential_base: 2
    max_wait_seconds: 60
```

### Migration Path C: Hybrid Approach

Use OpenRouter for web search tests, keep direct APIs for baseline/comparison:

**Strategy:**
- Direct APIs: Test with known/curated content (no web search)
- OpenRouter: Test content discovery via web search
- Compare: See how semantic markup affects discovery

**Configuration:** Use Path A above, then run evaluations selectively:

```bash
# Baseline: Direct API (no web search)
tclp-benchmark run --version-tag "baseline-direct" --providers openai anthropic

# Test: OpenRouter with web search
tclp-benchmark run --version-tag "test-openrouter" --providers openrouter

# Compare
tclp-benchmark report --version-a "baseline-direct" --version-b "test-openrouter"
```

## Step 3: Run Test Evaluation

Validate OpenRouter configuration with a small test:

```bash
# Run single category with OpenRouter
tclp-benchmark run \
  --version-tag "openrouter-test-$(date +%Y%m%d)" \
  --providers openrouter \
  --category direct_lookup \
  --verbose
```

Expected output:
```
Initialising evaluation run: openrouter-test-20250129
Loading configuration...
Loaded 2 test prompts for category: direct_lookup
Initialised providers: openrouter
Running evaluation...
  [1/2] Processing prompt: Find the net zero clause... ✓
  [2/2] Processing prompt: What is the biodiversity clause status... ✓
Evaluation complete!
```

If you see errors:
- **"API key not found"**: Ensure `OPENROUTER_API_KEY` is in `.env` and exported
- **"Model not available"**: Check model name on [openrouter.ai/models](https://openrouter.ai/models)
- **Rate limiting**: Reduce `requests_per_minute` in config

## Step 4: Compare Results

### Scenario 1: Before/After Semantic Markup

If you have evaluations from before adding JSON-LD:

```bash
# You ran this before adding semantic markup
# (Old run stored as "pre-jsonld")

# Run with OpenRouter after adding semantic markup
tclp-benchmark run --version-tag "post-jsonld-openrouter" --providers openrouter

# Compare
tclp-benchmark report \
  --version-a "pre-jsonld" \
  --version-b "post-jsonld-openrouter" \
  --format html \
  --output reports/semantic-markup-impact.html
```

### Scenario 2: Direct API vs OpenRouter Web Search

Compare how LLMs perform with/without web search:

```bash
# Run with direct API (no web search)
tclp-benchmark run --version-tag "baseline-openai-direct" --providers openai

# Run with OpenRouter (web search enabled)
tclp-benchmark run --version-tag "test-openrouter-online" --providers openrouter

# Compare
tclp-benchmark report \
  --version-a "baseline-openai-direct" \
  --version-b "test-openrouter-online" \
  --format html \
  --output reports/websearch-comparison.html
```

The report will highlight:
- Discovery improvements (did web search find TCLP content?)
- Citation accuracy (are clause names correct?)
- Metadata surfacing (jurisdiction, status, standards, etc.)

## Step 5: Optimise Configuration

Based on test results, fine-tune your OpenRouter configuration:

### Optimise Web Search Prompt

Adjust `custom_search_prompt` to target specific metadata:

```yaml
web_search:
  enabled: true
  max_results: 5

  # Basic prompt
  custom_search_prompt: "Search chancerylaneproject.org for clauses."

  # Metadata-focused prompt (better for semantic markup testing)
  custom_search_prompt: "Search chancerylaneproject.org for legal clauses with JSON-LD metadata. Include schema:creativeWorkStatus (maintenance status), schema:isPartOf (parent clauses), schema:jurisdiction, schema:industry (practice area), and references to standards like SBTi, GHG Protocol, or TCFD."

  # Use-case specific prompt
  custom_search_prompt: "Search chancerylaneproject.org for carbon reduction clauses suitable for UK commercial contracts. Include maintenance status, applicable standards, and related clauses."
```

### Adjust Model Selection

Choose models based on cost, performance, and availability:

```yaml
# Cost-optimised (development)
models:
  - "openai/gpt-4o-mini:online"
  - "anthropic/claude-3-5-haiku-20241022:online"

# Performance-optimised (formal benchmarks)
models:
  - "openai/gpt-4o:online"
  - "anthropic/claude-opus-4-5-20251101:online"
  - "google/gemini-2.0-flash-exp:online"

# Single model testing
models:
  - "anthropic/claude-opus-4-5-20251101:online"
```

### Configure Rate Limiting

Adjust based on OpenRouter plan and usage patterns:

```yaml
rate_limit:
  requests_per_minute: 50    # Conservative default
  tokens_per_minute: 100000  # Monitor in OpenRouter dashboard

# For faster development (if plan supports)
rate_limit:
  requests_per_minute: 100
  tokens_per_minute: 200000

# For strict budget control
rate_limit:
  requests_per_minute: 10
  tokens_per_minute: 50000
```

## Common Migration Issues

### Issue 1: Higher Costs

**Symptom:** OpenRouter costs more than expected

**Causes:**
- Using premium models (Claude Opus, GPT-4)
- `:online` models cost more due to web search
- Too many test runs

**Solutions:**
```yaml
# Use smaller models for development
models:
  - "openai/gpt-4o-mini:online"
  - "anthropic/claude-3-5-haiku-20241022:online"

# Reduce test runs during development
# Run full suite only for formal benchmarks
tclp-benchmark run --category direct_lookup --providers openrouter  # Small test
```

### Issue 2: Web Search Not Finding TCLP Content

**Symptom:** Metrics show low discovery/citation scores

**Causes:**
- Content not indexed by search engines
- Semantic markup not visible/correct
- Search prompt too generic

**Solutions:**

1. **Verify indexing:**
   ```bash
   # Check if Google has indexed your pages
   site:chancerylaneproject.org "net zero clause"
   ```

2. **Check JSON-LD visibility:**
   - Use [Google Rich Results Test](https://search.google.com/test/rich-results)
   - Verify structured data with browser DevTools

3. **Refine search prompt:**
   ```yaml
   custom_search_prompt: "Search chancerylaneproject.org specifically for maintained clauses (schema:creativeWorkStatus = 'Published'). Include full clause names, jurisdiction, and practice area."
   ```

### Issue 3: Rate Limiting

**Symptom:** Evaluation runs slowly or hits rate limits

**Solutions:**
```yaml
# Reduce rate limits
rate_limit:
  requests_per_minute: 20  # Down from 50

# Add delay between requests (for Gemini-style rate limiting)
rate_limit:
  requests_per_minute: 10
  delay_between_requests: 6  # 6 seconds = 10 req/min
```

### Issue 4: Model Unavailability

**Symptom:** "Model not available" errors

**Causes:**
- Model temporarily unavailable
- Model requires whitelisting
- Model name typo

**Solutions:**

1. **Check current availability:**
   - Visit [openrouter.ai/models](https://openrouter.ai/models)
   - Look for "Available" status

2. **Use fallback models:**
   ```yaml
   models:
     - "openai/gpt-4o:online"
     # Fallback if above unavailable
     - "openai/gpt-4o-mini:online"
   ```

3. **Verify model names:**
   - Use exact names from OpenRouter docs
   - Include `:online` suffix for web search

### Issue 5: Results Different from Direct APIs

**Symptom:** OpenRouter results differ from direct API calls

**This is expected!** Reasons:
- **Web search context:** `:online` models have additional web content
- **Model versions:** OpenRouter may use different model versions
- **System prompts:** OpenRouter adds web search instructions

**Not a bug - this is what we're testing!** The differences show:
- How web search affects discovery
- Impact of semantic markup on LLM responses
- Value of structured metadata

## Rollback Plan

If you need to rollback to direct APIs only:

1. **Disable OpenRouter in `config/providers.yaml`:**
   ```yaml
   openrouter:
     enabled: false
   ```

2. **Re-enable direct providers:**
   ```yaml
   openai:
     enabled: true
   anthropic:
     enabled: true
   gemini:
     enabled: true
   ```

3. **Run evaluation with direct providers:**
   ```bash
   tclp-benchmark run --version-tag "rollback-test" --providers openai anthropic
   ```

Your previous evaluation data is preserved in the database, so you can always compare historical runs.

## Best Practices

### 1. Phased Migration

```bash
# Week 1: Add OpenRouter, run parallel tests
tclp-benchmark run --version-tag "week1-direct" --providers openai
tclp-benchmark run --version-tag "week1-openrouter" --providers openrouter

# Week 2: Optimise OpenRouter config based on Week 1 results
# Update custom_search_prompt, adjust models

# Week 3: Expand to more providers via OpenRouter
# Add Anthropic, Google models through OpenRouter

# Week 4: Full comparison and decision
tclp-benchmark report --version-a "week1-direct" --version-b "week3-openrouter"
```

### 2. Version Tag Strategy

Use descriptive version tags that indicate provider and configuration:

```bash
# Good
--version-tag "pre-jsonld-openai-direct-20250129"
--version-tag "post-jsonld-openrouter-gpt4o-online-20250129"

# Less clear
--version-tag "test1"
--version-tag "v2"
```

### 3. Document Each Run

```bash
tclp-benchmark run \
  --version-tag "semantic-markup-test-v3" \
  --description "Testing JSON-LD with Schema.org hasPart relationships and SBTi standards. Using OpenRouter gpt-4o:online with custom search prompt v2." \
  --providers openrouter
```

### 4. Regular Comparison

Schedule regular comparisons to track improvements:

```bash
# Monthly comparison
tclp-benchmark report \
  --version-a "baseline-jan-2025" \
  --version-b "current-jan-2025" \
  --format html \
  --output reports/monthly-comparison-jan-2025.html
```

### 5. Cost Monitoring

- Check OpenRouter dashboard regularly
- Set up budget alerts in OpenRouter account
- Use smaller models for development, premium models for formal benchmarks

## Next Steps

After successful migration:

1. **Establish Baseline**
   - Run comprehensive evaluation with OpenRouter
   - Document current semantic markup state
   - Save as baseline for future comparisons

2. **Iterate on Semantic Markup**
   - Add/improve JSON-LD metadata
   - Run evaluation after each change
   - Compare against baseline

3. **Optimise Configuration**
   - Fine-tune `custom_search_prompt`
   - Adjust model selection based on results
   - Optimise rate limits and costs

4. **Automate Testing**
   - Set up CI/CD pipeline (see [OpenRouter Integration Guide](../openrouter-integration.md#integration-with-cicd))
   - Run benchmarks on semantic markup changes
   - Generate automatic comparison reports

## Support

- **OpenRouter Issues**: [OpenRouter Discord](https://discord.gg/openrouter)
- **TCLP Benchmark Issues**: GitHub Issues (if available)
- **General Questions**: Check [OpenRouter Integration Guide](../openrouter-integration.md)

## Changelog

Track changes to your OpenRouter configuration:

```markdown
## 2025-01-29
- Initial OpenRouter integration
- Configured gpt-4o-mini:online for testing
- Custom search prompt v1

## 2025-02-05
- Added Anthropic Claude models
- Updated custom search prompt to include schema:hasPart
- Increased max_results from 5 to 10

## 2025-02-12
- Optimised rate limits based on usage patterns
- Migrated from gpt-4o-mini to gpt-4o for formal benchmarks
```

## See Also

- [OpenRouter Integration Guide](../openrouter-integration.md) - Comprehensive OpenRouter documentation
- [CLAUDE.md](../../CLAUDE.md) - Project context and goals
- [README.md](../../README.md) - Quick start guide
