# OpenRouter Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace direct LLM API calls (Anthropic, OpenAI, Google) with OpenRouter API calls using web search plugin to inject chancerylaneproject.org content directly into test prompts.

**Architecture:** Create a new `OpenRouterProvider` that implements the `BaseLLMProvider` interface. Configure it to use OpenRouter's `:online` model variants or explicit web plugin configuration. Update configuration system to support OpenRouter settings alongside existing providers. The OpenRouter provider will handle web search automatically via OpenRouter's integrated search functionality, removing the need for separate search result formatting.

**Tech Stack:** OpenRouter API (OpenAI-compatible endpoint), Pydantic configuration models, async HTTP with existing provider patterns, tenacity retry logic.

---

## Task 1: Add OpenRouter Configuration Support

**Files:**
- Modify: `config/providers.yaml`
- Modify: `src/tclp_benchmark/config/models.py`
- Modify: `.env.example` (if exists, otherwise document in CLAUDE.md)

### Step 1: Write failing test for OpenRouter config model

**Create:** `tests/unit/config/test_openrouter_config.py`

```python
import pytest
from pydantic import ValidationError
from tclp_benchmark.config.models import OpenRouterConfig, WebSearchConfig


def test_openrouter_config_valid():
    """Test valid OpenRouter configuration"""
    config = OpenRouterConfig(
        enabled=True,
        api_endpoint="https://openrouter.ai/api/v1/chat/completions",
        models=["anthropic/claude-opus-4-5-20251101:online"],
        max_tokens=1000,
        temperature=0.1,
        web_search=WebSearchConfig(
            enabled=True,
            max_results=5,
            custom_search_prompt="Search chancerylaneproject.org for clauses."
        ),
        rate_limits={
            "requests_per_minute": 50,
            "tokens_per_minute": 100000
        },
        retry_config={
            "max_attempts": 3,
            "exponential_backoff_base": 2,
            "max_wait_seconds": 60
        }
    )

    assert config.enabled is True
    assert config.api_endpoint == "https://openrouter.ai/api/v1/chat/completions"
    assert len(config.models) == 1
    assert config.web_search.enabled is True
    assert config.web_search.max_results == 5


def test_openrouter_config_web_search_optional():
    """Test OpenRouter config with web search disabled"""
    config = OpenRouterConfig(
        enabled=True,
        api_endpoint="https://openrouter.ai/api/v1/chat/completions",
        models=["anthropic/claude-opus-4-5-20251101"],
        max_tokens=1000,
        temperature=0.1,
        web_search=WebSearchConfig(enabled=False)
    )

    assert config.web_search.enabled is False


def test_openrouter_config_invalid_temperature():
    """Test validation for invalid temperature"""
    with pytest.raises(ValidationError):
        OpenRouterConfig(
            enabled=True,
            api_endpoint="https://openrouter.ai/api/v1/chat/completions",
            models=["anthropic/claude-opus-4-5-20251101:online"],
            max_tokens=1000,
            temperature=3.0,  # Invalid: > 2.0
            web_search=WebSearchConfig(enabled=True)
        )


def test_web_search_config_default_max_results():
    """Test WebSearchConfig defaults"""
    config = WebSearchConfig(enabled=True)
    assert config.max_results == 5
    assert config.custom_search_prompt is None
```

### Step 2: Run test to verify it fails

```bash
pytest tests/unit/config/test_openrouter_config.py -v
```

Expected: FAIL - `ModuleNotFoundError: No module named 'tclp_benchmark.config.models'` or `ImportError: cannot import name 'OpenRouterConfig'`

### Step 3: Implement OpenRouter config models

**Modify:** `src/tclp_benchmark/config/models.py`

Add these classes before the `ProvidersConfig` class:

```python
class WebSearchConfig(BaseModel):
    """Configuration for OpenRouter web search plugin"""
    enabled: bool
    max_results: int = Field(default=5, ge=1, le=20)
    custom_search_prompt: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class OpenRouterConfig(BaseModel):
    """Configuration for OpenRouter LLM provider"""
    enabled: bool
    api_endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    models: List[str]
    max_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0.0, le=2.0)
    web_search: WebSearchConfig
    rate_limits: Dict[str, int] = Field(default_factory=dict)
    retry_config: Dict[str, int] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")
```

Then update the `ProvidersConfig` class to include OpenRouter:

```python
class ProvidersConfig(BaseModel):
    """Top-level providers configuration"""
    openai: OpenAIConfig
    anthropic: AnthropicConfig
    gemini: GeminiConfig
    openrouter: OpenRouterConfig  # Add this line

    model_config = ConfigDict(extra="forbid")
```

### Step 4: Run test to verify it passes

```bash
pytest tests/unit/config/test_openrouter_config.py -v
```

Expected: PASS (all tests green)

### Step 5: Add OpenRouter section to providers.yaml

**Modify:** `config/providers.yaml`

Add this section at the end of the file:

```yaml
openrouter:
  enabled: true
  api_endpoint: "https://openrouter.ai/api/v1/chat/completions"
  models:
    - "anthropic/claude-opus-4-5-20251101:online"
    - "openai/gpt-4o:online"
    - "google/gemini-2.0-flash-exp:online"
  max_tokens: 1000
  temperature: 0.1
  web_search:
    enabled: true
    max_results: 5
    custom_search_prompt: "Search chancerylaneproject.org for legal clauses matching this query. Include exact clause names, maintenance status (schema:creativeWorkStatus), jurisdiction, practice area, and any semantic relationships (parent clauses via schema:hasPart, variants via schema:isBasedOn, related standards like SBTi or GHG Protocol)."
  rate_limits:
    requests_per_minute: 50
    tokens_per_minute: 100000
  retry_config:
    max_attempts: 3
    exponential_backoff_base: 2
    max_wait_seconds: 60
```

### Step 6: Add OpenRouter API key to environment

**Modify:** `.env` (or create if doesn't exist)

Add:
```
OPENROUTER_API_KEY=your_key_here
```

**Update:** `CLAUDE.md` to document the new environment variable:

Add to the "Secrets" or environment variables section:
```markdown
- `OPENROUTER_API_KEY`: API key for OpenRouter unified API access
```

### Step 7: Test configuration loading

**Create:** `tests/integration/config/test_load_openrouter_config.py`

```python
from pathlib import Path
import pytest
from tclp_benchmark.config.loader import load_providers_config


def test_load_providers_config_with_openrouter():
    """Test loading providers.yaml with OpenRouter configuration"""
    config_path = Path("config/providers.yaml")

    if not config_path.exists():
        pytest.skip("providers.yaml not found")

    config = load_providers_config(config_path)

    # Verify OpenRouter config loaded
    assert hasattr(config, "openrouter")
    assert config.openrouter.enabled is True
    assert config.openrouter.api_endpoint == "https://openrouter.ai/api/v1/chat/completions"
    assert len(config.openrouter.models) >= 3
    assert config.openrouter.web_search.enabled is True
    assert config.openrouter.web_search.max_results == 5
    assert "chancerylaneproject.org" in config.openrouter.web_search.custom_search_prompt
```

### Step 8: Run integration test

```bash
pytest tests/integration/config/test_load_openrouter_config.py -v
```

Expected: PASS

### Step 9: Commit configuration changes

```bash
git add config/providers.yaml src/tclp_benchmark/config/models.py tests/unit/config/test_openrouter_config.py tests/integration/config/test_load_openrouter_config.py CLAUDE.md
git commit -m "feat: add OpenRouter configuration support with web search plugin"
```

---

## Task 2: Implement OpenRouter Provider

**Files:**
- Create: `src/tclp_benchmark/llm_providers/openrouter_provider.py`
- Create: `tests/unit/llm_providers/test_openrouter_provider.py`

### Step 1: Write failing test for OpenRouter provider initialization

**Create:** `tests/unit/llm_providers/test_openrouter_provider.py`

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tclp_benchmark.llm_providers.openrouter_provider import OpenRouterProvider
from tclp_benchmark.llm_providers.base import LLMRequest, LLMResponse


@pytest.fixture
def provider_config():
    """OpenRouter provider configuration fixture"""
    return {
        "api_endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "models": ["anthropic/claude-opus-4-5-20251101:online"],
        "max_tokens": 1000,
        "temperature": 0.1,
        "web_search": {
            "enabled": True,
            "max_results": 5,
            "custom_search_prompt": "Search chancerylaneproject.org"
        },
        "rate_limits": {
            "requests_per_minute": 50,
            "tokens_per_minute": 100000
        },
        "retry_config": {
            "max_attempts": 3,
            "exponential_backoff_base": 2,
            "max_wait_seconds": 60
        }
    }


@pytest.fixture
def api_key():
    return "sk-or-test-key-123"


def test_openrouter_provider_init(provider_config, api_key):
    """Test OpenRouter provider initialization"""
    provider = OpenRouterProvider(api_key=api_key, config=provider_config)

    assert provider.api_key == api_key
    assert provider.api_endpoint == "https://openrouter.ai/api/v1/chat/completions"
    assert provider.max_tokens == 1000
    assert provider.temperature == 0.1
    assert provider.web_search_enabled is True
    assert provider.web_search_max_results == 5


def test_openrouter_provider_name(provider_config, api_key):
    """Test provider name property"""
    provider = OpenRouterProvider(api_key=api_key, config=provider_config)
    assert provider.name == "openrouter"
```

### Step 2: Run test to verify it fails

```bash
pytest tests/unit/llm_providers/test_openrouter_provider.py -v
```

Expected: FAIL - `ModuleNotFoundError: No module named 'tclp_benchmark.llm_providers.openrouter_provider'`

### Step 3: Implement basic OpenRouter provider structure

**Create:** `src/tclp_benchmark/llm_providers/openrouter_provider.py`

```python
"""OpenRouter LLM Provider with web search integration"""

import logging
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .base import BaseLLMProvider, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter LLM provider with integrated web search"""

    def __init__(self, api_key: str, config: Dict[str, Any]):
        """
        Initialize OpenRouter provider

        Args:
            api_key: OpenRouter API key
            config: Provider configuration from providers.yaml
        """
        self.api_key = api_key
        self.api_endpoint = config.get(
            "api_endpoint",
            "https://openrouter.ai/api/v1/chat/completions"
        )
        self.max_tokens = config.get("max_tokens", 1000)
        self.temperature = config.get("temperature", 0.1)

        # Web search configuration
        web_search_config = config.get("web_search", {})
        self.web_search_enabled = web_search_config.get("enabled", False)
        self.web_search_max_results = web_search_config.get("max_results", 5)
        self.custom_search_prompt = web_search_config.get("custom_search_prompt")

        # Retry configuration
        retry_config = config.get("retry_config", {})
        self.max_attempts = retry_config.get("max_attempts", 3)
        self.backoff_base = retry_config.get("exponential_backoff_base", 2)
        self.max_wait = retry_config.get("max_wait_seconds", 60)

        # Initialize OpenAI-compatible client pointing to OpenRouter
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.api_endpoint.replace("/chat/completions", ""),
        )

        logger.info(
            f"Initialized OpenRouter provider (endpoint: {self.api_endpoint}, "
            f"web_search: {self.web_search_enabled})"
        )

    @property
    def name(self) -> str:
        """Provider name identifier"""
        return "openrouter"

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response using OpenRouter API

        Args:
            request: LLM request with prompt and configuration

        Returns:
            LLMResponse with text and metadata
        """
        # Implementation in next step
        pass
```

### Step 4: Run test to verify basic structure passes

```bash
pytest tests/unit/llm_providers/test_openrouter_provider.py::test_openrouter_provider_init -v
pytest tests/unit/llm_providers/test_openrouter_provider.py::test_openrouter_provider_name -v
```

Expected: PASS for init and name tests

### Step 5: Write failing test for response generation

Add to `tests/unit/llm_providers/test_openrouter_provider.py`:

```python
@pytest.mark.asyncio
async def test_generate_response_with_web_search(provider_config, api_key):
    """Test response generation with web search enabled"""
    provider = OpenRouterProvider(api_key=api_key, config=provider_config)

    request = LLMRequest(
        prompt="Find the net zero carbon commitment clause",
        model="anthropic/claude-opus-4-5-20251101:online",
        max_tokens=1000,
        temperature=0.1,
    )

    # Mock the OpenAI client response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Found Evie's Clause at chancerylaneproject.org"
    mock_response.usage.completion_tokens = 50
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.total_tokens = 150

    with patch.object(provider.client.chat.completions, 'create', new=AsyncMock(return_value=mock_response)):
        response = await provider.generate_response(request)

    assert response.text == "Found Evie's Clause at chancerylaneproject.org"
    assert response.model == "anthropic/claude-opus-4-5-20251101:online"
    assert response.error is None
    assert response.metadata["completion_tokens"] == 50
    assert response.metadata["prompt_tokens"] == 100


@pytest.mark.asyncio
async def test_generate_response_without_web_search(provider_config, api_key):
    """Test response generation with web search disabled"""
    # Disable web search
    provider_config["web_search"]["enabled"] = False
    provider = OpenRouterProvider(api_key=api_key, config=provider_config)

    request = LLMRequest(
        prompt="What is a carbon commitment clause?",
        model="anthropic/claude-opus-4-5-20251101",  # No :online suffix
        max_tokens=1000,
        temperature=0.1,
    )

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "A carbon commitment clause is..."
    mock_response.usage.completion_tokens = 30
    mock_response.usage.prompt_tokens = 20
    mock_response.usage.total_tokens = 50

    with patch.object(provider.client.chat.completions, 'create', new=AsyncMock(return_value=mock_response)):
        response = await provider.generate_response(request)

    assert response.text == "A carbon commitment clause is..."
    assert response.error is None


@pytest.mark.asyncio
async def test_generate_response_handles_api_error(provider_config, api_key):
    """Test error handling when API call fails"""
    provider = OpenRouterProvider(api_key=api_key, config=provider_config)

    request = LLMRequest(
        prompt="Test prompt",
        model="anthropic/claude-opus-4-5-20251101:online",
        max_tokens=1000,
        temperature=0.1,
    )

    with patch.object(
        provider.client.chat.completions,
        'create',
        new=AsyncMock(side_effect=Exception("API Error"))
    ):
        response = await provider.generate_response(request)

    assert response.text == ""
    assert "API Error" in response.error
```

### Step 6: Run test to verify it fails

```bash
pytest tests/unit/llm_providers/test_openrouter_provider.py::test_generate_response_with_web_search -v
```

Expected: FAIL - `generate_response` returns None or raises error

### Step 7: Implement generate_response method

**Modify:** `src/tclp_benchmark/llm_providers/openrouter_provider.py`

Replace the `generate_response` method stub with:

```python
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response using OpenRouter API

        Args:
            request: LLM request with prompt and configuration

        Returns:
            LLMResponse with text and metadata
        """
        import time
        start_time = time.time()

        try:
            # Build messages
            messages = [
                {"role": "user", "content": request.prompt}
            ]

            # Determine if we're using web search
            # Models with :online suffix automatically use web search
            # Or we can explicitly add web plugin to tools array
            use_online_suffix = ":online" in request.model

            # Build API call parameters
            api_params = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens or self.max_tokens,
                "temperature": request.temperature if request.temperature is not None else self.temperature,
            }

            # Add web search plugin if enabled and not using :online suffix
            if self.web_search_enabled and not use_online_suffix:
                tools = [
                    {
                        "type": "web",
                        "max_results": self.web_search_max_results,
                    }
                ]
                if self.custom_search_prompt:
                    tools[0]["custom_search_prompt"] = self.custom_search_prompt

                api_params["tools"] = tools

            logger.debug(
                f"Calling OpenRouter API with model={request.model}, "
                f"web_search={'online' if use_online_suffix else 'plugin' if self.web_search_enabled else 'disabled'}"
            )

            # Make API call
            response = await self.client.chat.completions.create(**api_params)

            # Extract response text
            response_text = response.choices[0].message.content

            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)

            # Build metadata
            metadata = {
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
                "web_search_enabled": self.web_search_enabled or use_online_suffix,
            }

            logger.info(
                f"OpenRouter response: model={request.model}, "
                f"tokens={metadata['total_tokens']}, time={execution_time_ms}ms"
            )

            return LLMResponse(
                text=response_text,
                model=request.model,
                metadata=metadata,
                execution_time_ms=execution_time_ms,
                error=None,
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"OpenRouter API error: {str(e)}")

            return LLMResponse(
                text="",
                model=request.model,
                metadata={},
                execution_time_ms=execution_time_ms,
                error=f"OpenRouter API error: {str(e)}",
            )
```

### Step 8: Run tests to verify implementation passes

```bash
pytest tests/unit/llm_providers/test_openrouter_provider.py -v
```

Expected: PASS (all tests green)

### Step 9: Commit OpenRouter provider implementation

```bash
git add src/tclp_benchmark/llm_providers/openrouter_provider.py tests/unit/llm_providers/test_openrouter_provider.py
git commit -m "feat: implement OpenRouter provider with web search support"
```

---

## Task 3: Integrate OpenRouter Provider into Orchestrator

**Files:**
- Modify: `src/tclp_benchmark/evaluators/orchestrator.py`
- Create: `tests/integration/evaluators/test_orchestrator_openrouter.py`

### Step 1: Write failing integration test

**Create:** `tests/integration/evaluators/test_orchestrator_openrouter.py`

```python
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from tclp_benchmark.evaluators.orchestrator import EvaluationOrchestrator
from tclp_benchmark.config.loader import load_providers_config, load_prompts_config
from tclp_benchmark.storage.database import DatabaseManager


@pytest.fixture
def db_manager(tmp_path):
    """Create temporary test database"""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    return DatabaseManager(db_url)


@pytest.fixture
def providers_config():
    """Load providers configuration"""
    return load_providers_config(Path("config/providers.yaml"))


@pytest.fixture
def prompts_config():
    """Load prompts configuration"""
    return load_prompts_config(Path("config/prompts.yaml"))


def test_orchestrator_recognizes_openrouter_provider(
    db_manager,
    providers_config,
    prompts_config
):
    """Test that orchestrator recognizes OpenRouter as a valid provider"""
    orchestrator = EvaluationOrchestrator(
        db_manager=db_manager,
        providers_config=providers_config,
        prompts_config=prompts_config,
        metrics_config=None,  # Simplified test
        verbose=False,
    )

    # Check that OpenRouter is in the provider map
    assert "openrouter" in orchestrator._provider_types

    # Verify OpenRouter provider class is registered
    from tclp_benchmark.llm_providers.openrouter_provider import OpenRouterProvider
    assert orchestrator._provider_types["openrouter"] == OpenRouterProvider


@pytest.mark.asyncio
async def test_orchestrator_creates_openrouter_provider_instance(
    db_manager,
    providers_config,
    prompts_config,
    monkeypatch
):
    """Test that orchestrator can instantiate OpenRouter provider"""
    # Mock environment variable
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")

    orchestrator = EvaluationOrchestrator(
        db_manager=db_manager,
        providers_config=providers_config,
        prompts_config=prompts_config,
        metrics_config=None,
        verbose=False,
    )

    # Get OpenRouter provider instance
    provider = orchestrator._get_provider("openrouter")

    assert provider is not None
    assert provider.name == "openrouter"
    assert provider.api_key == "test-key-123"
```

### Step 2: Run test to verify it fails

```bash
pytest tests/integration/evaluators/test_orchestrator_openrouter.py -v
```

Expected: FAIL - OpenRouter provider not registered in orchestrator

### Step 3: Register OpenRouter provider in orchestrator

**Modify:** `src/tclp_benchmark/evaluators/orchestrator.py`

Find the provider imports section (around line 10-15) and add:

```python
from ..llm_providers.openrouter_provider import OpenRouterProvider
```

Find the `_provider_types` mapping in the `__init__` method (around line 445-450) and update:

```python
        # Map provider names to their classes
        self._provider_types = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "gemini": GeminiProvider,
            "openrouter": OpenRouterProvider,  # Add this line
        }
```

Find the `api_key_map` dictionary (around line 470-475) and update:

```python
        api_key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",  # Add this line
        }
```

### Step 4: Run test to verify it passes

```bash
pytest tests/integration/evaluators/test_orchestrator_openrouter.py -v
```

Expected: PASS

### Step 5: Test provider filtering with OpenRouter

Add test to `tests/integration/evaluators/test_orchestrator_openrouter.py`:

```python
def test_orchestrator_can_filter_to_openrouter_only(
    db_manager,
    providers_config,
    prompts_config,
    monkeypatch
):
    """Test filtering evaluation to run OpenRouter only"""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")

    orchestrator = EvaluationOrchestrator(
        db_manager=db_manager,
        providers_config=providers_config,
        prompts_config=prompts_config,
        metrics_config=None,
        verbose=False,
        provider_filter=["openrouter"],  # Only OpenRouter
    )

    # Verify only OpenRouter is enabled
    enabled_providers = [
        name for name, config in [
            ("openai", providers_config.openai),
            ("anthropic", providers_config.anthropic),
            ("gemini", providers_config.gemini),
            ("openrouter", providers_config.openrouter),
        ]
        if config.enabled and (
            orchestrator.provider_filter is None
            or name in orchestrator.provider_filter
        )
    ]

    assert enabled_providers == ["openrouter"]
```

### Step 6: Run test

```bash
pytest tests/integration/evaluators/test_orchestrator_openrouter.py::test_orchestrator_can_filter_to_openrouter_only -v
```

Expected: PASS

### Step 7: Commit orchestrator integration

```bash
git add src/tclp_benchmark/evaluators/orchestrator.py tests/integration/evaluators/test_orchestrator_openrouter.py
git commit -m "feat: integrate OpenRouter provider into evaluation orchestrator"
```

---

## Task 4: Add OpenRouter-Specific CLI Support

**Files:**
- Modify: `src/tclp_benchmark/cli/commands.py`
- Create: `tests/integration/cli/test_openrouter_cli.py`

### Step 1: Write failing test for OpenRouter CLI usage

**Create:** `tests/integration/cli/test_openrouter_cli.py`

```python
import pytest
from click.testing import CliRunner
from pathlib import Path
from tclp_benchmark.cli.commands import cli


@pytest.fixture
def runner():
    """Click test runner"""
    return CliRunner()


def test_run_command_accepts_openrouter_provider(runner, monkeypatch, tmp_path):
    """Test run command with --providers openrouter"""
    # Set up environment
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")

    result = runner.invoke(
        cli,
        [
            "run",
            "--version-tag", "test-openrouter",
            "--providers", "openrouter",
            "--category", "status_checks",  # Use small category for speed
        ],
    )

    # Should not fail with "Invalid provider"
    assert "Invalid provider" not in result.output
    # OpenRouter should be recognized
    assert result.exit_code == 0 or "openrouter" in result.output.lower()


def test_run_command_requires_openrouter_api_key(runner, tmp_path):
    """Test that run command fails gracefully without OpenRouter API key"""
    result = runner.invoke(
        cli,
        [
            "run",
            "--version-tag", "test-no-key",
            "--providers", "openrouter",
        ],
        env={"DATABASE_URL": f"sqlite:///{tmp_path / 'test.db'}"},
    )

    # Should fail or warn about missing API key
    assert result.exit_code != 0 or "api key" in result.output.lower()
```

### Step 2: Run test to verify current behavior

```bash
pytest tests/integration/cli/test_openrouter_cli.py -v
```

Expected: May pass or fail depending on existing validation - document current state

### Step 3: Update CLI documentation and help text

**Modify:** `src/tclp_benchmark/cli/commands.py`

Find the `--providers` option in the `run` command (around line 20-30) and update help text:

```python
@click.option(
    "--providers",
    default=None,
    help="Comma-separated list of providers to run (e.g., 'openai,anthropic,openrouter'). "
         "Defaults to all enabled providers in config.",
)
```

### Step 4: Add validation for OpenRouter API key

**Modify:** `src/tclp_benchmark/cli/commands.py`

Find the API key validation section in the `run` command function (around line 50-70).

Add OpenRouter to the validation logic:

```python
    # Validate API keys for enabled providers
    required_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",  # Add this
    }

    # Parse provider filter
    provider_filter = None
    if providers:
        provider_filter = [p.strip() for p in providers.split(",")]

    # Check which providers are enabled and filtered
    enabled_providers = []
    for provider_name in ["openai", "anthropic", "gemini", "openrouter"]:  # Add openrouter
        provider_config = getattr(providers_config, provider_name, None)
        if provider_config and provider_config.enabled:
            if provider_filter is None or provider_name in provider_filter:
                enabled_providers.append(provider_name)

    # Validate API keys for enabled providers
    missing_keys = []
    for provider_name in enabled_providers:
        env_var = required_keys[provider_name]
        if not os.getenv(env_var):
            missing_keys.append(f"{provider_name} ({env_var})")

    if missing_keys:
        console.print(
            f"[red]Error:[/red] Missing API keys for: {', '.join(missing_keys)}"
        )
        raise click.Abort()
```

### Step 5: Run tests to verify CLI integration

```bash
pytest tests/integration/cli/test_openrouter_cli.py -v
```

Expected: PASS

### Step 6: Manual CLI test

```bash
# Test help text update
tclp-benchmark run --help | grep -A 2 "providers"

# Test with OpenRouter (with valid API key)
tclp-benchmark run \
  --version-tag "openrouter-test-$(date +%Y%m%d-%H%M%S)" \
  --providers openrouter \
  --category status_checks
```

Expected: Command executes successfully, shows OpenRouter in output

### Step 7: Commit CLI integration

```bash
git add src/tclp_benchmark/cli/commands.py tests/integration/cli/test_openrouter_cli.py
git commit -m "feat: add OpenRouter support to CLI with validation"
```

---

## Task 5: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Create: `docs/openrouter-integration.md`

### Step 1: Document OpenRouter in README

**Modify:** `README.md`

Find the "LLM Providers" or "Configuration" section and add:

```markdown
### OpenRouter Integration

OpenRouter provides unified API access to multiple LLM providers with integrated web search capabilities. This allows LLMs to fetch content from chancerylaneproject.org directly during evaluation.

**Configuration:**
```yaml
# config/providers.yaml
openrouter:
  enabled: true
  api_endpoint: "https://openrouter.ai/api/v1/chat/completions"
  models:
    - "anthropic/claude-opus-4-5-20251101:online"  # :online suffix enables web search
    - "openai/gpt-4o:online"
    - "google/gemini-2.0-flash-exp:online"
  web_search:
    enabled: true
    max_results: 5
    custom_search_prompt: "Search chancerylaneproject.org for clauses..."
```

**API Key Setup:**
```bash
export OPENROUTER_API_KEY="your-key-here"
```

**Usage:**
```bash
# Run evaluation with OpenRouter only
tclp-benchmark run \
  --version-tag "with-web-search" \
  --providers openrouter

# Compare OpenRouter vs direct API calls
tclp-benchmark run --version-tag "openrouter-comparison" --providers "openrouter,anthropic"
```

**Web Search Modes:**

1. **:online suffix** - Simplest approach, automatic web search
   - Model: `anthropic/claude-opus-4-5-20251101:online`
   - Web search automatically enabled by OpenRouter

2. **Explicit web plugin** - Fine-grained control
   - Model: `anthropic/claude-opus-4-5-20251101` (no suffix)
   - Set `web_search.enabled: true` in config
   - Customize search behavior with `custom_search_prompt`
```

### Step 2: Update CLAUDE.md with implementation notes

**Modify:** `CLAUDE.md`

Add new section after "Future Integration Points":

```markdown
## OpenRouter Integration (Implemented 2026-01-29)

We've integrated OpenRouter as a unified LLM provider with web search capabilities:

### What Changed
- **New provider**: `OpenRouterProvider` in `src/tclp_benchmark/llm_providers/openrouter_provider.py`
- **Configuration**: Added `openrouter` section to `config/providers.yaml`
- **Web search**: Models with `:online` suffix or explicit web plugin configuration
- **API access**: OpenAI-compatible endpoint at `https://openrouter.ai/api/v1/chat/completions`

### Why This Matters
Before: LLMs had to rely on web search APIs (Bing, Google) as separate steps
After: Web search results are injected directly into prompts by OpenRouter
Benefit: More accurate, contextual responses with fewer API calls

### Configuration Keys
- `OPENROUTER_API_KEY`: Required environment variable
- `web_search.enabled`: Enable/disable web search plugin
- `web_search.max_results`: Limit search results (default: 5)
- `web_search.custom_search_prompt`: Guide search to specific domains/content

### Model Naming Convention
- Standard: `anthropic/claude-opus-4-5-20251101` (no web search)
- Online: `anthropic/claude-opus-4-5-20251101:online` (automatic web search)
- Same pattern for OpenAI (`openai/gpt-4o:online`) and Google models

### Testing OpenRouter
```bash
# Quick test with single category
tclp-benchmark run --version-tag "openrouter-$(date +%Y%m%d)" --providers openrouter --category status_checks

# Compare with direct API
tclp-benchmark run --version-tag "direct-$(date +%Y%m%d)" --providers anthropic --category status_checks

# Generate comparison report
tclp-benchmark report --version-a "direct-20260129" --version-b "openrouter-20260129"
```

### Known Limitations
- OpenRouter rate limits may differ from direct API access
- Web search quality depends on OpenRouter's search implementation
- Additional latency due to web search step (typically 2-5 seconds)
- Costs may vary compared to direct API calls
```

### Step 3: Create detailed integration guide

**Create:** `docs/openrouter-integration.md`

```markdown
# OpenRouter Integration Guide

## Overview

OpenRouter provides unified access to multiple LLM providers (Anthropic, OpenAI, Google) with integrated web search capabilities. This integration allows our benchmarking system to test how LLMs discover and cite TCLP content when web search is embedded directly in the prompt context.

## Architecture

### Request Flow

```
User Prompt
    ↓
OpenRouter API
    ↓
    ├─→ Web Search (if :online or plugin enabled)
    │   ├─→ Search chancerylaneproject.org
    │   └─→ Extract relevant content
    ↓
LLM (Claude/GPT/Gemini)
    ├─→ User prompt
    ├─→ Web search results (injected as system message)
    └─→ Generate response
    ↓
Response with citations
```

### Provider Comparison

| Provider Type | Web Search | API Calls | Latency | Use Case |
|--------------|------------|-----------|---------|----------|
| Direct (Anthropic/OpenAI) | Separate | 2+ | Low | Baseline benchmark |
| OpenRouter :online | Integrated | 1 | Medium | Real-world usage |
| OpenRouter plugin | Customizable | 1 | Medium | Fine-tuned search |

## Configuration Deep Dive

### Basic Setup

```yaml
# config/providers.yaml
openrouter:
  enabled: true
  api_endpoint: "https://openrouter.ai/api/v1/chat/completions"
  models:
    - "anthropic/claude-opus-4-5-20251101:online"
  max_tokens: 1000
  temperature: 0.1
  web_search:
    enabled: true
    max_results: 5
  rate_limits:
    requests_per_minute: 50
    tokens_per_minute: 100000
  retry_config:
    max_attempts: 3
    exponential_backoff_base: 2
    max_wait_seconds: 60
```

### Web Search Configuration Options

#### 1. Using :online Suffix (Recommended for simplicity)

```yaml
models:
  - "anthropic/claude-opus-4-5-20251101:online"
```

**Pros:**
- Simplest configuration
- Automatic web search
- OpenRouter handles all details

**Cons:**
- Less control over search behavior
- Cannot customize search prompt

#### 2. Using Web Plugin (Recommended for control)

```yaml
models:
  - "anthropic/claude-opus-4-5-20251101"  # No :online suffix
web_search:
  enabled: true
  max_results: 5
  custom_search_prompt: >
    Search chancerylaneproject.org for legal clauses matching this query.
    Include exact clause names, maintenance status (schema:creativeWorkStatus),
    jurisdiction, practice area, and semantic relationships
    (parent clauses via schema:hasPart, variants via schema:isBasedOn,
    related standards like SBTi or GHG Protocol).
```

**Pros:**
- Fine-grained control
- Custom search instructions
- Can limit to specific domains
- Adjustable result count

**Cons:**
- More complex configuration
- Need to craft effective search prompts

### Custom Search Prompt Best Practices

Our custom search prompt guides OpenRouter to prioritize:

1. **Exact clause names** - For citation accuracy
2. **Metadata fields** - Status, jurisdiction, practice area
3. **Semantic relationships** - Parent/child, variants, standards
4. **Domain restriction** - chancerylaneproject.org only

Example:
```yaml
custom_search_prompt: >
  Search chancerylaneproject.org for legal clauses.
  Prioritize:
  - Exact clause names and titles
  - Maintenance status (draft/active/deprecated)
  - Jurisdiction and practice area
  - Parent clauses (schema:hasPart)
  - Variant clauses (schema:isBasedOn)
  - Related standards (SBTi, GHG Protocol, TCFD)
  Return structured information suitable for legal citation.
```

## Usage Patterns

### 1. OpenRouter-Only Evaluation

Test how well OpenRouter's web search performs:

```bash
tclp-benchmark run \
  --version-tag "openrouter-$(date +%Y%m%d)" \
  --providers openrouter
```

### 2. Cross-Provider Comparison

Compare OpenRouter against direct API access:

```bash
# Run with direct APIs (no web search)
tclp-benchmark run \
  --version-tag "direct-20260129" \
  --providers "anthropic,openai"

# Run with OpenRouter (with web search)
tclp-benchmark run \
  --version-tag "openrouter-20260129" \
  --providers openrouter

# Compare results
tclp-benchmark report \
  --version-a "direct-20260129" \
  --version-b "openrouter-20260129" \
  --format html \
  --output comparison.html
```

### 3. Multi-Model Testing

Test multiple models via OpenRouter:

```yaml
# config/providers.yaml
openrouter:
  models:
    - "anthropic/claude-opus-4-5-20251101:online"
    - "anthropic/claude-sonnet-4-5-20250129:online"
    - "openai/gpt-4o:online"
    - "google/gemini-2.0-flash-exp:online"
```

```bash
tclp-benchmark run \
  --version-tag "multi-model-$(date +%Y%m%d)" \
  --providers openrouter
```

## Implementation Details

### Provider Class Structure

```python
class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter with web search integration"""

    def __init__(self, api_key: str, config: Dict[str, Any]):
        # Initialize OpenAI-compatible client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        # Parse web search config
        self.web_search_enabled = config["web_search"]["enabled"]
        self.web_search_max_results = config["web_search"]["max_results"]
        self.custom_search_prompt = config["web_search"].get("custom_search_prompt")

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        # Detect :online suffix or add web plugin
        use_online = ":online" in request.model

        api_params = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        # Add explicit web plugin if not using :online
        if self.web_search_enabled and not use_online:
            api_params["tools"] = [{
                "type": "web",
                "max_results": self.web_search_max_results,
                "custom_search_prompt": self.custom_search_prompt,
            }]

        response = await self.client.chat.completions.create(**api_params)
        return self._format_response(response)
```

### Database Schema Changes

No database schema changes required - OpenRouter responses are stored using the same `LLMResponse` model:

- `provider_name`: "openrouter"
- `model_name`: Full model slug (e.g., "anthropic/claude-opus-4-5-20251101:online")
- `response_metadata`: Includes `web_search_enabled: true/false`

### Metric Evaluation

All existing metrics work unchanged:
- `TCLPCitationMetric` - Should score higher with web search
- `TaxonomyAwarenessMetric` - Should detect more taxonomy terms
- `EntityRetrievalAccuracyMetric` - Should identify specific clauses better
- `MetadataPrecisionMetric` - Should respect filters more accurately

## Troubleshooting

### Issue: "Missing API key for openrouter (OPENROUTER_API_KEY)"

**Solution:** Add API key to environment:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Or add to `.env` file:
```
OPENROUTER_API_KEY=sk-or-v1-...
```

### Issue: Web search not returning TCLP results

**Solution:** Verify custom search prompt mentions chancerylaneproject.org:
```yaml
web_search:
  custom_search_prompt: "Search chancerylaneproject.org for..."
```

### Issue: Rate limit errors

**Solution:** Adjust rate limits in config:
```yaml
rate_limits:
  requests_per_minute: 30  # Lower value
  tokens_per_minute: 50000
```

### Issue: High latency

**Cause:** Web search adds 2-5 seconds per request

**Solution:**
- Use :online suffix instead of explicit plugin (slightly faster)
- Reduce `max_results` to 3-5
- Run evaluations in parallel (future enhancement)

## Cost Considerations

OpenRouter pricing varies by model:
- Claude models: ~$15/1M input tokens, ~$75/1M output tokens
- GPT models: ~$5/1M input tokens, ~$15/1M output tokens
- Web search: May incur additional charges

**Cost optimization tips:**
1. Use smaller models (Claude Haiku, GPT-3.5) for initial testing
2. Filter to specific prompt categories (`--category status_checks`)
3. Reduce `max_tokens` to 500-750 for testing
4. Use provider filter (`--providers openrouter`) instead of running all providers

## Future Enhancements

### 1. Search Result Logging
Store web search results in database for analysis:
```python
# Future schema addition
class WebSearchResult(Base):
    id = Column(Integer, primary_key=True)
    llm_response_id = Column(Integer, ForeignKey("llm_responses.id"))
    url = Column(String)
    title = Column(String)
    snippet = Column(Text)
    rank = Column(Integer)
```

### 2. Domain-Specific Search Filters
Add support for restricting search to chancerylaneproject.org:
```yaml
web_search:
  domains: ["chancerylaneproject.org"]
  exclude_domains: []
```

### 3. Search Quality Metrics
New metric to evaluate search result relevance:
```python
class SearchQualityMetric(BaseMetric):
    """Measure how relevant web search results were"""
    def measure(self, test_case, response):
        # Check if search results contained expected clause names
        # Score based on result rank and relevance
        pass
```

### 4. A/B Testing Framework
Compare search configurations:
```bash
tclp-benchmark ab-test \
  --config-a config/openrouter-default.yaml \
  --config-b config/openrouter-custom-search.yaml \
  --category all
```

## References

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [OpenRouter Web Search Plugin](https://openrouter.ai/docs/web-search)
- [Model Pricing](https://openrouter.ai/models)
- [API Reference](https://openrouter.ai/docs/api-reference)
```

### Step 4: Commit documentation

```bash
git add README.md CLAUDE.md docs/openrouter-integration.md
git commit -m "docs: add OpenRouter integration guide and configuration examples"
```

---

## Task 6: Integration Testing and Validation

**Files:**
- Create: `tests/integration/end_to_end/test_openrouter_e2e.py`
- Create: `scripts/validate_openrouter.sh`

### Step 1: Create end-to-end integration test

**Create:** `tests/integration/end_to_end/test_openrouter_e2e.py`

```python
"""End-to-end integration test for OpenRouter"""

import pytest
import os
from pathlib import Path
from tclp_benchmark.config.loader import (
    load_providers_config,
    load_prompts_config,
)
from tclp_benchmark.storage.database import DatabaseManager
from tclp_benchmark.evaluators.orchestrator import EvaluationOrchestrator


@pytest.fixture
def test_db(tmp_path):
    """Create test database"""
    db_path = tmp_path / "e2e_test.db"
    db_url = f"sqlite:///{db_path}"
    return DatabaseManager(db_url)


@pytest.fixture
def skip_if_no_api_key():
    """Skip test if OpenRouter API key not available"""
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set - skipping live API test")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openrouter_live_api_call(test_db, skip_if_no_api_key):
    """
    Test live API call to OpenRouter with web search

    This test requires:
    - Valid OPENROUTER_API_KEY environment variable
    - Internet connection
    - OpenRouter service availability
    """
    # Load configurations
    providers_config = load_providers_config(Path("config/providers.yaml"))
    prompts_config = load_prompts_config(Path("config/prompts.yaml"))

    # Ensure OpenRouter is enabled
    assert providers_config.openrouter.enabled, "OpenRouter must be enabled in config"

    # Create orchestrator with OpenRouter only
    orchestrator = EvaluationOrchestrator(
        db_manager=test_db,
        providers_config=providers_config,
        prompts_config=prompts_config,
        metrics_config=None,  # Skip metrics for speed
        verbose=True,
        provider_filter=["openrouter"],
        category_filter=["status_checks"],  # Use smallest category
    )

    # Run evaluation
    version_tag = "e2e-test-openrouter"
    await orchestrator.run_evaluation(version_tag)

    # Verify results were stored
    with test_db.get_session() as session:
        from tclp_benchmark.storage.models import EvaluationRun, TestCase, LLMResponse

        # Get evaluation run
        run = session.query(EvaluationRun).filter_by(
            version_tag=version_tag
        ).first()

        assert run is not None, "Evaluation run should be created"
        assert run.status == "completed", "Evaluation should complete successfully"

        # Get test cases
        test_cases = session.query(TestCase).filter_by(
            evaluation_run_id=run.id
        ).all()

        assert len(test_cases) > 0, "Should have test cases"

        # Get LLM responses
        responses = session.query(LLMResponse).join(TestCase).filter(
            TestCase.evaluation_run_id == run.id,
            LLMResponse.provider_name == "openrouter"
        ).all()

        assert len(responses) > 0, "Should have OpenRouter responses"

        # Verify response quality
        for response in responses:
            assert response.response_text, "Response should have text"
            assert response.error is None, f"Response should not have error: {response.error}"
            assert response.metadata.get("web_search_enabled") is True, \
                "Web search should be enabled"
            assert response.metadata.get("total_tokens", 0) > 0, \
                "Should have token usage"

        print(f"\n✅ Successfully tested OpenRouter with {len(responses)} responses")
        print(f"   Models tested: {set(r.model_name for r in responses)}")
        print(f"   Average tokens: {sum(r.metadata.get('total_tokens', 0) for r in responses) / len(responses):.0f}")


@pytest.mark.integration
def test_openrouter_config_validation():
    """Test that OpenRouter configuration validates correctly"""
    providers_config = load_providers_config(Path("config/providers.yaml"))

    # Verify OpenRouter section exists and is valid
    assert hasattr(providers_config, "openrouter")
    or_config = providers_config.openrouter

    # Check required fields
    assert or_config.api_endpoint, "API endpoint must be set"
    assert "openrouter.ai" in or_config.api_endpoint, "Should use OpenRouter endpoint"
    assert len(or_config.models) > 0, "Must have at least one model"
    assert or_config.max_tokens > 0, "Max tokens must be positive"
    assert 0 <= or_config.temperature <= 2, "Temperature must be 0-2"

    # Check web search config
    assert hasattr(or_config, "web_search")
    assert or_config.web_search.max_results > 0, "Max results must be positive"

    # If web search enabled, check custom prompt
    if or_config.web_search.enabled:
        assert or_config.web_search.custom_search_prompt, \
            "Custom search prompt recommended when web search enabled"
        assert "chancerylaneproject.org" in or_config.web_search.custom_search_prompt, \
            "Search prompt should mention target domain"

    print(f"\n✅ OpenRouter config validation passed")
    print(f"   Models: {or_config.models}")
    print(f"   Web search: {'enabled' if or_config.web_search.enabled else 'disabled'}")
```

### Step 2: Create validation script

**Create:** `scripts/validate_openrouter.sh`

```bash
#!/usr/bin/env bash
#
# Validate OpenRouter integration
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "🔍 Validating OpenRouter Integration..."
echo

# 1. Check configuration files
echo "1. Checking configuration files..."
if ! grep -q "openrouter:" config/providers.yaml; then
    echo "❌ OpenRouter section missing from config/providers.yaml"
    exit 1
fi
echo "✅ providers.yaml has openrouter section"

# 2. Check environment variable
echo
echo "2. Checking environment variable..."
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "⚠️  OPENROUTER_API_KEY not set (optional for config validation)"
else
    echo "✅ OPENROUTER_API_KEY is set"
fi

# 3. Run unit tests
echo
echo "3. Running unit tests..."
if pytest tests/unit/config/test_openrouter_config.py -v; then
    echo "✅ Unit tests passed"
else
    echo "❌ Unit tests failed"
    exit 1
fi

# 4. Run integration tests (config only)
echo
echo "4. Running integration tests (config validation)..."
if pytest tests/integration/end_to_end/test_openrouter_e2e.py::test_openrouter_config_validation -v; then
    echo "✅ Integration tests passed"
else
    echo "❌ Integration tests failed"
    exit 1
fi

# 5. Test CLI help text
echo
echo "5. Testing CLI..."
if tclp-benchmark run --help | grep -q "openrouter"; then
    echo "✅ CLI recognizes OpenRouter"
else
    echo "⚠️  OpenRouter not mentioned in CLI help (may be expected)"
fi

# 6. Optional: Live API test
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
    echo
    echo "6. Running live API test (optional)..."
    echo "   This will make a real API call to OpenRouter"
    read -p "   Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if pytest tests/integration/end_to_end/test_openrouter_e2e.py::test_openrouter_live_api_call -v -s; then
            echo "✅ Live API test passed"
        else
            echo "❌ Live API test failed"
            exit 1
        fi
    else
        echo "⏭️  Skipped live API test"
    fi
else
    echo
    echo "6. Skipping live API test (no API key)"
fi

echo
echo "✅ All validation checks passed!"
echo
echo "Next steps:"
echo "  1. Set OPENROUTER_API_KEY environment variable"
echo "  2. Run a test evaluation:"
echo "     tclp-benchmark run --version-tag 'openrouter-test' --providers openrouter --category status_checks"
echo "  3. Generate a comparison report if you have baseline data"
echo
```

### Step 3: Make validation script executable

```bash
chmod +x scripts/validate_openrouter.sh
```

### Step 4: Run validation script (without live API test)

```bash
./scripts/validate_openrouter.sh
```

Expected: All checks pass except live API test (if API key not set)

### Step 5: Run config validation tests

```bash
pytest tests/integration/end_to_end/test_openrouter_e2e.py::test_openrouter_config_validation -v
```

Expected: PASS

### Step 6: Optional - Run live API test (if API key available)

```bash
# Only if OPENROUTER_API_KEY is set
pytest tests/integration/end_to_end/test_openrouter_e2e.py::test_openrouter_live_api_call -v -s
```

Expected: PASS (if API key valid and service available)

### Step 7: Commit integration tests and validation

```bash
git add tests/integration/end_to_end/test_openrouter_e2e.py scripts/validate_openrouter.sh
git commit -m "test: add OpenRouter integration tests and validation script"
```

---

## Task 7: Update Reporting for OpenRouter Comparison

**Files:**
- Modify: `src/tclp_benchmark/reporting/report_generator.py`
- Create: `tests/unit/reporting/test_openrouter_report.py`

### Step 1: Write test for OpenRouter in reports

**Create:** `tests/unit/reporting/test_openrouter_report.py`

```python
import pytest
from datetime import datetime
from tclp_benchmark.storage.database import DatabaseManager
from tclp_benchmark.storage.models import (
    EvaluationRun,
    TestCase,
    LLMResponse,
    MetricScore,
)
from tclp_benchmark.reporting.report_generator import ReportGenerator


@pytest.fixture
def test_db(tmp_path):
    """Create test database with OpenRouter data"""
    db_path = tmp_path / "test_report.db"
    db_url = f"sqlite:///{db_path}"
    db_manager = DatabaseManager(db_url)

    with db_manager.get_session() as session:
        # Create two evaluation runs for comparison
        run_direct = EvaluationRun(
            version_tag="direct-api",
            status="completed",
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )
        run_openrouter = EvaluationRun(
            version_tag="openrouter",
            status="completed",
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )
        session.add_all([run_direct, run_openrouter])
        session.flush()

        # Create test case
        test_case = TestCase(
            evaluation_run_id=run_direct.id,
            prompt_category="status_checks",
            prompt_text="Find clauses with maintenance status",
            expected_outcome={"tclp_mention": True},
        )
        session.add(test_case)
        session.flush()

        # Create LLM responses
        response_anthropic = LLMResponse(
            test_case_id=test_case.id,
            provider_name="anthropic",
            model_name="claude-3-7-sonnet-20250219",
            response_text="TCLP has clauses...",
            response_metadata={"total_tokens": 100},
            execution_time_ms=1500,
        )

        response_openrouter = LLMResponse(
            test_case_id=test_case.id,
            provider_name="openrouter",
            model_name="anthropic/claude-opus-4-5-20251101:online",
            response_text="According to chancerylaneproject.org, clauses include...",
            response_metadata={
                "total_tokens": 150,
                "web_search_enabled": True,
            },
            execution_time_ms=3200,
        )

        session.add_all([response_anthropic, response_openrouter])
        session.flush()

        # Add metric scores
        metric_direct = MetricScore(
            llm_response_id=response_anthropic.id,
            metric_name="TCLPCitationMetric",
            metric_category="custom",
            score=1.0,
            threshold=1.0,
            passed=True,
        )

        metric_openrouter = MetricScore(
            llm_response_id=response_openrouter.id,
            metric_name="TCLPCitationMetric",
            metric_category="custom",
            score=1.0,
            threshold=1.0,
            passed=True,
        )

        session.add_all([metric_direct, metric_openrouter])
        session.commit()

    return db_manager


def test_report_includes_openrouter_provider(test_db):
    """Test that OpenRouter appears in comparison reports"""
    generator = ReportGenerator(test_db)

    report_data = generator.generate_comparison_report(
        version_a="direct-api",
        version_b="openrouter",
    )

    # Check that OpenRouter is in provider list
    providers = set()
    for provider_results in report_data.get("providers", {}).values():
        providers.update(provider_results.keys())

    assert "openrouter" in providers, "OpenRouter should appear in report"


def test_report_highlights_web_search(test_db):
    """Test that report indicates web search capability"""
    generator = ReportGenerator(test_db)

    report_md = generator.generate_markdown_report(
        version_a="direct-api",
        version_b="openrouter",
    )

    # Report should mention web search
    assert "web" in report_md.lower() or "search" in report_md.lower(), \
        "Report should mention web search"
```

### Step 2: Run test to verify current behavior

```bash
pytest tests/unit/reporting/test_openrouter_report.py -v
```

Expected: May pass or fail depending on current report generator implementation

### Step 3: Update report generator to highlight OpenRouter

**Modify:** `src/tclp_benchmark/reporting/report_generator.py`

Find the provider comparison section (likely in `generate_markdown_report` method) and add:

```python
    def _format_provider_name(self, provider_name: str, model_name: str = None) -> str:
        """Format provider name with additional context"""
        display_name = provider_name.title()

        # Add web search indicator for OpenRouter
        if provider_name == "openrouter":
            if model_name and ":online" in model_name:
                display_name += " (web search)"
            else:
                display_name += " (OpenRouter)"

        return display_name
```

Then use this method when displaying provider names in reports:

```python
    # In generate_markdown_report method, when iterating providers:
    for provider_name, metrics in provider_data.items():
        display_name = self._format_provider_name(
            provider_name,
            model_name=metrics.get("model_name")
        )
        report_lines.append(f"### {display_name}\n")
        # ... rest of provider section
```

### Step 4: Add web search metadata to report summary

**Modify:** `src/tclp_benchmark/reporting/report_generator.py`

In the summary section, add web search statistics:

```python
    def _generate_summary_stats(self, run_a, run_b):
        """Generate summary statistics including web search info"""
        stats = {
            "version_a": run_a.version_tag,
            "version_b": run_b.version_tag,
            "providers": {},
        }

        # Count providers and track web search usage
        with self.db_manager.get_session() as session:
            # ... existing provider counting logic ...

            # Check for OpenRouter with web search
            openrouter_responses = session.query(LLMResponse).join(TestCase).filter(
                TestCase.evaluation_run_id.in_([run_a.id, run_b.id]),
                LLMResponse.provider_name == "openrouter"
            ).all()

            web_search_enabled = any(
                r.response_metadata.get("web_search_enabled", False)
                for r in openrouter_responses
            )

            if web_search_enabled:
                stats["web_search_enabled"] = True
                stats["web_search_provider"] = "openrouter"

        return stats
```

### Step 5: Run tests to verify report updates

```bash
pytest tests/unit/reporting/test_openrouter_report.py -v
```

Expected: PASS

### Step 6: Generate test report manually

```bash
# Assuming you have test data with OpenRouter results
tclp-benchmark report \
  --version-a "baseline-test" \
  --version-b "openrouter-test" \
  --format markdown \
  --output /tmp/openrouter-comparison.md

cat /tmp/openrouter-comparison.md
```

Expected: Report shows OpenRouter with web search indicator

### Step 7: Commit reporting updates

```bash
git add src/tclp_benchmark/reporting/report_generator.py tests/unit/reporting/test_openrouter_report.py
git commit -m "feat: enhance reporting to highlight OpenRouter web search capabilities"
```

---

## Task 8: Migration Guide for Existing Users

**Files:**
- Create: `docs/migration-guides/openrouter-migration.md`
- Modify: `CHANGELOG.md` (if exists, otherwise skip)

### Step 1: Create migration guide

**Create:** `docs/migration-guides/openrouter-migration.md`

```markdown
# Migrating to OpenRouter

This guide helps you migrate your existing TCLP benchmark workflows to use OpenRouter with web search capabilities.

## Why Migrate?

**Before (Direct API):**
```
Prompt → LLM → Response
```
- No automatic access to chancerylaneproject.org content
- LLM relies on training data (may be outdated)
- Separate web search step required for context

**After (OpenRouter):**
```
Prompt → Web Search → LLM with Context → Response
```
- Automatic web search of chancerylaneproject.org
- Fresh content from live website
- Single API call, integrated flow
- Better citation accuracy

## Step-by-Step Migration

### 1. Install/Update Dependencies

No new dependencies required - OpenRouter uses OpenAI-compatible API:

```bash
# Verify existing dependencies
pip list | grep openai
# Should show: openai>=1.0.0
```

### 2. Get OpenRouter API Key

1. Visit [https://openrouter.ai](https://openrouter.ai)
2. Sign up or log in
3. Go to API Keys section
4. Create new key
5. Add to environment:

```bash
# Add to .env
echo "OPENROUTER_API_KEY=sk-or-v1-..." >> .env

# Or export in shell
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### 3. Verify Configuration

Check that `config/providers.yaml` has OpenRouter section:

```bash
grep -A 10 "openrouter:" config/providers.yaml
```

Should show:
```yaml
openrouter:
  enabled: true
  api_endpoint: "https://openrouter.ai/api/v1/chat/completions"
  models:
    - "anthropic/claude-opus-4-5-20251101:online"
  web_search:
    enabled: true
    max_results: 5
```

If missing, update from latest version or add manually.

### 4. Run Validation

```bash
./scripts/validate_openrouter.sh
```

All checks should pass except live API test (until you add API key).

### 5. Run Test Evaluation

Run a small test to verify everything works:

```bash
tclp-benchmark run \
  --version-tag "openrouter-test-$(date +%Y%m%d)" \
  --providers openrouter \
  --category status_checks
```

Expected output:
```
✅ Evaluation completed
📊 Results: X test cases, Y responses
```

### 6. Compare with Baseline

If you have existing baseline data:

```bash
# Run OpenRouter version
tclp-benchmark run \
  --version-tag "with-openrouter" \
  --providers openrouter

# Compare with previous run
tclp-benchmark report \
  --version-a "your-baseline-tag" \
  --version-b "with-openrouter" \
  --format html \
  --output comparison.html
```

## Configuration Options

### Option 1: Use :online Suffix (Simplest)

Best for: Quick setup, minimal configuration

```yaml
openrouter:
  enabled: true
  models:
    - "anthropic/claude-opus-4-5-20251101:online"
  web_search:
    enabled: false  # Handled by :online suffix
```

### Option 2: Explicit Web Plugin (Recommended)

Best for: Fine-tuned control, custom search prompts

```yaml
openrouter:
  enabled: true
  models:
    - "anthropic/claude-opus-4-5-20251101"  # No :online
  web_search:
    enabled: true
    max_results: 5
    custom_search_prompt: "Search chancerylaneproject.org for legal clauses..."
```

### Option 3: Mixed Approach

Test both for comparison:

```yaml
openrouter:
  enabled: true
  models:
    - "anthropic/claude-opus-4-5-20251101:online"
    - "anthropic/claude-opus-4-5-20251101"  # Same model, no web search
  web_search:
    enabled: true  # Applies to non-:online models
```

## Common Migration Patterns

### Pattern 1: Replace Direct Anthropic with OpenRouter

**Before:**
```bash
tclp-benchmark run \
  --version-tag "baseline" \
  --providers anthropic
```

**After:**
```bash
tclp-benchmark run \
  --version-tag "with-web-search" \
  --providers openrouter
```

### Pattern 2: Side-by-Side Comparison

Run both for comparison:

```bash
# Direct API (no web search)
tclp-benchmark run \
  --version-tag "direct-$(date +%Y%m%d)" \
  --providers "anthropic,openai"

# OpenRouter (with web search)
tclp-benchmark run \
  --version-tag "openrouter-$(date +%Y%m%d)" \
  --providers openrouter

# Compare
tclp-benchmark report \
  --version-a "direct-20260129" \
  --version-b "openrouter-20260129"
```

### Pattern 3: Gradual Rollout

Test on subset before full migration:

```bash
# Step 1: Test on status checks only
tclp-benchmark run \
  --version-tag "openrouter-pilot" \
  --providers openrouter \
  --category status_checks

# Step 2: Expand to more categories
tclp-benchmark run \
  --version-tag "openrouter-expanded" \
  --providers openrouter \
  --category "status_checks,multi_dimensional"

# Step 3: Full evaluation
tclp-benchmark run \
  --version-tag "openrouter-full" \
  --providers openrouter
```

## Cost Impact

### Direct API Pricing (Example)
- Claude Opus 4.5: $15/1M input, $75/1M output
- GPT-4o: $5/1M input, $15/1M output

### OpenRouter Pricing
- Same base rates as direct API
- Possible small markup (check openrouter.ai/models)
- Web search may incur additional charges

**Cost optimization:**
```yaml
# Reduce tokens for testing
openrouter:
  max_tokens: 500  # Down from 1000

# Limit search results
web_search:
  max_results: 3  # Down from 5
```

## Performance Impact

### Latency Changes

| Metric | Direct API | OpenRouter :online |
|--------|------------|-------------------|
| Typical response | 1-3 seconds | 3-7 seconds |
| Additional latency | - | +2-5 seconds (web search) |
| Variance | Low | Medium (depends on search) |

**Mitigation strategies:**
1. Run evaluations asynchronously (already implemented)
2. Use smaller prompt categories during development
3. Accept higher latency for better accuracy trade-off

## Troubleshooting Migration Issues

### Issue: "Invalid model name"

**Symptom:** Error about model not found

**Solution:** Verify model slug format:
- ✅ Correct: `anthropic/claude-opus-4-5-20251101:online`
- ❌ Wrong: `claude-opus-4-5-20251101:online` (missing provider prefix)

### Issue: Web search not working

**Symptom:** Responses don't include TCLP citations

**Diagnostic:**
```bash
# Check response metadata
sqlite3 data/tclp_benchmark.db \
  "SELECT model_name, response_metadata FROM llm_responses WHERE provider_name='openrouter' LIMIT 1"
```

Should show: `"web_search_enabled": true`

**Solutions:**
1. Verify :online suffix in model name
2. Check `web_search.enabled: true` in config
3. Add custom_search_prompt mentioning chancerylaneproject.org

### Issue: Rate limiting errors

**Symptom:** 429 errors, "rate limit exceeded"

**Solution:** Adjust rate limits:
```yaml
openrouter:
  rate_limits:
    requests_per_minute: 30  # Reduced from 50
    tokens_per_minute: 50000  # Reduced from 100000
  retry_config:
    max_attempts: 5  # Increased from 3
    max_wait_seconds: 120  # Increased from 60
```

### Issue: Higher costs than expected

**Symptom:** OpenRouter costs exceed direct API

**Explanation:**
- Web search adds token usage (search results injected into prompt)
- Typical increase: 200-500 tokens per request

**Solution:**
```yaml
# Reduce max_tokens to compensate
openrouter:
  max_tokens: 750  # Reduced to offset search token cost

# Limit search results
web_search:
  max_results: 3
```

## Rollback Plan

If you need to revert to direct APIs:

### 1. Disable OpenRouter

```yaml
# config/providers.yaml
openrouter:
  enabled: false  # Disable OpenRouter

# Re-enable direct providers
anthropic:
  enabled: true
openai:
  enabled: true
```

### 2. Continue Using Existing Data

All previous evaluation runs remain intact:

```bash
# List all runs
sqlite3 data/tclp_benchmark.db "SELECT version_tag FROM evaluation_runs ORDER BY created_at DESC"

# Compare old runs (no code changes needed)
tclp-benchmark report --version-a "old-run" --version-b "another-old-run"
```

### 3. Restore Previous Workflow

```bash
# Use original command format
tclp-benchmark run \
  --version-tag "back-to-baseline" \
  --providers "anthropic,openai"
```

## Best Practices Post-Migration

### 1. Tag Evaluation Runs Clearly

```bash
# Good tags
--version-tag "openrouter-web-search-2026-01-29"
--version-tag "or-v1-status-checks"

# Poor tags
--version-tag "test"
--version-tag "run123"
```

### 2. Document Configuration Changes

Keep notes on what you're testing:

```bash
# Add to git commit message
git commit -m "config: increase OpenRouter search results to 7 for better coverage"
```

### 3. Run Comparisons Regularly

```bash
# Weekly comparison
tclp-benchmark run --version-tag "weekly-$(date +%Y-W%V)"
```

### 4. Monitor Costs

Track token usage in reports:

```bash
sqlite3 data/tclp_benchmark.db \
  "SELECT
    provider_name,
    AVG(json_extract(response_metadata, '$.total_tokens')) as avg_tokens
   FROM llm_responses
   GROUP BY provider_name"
```

## Support

If you encounter issues during migration:

1. Check validation script: `./scripts/validate_openrouter.sh`
2. Review logs: `tail -f logs/tclp_benchmark.log` (if logging enabled)
3. Open issue: [GitHub Issues](https://github.com/your-org/llm_eval/issues)
4. Check OpenRouter status: [https://status.openrouter.ai](https://status.openrouter.ai)

## Next Steps

After successful migration:

1. ✅ Run baseline comparison
2. ✅ Document results for stakeholders
3. ✅ Adjust web search prompts based on results
4. ✅ Explore multi-model testing via OpenRouter
5. ✅ Integrate into CI/CD (if applicable)
```

### Step 2: Update CHANGELOG (if exists)

**Create or modify:** `CHANGELOG.md`

```markdown
# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **OpenRouter Integration** (2026-01-29)
  - New `OpenRouterProvider` for unified LLM access with web search
  - Web search plugin configuration with customizable search prompts
  - Support for `:online` model suffix for automatic web search
  - OpenRouter-specific configuration in `config/providers.yaml`
  - Environment variable: `OPENROUTER_API_KEY`
  - Integration tests and validation script
  - Comprehensive documentation in `docs/openrouter-integration.md`
  - Migration guide in `docs/migration-guides/openrouter-migration.md`

### Changed
- Orchestrator now supports `openrouter` as a provider option
- CLI accepts `--providers openrouter` for filtering
- Report generator highlights web search capabilities
- Configuration models extended with `OpenRouterConfig` and `WebSearchConfig`

### Documentation
- Added OpenRouter section to README
- Updated CLAUDE.md with implementation notes
- Created detailed integration guide
- Added migration guide for existing users

## Previous Versions
[Previous changelog entries...]
```

### Step 3: Commit migration guide

```bash
git add docs/migration-guides/openrouter-migration.md CHANGELOG.md
git commit -m "docs: add OpenRouter migration guide and changelog"
```

---

## Task 9: Final Verification and Documentation Review

**Files:**
- Review all modified files
- Run full test suite
- Generate final validation report

### Step 1: Run complete test suite

```bash
# Run all tests (unit + integration)
pytest tests/ -v --tb=short

# Generate coverage report (if pytest-cov installed)
pytest tests/ --cov=src/tclp_benchmark --cov-report=html --cov-report=term
```

Expected: All tests pass with good coverage

### Step 2: Validate all configuration files

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/providers.yaml'))"

# Validate with Pydantic models
python -c "
from pathlib import Path
from tclp_benchmark.config.loader import load_providers_config
config = load_providers_config(Path('config/providers.yaml'))
print(f'✅ Configuration valid')
print(f'   OpenRouter enabled: {config.openrouter.enabled}')
print(f'   Models: {config.openrouter.models}')
print(f'   Web search: {config.openrouter.web_search.enabled}')
"
```

Expected: Configuration valid, no errors

### Step 3: Run full validation script

```bash
./scripts/validate_openrouter.sh
```

Expected: All validation checks pass

### Step 4: Test CLI commands

```bash
# Test help text
tclp-benchmark run --help | grep -A 5 "providers"

# Test invalid provider handling
tclp-benchmark run --version-tag "test" --providers invalid-provider || echo "Expected failure: invalid provider"

# Test OpenRouter recognition (without API key)
tclp-benchmark run --version-tag "test" --providers openrouter 2>&1 | grep -i "openrouter\|api key"
```

Expected: Help text updated, proper error messages

### Step 5: Review all documentation files

```bash
# List all documentation
find docs/ -name "*.md" -type f | sort

# Check for broken internal links (basic check)
grep -r "docs/" README.md CLAUDE.md docs/ | grep -v "Binary" | grep "\.md"

# Verify code examples in docs have correct syntax
# (Manual review recommended)
```

Expected: All docs present, no obvious broken links

### Step 6: Create verification checklist

**Create:** `docs/openrouter-verification-checklist.md`

```markdown
# OpenRouter Integration Verification Checklist

Use this checklist to verify OpenRouter integration is working correctly.

## Configuration ✅

- [ ] `config/providers.yaml` has `openrouter` section
- [ ] At least one model with `:online` suffix configured
- [ ] `web_search.enabled: true` in configuration
- [ ] `custom_search_prompt` mentions `chancerylaneproject.org`
- [ ] Rate limits and retry config present

## Environment ✅

- [ ] `.env` file has `OPENROUTER_API_KEY` entry (or documented in README)
- [ ] API key is valid (test at openrouter.ai)
- [ ] No hardcoded API keys in codebase

## Code ✅

- [ ] `src/tclp_benchmark/llm_providers/openrouter_provider.py` exists
- [ ] `OpenRouterProvider` inherits from `BaseLLMProvider`
- [ ] `generate_response` method implemented with retry logic
- [ ] Orchestrator imports and registers `OpenRouterProvider`
- [ ] CLI validates `OPENROUTER_API_KEY` when OpenRouter selected

## Tests ✅

- [ ] Unit tests for `OpenRouterConfig` model pass
- [ ] Unit tests for `OpenRouterProvider` pass
- [ ] Integration tests for config loading pass
- [ ] Integration tests for orchestrator pass
- [ ] E2E test validates configuration (without API call)

## Documentation ✅

- [ ] README mentions OpenRouter integration
- [ ] CLAUDE.md documents implementation details
- [ ] `docs/openrouter-integration.md` provides comprehensive guide
- [ ] `docs/migration-guides/openrouter-migration.md` helps existing users
- [ ] CHANGELOG.md lists all changes (if applicable)

## Validation ✅

- [ ] `./scripts/validate_openrouter.sh` runs successfully
- [ ] All unit tests pass: `pytest tests/unit/ -v`
- [ ] All integration tests pass: `pytest tests/integration/ -v`
- [ ] CLI help text mentions OpenRouter: `tclp-benchmark run --help`

## Optional: Live API Test ✅

- [ ] Set `OPENROUTER_API_KEY` in environment
- [ ] Run test evaluation: `tclp-benchmark run --version-tag "or-test" --providers openrouter --category status_checks`
- [ ] Verify responses in database have `web_search_enabled: true`
- [ ] Check responses mention "chancerylaneproject.org" or specific clauses
- [ ] Generate comparison report: `tclp-benchmark report --version-a X --version-b Y`

## Git Status ✅

- [ ] All changes committed to feature branch
- [ ] Commit messages follow convention (feat:, docs:, test:)
- [ ] No uncommitted changes: `git status` is clean
- [ ] Branch ready for merge/PR

## Pre-Merge Review ✅

- [ ] Code follows existing patterns (async/await, tenacity retry, etc.)
- [ ] No security issues (API keys not committed)
- [ ] Configuration schema maintains backward compatibility
- [ ] New dependencies documented (none expected - uses existing OpenAI client)
- [ ] Performance impact documented (2-5s additional latency for web search)
```

### Step 7: Run checklist verification

```bash
# Create a simple verification script
cat > /tmp/verify_openrouter_checklist.sh << 'EOF'
#!/usr/bin/env bash
echo "🔍 OpenRouter Integration Verification"
echo

# Configuration
echo "✅ Configuration files"
test -f config/providers.yaml && echo "  ✅ providers.yaml exists"
grep -q "openrouter:" config/providers.yaml && echo "  ✅ openrouter section present"

# Code
echo
echo "✅ Code files"
test -f src/tclp_benchmark/llm_providers/openrouter_provider.py && echo "  ✅ openrouter_provider.py exists"
grep -q "class OpenRouterProvider" src/tclp_benchmark/llm_providers/openrouter_provider.py && echo "  ✅ Provider class defined"

# Tests
echo
echo "✅ Test files"
test -f tests/unit/config/test_openrouter_config.py && echo "  ✅ Config tests exist"
test -f tests/unit/llm_providers/test_openrouter_provider.py && echo "  ✅ Provider tests exist"
test -f tests/integration/end_to_end/test_openrouter_e2e.py && echo "  ✅ E2E tests exist"

# Documentation
echo
echo "✅ Documentation"
grep -q "OpenRouter" README.md && echo "  ✅ README mentions OpenRouter"
grep -q "OpenRouter" CLAUDE.md && echo "  ✅ CLAUDE.md mentions OpenRouter"
test -f docs/openrouter-integration.md && echo "  ✅ Integration guide exists"
test -f docs/migration-guides/openrouter-migration.md && echo "  ✅ Migration guide exists"

echo
echo "✅ All verification items checked!"
EOF

chmod +x /tmp/verify_openrouter_checklist.sh
/tmp/verify_openrouter_checklist.sh
```

Expected: All checks pass

### Step 8: Final commit

```bash
git add docs/openrouter-verification-checklist.md
git commit -m "docs: add verification checklist for OpenRouter integration"
```

### Step 9: Create summary of changes

```bash
# Generate list of all commits for this feature
git log --oneline --reverse HEAD~20..HEAD | grep -E "(feat|docs|test|fix).*[Oo]pen[Rr]outer"

# Show files changed
git diff --name-only HEAD~20..HEAD | grep -E "(openrouter|providers\.yaml)"

# Show test coverage (if available)
pytest tests/ --cov=src/tclp_benchmark --cov-report=term | grep -A 20 "TOTAL"
```

### Step 10: Final verification command

Run this comprehensive command to verify everything:

```bash
echo "🔍 Final OpenRouter Integration Verification" && \
echo && \
echo "1. Configuration validation..." && \
python -c "from pathlib import Path; from tclp_benchmark.config.loader import load_providers_config; c = load_providers_config(Path('config/providers.yaml')); print(f'✅ Config valid: {len(c.openrouter.models)} models')" && \
echo && \
echo "2. Unit tests..." && \
pytest tests/unit/config/test_openrouter_config.py tests/unit/llm_providers/test_openrouter_provider.py -v --tb=line && \
echo && \
echo "3. Integration tests..." && \
pytest tests/integration/evaluators/test_orchestrator_openrouter.py tests/integration/end_to_end/test_openrouter_e2e.py::test_openrouter_config_validation -v --tb=line && \
echo && \
echo "4. Git status..." && \
git status --short && \
echo && \
echo "✅ OpenRouter integration verification complete!"
```

Expected: All tests pass, clean git status

---

## Summary

This implementation plan provides a complete, test-driven approach to integrating OpenRouter into the TCLP benchmarking system.

**Key Deliverables:**

1. ✅ Configuration system extended with OpenRouter support
2. ✅ New `OpenRouterProvider` with web search capabilities
3. ✅ Orchestrator integration for seamless provider switching
4. ✅ CLI updates with validation and documentation
5. ✅ Comprehensive test suite (unit + integration + E2E)
6. ✅ Complete documentation (README, guides, migration)
7. ✅ Validation tooling and verification checklists

**Testing Strategy:**

- Unit tests for configuration and provider logic
- Integration tests for orchestrator and CLI
- E2E tests for full workflow validation
- Manual validation script for deployment verification

**Documentation Strategy:**

- User-facing: README updates, migration guide
- Developer-facing: CLAUDE.md implementation notes, integration guide
- Verification: Checklist, validation scripts

**Implementation Approach:**

- TDD: Write failing tests first, implement to pass
- Incremental: Each task is self-contained and committable
- DRY: Reuse existing patterns (AsyncOpenAI, tenacity retry)
- YAGNI: No speculative features, focus on requirements

**Estimated Task Complexity:**

- Task 1: 15-20 minutes (config models)
- Task 2: 20-25 minutes (provider implementation)
- Task 3: 10-15 minutes (orchestrator integration)
- Task 4: 10-15 minutes (CLI updates)
- Task 5: 15-20 minutes (documentation)
- Task 6: 15-20 minutes (integration testing)
- Task 7: 10-15 minutes (reporting updates)
- Task 8: 10-15 minutes (migration guide)
- Task 9: 15-20 minutes (final verification)

**Total: ~2-3 hours for complete implementation**
