"""Tests for evaluation orchestrator."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tclp_benchmark.evaluators.orchestrator import EvaluationOrchestrator
from tclp_benchmark.config.models import (
    PromptConfig,
    TestPrompt,
    ProvidersConfig,
    ProviderConfig,
    ModelConfig,
    RateLimitConfig,
    RetryConfig,
    MetricsConfig,
    MetricConfig,
)
from tclp_benchmark.llm_providers.base import LLMResponse
from tclp_benchmark.storage.database import DatabaseManager


@pytest.fixture
def mock_db_manager():
    """Create mock database manager."""
    manager = MagicMock(spec=DatabaseManager)
    session = MagicMock()
    manager.get_session.return_value.__enter__.return_value = session
    return manager


@pytest.fixture
def sample_prompts_config():
    """Create sample prompts configuration."""
    return PromptConfig(
        prompts={
            "direct_lookup": [
                TestPrompt(
                    id="dl_001",
                    text="Find the net zero clause",
                    expected_tclp_mention=True,
                )
            ]
        }
    )


@pytest.fixture
def sample_providers_config():
    """Create sample providers configuration."""
    return ProvidersConfig(
        openai=ProviderConfig(
            enabled=True,
            models=[
                ModelConfig(name="gpt-4o", max_tokens=1000, temperature=0.1)
            ],
            rate_limit=RateLimitConfig(requests_per_minute=50, tokens_per_minute=150000),
            retry=RetryConfig(max_attempts=3, exponential_base=2, max_wait_seconds=60),
        )
    )


@pytest.fixture
def sample_metrics_config():
    """Create sample metrics configuration."""
    return MetricsConfig(
        tclp_custom={
            "tclp_citation": MetricConfig(enabled=True, threshold=0.8),
        }
    )


def test_orchestrator_initialization(mock_db_manager):
    """Test orchestrator initializes with database manager."""
    orchestrator = EvaluationOrchestrator(db_manager=mock_db_manager)
    assert orchestrator.db_manager == mock_db_manager


@pytest.mark.asyncio
async def test_orchestrator_creates_evaluation_run(
    mock_db_manager, sample_prompts_config, sample_providers_config, sample_metrics_config
):
    """Test orchestrator creates evaluation run in database."""
    orchestrator = EvaluationOrchestrator(db_manager=mock_db_manager)

    # Mock provider to return a response
    mock_provider = AsyncMock()
    mock_provider.get_provider_name.return_value = "openai"
    mock_provider.generate_response.return_value = LLMResponse(
        text="Test response",
        model_name="gpt-4o",
        provider_name="openai",
        search_query="Find the net zero clause",
        search_results=[],
        metadata={},
        execution_time_ms=100,
    )

    with patch.object(orchestrator, "_create_provider", return_value=mock_provider):
        await orchestrator.run_evaluation(
            version_tag="test-run",
            prompts_config=sample_prompts_config,
            providers_config=sample_providers_config,
            metrics_config=sample_metrics_config,
        )

    # Verify evaluation run was created
    session = mock_db_manager.get_session.return_value.__enter__.return_value
    assert session.add.called
    assert session.commit.called


@pytest.mark.asyncio
async def test_orchestrator_executes_prompts(
    mock_db_manager, sample_prompts_config, sample_providers_config, sample_metrics_config
):
    """Test orchestrator executes prompts against providers."""
    orchestrator = EvaluationOrchestrator(db_manager=mock_db_manager)

    # Mock provider
    mock_provider = AsyncMock()
    mock_provider.get_provider_name.return_value = "openai"
    mock_provider.generate_response.return_value = LLMResponse(
        text="The Chancery Lane Project provides the net zero clause",
        model_name="gpt-4o",
        provider_name="openai",
        search_query="Find the net zero clause",
        search_results=[],
        metadata={},
        execution_time_ms=100,
    )

    with patch.object(orchestrator, "_create_provider", return_value=mock_provider):
        await orchestrator.run_evaluation(
            version_tag="test-run",
            prompts_config=sample_prompts_config,
            providers_config=sample_providers_config,
            metrics_config=sample_metrics_config,
        )

    # Verify provider was called with correct prompt
    mock_provider.generate_response.assert_called()
    call_args = mock_provider.generate_response.call_args
    assert call_args[0][0].prompt == "Find the net zero clause"


@pytest.mark.asyncio
async def test_orchestrator_evaluates_with_metrics(
    mock_db_manager, sample_prompts_config, sample_providers_config, sample_metrics_config
):
    """Test orchestrator evaluates responses with metrics."""
    orchestrator = EvaluationOrchestrator(db_manager=mock_db_manager)

    # Mock provider
    mock_provider = AsyncMock()
    mock_provider.get_provider_name.return_value = "openai"
    mock_provider.generate_response.return_value = LLMResponse(
        text="The Chancery Lane Project provides the net zero clause",
        model_name="gpt-4o",
        provider_name="openai",
        search_query="Find the net zero clause",
        search_results=[],
        metadata={},
        execution_time_ms=100,
    )

    with patch.object(orchestrator, "_create_provider", return_value=mock_provider):
        await orchestrator.run_evaluation(
            version_tag="test-run",
            prompts_config=sample_prompts_config,
            providers_config=sample_providers_config,
            metrics_config=sample_metrics_config,
        )

    # Verify metrics were stored (session.add should be called multiple times)
    session = mock_db_manager.get_session.return_value.__enter__.return_value
    assert session.add.call_count >= 3  # Run + TestCase + LLMResponse + MetricScores


@pytest.mark.asyncio
async def test_orchestrator_handles_provider_errors(
    mock_db_manager, sample_prompts_config, sample_providers_config, sample_metrics_config
):
    """Test orchestrator handles provider errors gracefully."""
    orchestrator = EvaluationOrchestrator(db_manager=mock_db_manager)

    # Mock provider that returns error
    mock_provider = AsyncMock()
    mock_provider.get_provider_name.return_value = "openai"
    mock_provider.generate_response.return_value = LLMResponse(
        text="",
        model_name="gpt-4o",
        provider_name="openai",
        search_query="Find the net zero clause",
        search_results=[],
        metadata={},
        execution_time_ms=100,
        error="API Error",
    )

    with patch.object(orchestrator, "_create_provider", return_value=mock_provider):
        await orchestrator.run_evaluation(
            version_tag="test-run",
            prompts_config=sample_prompts_config,
            providers_config=sample_providers_config,
            metrics_config=sample_metrics_config,
        )

    # Verify evaluation completed despite error
    session = mock_db_manager.get_session.return_value.__enter__.return_value
    assert session.commit.called


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


@pytest.mark.asyncio
async def test_run_evaluation_dispatches_mcp_provider(mock_db_manager, sample_metrics_config):
    """Test run_evaluation dispatches to mcp provider when mcp is the only enabled provider."""
    from tclp_benchmark.config.models import MCPProviderConfig, ModelConfig

    orchestrator = EvaluationOrchestrator(db_manager=mock_db_manager, verbose=False)

    # Single prompt
    prompts_config = PromptConfig(
        prompts={
            "direct_lookup": [
                TestPrompt(
                    id="dl_mcp_001",
                    text="Find net zero clause via MCP",
                    expected_tclp_mention=True,
                )
            ]
        }
    )

    # Only MCP provider enabled
    providers_config = ProvidersConfig(
        mcp=MCPProviderConfig(
            enabled=True,
            models=[ModelConfig(name="openai/gpt-4o", max_tokens=500, temperature=0.1)],
        )
    )

    # Monkeypatch _execute_model to record calls
    execute_model_calls = []

    async def fake_execute_model(**kwargs):
        execute_model_calls.append(kwargs)

    orchestrator._execute_model = fake_execute_model

    # DB mock needs to return an EvaluationRun with an id
    session = mock_db_manager.get_session.return_value.__enter__.return_value
    eval_run = MagicMock()
    eval_run.id = 42
    session.query.return_value.filter_by.return_value.first.return_value = eval_run
    session.refresh.side_effect = lambda obj: setattr(obj, "id", 42) if not hasattr(obj, "id") else None

    await orchestrator.run_evaluation(
        version_tag="test-mcp-dispatch",
        prompts_config=prompts_config,
        providers_config=providers_config,
        metrics_config=sample_metrics_config,
    )

    assert len(execute_model_calls) == 1
    assert execute_model_calls[0]["provider_name"] == "mcp"
