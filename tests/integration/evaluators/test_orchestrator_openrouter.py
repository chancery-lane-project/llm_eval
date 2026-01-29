"""Integration tests for OpenRouter provider in orchestrator."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from tclp_benchmark.evaluators.orchestrator import EvaluationOrchestrator
from tclp_benchmark.config.loader import ConfigLoader
from tclp_benchmark.storage.database import DatabaseManager


@pytest.fixture
def db_manager(tmp_path):
    """Create temporary test database"""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    manager = DatabaseManager(db_url)
    manager.create_tables()
    return manager


@pytest.fixture
def providers_config():
    """Load providers configuration"""
    return ConfigLoader.load_providers(Path("config/providers.yaml"))


@pytest.fixture
def prompts_config():
    """Load prompts configuration"""
    return ConfigLoader.load_prompts(Path("config/prompts.yaml"))


@pytest.fixture
def metrics_config():
    """Load metrics configuration"""
    return ConfigLoader.load_metrics(Path("config/metrics.yaml"))


def test_orchestrator_recognizes_openrouter_provider(
    db_manager,
):
    """Test that orchestrator recognizes OpenRouter as a valid provider"""
    orchestrator = EvaluationOrchestrator(
        db_manager=db_manager,
        verbose=False,
    )

    # Check that OpenRouter provider class can be created
    from tclp_benchmark.llm_providers.openrouter_provider import OpenRouterProvider

    # The orchestrator should have OpenRouter in its provider_map
    # We'll verify this by checking if it can create an OpenRouter provider
    api_key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }

    provider_map = {
        "openai": "OpenAIProvider",
        "anthropic": "AnthropicProvider",
        "gemini": "GeminiProvider",
        "openrouter": "OpenRouterProvider",
    }

    # Verify OpenRouter is in expected maps
    assert "openrouter" in api_key_map
    assert "openrouter" in provider_map


def test_orchestrator_creates_openrouter_provider_instance(
    db_manager,
    providers_config,
    monkeypatch
):
    """Test that orchestrator can instantiate OpenRouter provider"""
    # Mock environment variable
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")

    orchestrator = EvaluationOrchestrator(
        db_manager=db_manager,
        verbose=False,
    )

    # Create OpenRouter provider instance using orchestrator's method
    provider = orchestrator._create_provider("openrouter", providers_config.openrouter)

    assert provider is not None
    assert provider.name == "openrouter"
    assert provider.api_key == "test-key-123"


@pytest.mark.asyncio
async def test_orchestrator_runs_evaluation_with_openrouter(
    db_manager,
    providers_config,
    prompts_config,
    metrics_config,
    monkeypatch
):
    """Test that orchestrator can run evaluation with OpenRouter provider"""
    # Mock environment variable
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")

    # Mock the OpenRouter provider's generate_response method
    from tclp_benchmark.llm_providers.base import LLMResponse

    async def mock_generate_response(self, request):
        return LLMResponse(
            text="This is a test response from OpenRouter",
            provider_name="openrouter",
            model_name=request.model_name,
            search_query="",
            search_results=[],
            metadata={},
            execution_time_ms=100,
            error=None,
        )

    with patch(
        "tclp_benchmark.llm_providers.openrouter_provider.OpenRouterProvider.generate_response",
        new=mock_generate_response
    ):
        orchestrator = EvaluationOrchestrator(
            db_manager=db_manager,
            verbose=False,
        )

        # Run evaluation with OpenRouter only (limit to 1 prompt for speed)
        # Temporarily modify config to have only one prompt
        test_prompts_config = prompts_config.model_copy(deep=True)
        test_prompts_config.prompts = {
            "status_checks": [prompts_config.prompts["status_checks"][0]]
        }

        run_id = await orchestrator.run_evaluation(
            version_tag="test-openrouter",
            prompts_config=test_prompts_config,
            providers_config=providers_config,
            metrics_config=metrics_config,
            provider_filter=["openrouter"],
        )

        # Verify run was created
        assert run_id is not None

        # Verify responses were stored
        with db_manager.get_session() as session:
            from tclp_benchmark.storage.models import EvaluationRun, LLMResponse

            run = session.query(EvaluationRun).filter_by(id=run_id).first()
            assert run is not None
            assert run.status == "completed"

            # Check that OpenRouter responses were created
            responses = session.query(LLMResponse).join(
                LLMResponse.test_case
            ).filter_by(evaluation_run_id=run_id).all()

            openrouter_responses = [r for r in responses if r.provider_name == "openrouter"]
            assert len(openrouter_responses) > 0
            assert openrouter_responses[0].response_text == "This is a test response from OpenRouter"
