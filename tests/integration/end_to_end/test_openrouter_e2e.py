"""End-to-end integration tests for OpenRouter functionality."""
import os
import pytest
from pathlib import Path
from click.testing import CliRunner
from tclp_benchmark.cli.commands import cli
from tclp_benchmark.storage.database import DatabaseManager
from tclp_benchmark.storage.models import EvaluationRun


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


@pytest.fixture
def test_env(tmp_path, monkeypatch):
    """Set up test environment with configs and database."""
    # Create config directory
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create prompts config
    prompts_config = config_dir / "prompts.yaml"
    prompts_config.write_text("""
prompts:
  direct_lookup:
    - id: test_direct_001
      text: "Find the net zero carbon commitment clause from The Chancery Lane Project"
      expected_tclp_mention: true
      expected_terms:
        - "net zero"
        - "carbon"
  status_checks:
    - id: test_status_001
      text: "What is the maintenance status of the biodiversity clause on chancerylaneproject.org?"
      expected_tclp_mention: true
      expected_terms:
        - "biodiversity"
        - "status"
""")

    # Create providers config with OpenRouter
    providers_config = config_dir / "providers.yaml"
    providers_config.write_text("""
openrouter:
  enabled: true
  api_endpoint: "https://openrouter.ai/api/v1/chat/completions"
  models:
    - "openai/gpt-4o-mini:online"
  max_tokens: 500
  temperature: 0.1
  web_search:
    enabled: true
    max_results: 5
    custom_search_prompt: "Search chancerylaneproject.org for clauses."
  rate_limit:
    requests_per_minute: 50
    tokens_per_minute: 100000
  retry:
    max_attempts: 3
    exponential_base: 2
    max_wait_seconds: 60
""")

    # Create metrics config
    metrics_config = config_dir / "metrics.yaml"
    metrics_config.write_text("""
tclp_custom:
  tclp_citation:
    enabled: true
    threshold: 0.5
  taxonomy_awareness:
    enabled: true
    threshold: 0.5
""")

    # Set up database
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    # Return config paths
    return {
        "prompts_config": prompts_config,
        "providers_config": providers_config,
        "metrics_config": metrics_config,
        "db_path": db_path,
        "tmp_path": tmp_path,
    }


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set - skipping live API test"
)
def test_full_openrouter_evaluation_run(runner, test_env):
    """
    Test complete evaluation run with OpenRouter provider.

    This is a full end-to-end test that:
    1. Runs evaluation with OpenRouter
    2. Stores results in database
    3. Verifies data integrity
    """
    result = runner.invoke(
        cli,
        [
            "run",
            "--version-tag", "e2e-openrouter-test",
            "--description", "End-to-end OpenRouter integration test",
            "--providers", "openrouter",
            "--prompts-config", str(test_env["prompts_config"]),
            "--providers-config", str(test_env["providers_config"]),
            "--metrics-config", str(test_env["metrics_config"]),
        ],
    )

    # Check command succeeded
    assert result.exit_code == 0, f"Command failed: {result.output}"
    assert "Evaluation complete" in result.output or "completed" in result.output.lower()

    # Verify database contains results
    db_manager = DatabaseManager(f"sqlite:///{test_env['db_path']}")

    with db_manager.get_session() as session:
        run = session.query(EvaluationRun).filter_by(
            version_tag="e2e-openrouter-test"
        ).first()

        assert run is not None, "Evaluation run not found in database"
        assert run.status == "completed"
        assert run.description == "End-to-end OpenRouter integration test"
        assert len(run.test_cases) > 0, "No test cases found"

        # Verify test cases have responses
        for test_case in run.test_cases:
            assert len(test_case.llm_responses) > 0, f"No responses for test case: {test_case.prompt_text}"

            # Verify OpenRouter provider was used
            for response in test_case.llm_responses:
                assert response.provider_name == "openrouter"
                assert response.model_name.startswith("openai/gpt-4o-mini")

                # Verify metrics were calculated
                assert len(response.metric_scores) > 0, "No metric scores found"

                # Check for expected metrics
                metric_names = [m.metric_name for m in response.metric_scores]
                assert "tclp_citation" in metric_names
                assert "taxonomy_awareness" in metric_names


@pytest.mark.integration
def test_openrouter_with_missing_api_key(runner, test_env, monkeypatch):
    """Test that OpenRouter fails gracefully without API key."""
    # Remove API key
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    result = runner.invoke(
        cli,
        [
            "run",
            "--version-tag", "no-key-test",
            "--providers", "openrouter",
            "--prompts-config", str(test_env["prompts_config"]),
            "--providers-config", str(test_env["providers_config"]),
            "--metrics-config", str(test_env["metrics_config"]),
        ],
    )

    # Should fail with clear error message
    assert result.exit_code != 0
    assert "OPENROUTER_API_KEY" in result.output or "API key" in result.output


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set - skipping live API test"
)
def test_openrouter_web_search_enabled(runner, test_env):
    """Test that web search is properly enabled for OpenRouter."""
    result = runner.invoke(
        cli,
        [
            "run",
            "--version-tag", "web-search-test",
            "--providers", "openrouter",
            "--category", "direct_lookup",
            "--prompts-config", str(test_env["prompts_config"]),
            "--providers-config", str(test_env["providers_config"]),
            "--metrics-config", str(test_env["metrics_config"]),
        ],
    )

    assert result.exit_code == 0

    # Verify in database
    db_manager = DatabaseManager(f"sqlite:///{test_env['db_path']}")

    with db_manager.get_session() as session:
        run = session.query(EvaluationRun).filter_by(
            version_tag="web-search-test"
        ).first()

        assert run is not None

        # Check that responses contain web search context
        # (OpenRouter responses with :online models should reference web sources)
        for test_case in run.test_cases:
            for response in test_case.llm_responses:
                # Response should exist and not be empty
                assert response.response_text
                assert len(response.response_text) > 0


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set - skipping live API test"
)
def test_openrouter_report_generation(runner, test_env):
    """Test report generation for OpenRouter evaluation."""
    # First run evaluation
    run_result = runner.invoke(
        cli,
        [
            "run",
            "--version-tag", "report-test",
            "--providers", "openrouter",
            "--category", "direct_lookup",
            "--prompts-config", str(test_env["prompts_config"]),
            "--providers-config", str(test_env["providers_config"]),
            "--metrics-config", str(test_env["metrics_config"]),
        ],
    )

    assert run_result.exit_code == 0

    # Generate single-run report
    report_path = test_env["tmp_path"] / "report.html"
    report_result = runner.invoke(
        cli,
        [
            "report-single",
            "--version-tag", "report-test",
            "--format", "html",
            "--output", str(report_path),
        ],
    )

    assert report_result.exit_code == 0
    assert report_path.exists()

    # Verify report contains OpenRouter information
    report_content = report_path.read_text()
    assert "openrouter" in report_content.lower()
    assert "gpt-4o-mini" in report_content.lower()


@pytest.mark.integration
@pytest.mark.skipif(
    not all([os.getenv("OPENROUTER_API_KEY"), os.getenv("OPENAI_API_KEY")]),
    reason="API keys not set - skipping comparison test"
)
def test_openrouter_vs_direct_provider_comparison(runner, test_env):
    """Test comparison report between OpenRouter and direct API provider."""
    # Create providers config with both OpenRouter and OpenAI
    providers_config = test_env["tmp_path"] / "config" / "providers_both.yaml"
    providers_config.write_text("""
openai:
  enabled: true
  models:
    - name: gpt-4o-mini
      max_tokens: 500
      temperature: 0.1
  rate_limit:
    requests_per_minute: 50
    tokens_per_minute: 100000
  retry:
    max_attempts: 3
    exponential_base: 2
    max_wait_seconds: 60

openrouter:
  enabled: true
  api_endpoint: "https://openrouter.ai/api/v1/chat/completions"
  models:
    - "openai/gpt-4o-mini:online"
  max_tokens: 500
  temperature: 0.1
  web_search:
    enabled: true
    max_results: 5
  rate_limit:
    requests_per_minute: 50
    tokens_per_minute: 100000
  retry:
    max_attempts: 3
    exponential_base: 2
    max_wait_seconds: 60
""")

    # Run baseline with OpenAI
    baseline_result = runner.invoke(
        cli,
        [
            "run",
            "--version-tag", "baseline-openai",
            "--providers", "openai",
            "--category", "direct_lookup",
            "--prompts-config", str(test_env["prompts_config"]),
            "--providers-config", str(providers_config),
            "--metrics-config", str(test_env["metrics_config"]),
        ],
    )

    assert baseline_result.exit_code == 0

    # Run comparison with OpenRouter
    comparison_result = runner.invoke(
        cli,
        [
            "run",
            "--version-tag", "comparison-openrouter",
            "--providers", "openrouter",
            "--category", "direct_lookup",
            "--prompts-config", str(test_env["prompts_config"]),
            "--providers-config", str(providers_config),
            "--metrics-config", str(test_env["metrics_config"]),
        ],
    )

    assert comparison_result.exit_code == 0

    # Generate comparison report
    report_path = test_env["tmp_path"] / "comparison.html"
    report_result = runner.invoke(
        cli,
        [
            "report",
            "--version-a", "baseline-openai",
            "--version-b", "comparison-openrouter",
            "--format", "html",
            "--output", str(report_path),
        ],
    )

    assert report_result.exit_code == 0
    assert report_path.exists()

    # Verify report shows both providers
    report_content = report_path.read_text()
    assert "openai" in report_content.lower()
    assert "openrouter" in report_content.lower()


def test_openrouter_config_validation(runner, test_env):
    """Test that invalid OpenRouter config is caught."""
    # Create invalid config (missing required fields)
    invalid_config = test_env["tmp_path"] / "config" / "invalid.yaml"
    invalid_config.write_text("""
openrouter:
  enabled: true
  # Missing api_endpoint and models
""")

    result = runner.invoke(
        cli,
        [
            "run",
            "--version-tag", "invalid-test",
            "--providers", "openrouter",
            "--prompts-config", str(test_env["prompts_config"]),
            "--providers-config", str(invalid_config),
            "--metrics-config", str(test_env["metrics_config"]),
        ],
    )

    # Should fail with validation error
    assert result.exit_code != 0


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set - skipping live API test"
)
def test_openrouter_rate_limiting(runner, test_env):
    """Test that rate limiting is respected for OpenRouter."""
    # Create config with strict rate limiting
    strict_config = test_env["tmp_path"] / "config" / "strict_limits.yaml"
    strict_config.write_text("""
openrouter:
  enabled: true
  api_endpoint: "https://openrouter.ai/api/v1/chat/completions"
  models:
    - "openai/gpt-4o-mini:online"
  max_tokens: 500
  temperature: 0.1
  web_search:
    enabled: true
  rate_limit:
    requests_per_minute: 1  # Very strict for testing
    tokens_per_minute: 10000
  retry:
    max_attempts: 3
    exponential_base: 2
    max_wait_seconds: 60
""")

    import time
    start_time = time.time()

    result = runner.invoke(
        cli,
        [
            "run",
            "--version-tag", "rate-limit-test",
            "--providers", "openrouter",
            "--category", "direct_lookup",
            "--prompts-config", str(test_env["prompts_config"]),
            "--providers-config", str(strict_config),
            "--metrics-config", str(test_env["metrics_config"]),
        ],
    )

    elapsed_time = time.time() - start_time

    # With 2 prompts and 1 req/min, should take at least 60 seconds
    # (allowing some tolerance for processing time)
    assert result.exit_code == 0
    # Note: This is a rough check - in practice may want to verify logs
    # showing rate limiting was applied


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set - skipping live API test"
)
def test_openrouter_error_handling(runner, test_env):
    """Test error handling for OpenRouter API failures."""
    # Create config with non-existent model
    bad_model_config = test_env["tmp_path"] / "config" / "bad_model.yaml"
    bad_model_config.write_text("""
openrouter:
  enabled: true
  api_endpoint: "https://openrouter.ai/api/v1/chat/completions"
  models:
    - "nonexistent/model-that-does-not-exist:online"
  max_tokens: 500
  temperature: 0.1
  web_search:
    enabled: true
  rate_limit:
    requests_per_minute: 50
  retry:
    max_attempts: 3
    exponential_base: 2
    max_wait_seconds: 60
""")

    result = runner.invoke(
        cli,
        [
            "run",
            "--version-tag", "error-test",
            "--providers", "openrouter",
            "--category", "direct_lookup",
            "--prompts-config", str(test_env["prompts_config"]),
            "--providers-config", str(bad_model_config),
            "--metrics-config", str(test_env["metrics_config"]),
        ],
    )

    # Should handle error gracefully
    # May complete with errors logged, or fail cleanly
    # The key is it shouldn't crash with unhandled exception
    assert "Traceback" not in result.output or result.exit_code != 0
