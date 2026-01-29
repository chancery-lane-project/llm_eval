"""Integration tests for OpenRouter CLI support."""
import pytest
from click.testing import CliRunner
from pathlib import Path
from tclp_benchmark.cli.commands import cli


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


def test_run_command_accepts_openrouter_provider(runner, monkeypatch, tmp_path):
    """Test run command with --providers openrouter."""
    # Set up environment
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")

    # Create minimal config files
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Minimal prompts config
    prompts_config = config_dir / "prompts.yaml"
    prompts_config.write_text("""
prompts:
  status_checks:
    - text: "What is the status of TCLP clauses?"
      expected_tclp_mention: true
""")

    # Minimal providers config with openrouter
    providers_config = config_dir / "providers.yaml"
    providers_config.write_text("""
providers:
  openrouter:
    enabled: true
    models:
      - "openai/gpt-4o-mini"
    max_tokens: 500
    temperature: 0.0
""")

    # Minimal metrics config
    metrics_config = config_dir / "metrics.yaml"
    metrics_config.write_text("""
metrics:
  tclp_custom:
    tclp_citation:
      enabled: true
      threshold: 0.5
""")

    result = runner.invoke(
        cli,
        [
            "run",
            "--version-tag", "test-openrouter",
            "--providers", "openrouter",
            "--category", "status_checks",
            "--prompts-config", str(prompts_config),
            "--providers-config", str(providers_config),
            "--metrics-config", str(metrics_config),
        ],
    )

    # Should not fail with "Invalid provider"
    assert "Invalid provider" not in result.output
    # Should recognize openrouter - will fail at API call but that's expected
    # The key test is that the provider is accepted
    if result.exit_code != 0:
        # If it failed, it should be because of API issues, not provider validation
        assert "openrouter" not in result.output.lower() or "API key not found" not in result.output


def test_run_command_requires_openrouter_api_key(runner, tmp_path):
    """Test that run command fails gracefully without OpenRouter API key."""
    # Create minimal config files
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Minimal prompts config
    prompts_config = config_dir / "prompts.yaml"
    prompts_config.write_text("""
prompts:
  status_checks:
    - text: "What is the status of TCLP clauses?"
      expected_tclp_mention: true
""")

    # Minimal providers config with openrouter
    providers_config = config_dir / "providers.yaml"
    providers_config.write_text("""
providers:
  openrouter:
    enabled: true
    models:
      - "openai/gpt-4o-mini"
    max_tokens: 500
    temperature: 0.0
""")

    # Minimal metrics config
    metrics_config = config_dir / "metrics.yaml"
    metrics_config.write_text("""
metrics:
  tclp_custom:
    tclp_citation:
      enabled: true
      threshold: 0.5
""")

    result = runner.invoke(
        cli,
        [
            "run",
            "--version-tag", "test-no-key",
            "--providers", "openrouter",
            "--prompts-config", str(prompts_config),
            "--providers-config", str(providers_config),
            "--metrics-config", str(metrics_config),
        ],
        env={"DATABASE_URL": f"sqlite:///{tmp_path / 'test.db'}"},
    )

    # Should fail or warn about missing API key
    assert result.exit_code != 0
    # Should mention the missing API key
    assert "OPENROUTER_API_KEY" in result.output or "API key not found" in result.output


def test_providers_help_text_mentions_openrouter(runner):
    """Test that --providers help text includes openrouter as an example."""
    result = runner.invoke(cli, ["run", "--help"])

    assert result.exit_code == 0
    # Should mention openrouter in the help text
    assert "openrouter" in result.output.lower()
