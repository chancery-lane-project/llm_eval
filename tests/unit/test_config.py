"""Tests for configuration loading and validation."""
import pytest
from pathlib import Path
from pydantic import ValidationError
from tclp_benchmark.config.models import (
    PromptConfig,
    TestPrompt,
    ProviderConfig,
    MetricConfig,
)


def test_test_prompt_validation():
    """Test TestPrompt model validates required fields."""
    # Valid prompt
    prompt = TestPrompt(
        id="dl_001",
        text="Find the net zero carbon commitment clause",
        expected_tclp_mention=True,
    )
    assert prompt.id == "dl_001"
    assert prompt.expected_tclp_mention is True

    # Missing required field
    with pytest.raises(ValidationError):
        TestPrompt(id="dl_001")  # Missing 'text'


def test_prompt_config_loads_categories():
    """Test PromptConfig organises prompts by category."""
    config = PromptConfig(
        prompts={
            "direct_lookup": [
                TestPrompt(id="dl_001", text="Test prompt", expected_tclp_mention=True)
            ],
            "filtered_discovery": [
                TestPrompt(id="fd_001", text="Another prompt", expected_tclp_mention=False)
            ],
        }
    )
    assert len(config.prompts["direct_lookup"]) == 1
    assert len(config.prompts["filtered_discovery"]) == 1


def test_provider_config_validation():
    """Test ProviderConfig validates rate limits."""
    config = ProviderConfig(
        enabled=True,
        models=[{"name": "gpt-4o", "max_tokens": 1000, "temperature": 0.1}],
        rate_limit={"requests_per_minute": 50, "tokens_per_minute": 150000},
        retry={"max_attempts": 3, "exponential_base": 2, "max_wait_seconds": 60},
    )
    assert config.enabled is True
    assert config.rate_limit.requests_per_minute == 50


def test_metric_config_threshold_validation():
    """Test MetricConfig validates threshold range."""
    # Valid threshold
    config = MetricConfig(enabled=True, threshold=0.7)
    assert config.threshold == 0.7

    # Invalid threshold (out of range)
    with pytest.raises(ValidationError):
        MetricConfig(enabled=True, threshold=1.5)  # > 1.0


from tclp_benchmark.config.loader import ConfigLoader


def test_config_loader_loads_yaml(tmp_path):
    """Test ConfigLoader loads valid YAML."""
    # Create temporary YAML file
    yaml_content = """
prompts:
  direct_lookup:
    - id: dl_001
      text: Find the net zero carbon commitment clause
      expected_tclp_mention: true
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)

    # Load config
    config = ConfigLoader.load_prompts(config_file)

    assert "direct_lookup" in config.prompts
    assert len(config.prompts["direct_lookup"]) == 1
    assert config.prompts["direct_lookup"][0].id == "dl_001"


def test_config_loader_raises_on_missing_file():
    """Test ConfigLoader raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        ConfigLoader.load_prompts(Path("/nonexistent/config.yaml"))


def test_config_loader_raises_on_invalid_yaml(tmp_path):
    """Test ConfigLoader raises ValidationError for invalid YAML."""
    # Create invalid YAML
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("prompts:\n  - invalid structure")

    with pytest.raises(ValidationError):
        ConfigLoader.load_prompts(config_file)
