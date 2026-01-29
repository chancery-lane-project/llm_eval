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
