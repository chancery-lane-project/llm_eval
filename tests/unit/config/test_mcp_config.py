"""Config wiring tests for the MCP provider arm."""
import pytest
from pydantic import ValidationError

from tclp_benchmark.config.loader import ConfigLoader
from tclp_benchmark.config.models import ProvidersConfig, MCPProviderConfig


def test_providers_config_accepts_mcp_block():
    data = {
        "mcp": {
            "enabled": True,
            "max_tool_iterations": 8,
            "models": [
                {"name": "openai/gpt-4o", "max_tokens": 1500, "temperature": 0.1},
            ],
        }
    }
    cfg = ProvidersConfig(**data)
    assert isinstance(cfg.mcp, MCPProviderConfig)
    assert cfg.mcp.models[0].name == "openai/gpt-4o"
    assert cfg.mcp.max_tool_iterations == 8


def test_loader_round_trips_mcp_block(tmp_path):
    """The loader allowlist must not silently drop the mcp block."""
    yaml_content = """
mcp:
  enabled: true
  max_tool_iterations: 8
  models:
    - name: openai/gpt-4o
      max_tokens: 1500
      temperature: 0.1
"""
    config_file = tmp_path / "providers.yaml"
    config_file.write_text(yaml_content)

    result = ConfigLoader.load_providers(config_file)

    assert result.mcp is not None
    assert result.mcp.models[0].name == "openai/gpt-4o"


def test_mcp_provider_config_rejects_unknown_key():
    """extra='forbid' must catch yaml typos in the mcp block."""
    with pytest.raises(ValidationError):
        MCPProviderConfig(
            enabled=True,
            max_tool_iterations=8,
            models=[{"name": "openai/gpt-4o"}],
            web_searchh=True,  # typo: unknown key
        )
