"""Config wiring tests for the MCP provider arm."""
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
