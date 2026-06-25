"""Configuration loader for YAML files."""
import yaml
from pathlib import Path
from typing import Type, TypeVar
from pydantic import BaseModel
from .models import PromptConfig, ProvidersConfig, MetricsConfig

T = TypeVar("T", bound=BaseModel)


class ConfigLoader:
    """Loads and validates YAML configuration files."""

    @staticmethod
    def _load_yaml(file_path: Path) -> dict:
        """Load YAML file and return dict."""
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in {file_path}: {e}")

        return data

    @staticmethod
    def _validate_config(data: dict, model: Type[T]) -> T:
        """Validate data against Pydantic model."""
        return model(**data)

    @classmethod
    def load_prompts(cls, file_path: Path) -> PromptConfig:
        """Load and validate prompts configuration."""
        data = cls._load_yaml(file_path)
        return cls._validate_config(data, PromptConfig)

    @classmethod
    def load_providers(cls, file_path: Path) -> ProvidersConfig:
        """Load and validate providers configuration."""
        data = cls._load_yaml(file_path)
        # Extract only provider configs (not 'search' section)
        provider_data = {
            k: v for k, v in data.items()
            if k in ["openai", "anthropic", "gemini", "openrouter"]
        }
        return cls._validate_config(provider_data, ProvidersConfig)

    @classmethod
    def load_metrics(cls, file_path: Path) -> MetricsConfig:
        """Load and validate metrics configuration."""
        data = cls._load_yaml(file_path)
        return cls._validate_config(data, MetricsConfig)


# Convenience functions for direct access
def load_providers_config(file_path: Path) -> ProvidersConfig:
    """Load and validate providers configuration."""
    return ConfigLoader.load_providers(file_path)
