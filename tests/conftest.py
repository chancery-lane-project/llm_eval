"""Shared pytest fixtures and setup.

Loads the project .env so env-gated integration tests (e.g. the live MCP test,
which needs TCLP_MCP_URL and OPENROUTER_API_KEY) can see credentials without
requiring the caller to export them manually. Unit tests do not depend on it.
"""
from pathlib import Path

from dotenv import load_dotenv

# Repo root is two levels up from this file (tests/conftest.py).
load_dotenv(Path(__file__).parent.parent / ".env")
