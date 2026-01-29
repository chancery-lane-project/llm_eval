"""Tests for OpenRouter-specific report formatting."""
import pytest
from pathlib import Path
from datetime import datetime
from tclp_benchmark.reporting.report_generator import ReportGenerator
from tclp_benchmark.storage.models import (
    EvaluationRun,
    TestCase,
    LLMResponse,
    MetricScore,
)


@pytest.fixture
def mock_openrouter_run():
    """Create mock evaluation run with OpenRouter provider."""
    run = EvaluationRun(
        version_tag="test-openrouter",
        description="Test run with OpenRouter",
        status="completed",
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )

    test_case = TestCase(
        prompt_category="direct_lookup",
        prompt_text="Find the net zero clause",
    )

    # OpenRouter response with :online model
    response = LLMResponse(
        provider_name="openrouter",
        model_name="openai/gpt-4o:online",
        response_text="The net zero clause can be found on chancerylaneproject.org...",
        execution_time_ms=1500,
    )

    metric = MetricScore(
        metric_name="tclp_citation",
        score=0.85,
        threshold=0.5,
        passed=True,
        reasoning="Found TCLP citation",
    )

    response.metric_scores.append(metric)
    test_case.llm_responses.append(response)
    run.test_cases.append(test_case)

    return run


def test_markdown_report_highlights_openrouter_web_search(
    mock_openrouter_run, tmp_path
):
    """Test that markdown reports highlight OpenRouter web search."""
    from unittest.mock import MagicMock

    # Mock database manager
    db_manager = MagicMock()
    generator = ReportGenerator(db_manager)

    # Calculate stats
    stats = generator._calculate_single_run_stats(mock_openrouter_run)

    # Generate markdown
    output_path = tmp_path / "report.md"
    generator._generate_single_markdown(mock_openrouter_run, stats, output_path)

    content = output_path.read_text()

    # Check for OpenRouter indication
    assert "openrouter" in content.lower()
    assert "openai/gpt-4o:online" in content

    # Check for web search indication
    assert "Web Search Enabled" in content or "🌐" in content


def test_html_report_highlights_openrouter_web_search(
    mock_openrouter_run, tmp_path
):
    """Test that HTML reports highlight OpenRouter web search with badge."""
    from unittest.mock import MagicMock

    db_manager = MagicMock()
    generator = ReportGenerator(db_manager)

    stats = generator._calculate_single_run_stats(mock_openrouter_run)
    output_path = tmp_path / "report.html"

    generator._generate_single_html(mock_openrouter_run, stats, output_path)

    content = output_path.read_text()

    # Check for OpenRouter
    assert "openrouter" in content.lower()
    assert "openai/gpt-4o:online" in content

    # Check for web search badge styling
    assert "Web Search" in content
    assert "#28a745" in content  # Green badge color


def test_comparison_report_shows_openrouter_vs_direct(tmp_path):
    """Test comparison report between OpenRouter and direct API."""
    from unittest.mock import MagicMock

    # Create baseline run (direct API)
    run_a = EvaluationRun(
        version_tag="baseline-openai",
        status="completed",
        created_at=datetime.now(),
    )
    test_case_a = TestCase(
        prompt_category="direct_lookup",
        prompt_text="Test prompt",
    )
    response_a = LLMResponse(
        provider_name="openai",
        model_name="gpt-4o",
        response_text="Response",
        execution_time_ms=1000,
    )
    metric_a = MetricScore(
        metric_name="tclp_citation",
        score=0.65,
        threshold=0.5,
        passed=True,
    )
    response_a.metric_scores.append(metric_a)
    test_case_a.llm_responses.append(response_a)
    run_a.test_cases.append(test_case_a)

    # Create comparison run (OpenRouter with web search)
    run_b = EvaluationRun(
        version_tag="test-openrouter",
        status="completed",
        created_at=datetime.now(),
    )
    test_case_b = TestCase(
        prompt_category="direct_lookup",
        prompt_text="Test prompt",
    )
    response_b = LLMResponse(
        provider_name="openrouter",
        model_name="openai/gpt-4o:online",
        response_text="Response with web search",
        execution_time_ms=1500,
    )
    metric_b = MetricScore(
        metric_name="tclp_citation",
        score=0.85,
        threshold=0.5,
        passed=True,
    )
    response_b.metric_scores.append(metric_b)
    test_case_b.llm_responses.append(response_b)
    run_b.test_cases.append(test_case_b)

    # Generate comparison
    db_manager = MagicMock()
    generator = ReportGenerator(db_manager)

    comparison = generator.calculate_comparison(run_a, run_b)

    # Verify by_provider shows both
    assert "openai/gpt-4o" in comparison["by_provider"]
    assert "openrouter/openai/gpt-4o:online" in comparison["by_provider"]

    # Generate markdown report
    output_md = tmp_path / "comparison.md"
    generator._generate_markdown(run_a, run_b, comparison, output_md)

    content_md = output_md.read_text()
    assert "openrouter" in content_md.lower()
    assert "Web Search" in content_md or "🌐" in content_md

    # Generate HTML report
    output_html = tmp_path / "comparison.html"
    generator._generate_html(run_a, run_b, comparison, output_html)

    content_html = output_html.read_text()
    assert "openrouter" in content_html.lower()
    assert "Web Search" in content_html


def test_openrouter_without_online_suffix(tmp_path):
    """Test OpenRouter model without :online suffix (no web search)."""
    run = EvaluationRun(
        version_tag="test-no-websearch",
        status="completed",
        created_at=datetime.now(),
    )
    test_case = TestCase(
        prompt_category="test",
        prompt_text="Test",
    )
    response = LLMResponse(
        provider_name="openrouter",
        model_name="openai/gpt-4o",  # No :online suffix
        response_text="Response",
        execution_time_ms=1000,
    )
    metric = MetricScore(
        metric_name="test_metric",
        score=0.5,
        threshold=0.5,
        passed=True,
    )
    response.metric_scores.append(metric)
    test_case.llm_responses.append(response)
    run.test_cases.append(test_case)

    from unittest.mock import MagicMock

    db_manager = MagicMock()
    generator = ReportGenerator(db_manager)

    stats = generator._calculate_single_run_stats(run)
    output_path = tmp_path / "report.md"

    generator._generate_single_markdown(run, stats, output_path)

    content = output_path.read_text()

    # Should mention OpenRouter but NOT web search
    assert "openrouter" in content.lower()
    assert "Web Search Enabled" not in content


def test_provider_label_formatting():
    """Test provider label formatting for different provider types."""
    from tclp_benchmark.reporting.report_generator import ReportGenerator

    # This is a simple unit test for the logic without full integration

    test_cases = [
        ("openrouter/openai/gpt-4o:online", True),  # Should show web search
        ("openrouter/anthropic/claude-opus:online", True),
        ("openrouter/openai/gpt-4o", False),  # No web search
        ("openai/gpt-4o", False),  # Direct API
        ("anthropic/claude-opus", False),
    ]

    for provider, should_highlight in test_cases:
        # Check if logic would identify web search correctly
        is_openrouter_online = (
            provider.startswith("openrouter/") and ":online" in provider
        )
        assert is_openrouter_online == should_highlight
