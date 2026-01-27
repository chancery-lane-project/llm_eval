"""Tests for report generation."""
import pytest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock
from tclp_benchmark.reporting.report_generator import ReportGenerator
from tclp_benchmark.storage.database import DatabaseManager
from tclp_benchmark.storage.models import EvaluationRun, TestCase, LLMResponse, MetricScore


@pytest.fixture
def mock_db_manager():
    """Create mock database manager with sample data."""
    manager = MagicMock(spec=DatabaseManager)
    session = MagicMock()

    # Create sample evaluation runs
    run_a = EvaluationRun(
        id=1,
        version_tag="pre-jsonld",
        description="Baseline before semantic markup",
        status="completed",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        completed_at=datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc),
    )

    run_b = EvaluationRun(
        id=2,
        version_tag="post-jsonld",
        description="After semantic markup",
        status="completed",
        created_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        completed_at=datetime(2024, 1, 2, 1, 0, 0, tzinfo=timezone.utc),
    )

    # Create sample test cases with responses and metrics
    test_case_a = TestCase(
        id=1,
        evaluation_run_id=1,
        prompt_category="direct_lookup",
        prompt_text="Find the net zero clause",
    )

    response_a = LLMResponse(
        id=1,
        test_case_id=1,
        provider_name="openai",
        model_name="gpt-4o",
        response_text="Test response without TCLP",
        execution_time_ms=100,
    )

    metric_a1 = MetricScore(
        id=1,
        llm_response_id=1,
        metric_name="tclp_citation",
        metric_category="tclp_custom",
        score=0.0,
        threshold=0.8,
        passed=0,
    )

    metric_a2 = MetricScore(
        id=2,
        llm_response_id=1,
        metric_name="taxonomy_awareness",
        metric_category="tclp_custom",
        score=0.3,
        threshold=0.6,
        passed=0,
    )

    # Setup relationships
    run_a.test_cases = [test_case_a]
    test_case_a.llm_responses = [response_a]
    response_a.metric_scores = [metric_a1, metric_a2]

    # Test case for run B
    test_case_b = TestCase(
        id=2,
        evaluation_run_id=2,
        prompt_category="direct_lookup",
        prompt_text="Find the net zero clause",
    )

    response_b = LLMResponse(
        id=2,
        test_case_id=2,
        provider_name="openai",
        model_name="gpt-4o",
        response_text="The Chancery Lane Project provides...",
        execution_time_ms=100,
    )

    metric_b1 = MetricScore(
        id=3,
        llm_response_id=2,
        metric_name="tclp_citation",
        metric_category="tclp_custom",
        score=1.0,
        threshold=0.8,
        passed=1,
    )

    metric_b2 = MetricScore(
        id=4,
        llm_response_id=2,
        metric_name="taxonomy_awareness",
        metric_category="tclp_custom",
        score=0.6,
        threshold=0.6,
        passed=1,
    )

    # Setup relationships
    run_b.test_cases = [test_case_b]
    test_case_b.llm_responses = [response_b]
    response_b.metric_scores = [metric_b1, metric_b2]

    # Mock query methods
    def mock_filter_by(**kwargs):
        if kwargs.get("version_tag") == "pre-jsonld":
            result = MagicMock()
            result.first.return_value = run_a
            return result
        elif kwargs.get("version_tag") == "post-jsonld":
            result = MagicMock()
            result.first.return_value = run_b
            return result
        elif kwargs.get("id") == 1:
            result = MagicMock()
            result.first.return_value = run_a
            return result
        elif kwargs.get("id") == 2:
            result = MagicMock()
            result.first.return_value = run_b
            return result
        # Return None for nonexistent runs
        result = MagicMock()
        result.first.return_value = None
        return result

    session.query.return_value.filter_by = mock_filter_by
    manager.get_session.return_value.__enter__.return_value = session

    return manager


def test_report_generator_initialization(mock_db_manager):
    """Test report generator initializes with database manager."""
    generator = ReportGenerator(db_manager=mock_db_manager)
    assert generator.db_manager == mock_db_manager


def test_report_generator_finds_evaluation_runs(mock_db_manager):
    """Test report generator finds evaluation runs by version tag."""
    generator = ReportGenerator(db_manager=mock_db_manager)

    run_a, run_b = generator.get_evaluation_runs("pre-jsonld", "post-jsonld")

    assert run_a.version_tag == "pre-jsonld"
    assert run_b.version_tag == "post-jsonld"


def test_report_generator_calculates_improvements(mock_db_manager):
    """Test report generator calculates metric improvements."""
    generator = ReportGenerator(db_manager=mock_db_manager)

    run_a, run_b = generator.get_evaluation_runs("pre-jsonld", "post-jsonld")
    comparison = generator.calculate_comparison(run_a, run_b)

    # Should have overall metrics
    assert "overall" in comparison
    assert "tclp_citation" in comparison["overall"]
    assert "taxonomy_awareness" in comparison["overall"]

    # Check improvements (0.0 -> 1.0 for tclp_citation)
    assert comparison["overall"]["tclp_citation"]["baseline"] == 0.0
    assert comparison["overall"]["tclp_citation"]["comparison"] == 1.0
    assert comparison["overall"]["tclp_citation"]["improvement"] == 1.0


def test_report_generator_markdown_output(mock_db_manager, tmp_path):
    """Test report generator creates markdown report."""
    generator = ReportGenerator(db_manager=mock_db_manager)

    output_path = tmp_path / "report.md"
    generator.generate_report(
        version_a="pre-jsonld",
        version_b="post-jsonld",
        output_path=output_path,
        format="markdown",
    )

    assert output_path.exists()

    # Check report contains expected sections
    content = output_path.read_text()
    assert "# Evaluation Comparison Report" in content
    assert "pre-jsonld" in content
    assert "post-jsonld" in content
    assert "tclp_citation" in content
    assert "taxonomy_awareness" in content


def test_report_generator_handles_missing_runs(mock_db_manager):
    """Test report generator raises error for missing runs."""
    generator = ReportGenerator(db_manager=mock_db_manager)

    with pytest.raises(ValueError, match="Evaluation run not found"):
        generator.get_evaluation_runs("nonexistent", "post-jsonld")
