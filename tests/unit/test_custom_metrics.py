"""Tests for custom TCLP metrics."""
import pytest
from deepeval.test_case import LLMTestCase
from tclp_benchmark.evaluators.custom_metrics import (
    TCLPCitationMetric,
    TaxonomyAwarenessMetric,
)


def test_tclp_citation_metric_detects_mention():
    """Test TCLPCitationMetric detects TCLP mentions."""
    metric = TCLPCitationMetric(threshold=0.8)

    test_case = LLMTestCase(
        input="Find the net zero clause",
        actual_output="The Chancery Lane Project provides a Net Zero Carbon Commitment clause.",
        expected_output="",
    )

    score = metric.measure(test_case)

    assert score == 1.0
    assert metric.is_successful()
    assert "TCLP mentioned" in metric.reason


def test_tclp_citation_metric_detects_acronym():
    """Test TCLPCitationMetric detects TCLP acronym."""
    metric = TCLPCitationMetric(threshold=0.8)

    test_case = LLMTestCase(
        input="Find clauses",
        actual_output="TCLP offers various climate-aligned contract clauses.",
        expected_output="",
    )

    score = metric.measure(test_case)

    assert score == 1.0
    assert metric.is_successful()


def test_tclp_citation_metric_no_mention():
    """Test TCLPCitationMetric returns 0.0 when not mentioned."""
    metric = TCLPCitationMetric(threshold=0.8)

    test_case = LLMTestCase(
        input="Find clauses",
        actual_output="There are various contract clauses available.",
        expected_output="",
    )

    score = metric.measure(test_case)

    assert score == 0.0
    assert not metric.is_successful()
    assert "not mentioned" in metric.reason


def test_taxonomy_awareness_metric_scores_mentions():
    """Test TaxonomyAwarenessMetric scores based on taxonomy term mentions."""
    metric = TaxonomyAwarenessMetric(threshold=0.6)

    # 3+ mentions should give 1.0
    test_case = LLMTestCase(
        input="Find clauses",
        actual_output="UK construction contracts for carbon reduction in commercial real estate.",
        expected_output="",
    )

    score = metric.measure(test_case)

    assert score == 1.0
    assert metric.is_successful()


def test_taxonomy_awareness_metric_partial_score():
    """Test TaxonomyAwarenessMetric gives partial score for few mentions."""
    metric = TaxonomyAwarenessMetric(threshold=0.6)

    # 2 mentions should give 0.6
    test_case = LLMTestCase(
        input="Find clauses",
        actual_output="UK construction contracts.",
        expected_output="",
    )

    score = metric.measure(test_case)

    assert score == 0.6
    assert metric.is_successful()


def test_taxonomy_awareness_metric_no_mentions():
    """Test TaxonomyAwarenessMetric returns 0.0 for no mentions."""
    metric = TaxonomyAwarenessMetric(threshold=0.6)

    test_case = LLMTestCase(
        input="Find clauses",
        actual_output="Various legal clauses exist.",
        expected_output="",
    )

    score = metric.measure(test_case)

    assert score == 0.0
    assert not metric.is_successful()
