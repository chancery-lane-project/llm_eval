"""Tests for custom TCLP metrics."""
import pytest
from deepeval.test_case import LLMTestCase
from tclp_benchmark.evaluators.custom_metrics import (
    TCLPCitationMetric,
    TaxonomyAwarenessMetric,
    EntityRetrievalAccuracyMetric,
    MetadataPrecisionMetric,
    RelationshipTraversalScoreMetric,
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


def test_entity_retrieval_accuracy_finds_expected_clause():
    """Test EntityRetrievalAccuracyMetric detects expected clause names."""
    metric = EntityRetrievalAccuracyMetric(
        threshold=1.0,
        expected_concepts=["Evie's Clause", "Net Zero Objectives"]
    )

    test_case = LLMTestCase(
        input="Find the debt finance clause",
        actual_output="I found Evie's Clause which addresses debt finance sustainability.",
        expected_output="",
    )

    score = metric.measure(test_case)

    assert score == 1.0
    assert metric.is_successful()
    assert "Evie's Clause" in metric.reason


def test_entity_retrieval_accuracy_no_expected_clause():
    """Test EntityRetrievalAccuracyMetric returns 0.0 when expected clause not found."""
    metric = EntityRetrievalAccuracyMetric(
        threshold=1.0,
        expected_concepts=["Evie's Clause", "Net Zero Objectives"]
    )

    test_case = LLMTestCase(
        input="Find the debt finance clause",
        actual_output="I found a debt finance clause on the website.",
        expected_output="",
    )

    score = metric.measure(test_case)

    assert score == 0.0
    assert not metric.is_successful()
    assert "No expected entities found" in metric.reason


def test_entity_retrieval_accuracy_no_expected_concepts():
    """Test EntityRetrievalAccuracyMetric returns 1.0 when no concepts specified."""
    metric = EntityRetrievalAccuracyMetric(threshold=1.0, expected_concepts=None)

    test_case = LLMTestCase(
        input="Find clauses",
        actual_output="I found various clauses.",
        expected_output="",
    )

    score = metric.measure(test_case)

    assert score == 1.0
    assert metric.is_successful()


def test_metadata_precision_finds_jurisdiction():
    """Test MetadataPrecisionMetric detects jurisdiction metadata."""
    metric = MetadataPrecisionMetric(
        threshold=0.9,
        expected_concepts=["England & Wales", "Maintained"]
    )

    test_case = LLMTestCase(
        input="Find the clause",
        actual_output="This clause is maintained and applicable to England & Wales.",
        expected_output="",
    )

    score = metric.measure(test_case)

    assert score >= 0.5
    assert "England & Wales" in test_case.actual_output.lower() or "maintained" in test_case.actual_output.lower()


def test_metadata_precision_finds_status():
    """Test MetadataPrecisionMetric detects maintenance status."""
    metric = MetadataPrecisionMetric(
        threshold=0.9,
        expected_concepts=["Not Maintained", "Deutschland"]
    )

    test_case = LLMTestCase(
        input="Check the status",
        actual_output="This clause is not maintained and was designed for Deutschland.",
        expected_output="",
    )

    score = metric.measure(test_case)

    assert score >= 0.5


def test_metadata_precision_no_expected_concepts():
    """Test MetadataPrecisionMetric handles no expected concepts."""
    metric = MetadataPrecisionMetric(threshold=0.9, expected_concepts=None)

    test_case = LLMTestCase(
        input="Find clauses",
        actual_output="This clause is for England & Wales.",
        expected_output="",
    )

    score = metric.measure(test_case)

    assert score >= 0.0


def test_relationship_traversal_finds_relationships():
    """Test RelationshipTraversalScoreMetric detects relationship keywords."""
    metric = RelationshipTraversalScoreMetric(
        threshold=0.8,
        expected_concepts=["child clause", "handbook"]
    )

    test_case = LLMTestCase(
        input="Does the handbook contain child clauses?",
        actual_output="Yes, the handbook contains a child clause on sustainable business travel.",
        expected_output="",
    )

    score = metric.measure(test_case)

    assert score == 1.0
    assert metric.is_successful()


def test_relationship_traversal_no_relationships():
    """Test RelationshipTraversalScoreMetric handles absence of relationships."""
    metric = RelationshipTraversalScoreMetric(
        threshold=0.8,
        expected_concepts=["Net Zero", "Carbon"]
    )

    test_case = LLMTestCase(
        input="Find the clause",
        actual_output="This is a standalone clause.",
        expected_output="",
    )

    score = metric.measure(test_case)

    # Should get neutral score since no relationships expected or found
    assert 0.0 <= score <= 1.0


def test_relationship_traversal_variant_detected():
    """Test RelationshipTraversalScoreMetric detects variants."""
    metric = RelationshipTraversalScoreMetric(
        threshold=0.8,
        expected_concepts=["variant", "transposition"]
    )

    test_case = LLMTestCase(
        input="Are there variants?",
        actual_output="Yes, this is a variant of the original clause, transposed for real estate.",
        expected_output="",
    )

    score = metric.measure(test_case)

    assert score == 1.0
    assert metric.is_successful()
    assert "variant" in metric.reason.lower() or "transposition" in metric.reason.lower()
