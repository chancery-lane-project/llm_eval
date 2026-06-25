"""Tests for world knowledge eval script."""
import sys
import pytest
from pathlib import Path

# Add scripts/ to path so we can import the standalone script
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from world_knowledge_eval import (
    detect_awareness,
    extract_clause_count,
    ModelResult,
    generate_report,
)


# --- detect_awareness ---

def test_detect_awareness_finds_full_name():
    text = "The Chancery Lane Project is a UK-based legal organisation."
    assert detect_awareness(text) is True


def test_detect_awareness_finds_abbreviation():
    text = "TCLP publishes open-source climate contract clauses."
    assert detect_awareness(text) is True


def test_detect_awareness_finds_url():
    text = "Visit chancerylaneproject.org for more information."
    assert detect_awareness(text) is True


def test_detect_awareness_returns_false_when_absent():
    text = "I don't have any information about that organisation."
    assert detect_awareness(text) is False


def test_detect_awareness_is_case_insensitive():
    text = "the chancery lane project focuses on climate law."
    assert detect_awareness(text) is True


def test_detect_awareness_finds_name_without_article():
    text = "Chancery Lane Project has published several clauses."
    assert detect_awareness(text) is True


# --- extract_clause_count ---

def test_extract_clause_count_finds_number_before_clauses():
    assert extract_clause_count("They have published 183 clauses.") == 183


def test_extract_clause_count_finds_number_after_published():
    assert extract_clause_count("The project has published around 200 resources.") == 200


def test_extract_clause_count_finds_approximate():
    assert extract_clause_count("Approximately 150 clauses are available.") == 150


def test_extract_clause_count_returns_none_for_unknown():
    assert extract_clause_count("I don't know how many clauses they have published.") is None


def test_extract_clause_count_finds_over_pattern():
    assert extract_clause_count("The project has over 100 documents.") == 100


def test_extract_clause_count_finds_number_with_items():
    # "items" is a recognised keyword so 50 is extracted
    assert extract_clause_count("More than 50 items are listed.") == 50


def test_extract_clause_count_ignores_large_numbers():
    # Numbers > 9999 (4 digits) should not match — not a plausible clause count
    result = extract_clause_count("It was founded in 2019 and has published 50 clauses.")
    assert result == 50


# --- ModelResult + generate_report ---

def _make_result(model_id, knows_tclp, clause_count_reported):
    return ModelResult(
        model_id=model_id,
        awareness_response="Some response text.",
        clause_count_response="They have published about 150 clauses.",
        knows_tclp=knows_tclp,
        clause_count_reported=clause_count_reported,
        error=None,
    )


def test_generate_report_contains_model_ids():
    results = [
        _make_result("openai/gpt-4o", True, 150),
        _make_result("mistralai/mistral-7b-instruct", False, None),
    ]
    report = generate_report(results, run_date="2026-02-18")
    assert "openai/gpt-4o" in report
    assert "mistralai/mistral-7b-instruct" in report


def test_generate_report_contains_statistics_section():
    results = [
        _make_result("openai/gpt-4o", True, 150),
        _make_result("mistralai/mistral-7b-instruct", False, None),
    ]
    report = generate_report(results, run_date="2026-02-18")
    assert "Statistical Overview" in report
    assert "50%" in report  # 1 of 2 models knows TCLP


def test_generate_report_sorted_by_clause_count():
    results = [
        _make_result("model-a", True, 50),
        _make_result("model-b", True, 200),
        _make_result("model-c", False, None),
    ]
    report = generate_report(results, run_date="2026-02-18")
    # Check order using rows that include the clause count — these only appear
    # in the results table, not the family breakdown, so position is unambiguous
    assert report.index("| model-b | ✓ | 200 |") < report.index("| model-a | ✓ | 50 |")


def test_generate_report_shows_check_and_cross():
    results = [
        _make_result("aware-model", True, 100),
        _make_result("unaware-model", False, None),
    ]
    report = generate_report(results, run_date="2026-02-18")
    assert "✓" in report
    assert "✗" in report


def test_generate_report_contains_per_model_detail():
    results = [_make_result("openai/gpt-4o", True, 150)]
    report = generate_report(results, run_date="2026-02-18")
    assert "Some response text." in report  # awareness response verbatim


def test_generate_report_shows_clause_count():
    results = [_make_result("openai/gpt-4o", True, 183)]
    report = generate_report(results, run_date="2026-02-18")
    assert "183" in report
