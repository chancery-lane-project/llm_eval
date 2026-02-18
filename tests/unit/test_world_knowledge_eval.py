"""Tests for world knowledge eval script."""
import sys
import pytest
from pathlib import Path

# Add scripts/ to path so we can import the standalone script
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from world_knowledge_eval import (
    load_ground_truth,
    detect_awareness,
    score_enumeration,
    ModelResult,
    generate_report,
)


def test_load_ground_truth_returns_dict_with_names_set(tmp_path):
    """load_ground_truth returns a dict with a 'names' set."""
    clause_file = tmp_path / "test-clause.md"
    clause_file.write_text(
        "---\n"
        "title: Green Termination Provision\n"
        "eleventyComputed:\n"
        '  clause_child_name: "Annie\'s Clause"\n'
        "---\n\n# Green Termination Provision\n"
    )
    result = load_ground_truth(tmp_path)
    assert isinstance(result["names"], set)
    assert "green termination provision" in result["names"]
    assert "annie's clause" in result["names"]


def test_load_ground_truth_lowercases_names(tmp_path):
    """All names stored in lowercase for case-insensitive matching."""
    clause_file = tmp_path / "test-clause.md"
    clause_file.write_text(
        "---\n"
        "title: Net Zero Culture Employment Handbook\n"
        "eleventyComputed:\n"
        '  clause_child_name: "Elliot\'s Handbook"\n'
        "---\n"
    )
    result = load_ground_truth(tmp_path)
    assert "net zero culture employment handbook" in result["names"]
    assert "elliot's handbook" in result["names"]


def test_load_ground_truth_handles_missing_child_name(tmp_path):
    """Files without clause_child_name are handled gracefully."""
    clause_file = tmp_path / "test-clause.md"
    clause_file.write_text(
        "---\n"
        "title: Some Guide\n"
        "eleventyComputed:\n"
        "  clause_summary: A summary.\n"
        "---\n"
    )
    result = load_ground_truth(tmp_path)
    assert "some guide" in result["names"]


def test_load_ground_truth_counts_total_clauses(tmp_path):
    """load_ground_truth returns the total number of source files."""
    for i in range(3):
        f = tmp_path / f"clause-{i}.md"
        f.write_text(f"---\ntitle: Clause {i}\neleventyComputed:\n  clause_child_name: \"Child {i}\"\n---\n")
    result = load_ground_truth(tmp_path)
    assert result["total_clauses"] == 3


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


# --- score_enumeration ---

def test_score_enumeration_counts_real_clauses():
    ground_truth = {"net zero culture employment handbook", "elliot's handbook",
                    "annie's clause", "green termination provision"}
    text = "- Elliot's Handbook\n- Annie's Clause\n- Some Made Up Clause"
    result = score_enumeration(text, ground_truth)
    assert result["real_count"] == 2
    assert "elliot's handbook" in result["real_matches"]
    assert "annie's clause" in result["real_matches"]


def test_score_enumeration_counts_hallucinations():
    ground_truth = {"annie's clause"}
    # "Charlie's Clause" looks like a TCLP clause name but isn't real
    text = "- Annie's Clause\n- Charlie's Clause\n- Some Random Thing"
    result = score_enumeration(text, ground_truth)
    assert result["hallucination_count"] == 1
    assert "charlie's clause" in result["hallucinations"]


def test_score_enumeration_no_matches_returns_zeros():
    ground_truth = {"annie's clause", "elliot's handbook"}
    text = "I don't know any clauses from that organisation."
    result = score_enumeration(text, ground_truth)
    assert result["real_count"] == 0
    assert result["hallucination_count"] == 0


def test_score_enumeration_accuracy_calculation():
    ground_truth = {"annie's clause", "elliot's handbook"}
    text = "- Annie's Clause\n- Fake Clause\n- Another Invented Clause"
    result = score_enumeration(text, ground_truth)
    # 1 real, 2 hallucinated → accuracy = 1/3
    assert result["accuracy"] == pytest.approx(1 / 3)


def test_score_enumeration_accuracy_none_when_nothing_found():
    ground_truth = {"annie's clause"}
    text = "No relevant clauses found."
    result = score_enumeration(text, ground_truth)
    assert result["accuracy"] is None


# --- ModelResult + generate_report ---


def _make_result(model_id, knows_tclp, real_count, hallucination_count):
    return ModelResult(
        model_id=model_id,
        awareness_response="Some response text.",
        enumeration_response="- Annie's Clause\n- Fake Clause",
        knows_tclp=knows_tclp,
        real_count=real_count,
        real_matches=["annie's clause"] * real_count,
        hallucination_count=hallucination_count,
        hallucinations=["fake clause"] * hallucination_count,
        accuracy=(real_count / (real_count + hallucination_count))
            if (real_count + hallucination_count) > 0 else None,
        error=None,
    )


def test_generate_report_contains_model_ids():
    results = [
        _make_result("openai/gpt-4o", True, 5, 1),
        _make_result("mistralai/mistral-7b-instruct", False, 0, 0),
    ]
    report = generate_report(results, total_clauses=183, run_date="2026-02-18")
    assert "openai/gpt-4o" in report
    assert "mistralai/mistral-7b-instruct" in report


def test_generate_report_contains_statistics_section():
    results = [
        _make_result("openai/gpt-4o", True, 5, 1),
        _make_result("mistralai/mistral-7b-instruct", False, 0, 0),
    ]
    report = generate_report(results, total_clauses=183, run_date="2026-02-18")
    assert "Statistical Overview" in report
    assert "50%" in report  # 1 of 2 models knows TCLP


def test_generate_report_sorted_by_clauses_found():
    results = [
        _make_result("model-a", True, 2, 0),
        _make_result("model-b", True, 10, 1),
        _make_result("model-c", False, 0, 0),
    ]
    report = generate_report(results, total_clauses=183, run_date="2026-02-18")
    # model-b should appear before model-a in the table
    assert report.index("model-b") < report.index("model-a")


def test_generate_report_shows_check_and_cross():
    results = [
        _make_result("aware-model", True, 3, 1),
        _make_result("unaware-model", False, 0, 0),
    ]
    report = generate_report(results, total_clauses=183, run_date="2026-02-18")
    assert "✓" in report  # knows TCLP
    assert "✗" in report  # doesn't know TCLP


def test_generate_report_contains_per_model_detail():
    results = [_make_result("openai/gpt-4o", True, 3, 1)]
    report = generate_report(results, total_clauses=183, run_date="2026-02-18")
    assert "Some response text." in report  # awareness response verbatim
