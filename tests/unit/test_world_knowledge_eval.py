"""Tests for world knowledge eval script."""
import sys
from pathlib import Path

# Add scripts/ to path so we can import the standalone script
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from world_knowledge_eval import load_ground_truth


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
