"""
World Knowledge Eval
====================
Tests which LLMs have The Chancery Lane Project in their training data,
and how many TCLP clauses they can name — using NO web search.

Usage:
    python scripts/world_knowledge_eval.py

Requirements:
    OPENROUTER_API_KEY in .env
"""
import re
import sys
import asyncio
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

load_dotenv()
console = Console()

# Paths (relative to project root, not scripts/)
PROJECT_ROOT = Path(__file__).parent.parent
CLAUSES_DIR = PROJECT_ROOT / "clauses"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIG_FILE = Path(__file__).parent / "world_knowledge_models.yaml"


def load_ground_truth(clauses_dir: Path) -> dict:
    """
    Parse all .md files in clauses_dir, extract title and clause_child_name.

    Returns:
        {
            "names": set of lowercase strings (titles + child names),
            "total_clauses": int
        }
    """
    names = set()
    total = 0

    for md_file in clauses_dir.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")

        # Extract YAML frontmatter between first two --- delimiters
        parts = text.split("---")
        if len(parts) < 3:
            continue

        try:
            frontmatter = yaml.safe_load(parts[1])
        except yaml.YAMLError:
            continue

        if not isinstance(frontmatter, dict):
            continue

        total += 1

        # Add title
        title = frontmatter.get("title")
        if title:
            names.add(str(title).lower())

        # Add clause_child_name (nested under eleventyComputed)
        computed = frontmatter.get("eleventyComputed", {})
        if isinstance(computed, dict):
            child_name = computed.get("clause_child_name")
            if child_name:
                names.add(str(child_name).lower())

    return {"names": names, "total_clauses": total}


# Keywords that flag something as a "clause-like" name (for hallucination detection)
CLAUSE_KEYWORDS = re.compile(
    r"\b(clause|ddq|handbook|annex|minutes|protocol|guide|checklist|questionnaire)\b",
    re.IGNORECASE,
)


def detect_awareness(text: str) -> bool:
    """Return True if response mentions TCLP in any recognisable form."""
    text_lower = text.lower()
    return (
        "the chancery lane project" in text_lower
        or "chancerylaneproject.org" in text_lower
        or bool(re.search(r"\btclp\b", text_lower))
    )


def score_enumeration(text: str, ground_truth_names: set) -> dict:
    """
    Score a model's enumeration response against ground truth.

    Strategy: scan every line; for each ground truth name, check if it
    appears anywhere in the response (substring match). For hallucinations,
    extract lines that look like clause names but don't match anything.

    Returns:
        {
            real_count: int,
            real_matches: list[str],
            hallucination_count: int,
            hallucinations: list[str],
            accuracy: float | None  (None if nothing clause-like found)
        }
    """
    text_lower = text.lower()

    # Find which ground truth names appear in the response
    real_matches = [name for name in ground_truth_names if name in text_lower]

    # Extract candidate names: lines that contain clause-like keywords
    candidates = []
    for line in text.splitlines():
        # Strip list markers (-, *, 1., •, etc.)
        cleaned = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", line).strip()
        # Strip surrounding quotes
        cleaned = cleaned.strip("\"'\u201c\u201d\u2018\u2019")
        if cleaned and CLAUSE_KEYWORDS.search(cleaned):
            candidates.append(cleaned.lower())

    # Hallucinations: candidates not matching any ground truth name
    hallucinations = [
        c for c in candidates
        if c not in ground_truth_names
        and not any(c in gt or gt in c for gt in ground_truth_names)
    ]
    # Deduplicate
    hallucinations = list(dict.fromkeys(hallucinations))

    total = len(real_matches) + len(hallucinations)
    accuracy = len(real_matches) / total if total > 0 else None

    return {
        "real_count": len(real_matches),
        "real_matches": real_matches,
        "hallucination_count": len(hallucinations),
        "hallucinations": hallucinations,
        "accuracy": accuracy,
    }


@dataclass
class ModelResult:
    """Result of running both prompts against a single model."""
    model_id: str
    awareness_response: str
    enumeration_response: str
    knows_tclp: bool
    real_count: int
    real_matches: list
    hallucination_count: int
    hallucinations: list
    accuracy: Optional[float]
    error: Optional[str] = None


def generate_report(
    results: list,
    total_clauses: int,
    run_date: str,
) -> str:
    """Generate markdown report from eval results."""
    lines = []

    # Header
    lines.append(f"# TCLP World Knowledge Eval")
    lines.append(f"")
    lines.append(f"**Date:** {run_date}  ")
    lines.append(f"**Models tested:** {len(results)}  ")
    lines.append(f"**Ground truth clauses:** {total_clauses}  ")
    lines.append(f"**Method:** Pure training knowledge (no web search)  ")
    lines.append(f"")

    # --- Statistical Overview ---
    knows_count = sum(1 for r in results if r.knows_tclp)
    knows_pct = round(knows_count / len(results) * 100) if results else 0
    real_counts = [r.real_count for r in results if not r.error]
    mean_clauses = round(sum(real_counts) / len(real_counts), 1) if real_counts else 0
    sorted_by_real = sorted(results, key=lambda r: r.real_count, reverse=True)
    best = sorted_by_real[0] if sorted_by_real else None
    # Most accurate: highest accuracy among models that found >=3 clauses
    accurate_candidates = [r for r in results if r.real_count >= 3 and r.accuracy is not None]
    most_accurate = max(accurate_candidates, key=lambda r: r.accuracy) if accurate_candidates else None

    lines.append(f"## Statistical Overview")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Models that know TCLP | {knows_count}/{len(results)} ({knows_pct}%) |")
    lines.append(f"| Mean clauses found (real) | {mean_clauses} |")
    lines.append(f"| Median clauses found | {_median(real_counts)} |")
    if best:
        lines.append(f"| Best performer (most clauses) | {best.model_id} ({best.real_count} clauses) |")
    if most_accurate:
        lines.append(f"| Most accurate (fewest hallucinations) | {most_accurate.model_id} ({round(most_accurate.accuracy * 100)}%) |")
    lines.append(f"")

    # Family breakdown
    families = {}
    for r in results:
        family = r.model_id.split("/")[0]
        if family not in families:
            families[family] = {"total": 0, "aware": 0, "clauses": []}
        families[family]["total"] += 1
        if r.knows_tclp:
            families[family]["aware"] += 1
        families[family]["clauses"].append(r.real_count)

    lines.append(f"### By Model Family")
    lines.append(f"")
    lines.append(f"| Family | Models | Aware | Avg Clauses |")
    lines.append(f"|--------|--------|-------|-------------|")
    for family, data in sorted(families.items()):
        avg = round(sum(data["clauses"]) / len(data["clauses"]), 1) if data["clauses"] else 0
        lines.append(f"| {family} | {data['total']} | {data['aware']} | {avg} |")
    lines.append(f"")

    # --- Summary Table ---
    lines.append(f"## Results by Model")
    lines.append(f"")
    lines.append(f"Sorted by clauses found (real), descending.")
    lines.append(f"")
    lines.append(f"| Model | Knows TCLP | Clauses Found | Real | Hallucinated | Accuracy |")
    lines.append(f"|-------|-----------|---------------|------|--------------|----------|")

    for r in sorted_by_real:
        if r.error:
            lines.append(f"| {r.model_id} | ⚠ Error | — | — | — | — |")
            continue
        aware = "✓" if r.knows_tclp else "✗"
        acc = f"{round(r.accuracy * 100)}%" if r.accuracy is not None else "—"
        lines.append(
            f"| {r.model_id} | {aware} | {r.real_count + r.hallucination_count} "
            f"| {r.real_count} | {r.hallucination_count} | {acc} |"
        )
    lines.append(f"")

    # --- Per-model detail ---
    lines.append(f"## Per-Model Detail")
    lines.append(f"")

    for r in sorted_by_real:
        lines.append(f"### {r.model_id}")
        lines.append(f"")
        if r.error:
            lines.append(f"**Error:** {r.error}")
            lines.append(f"")
            continue

        aware = "Yes" if r.knows_tclp else "No"
        lines.append(f"**Knows TCLP:** {aware}  ")
        lines.append(f"**Real clauses:** {r.real_count}  ")
        lines.append(f"**Hallucinations:** {r.hallucination_count}  ")
        if r.real_matches:
            lines.append(f"**Matched:** {', '.join(r.real_matches[:10])}"
                         + (" …" if len(r.real_matches) > 10 else ""))
        lines.append(f"")
        lines.append(f"<details>")
        lines.append(f"<summary>Awareness response</summary>")
        lines.append(f"")
        lines.append(r.awareness_response)
        lines.append(f"")
        lines.append(f"</details>")
        lines.append(f"")
        lines.append(f"<details>")
        lines.append(f"<summary>Enumeration response</summary>")
        lines.append(f"")
        lines.append(r.enumeration_response)
        lines.append(f"")
        lines.append(f"</details>")
        lines.append(f"")

    return "\n".join(lines)


def _median(values: list) -> float:
    """Return median of a list of numbers."""
    if not values:
        return 0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
