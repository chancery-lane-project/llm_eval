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
        or "chancery lane project" in text_lower
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
        cleaned = cleaned.strip("\"'")
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
