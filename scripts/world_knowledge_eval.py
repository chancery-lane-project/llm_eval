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
