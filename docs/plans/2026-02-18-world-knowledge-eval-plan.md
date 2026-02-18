# World Knowledge Eval Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Standalone script that sends 2 prompts to ~30 models via OpenRouter (no web search) to measure which models know about The Chancery Lane Project and how many of its clauses they can name.

**Architecture:** Single script `scripts/world_knowledge_eval.py` with a companion `scripts/world_knowledge_models.yaml` config. Ground truth is parsed at runtime from the `clauses/` folder (YAML frontmatter). Results are written to `reports/world-knowledge-YYYY-MM-DD.md`. No changes to `src/`.

**Tech Stack:** Python 3.13, `openai` SDK (OpenAI-compatible, pointed at OpenRouter), `httpx` for async, `pyyaml` for frontmatter parsing, `python-dotenv` for API key, `rich` for progress output. All already installed in `.venv`.

---

### Task 1: Create the model config YAML

**Files:**
- Create: `scripts/world_knowledge_models.yaml`

**Step 1: Create the config file**

```yaml
# World Knowledge Eval — Model List
# All models run WITHOUT web search (pure training knowledge)

models:
  # Frontier / commercial
  - openai/gpt-4o
  - openai/gpt-4o-mini
  - openai/gpt-4-turbo
  - openai/gpt-3.5-turbo
  - openai/o1-mini
  - anthropic/claude-3.5-sonnet
  - anthropic/claude-3-opus
  - anthropic/claude-3-haiku
  - anthropic/claude-3.5-haiku
  - google/gemini-2.0-flash-001
  - google/gemini-pro-1.5
  - google/gemini-flash-1.5
  - x-ai/grok-2
  - amazon/nova-pro-v1

  # Large open-weight
  - meta-llama/llama-3.3-70b-instruct
  - meta-llama/llama-3.1-405b-instruct
  - meta-llama/llama-3.1-70b-instruct
  - mistralai/mistral-large
  - mistralai/mixtral-8x22b-instruct
  - mistralai/mixtral-8x7b-instruct
  - qwen/qwen-2.5-72b-instruct
  - cohere/command-r-plus
  - cohere/command-r
  - databricks/dbrx-instruct
  - nvidia/llama-3.1-nemotron-70b-instruct
  - deepseek/deepseek-chat
  - deepseek/deepseek-r1

  # Smaller models
  - mistralai/mistral-7b-instruct
  - microsoft/phi-4
  - qwen/qwen-2.5-7b-instruct
  - allenai/olmo-2-0325-32b-instruct
  - 01-ai/yi-large

settings:
  max_tokens: 1000
  temperature: 0
  delay_between_requests: 3  # seconds, ~20 req/min
```

**Step 2: Commit**

```bash
git add scripts/world_knowledge_models.yaml
git commit -m "feat: add world knowledge eval model config"
```

---

### Task 2: Ground truth loader (TDD)

**Files:**
- Create: `tests/unit/test_world_knowledge_eval.py`
- Create: `scripts/world_knowledge_eval.py` (skeleton + loader only)

**Step 1: Write the failing tests**

Create `tests/unit/test_world_knowledge_eval.py`:

```python
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
    assert "Green Termination Provision" in result["names"]
    assert "Annie's Clause" in result["names"]


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
```

**Step 2: Run tests to confirm they fail**

```bash
cd /Users/felix/Sites/tclp/llm_eval && .venv/bin/pytest tests/unit/test_world_knowledge_eval.py -v
```

Expected: `ModuleNotFoundError: No module named 'world_knowledge_eval'`

**Step 3: Create the script skeleton with just the loader**

Create `scripts/world_knowledge_eval.py`:

```python
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
```

**Step 4: Run tests to confirm they pass**

```bash
cd /Users/felix/Sites/tclp/llm_eval && .venv/bin/pytest tests/unit/test_world_knowledge_eval.py -v
```

Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add scripts/world_knowledge_eval.py tests/unit/test_world_knowledge_eval.py
git commit -m "feat: add world knowledge eval skeleton with ground truth loader"
```

---

### Task 3: Scoring functions (TDD)

**Files:**
- Modify: `tests/unit/test_world_knowledge_eval.py` (add tests)
- Modify: `scripts/world_knowledge_eval.py` (add scoring functions)

**Step 1: Write the failing tests**

Append to `tests/unit/test_world_knowledge_eval.py`:

```python
from world_knowledge_eval import detect_awareness, score_enumeration


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
```

Add `import pytest` at the top of the test file.

**Step 2: Run tests to confirm they fail**

```bash
cd /Users/felix/Sites/tclp/llm_eval && .venv/bin/pytest tests/unit/test_world_knowledge_eval.py -v
```

Expected: ImportError on `detect_awareness`, `score_enumeration`

**Step 3: Implement scoring functions**

Append to `scripts/world_knowledge_eval.py`:

```python
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
        or "\btclp\b" in text_lower  # handled below with regex
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
        cleaned = re.sub(r"^\s*[-*•\d]+[.)]\s*", "", line).strip()
        # Strip surrounding quotes
        cleaned = cleaned.strip("\"'""''")
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
```

**Step 4: Run tests to confirm they pass**

```bash
cd /Users/felix/Sites/tclp/llm_eval && .venv/bin/pytest tests/unit/test_world_knowledge_eval.py -v
```

Expected: all tests PASS

**Step 5: Commit**

```bash
git add scripts/world_knowledge_eval.py tests/unit/test_world_knowledge_eval.py
git commit -m "feat: add TCLP awareness detection and clause scoring functions"
```

---

### Task 4: Data structures and report generator (TDD)

**Files:**
- Modify: `tests/unit/test_world_knowledge_eval.py` (add tests)
- Modify: `scripts/world_knowledge_eval.py` (add dataclasses + report generator)

**Step 1: Write the failing tests**

Append to `tests/unit/test_world_knowledge_eval.py`:

```python
from world_knowledge_eval import ModelResult, generate_report


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
```

**Step 2: Run tests to confirm they fail**

```bash
cd /Users/felix/Sites/tclp/llm_eval && .venv/bin/pytest tests/unit/test_world_knowledge_eval.py::test_generate_report_contains_model_ids -v
```

Expected: `ImportError: cannot import name 'ModelResult'`

**Step 3: Implement dataclass and report generator**

Append to `scripts/world_knowledge_eval.py`:

```python
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
```

**Step 4: Run all tests**

```bash
cd /Users/felix/Sites/tclp/llm_eval && .venv/bin/pytest tests/unit/test_world_knowledge_eval.py -v
```

Expected: all tests PASS

**Step 5: Commit**

```bash
git add scripts/world_knowledge_eval.py tests/unit/test_world_knowledge_eval.py
git commit -m "feat: add ModelResult dataclass and markdown report generator"
```

---

### Task 5: OpenRouter API client and main orchestrator

**Files:**
- Modify: `scripts/world_knowledge_eval.py` (add API client + main)

No unit tests for the API client (would require live credentials); instead we add a dry-run mode that can be tested manually.

**Step 1: Append the API client and main orchestrator to the script**

Append to `scripts/world_knowledge_eval.py`:

```python
PROMPT_AWARENESS = (
    "What is The Chancery Lane Project? "
    "Please describe what it does, what it publishes, and who it is aimed at."
)

PROMPT_ENUMERATION = (
    "List every clause, guide, questionnaire, handbook, or other resource "
    "published by The Chancery Lane Project that you know about. "
    "For each one, give its full name. "
    "Do not search the web — use only your training knowledge."
)


async def call_model(
    client: AsyncOpenAI,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Call a model via OpenRouter and return the response text.
    Raises on API error (caller handles).
    """
    response = await client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


async def eval_model(
    client: AsyncOpenAI,
    model_id: str,
    ground_truth_names: set,
    settings: dict,
    delay: float,
) -> ModelResult:
    """Run both prompts against a single model and return scored result."""
    max_tokens = settings.get("max_tokens", 1000)
    temperature = settings.get("temperature", 0)

    try:
        # Prompt 1: Awareness
        awareness_response = await call_model(
            client, model_id, PROMPT_AWARENESS, max_tokens, temperature
        )
        await asyncio.sleep(delay)

        # Prompt 2: Enumeration
        enumeration_response = await call_model(
            client, model_id, PROMPT_ENUMERATION, max_tokens, temperature
        )

        knows_tclp = detect_awareness(awareness_response)
        scores = score_enumeration(enumeration_response, ground_truth_names)

        return ModelResult(
            model_id=model_id,
            awareness_response=awareness_response,
            enumeration_response=enumeration_response,
            knows_tclp=knows_tclp,
            real_count=scores["real_count"],
            real_matches=scores["real_matches"],
            hallucination_count=scores["hallucination_count"],
            hallucinations=scores["hallucinations"],
            accuracy=scores["accuracy"],
        )

    except Exception as e:
        return ModelResult(
            model_id=model_id,
            awareness_response="",
            enumeration_response="",
            knows_tclp=False,
            real_count=0,
            real_matches=[],
            hallucination_count=0,
            hallucinations=[],
            accuracy=None,
            error=str(e),
        )


async def run_eval(models: list, settings: dict, dry_run: bool = False):
    """Main evaluation loop."""
    import os

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENROUTER_API_KEY not set in .env[/red]")
        sys.exit(1)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    ground_truth = load_ground_truth(CLAUSES_DIR)
    ground_truth_names = ground_truth["names"]
    total_clauses = ground_truth["total_clauses"]
    delay = float(settings.get("delay_between_requests", 3))

    console.print(f"\n[bold]TCLP World Knowledge Eval[/bold]")
    console.print(f"Models: {len(models)}  |  Ground truth clauses: {total_clauses}")
    console.print(f"Mode: {'[yellow]DRY RUN[/yellow]' if dry_run else '[green]LIVE[/green]'}\n")

    if dry_run:
        console.print("[yellow]Dry run — no API calls made.[/yellow]")
        console.print(f"Would test {len(models)} models:")
        for m in models:
            console.print(f"  • {m}")
        return

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running eval...", total=len(models))

        for model_id in models:
            progress.update(task, description=f"[cyan]{model_id}[/cyan]")
            result = await eval_model(
                client, model_id, ground_truth_names, settings, delay
            )
            results.append(result)

            # Live summary per model
            status = "[green]✓[/green]" if result.knows_tclp else "[red]✗[/red]"
            err = f" [red]ERROR: {result.error[:60]}[/red]" if result.error else ""
            console.print(
                f"  {status} {model_id} — "
                f"{result.real_count} real, {result.hallucination_count} hallucinated{err}"
            )

            progress.advance(task)
            await asyncio.sleep(delay)

    # Write report
    REPORTS_DIR.mkdir(exist_ok=True)
    run_date = date.today().isoformat()
    report_path = REPORTS_DIR / f"world-knowledge-{run_date}.md"
    report = generate_report(results, total_clauses=total_clauses, run_date=run_date)
    report_path.write_text(report, encoding="utf-8")

    console.print(f"\n[bold green]Done![/bold green] Report saved to: {report_path}")
    console.print(
        f"Summary: {sum(1 for r in results if r.knows_tclp)}/{len(results)} models know TCLP"
    )


def main():
    """Entry point."""
    dry_run = "--dry-run" in sys.argv

    config_path = CONFIG_FILE
    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/red]")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    models = config.get("models", [])
    settings = config.get("settings", {})

    if not models:
        console.print("[red]No models configured in world_knowledge_models.yaml[/red]")
        sys.exit(1)

    asyncio.run(run_eval(models, settings, dry_run=dry_run))


if __name__ == "__main__":
    main()
```

**Step 2: Run the full test suite to confirm nothing is broken**

```bash
cd /Users/felix/Sites/tclp/llm_eval && .venv/bin/pytest tests/unit/test_world_knowledge_eval.py -v
```

Expected: all tests PASS

**Step 3: Run the dry-run to verify the script works without API calls**

```bash
cd /Users/felix/Sites/tclp/llm_eval && .venv/bin/python scripts/world_knowledge_eval.py --dry-run
```

Expected output:
```
TCLP World Knowledge Eval
Models: 32  |  Ground truth clauses: 183
Mode: DRY RUN

Dry run — no API calls made.
Would test 32 models:
  • openai/gpt-4o
  ...
```

**Step 4: Commit**

```bash
git add scripts/world_knowledge_eval.py
git commit -m "feat: add OpenRouter client and main orchestrator for world knowledge eval"
```

---

### Task 6: Live run and report review

**Step 1: Run against a single model first to verify end-to-end**

Temporarily edit `scripts/world_knowledge_models.yaml` to contain just one model:

```yaml
models:
  - anthropic/claude-3.5-sonnet
settings:
  max_tokens: 1000
  temperature: 0
  delay_between_requests: 2
```

Run:
```bash
cd /Users/felix/Sites/tclp/llm_eval && .venv/bin/python scripts/world_knowledge_eval.py
```

Verify: a report appears in `reports/world-knowledge-YYYY-MM-DD.md` with correct structure.

**Step 2: Restore the full model list**

Revert `scripts/world_knowledge_models.yaml` to the full 32-model list from Task 1.

**Step 3: Run the full eval**

```bash
cd /Users/felix/Sites/tclp/llm_eval && .venv/bin/python scripts/world_knowledge_eval.py
```

This will take ~5-10 minutes (64 API calls with 3s delays). The progress bar shows live results.

**Step 4: Review the report**

Open `reports/world-knowledge-YYYY-MM-DD.md` and verify:
- Summary table is populated and sorted correctly
- Statistical overview percentages look sensible
- Per-model detail sections contain the raw responses

**Step 5: Commit the report**

```bash
git add reports/world-knowledge-*.md scripts/world_knowledge_models.yaml
git commit -m "feat: add world knowledge eval results"
```
