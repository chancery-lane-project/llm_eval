"""
World Knowledge Eval
====================
Tests which LLMs have The Chancery Lane Project in their training data,
and how many TCLP clauses they believe have been published — using NO web search.

Usage:
    python scripts/world_knowledge_eval.py

Requirements:
    OPENROUTER_API_KEY in .env
"""
import os
import re
import sys
import asyncio
from dataclasses import dataclass
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
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIG_FILE = Path(__file__).parent / "world_knowledge_models.yaml"


def detect_awareness(text: str) -> bool:
    """Return True if response mentions TCLP in any recognisable form."""
    text_lower = text.lower()
    return (
        "the chancery lane project" in text_lower
        or "chancery lane project" in text_lower
        or "chancerylaneproject.org" in text_lower
        or bool(re.search(r"\btclp\b", text_lower))
    )


def extract_clause_count(text: str) -> Optional[int]:
    """
    Extract a number from a clause-count response.

    Looks for patterns like "183 clauses", "published 200", "around 150", etc.
    Returns the first plausible integer found, or None if none found.
    """
    # Match numbers adjacent to clause-related words
    match = re.search(
        r"\b(\d{1,4})\s*(?:clauses?|resources?|documents?|items?|publications?)\b",
        text,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1))

    # Match "published/released/has X" patterns
    match = re.search(
        r"(?:published|released|produced|has|have|contains?|includes?|over|around|approximately|about|roughly)\s+(\d{1,4})\b",
        text,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1))

    # Last resort: first standalone number in the response
    match = re.search(r"\b(\d{1,4})\b", text)
    if match:
        return int(match.group(1))

    return None


@dataclass
class ModelResult:
    """Result of running both prompts against a single model."""
    model_id: str
    awareness_response: str
    clause_count_response: str
    knows_tclp: bool
    clause_count_reported: Optional[int]
    error: Optional[str] = None


def generate_report(results: list, run_date: str) -> str:
    """Generate markdown report from eval results."""
    lines = []

    # Header
    lines.append("# TCLP World Knowledge Eval")
    lines.append("")
    lines.append(f"**Date:** {run_date}  ")
    lines.append(f"**Models tested:** {len(results)}  ")
    lines.append("**Method:** Pure training knowledge (no web search)  ")
    lines.append("")

    # --- Statistical Overview ---
    knows_count = sum(1 for r in results if r.knows_tclp)
    knows_pct = round(knows_count / len(results) * 100) if results else 0
    counts = [r.clause_count_reported for r in results if r.clause_count_reported is not None]
    mean_count = round(sum(counts) / len(counts), 1) if counts else None

    lines.append("## Statistical Overview")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Models that know TCLP | {knows_count}/{len(results)} ({knows_pct}%) |")
    if mean_count is not None:
        lines.append(f"| Mean clause count reported | {mean_count} |")
        lines.append(f"| Median clause count reported | {_median(counts)} |")
    lines.append("")

    # Family breakdown
    families: dict = {}
    for r in results:
        family = r.model_id.split("/")[0]
        if family not in families:
            families[family] = {"total": 0, "aware": 0}
        families[family]["total"] += 1
        if r.knows_tclp:
            families[family]["aware"] += 1

    lines.append("### By Model Family")
    lines.append("")
    lines.append("| Family | Models | Aware |")
    lines.append("|--------|--------|-------|")
    for family, data in sorted(families.items()):
        lines.append(f"| {family} | {data['total']} | {data['aware']} |")
    lines.append("")

    # --- Summary Table ---
    lines.append("## Results by Model")
    lines.append("")
    lines.append("Sorted by clause count reported, descending.")
    lines.append("")
    lines.append("| Model | Knows TCLP | Clauses Reported |")
    lines.append("|-------|-----------|-----------------|")

    sorted_results = sorted(
        results,
        key=lambda r: (r.clause_count_reported or 0),
        reverse=True,
    )

    for r in sorted_results:
        if r.error:
            lines.append(f"| {r.model_id} | ⚠ Error | — |")
            continue
        aware = "✓" if r.knows_tclp else "✗"
        count = str(r.clause_count_reported) if r.clause_count_reported is not None else "—"
        lines.append(f"| {r.model_id} | {aware} | {count} |")
    lines.append("")

    # --- Per-model detail ---
    lines.append("## Per-Model Detail")
    lines.append("")

    for r in sorted_results:
        lines.append(f"### {r.model_id}")
        lines.append("")
        if r.error:
            lines.append(f"**Error:** {r.error}")
            lines.append("")
            continue

        aware = "Yes" if r.knows_tclp else "No"
        lines.append(f"**Knows TCLP:** {aware}  ")
        count = str(r.clause_count_reported) if r.clause_count_reported is not None else "Unknown"
        lines.append(f"**Clauses reported:** {count}  ")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Awareness response</summary>")
        lines.append("")
        lines.append(r.awareness_response)
        lines.append("")
        lines.append("</details>")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Clause count response</summary>")
        lines.append("")
        lines.append(r.clause_count_response)
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


def _median(values: list) -> float:
    """Return median of a list of numbers."""
    if not values:
        return 0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


PROMPT_AWARENESS = (
    "What is The Chancery Lane Project? "
    "Please describe what it does, what it publishes, and who it is aimed at."
)

PROMPT_CLAUSE_COUNT = (
    "How many clauses has The Chancery Lane Project published? "
    "Give a number if you know it, or say you don't know. "
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
    settings: dict,
    delay: float,
) -> ModelResult:
    """Run both prompts against a single model and return result."""
    max_tokens = settings.get("max_tokens", 1000)
    temperature = settings.get("temperature", 0)

    try:
        # Prompt 1: Awareness
        awareness_response = await call_model(
            client, model_id, PROMPT_AWARENESS, max_tokens, temperature
        )
        await asyncio.sleep(delay)

        # Prompt 2: Clause count
        clause_count_response = await call_model(
            client, model_id, PROMPT_CLAUSE_COUNT, max_tokens, temperature
        )

        knows_tclp = detect_awareness(awareness_response)
        clause_count_reported = extract_clause_count(clause_count_response) if knows_tclp else None

        return ModelResult(
            model_id=model_id,
            awareness_response=awareness_response,
            clause_count_response=clause_count_response,
            knows_tclp=knows_tclp,
            clause_count_reported=clause_count_reported,
        )

    except Exception as e:
        return ModelResult(
            model_id=model_id,
            awareness_response="",
            clause_count_response="",
            knows_tclp=False,
            clause_count_reported=None,
            error=str(e),
        )


async def run_eval(models: list, settings: dict, dry_run: bool = False):
    """Main evaluation loop."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENROUTER_API_KEY not set in .env[/red]")
        sys.exit(1)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    delay = float(settings.get("delay_between_requests", 3))

    console.print("\n[bold]TCLP World Knowledge Eval[/bold]")
    console.print(f"Models: {len(models)}")
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
            result = await eval_model(client, model_id, settings, delay)
            results.append(result)

            # Live summary per model
            status = "[green]✓[/green]" if result.knows_tclp else "[red]✗[/red]"
            count_str = f"{result.clause_count_reported} clauses" if result.clause_count_reported is not None else "count unknown"
            err = f" [red]ERROR: {result.error[:60]}[/red]" if result.error else ""
            console.print(f"  {status} {model_id} — {count_str}{err}")

            progress.advance(task)

    # Write report
    REPORTS_DIR.mkdir(exist_ok=True)
    run_date = date.today().isoformat()
    report_path = REPORTS_DIR / f"world-knowledge-{run_date}.md"
    report = generate_report(results, run_date=run_date)
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
