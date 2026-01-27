"""CLI commands for TCLP Benchmark."""
import click
from pathlib import Path
from rich.console import Console

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """TCLP Semantic Benchmark - Measure LLM discovery impact from semantic markup."""
    pass


@cli.command()
@click.option(
    "--version-tag",
    required=True,
    help="Version tag for this evaluation run (e.g., 'pre-jsonld', 'post-jsonld-v1')",
)
@click.option(
    "--prompts-config",
    type=click.Path(exists=True, path_type=Path),
    default=Path("config/prompts.yaml"),
    help="Path to prompts configuration file",
)
@click.option(
    "--providers-config",
    type=click.Path(exists=True, path_type=Path),
    default=Path("config/providers.yaml"),
    help="Path to providers configuration file",
)
@click.option(
    "--metrics-config",
    type=click.Path(exists=True, path_type=Path),
    default=Path("config/metrics.yaml"),
    help="Path to metrics configuration file",
)
@click.option(
    "--category",
    type=str,
    help="Run only prompts from this category (e.g., 'direct_lookup')",
)
@click.option(
    "--providers",
    type=str,
    help="Comma-separated list of providers to use (e.g., 'openai,anthropic')",
)
def run(version_tag, prompts_config, providers_config, metrics_config, category, providers):
    """Run evaluation with test prompts against LLM providers."""
    console.print(f"[bold blue]Running evaluation:[/bold blue] {version_tag}")
    console.print(f"  Prompts config: {prompts_config}")
    console.print(f"  Providers config: {providers_config}")
    console.print(f"  Metrics config: {metrics_config}")

    if category:
        console.print(f"  Filtering category: {category}")
    if providers:
        console.print(f"  Using providers: {providers}")

    # TODO: Implement evaluation workflow
    console.print("[yellow]Evaluation not yet implemented[/yellow]")


@cli.command()
@click.option(
    "--version-a",
    required=True,
    help="First version tag to compare (baseline)",
)
@click.option(
    "--version-b",
    required=True,
    help="Second version tag to compare (comparison)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for comparison report",
)
@click.option(
    "--format",
    type=click.Choice(["markdown", "html"]),
    default="markdown",
    help="Report format",
)
def report(version_a, version_b, output, format):
    """Generate comparison report between two evaluation runs."""
    console.print(f"[bold blue]Generating {format} report[/bold blue]")
    console.print(f"  Comparing: {version_a} vs {version_b}")
    console.print(f"  Output: {output}")

    # TODO: Implement report generation
    console.print("[yellow]Report generation not yet implemented[/yellow]")


@cli.command()
@click.option(
    "--status",
    type=click.Choice(["running", "completed", "failed"]),
    help="Filter by status",
)
def list(status):
    """List all evaluation runs."""
    console.print("[bold blue]Evaluation Runs[/bold blue]")

    if status:
        console.print(f"  Filtering by status: {status}")

    # TODO: Implement listing
    console.print("[yellow]Listing not yet implemented[/yellow]")


if __name__ == "__main__":
    cli()
