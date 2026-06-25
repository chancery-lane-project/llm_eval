"""CLI commands for TCLP Benchmark."""
import os
import asyncio
import click
from pathlib import Path
from rich.console import Console

from ..config.loader import ConfigLoader
from ..storage.database import DatabaseManager
from ..evaluators.orchestrator import EvaluationOrchestrator
from ..reporting.report_generator import ReportGenerator

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
    "--description",
    type=str,
    help="Optional description for this evaluation run",
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
    help="Comma-separated list of providers to use (e.g., 'openai,anthropic,openrouter'). "
         "Defaults to all enabled providers in config.",
)
def run(version_tag, description, prompts_config, providers_config, metrics_config, category, providers):
    """Run evaluation with test prompts against LLM providers."""
    console.print(f"[bold blue]Running evaluation:[/bold blue] {version_tag}")
    console.print(f"  Prompts config: {prompts_config}")
    console.print(f"  Providers config: {providers_config}")
    console.print(f"  Metrics config: {metrics_config}")

    if category:
        console.print(f"  Filtering category: {category}")
    if providers:
        console.print(f"  Using providers: {providers}")

    try:
        # Load configurations
        with console.status("[bold blue]Loading configurations..."):
            prompts_cfg = ConfigLoader.load_prompts(prompts_config)
            providers_cfg = ConfigLoader.load_providers(providers_config)
            metrics_cfg = ConfigLoader.load_metrics(metrics_config)

        console.print("[green]✓[/green] Configurations loaded")

        # Get database URL from environment or use default
        db_url = os.getenv("DATABASE_URL", "sqlite:///data/tclp_benchmark.db")

        # Initialize database
        with console.status("[bold blue]Initialising database..."):
            db_manager = DatabaseManager(db_url)
            db_manager.create_tables()

        console.print("[green]✓[/green] Database initialized")

        # Parse provider filter
        provider_filter = providers.split(",") if providers else None

        # Run evaluation
        console.print("\n[bold blue]Starting evaluation...[/bold blue]")

        orchestrator = EvaluationOrchestrator(db_manager=db_manager, verbose=True)

        run_id = asyncio.run(
            orchestrator.run_evaluation(
                version_tag=version_tag,
                prompts_config=prompts_cfg,
                providers_config=providers_cfg,
                metrics_config=metrics_cfg,
                description=description,
                category_filter=category,
                provider_filter=provider_filter,
            )
        )

        console.print("\n[green]✓[/green] Evaluation completed successfully")
        console.print(f"[bold]Run ID:[/bold] {run_id}")
        console.print(f"[bold]Version Tag:[/bold] {version_tag}")

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] Configuration file not found: {e}")
        raise click.Abort()
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("\n[yellow]Tip:[/yellow] Make sure API keys are set in .env file:")
        console.print("  OPENAI_API_KEY=...")
        console.print("  ANTHROPIC_API_KEY=...")
        console.print("  GOOGLE_API_KEY=...")
        console.print("  OPENROUTER_API_KEY=...")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


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

    try:
        # Get database URL from environment or use default
        db_url = os.getenv("DATABASE_URL", "sqlite:///data/tclp_benchmark.db")

        # Initialize database
        with console.status("[bold blue]Connecting to database..."):
            db_manager = DatabaseManager(db_url)

        console.print("[green]✓[/green] Connected to database")

        # Generate report
        with console.status(f"[bold blue]Generating {format} report..."):
            generator = ReportGenerator(db_manager=db_manager)
            generator.generate_report(
                version_a=version_a,
                version_b=version_b,
                output_path=output,
                format=format,
            )

        console.print("\n[green]✓[/green] Report generated successfully")
        console.print(f"[bold]Output:[/bold] {output}")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


@cli.command()
@click.option(
    "--version-tag",
    required=True,
    help="Version tag to generate report for",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for report",
)
@click.option(
    "--format",
    type=click.Choice(["markdown", "html"]),
    default="html",
    help="Report format",
)
def report_single(version_tag, output, format):
    """Generate detailed report for a single evaluation run."""
    console.print(f"[bold blue]Generating {format} report for single run[/bold blue]")
    console.print(f"  Version: {version_tag}")
    console.print(f"  Output: {output}")

    try:
        # Get database URL from environment or use default
        db_url = os.getenv("DATABASE_URL", "sqlite:///data/tclp_benchmark.db")

        # Initialize database
        with console.status("[bold blue]Connecting to database..."):
            db_manager = DatabaseManager(db_url)

        console.print("[green]✓[/green] Connected to database")

        # Generate report
        with console.status(f"[bold blue]Generating {format} report..."):
            generator = ReportGenerator(db_manager=db_manager)
            generator.generate_single_run_report(
                version_tag=version_tag,
                output_path=output,
                format=format,
            )

        console.print("\n[green]✓[/green] Report generated successfully")
        console.print(f"[bold]Output:[/bold] {output}")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


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
