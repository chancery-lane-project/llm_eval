"""Report generation for evaluation comparisons."""
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict

from ..storage.database import DatabaseManager
from ..storage.models import EvaluationRun


class ReportGenerator:
    """Generates comparison reports between evaluation runs."""

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialise report generator.

        Args:
            db_manager: Database manager for querying results
        """
        self.db_manager = db_manager

    def get_evaluation_runs(
        self, version_a: str, version_b: str
    ) -> Tuple[EvaluationRun, EvaluationRun]:
        """
        Get evaluation runs by version tag.

        Args:
            version_a: First version tag (baseline)
            version_b: Second version tag (comparison)

        Returns:
            Tuple of (run_a, run_b)

        Raises:
            ValueError: If either run is not found
        """
        with self.db_manager.get_session() as session:
            run_a = session.query(EvaluationRun).filter_by(version_tag=version_a).first()
            run_b = session.query(EvaluationRun).filter_by(version_tag=version_b).first()

            if not run_a:
                raise ValueError(f"Evaluation run not found: {version_a}")
            if not run_b:
                raise ValueError(f"Evaluation run not found: {version_b}")

            # Refresh to load relationships
            session.refresh(run_a)
            session.refresh(run_b)

            return run_a, run_b

    def calculate_comparison(
        self, run_a: EvaluationRun, run_b: EvaluationRun
    ) -> Dict[str, Any]:
        """
        Calculate metric comparisons between two runs.

        Args:
            run_a: Baseline evaluation run
            run_b: Comparison evaluation run

        Returns:
            Dictionary with comparison data
        """
        comparison = {
            "overall": {},
            "by_category": {},
            "by_provider": {},
        }

        # Collect all metrics from both runs
        metrics_a = defaultdict(list)
        metrics_b = defaultdict(list)

        # Collect metrics from run A
        for test_case in run_a.test_cases:
            for response in test_case.llm_responses:
                for metric_score in response.metric_scores:
                    metrics_a[metric_score.metric_name].append(metric_score.score)

        # Collect metrics from run B
        for test_case in run_b.test_cases:
            for response in test_case.llm_responses:
                for metric_score in response.metric_scores:
                    metrics_b[metric_score.metric_name].append(metric_score.score)

        # Calculate overall averages and improvements
        for metric_name in set(metrics_a.keys()) | set(metrics_b.keys()):
            avg_a = sum(metrics_a[metric_name]) / len(metrics_a[metric_name]) if metrics_a[metric_name] else 0.0
            avg_b = sum(metrics_b[metric_name]) / len(metrics_b[metric_name]) if metrics_b[metric_name] else 0.0
            improvement = avg_b - avg_a

            comparison["overall"][metric_name] = {
                "baseline": round(avg_a, 3),
                "comparison": round(avg_b, 3),
                "improvement": round(improvement, 3),
                "improvement_pct": round((improvement / avg_a * 100) if avg_a > 0 else 0, 1),
            }

        # Calculate by category
        categories_a = defaultdict(lambda: defaultdict(list))
        categories_b = defaultdict(lambda: defaultdict(list))

        for test_case in run_a.test_cases:
            category = test_case.prompt_category
            for response in test_case.llm_responses:
                for metric_score in response.metric_scores:
                    categories_a[category][metric_score.metric_name].append(metric_score.score)

        for test_case in run_b.test_cases:
            category = test_case.prompt_category
            for response in test_case.llm_responses:
                for metric_score in response.metric_scores:
                    categories_b[category][metric_score.metric_name].append(metric_score.score)

        for category in set(categories_a.keys()) | set(categories_b.keys()):
            comparison["by_category"][category] = {}
            for metric_name in set(categories_a[category].keys()) | set(categories_b[category].keys()):
                avg_a = sum(categories_a[category][metric_name]) / len(categories_a[category][metric_name]) if categories_a[category][metric_name] else 0.0
                avg_b = sum(categories_b[category][metric_name]) / len(categories_b[category][metric_name]) if categories_b[category][metric_name] else 0.0
                improvement = avg_b - avg_a

                comparison["by_category"][category][metric_name] = {
                    "baseline": round(avg_a, 3),
                    "comparison": round(avg_b, 3),
                    "improvement": round(improvement, 3),
                    "improvement_pct": round((improvement / avg_a * 100) if avg_a > 0 else 0, 1),
                }

        # Calculate by provider
        providers_a = defaultdict(lambda: defaultdict(list))
        providers_b = defaultdict(lambda: defaultdict(list))

        for test_case in run_a.test_cases:
            for response in test_case.llm_responses:
                provider = f"{response.provider_name}/{response.model_name}"
                for metric_score in response.metric_scores:
                    providers_a[provider][metric_score.metric_name].append(metric_score.score)

        for test_case in run_b.test_cases:
            for response in test_case.llm_responses:
                provider = f"{response.provider_name}/{response.model_name}"
                for metric_score in response.metric_scores:
                    providers_b[provider][metric_score.metric_name].append(metric_score.score)

        for provider in set(providers_a.keys()) | set(providers_b.keys()):
            comparison["by_provider"][provider] = {}
            for metric_name in set(providers_a[provider].keys()) | set(providers_b[provider].keys()):
                avg_a = sum(providers_a[provider][metric_name]) / len(providers_a[provider][metric_name]) if providers_a[provider][metric_name] else 0.0
                avg_b = sum(providers_b[provider][metric_name]) / len(providers_b[provider][metric_name]) if providers_b[provider][metric_name] else 0.0
                improvement = avg_b - avg_a

                comparison["by_provider"][provider][metric_name] = {
                    "baseline": round(avg_a, 3),
                    "comparison": round(avg_b, 3),
                    "improvement": round(improvement, 3),
                    "improvement_pct": round((improvement / avg_a * 100) if avg_a > 0 else 0, 1),
                }

        return comparison

    def generate_report(
        self,
        version_a: str,
        version_b: str,
        output_path: Path,
        format: str = "markdown",
    ):
        """
        Generate comparison report.

        Args:
            version_a: First version tag (baseline)
            version_b: Second version tag (comparison)
            output_path: Path to save report
            format: Report format ("markdown" or "html")
        """
        # Get evaluation runs
        run_a, run_b = self.get_evaluation_runs(version_a, version_b)

        # Calculate comparison
        comparison = self.calculate_comparison(run_a, run_b)

        # Generate report based on format
        if format == "markdown":
            self._generate_markdown(run_a, run_b, comparison, output_path)
        elif format == "html":
            self._generate_html(run_a, run_b, comparison, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_markdown(
        self,
        run_a: EvaluationRun,
        run_b: EvaluationRun,
        comparison: Dict[str, Any],
        output_path: Path,
    ):
        """Generate markdown report."""
        lines = []

        # Header
        lines.append("# Evaluation Comparison Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Run details
        lines.append("## Evaluation Runs")
        lines.append("")
        lines.append(f"### Baseline: {run_a.version_tag}")
        lines.append(f"- **Description:** {run_a.description or 'N/A'}")
        lines.append(f"- **Status:** {run_a.status}")
        lines.append(f"- **Created:** {run_a.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **Test Cases:** {len(run_a.test_cases)}")
        lines.append("")

        lines.append(f"### Comparison: {run_b.version_tag}")
        lines.append(f"- **Description:** {run_b.description or 'N/A'}")
        lines.append(f"- **Status:** {run_b.status}")
        lines.append(f"- **Created:** {run_b.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **Test Cases:** {len(run_b.test_cases)}")
        lines.append("")

        # Overall metrics
        lines.append("## Overall Metrics")
        lines.append("")
        lines.append("| Metric | Baseline | Comparison | Improvement | Improvement % |")
        lines.append("|--------|----------|------------|-------------|---------------|")

        for metric_name, data in comparison["overall"].items():
            improvement_sign = "+" if data["improvement"] >= 0 else ""
            lines.append(
                f"| {metric_name} | {data['baseline']:.3f} | {data['comparison']:.3f} | "
                f"{improvement_sign}{data['improvement']:.3f} | {improvement_sign}{data['improvement_pct']:.1f}% |"
            )

        lines.append("")

        # By category
        if comparison["by_category"]:
            lines.append("## Metrics by Category")
            lines.append("")

            for category, metrics in comparison["by_category"].items():
                lines.append(f"### {category}")
                lines.append("")
                lines.append("| Metric | Baseline | Comparison | Improvement | Improvement % |")
                lines.append("|--------|----------|------------|-------------|---------------|")

                for metric_name, data in metrics.items():
                    improvement_sign = "+" if data["improvement"] >= 0 else ""
                    lines.append(
                        f"| {metric_name} | {data['baseline']:.3f} | {data['comparison']:.3f} | "
                        f"{improvement_sign}{data['improvement']:.3f} | {improvement_sign}{data['improvement_pct']:.1f}% |"
                    )

                lines.append("")

        # By provider
        if comparison["by_provider"]:
            lines.append("## Metrics by Provider")
            lines.append("")

            for provider, metrics in comparison["by_provider"].items():
                lines.append(f"### {provider}")
                lines.append("")
                lines.append("| Metric | Baseline | Comparison | Improvement | Improvement % |")
                lines.append("|--------|----------|------------|-------------|---------------|")

                for metric_name, data in metrics.items():
                    improvement_sign = "+" if data["improvement"] >= 0 else ""
                    lines.append(
                        f"| {metric_name} | {data['baseline']:.3f} | {data['comparison']:.3f} | "
                        f"{improvement_sign}{data['improvement']:.3f} | {improvement_sign}{data['improvement_pct']:.1f}% |"
                    )

                lines.append("")

        # Write to file
        output_path.write_text("\n".join(lines))

    def _generate_html(
        self,
        run_a: EvaluationRun,
        run_b: EvaluationRun,
        comparison: Dict[str, Any],
        output_path: Path,
    ):
        """Generate HTML report."""
        # Placeholder for HTML generation
        # For now, just create a basic HTML version of the markdown
        markdown_path = output_path.with_suffix(".md")
        self._generate_markdown(run_a, run_b, comparison, markdown_path)

        # Simple HTML wrapper
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        h1, h2, h3 {{ color: #333; }}
    </style>
</head>
<body>
    <p>HTML report generation not fully implemented. See markdown version at: {markdown_path}</p>
</body>
</html>"""

        output_path.write_text(html)
