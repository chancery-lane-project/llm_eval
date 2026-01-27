"""Report generation for evaluation comparisons."""
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict

from ..storage.database import DatabaseManager
from ..storage.models import EvaluationRun, TestCase, LLMResponse


class ReportGenerator:
    """Generates comparison reports between evaluation runs."""

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialise report generator.

        Args:
            db_manager: Database manager for querying results
        """
        self.db_manager = db_manager

    def get_evaluation_run(self, version_tag: str) -> EvaluationRun:
        """
        Get single evaluation run by version tag.

        Args:
            version_tag: Version tag to fetch

        Returns:
            EvaluationRun instance

        Raises:
            ValueError: If run is not found
        """
        from sqlalchemy.orm import joinedload

        with self.db_manager.get_session() as session:
            run = (
                session.query(EvaluationRun)
                .filter_by(version_tag=version_tag)
                .options(
                    joinedload(EvaluationRun.test_cases)
                    .joinedload(TestCase.llm_responses)
                    .joinedload(LLMResponse.metric_scores)
                )
                .first()
            )

            if not run:
                raise ValueError(f"Evaluation run not found: {version_tag}")

            # Force load all relationships while in session
            _ = run.test_cases
            for tc in run.test_cases:
                _ = tc.llm_responses
                for resp in tc.llm_responses:
                    _ = resp.metric_scores

            return run

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
        from sqlalchemy.orm import joinedload

        with self.db_manager.get_session() as session:
            # Use eager loading to load all relationships
            run_a = (
                session.query(EvaluationRun)
                .filter_by(version_tag=version_a)
                .options(
                    joinedload(EvaluationRun.test_cases)
                    .joinedload(TestCase.llm_responses)
                    .joinedload(LLMResponse.metric_scores)
                )
                .first()
            )

            run_b = (
                session.query(EvaluationRun)
                .filter_by(version_tag=version_b)
                .options(
                    joinedload(EvaluationRun.test_cases)
                    .joinedload(TestCase.llm_responses)
                    .joinedload(LLMResponse.metric_scores)
                )
                .first()
            )

            if not run_a:
                raise ValueError(f"Evaluation run not found: {version_a}")
            if not run_b:
                raise ValueError(f"Evaluation run not found: {version_b}")

            # Make instances persistent by accessing attributes while in session
            # This forces SQLAlchemy to load all relationships
            _ = run_a.test_cases
            _ = run_b.test_cases
            for tc in run_a.test_cases:
                _ = tc.llm_responses
                for resp in tc.llm_responses:
                    _ = resp.metric_scores

            for tc in run_b.test_cases:
                _ = tc.llm_responses
                for resp in tc.llm_responses:
                    _ = resp.metric_scores

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

    def generate_single_run_report(
        self,
        version_tag: str,
        output_path: Path,
        format: str = "html",
    ):
        """
        Generate detailed report for a single evaluation run.

        Args:
            version_tag: Version tag to report on
            output_path: Path to save report
            format: Report format ("markdown" or "html")
        """
        # Get evaluation run
        run = self.get_evaluation_run(version_tag)

        # Calculate statistics
        stats = self._calculate_single_run_stats(run)

        # Generate report based on format
        if format == "markdown":
            self._generate_single_markdown(run, stats, output_path)
        elif format == "html":
            self._generate_single_html(run, stats, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _calculate_single_run_stats(self, run: EvaluationRun) -> Dict[str, Any]:
        """Calculate statistics for a single evaluation run."""
        stats = {
            "overall": {},
            "by_category": {},
            "by_provider": {},
            "test_cases": [],
        }

        # Collect all metrics
        all_metrics = defaultdict(list)

        for test_case in run.test_cases:
            # Collect test case details
            test_case_info = {
                "prompt_category": test_case.prompt_category,
                "prompt_text": test_case.prompt_text,
                "responses": [],
            }

            for response in test_case.llm_responses:
                response_info = {
                    "provider": response.provider_name,
                    "model": response.model_name,
                    "response_text": response.response_text,
                    "execution_time_ms": response.execution_time_ms,
                    "error": response.error,
                    "metrics": {},
                }

                for metric_score in response.metric_scores:
                    all_metrics[metric_score.metric_name].append(metric_score.score)

                    response_info["metrics"][metric_score.metric_name] = {
                        "score": metric_score.score,
                        "threshold": metric_score.threshold,
                        "passed": bool(metric_score.passed),
                        "reasoning": metric_score.reasoning,
                    }

                test_case_info["responses"].append(response_info)

            stats["test_cases"].append(test_case_info)

        # Calculate overall averages
        for metric_name, scores in all_metrics.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            pass_rate = sum(1 for tc in stats["test_cases"]
                          for resp in tc["responses"]
                          for m_name, m_data in resp["metrics"].items()
                          if m_name == metric_name and m_data["passed"]) / len(scores) if scores else 0.0

            stats["overall"][metric_name] = {
                "avg_score": round(avg_score, 3),
                "pass_rate": round(pass_rate, 3),
                "total_tests": len(scores),
            }

        # Calculate by category
        categories = defaultdict(lambda: defaultdict(list))
        for test_case in run.test_cases:
            category = test_case.prompt_category
            for response in test_case.llm_responses:
                for metric_score in response.metric_scores:
                    categories[category][metric_score.metric_name].append(metric_score.score)

        for category, metrics in categories.items():
            stats["by_category"][category] = {}
            for metric_name, scores in metrics.items():
                avg_score = sum(scores) / len(scores) if scores else 0.0
                stats["by_category"][category][metric_name] = {
                    "avg_score": round(avg_score, 3),
                    "total_tests": len(scores),
                }

        # Calculate by provider
        providers = defaultdict(lambda: defaultdict(list))
        for test_case in run.test_cases:
            for response in test_case.llm_responses:
                provider = f"{response.provider_name}/{response.model_name}"
                for metric_score in response.metric_scores:
                    providers[provider][metric_score.metric_name].append(metric_score.score)

        for provider, metrics in providers.items():
            stats["by_provider"][provider] = {}
            for metric_name, scores in metrics.items():
                avg_score = sum(scores) / len(scores) if scores else 0.0
                stats["by_provider"][provider][metric_name] = {
                    "avg_score": round(avg_score, 3),
                    "total_tests": len(scores),
                }

        return stats

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
        """Generate HTML report with charts and styling."""
        # Prepare data for charts
        metric_names = list(comparison["overall"].keys())
        baseline_scores = [comparison["overall"][m]["baseline"] for m in metric_names]
        comparison_scores = [comparison["overall"][m]["comparison"] for m in metric_names]
        improvements = [comparison["overall"][m]["improvement"] for m in metric_names]

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TCLP Evaluation Comparison Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .content {{
            padding: 40px;
        }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        .card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            border-left: 4px solid #667eea;
        }}

        .card h3 {{
            color: #495057;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }}

        .card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #212529;
        }}

        .section {{
            margin-bottom: 50px;
        }}

        .section h2 {{
            color: #212529;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
        }}

        .chart-container {{
            position: relative;
            height: 400px;
            margin-bottom: 40px;
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        th, td {{
            padding: 15px;
            text-align: left;
        }}

        tbody tr:hover {{
            background: #f8f9fa;
        }}

        tbody tr:nth-child(even) {{
            background: #f8f9fa;
        }}

        .improvement {{
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
        }}

        .improvement.positive {{
            background: #d4edda;
            color: #155724;
        }}

        .improvement.negative {{
            background: #f8d7da;
            color: #721c24;
        }}

        .improvement.neutral {{
            background: #e2e3e5;
            color: #383d41;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 TCLP Evaluation Comparison</h1>
            <p>Measuring Semantic Markup Impact on LLM Discovery</p>
            <p style="margin-top: 10px; font-size: 0.9em;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="content">
            <!-- Summary Cards -->
            <div class="summary-cards">
                <div class="card">
                    <h3>Baseline Version</h3>
                    <div class="value">{run_a.version_tag}</div>
                    <p style="margin-top: 10px; color: #6c757d;">{len(run_a.test_cases)} test cases</p>
                </div>
                <div class="card">
                    <h3>Comparison Version</h3>
                    <div class="value">{run_b.version_tag}</div>
                    <p style="margin-top: 10px; color: #6c757d;">{len(run_b.test_cases)} test cases</p>
                </div>
                <div class="card">
                    <h3>Metrics Tracked</h3>
                    <div class="value">{len(comparison["overall"])}</div>
                    <p style="margin-top: 10px; color: #6c757d;">Custom TCLP metrics</p>
                </div>
            </div>

            <!-- Overall Metrics Chart -->
            <div class="section">
                <h2>📈 Overall Performance Comparison</h2>
                <div class="chart-container">
                    <canvas id="overallChart"></canvas>
                </div>
            </div>

            <!-- Improvement Chart -->
            <div class="section">
                <h2>📊 Metric Improvements</h2>
                <div class="chart-container">
                    <canvas id="improvementChart"></canvas>
                </div>
            </div>

            <!-- Metrics Table -->
            <div class="section">
                <h2>📋 Detailed Metrics</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Baseline</th>
                            <th>Comparison</th>
                            <th>Improvement</th>
                            <th>% Change</th>
                        </tr>
                    </thead>
                    <tbody>
"""

        # Add metric rows
        for metric_name, data in comparison["overall"].items():
            improvement_class = "positive" if data["improvement"] > 0 else ("negative" if data["improvement"] < 0 else "neutral")
            improvement_sign = "+" if data["improvement"] >= 0 else ""

            html += f"""                        <tr>
                            <td><strong>{metric_name}</strong></td>
                            <td>{data['baseline']:.3f}</td>
                            <td>{data['comparison']:.3f}</td>
                            <td><span class="improvement {improvement_class}">{improvement_sign}{data['improvement']:.3f}</span></td>
                            <td><span class="improvement {improvement_class}">{improvement_sign}{data['improvement_pct']:.1f}%</span></td>
                        </tr>
"""

        html += """                    </tbody>
                </table>
            </div>
"""

        # Add category breakdowns if available
        if comparison["by_category"]:
            html += """            <div class="section">
                <h2>📂 Performance by Category</h2>
"""
            for category, metrics in comparison["by_category"].items():
                html += f"""                <h3 style="margin: 30px 0 15px 0; color: #495057;">{category}</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Baseline</th>
                            <th>Comparison</th>
                            <th>Improvement</th>
                        </tr>
                    </thead>
                    <tbody>
"""
                for metric_name, data in metrics.items():
                    improvement_class = "positive" if data["improvement"] > 0 else ("negative" if data["improvement"] < 0 else "neutral")
                    improvement_sign = "+" if data["improvement"] >= 0 else ""

                    html += f"""                        <tr>
                            <td>{metric_name}</td>
                            <td>{data['baseline']:.3f}</td>
                            <td>{data['comparison']:.3f}</td>
                            <td><span class="improvement {improvement_class}">{improvement_sign}{data['improvement']:.3f}</span></td>
                        </tr>
"""

                html += """                    </tbody>
                </table>
"""

            html += """            </div>
"""

        # Add Charts JavaScript
        html += f"""        </div>

        <div class="footer">
            Generated by TCLP Semantic Benchmark Tool |
            <a href="https://chancerylaneproject.org" style="color: #667eea; text-decoration: none;">The Chancery Lane Project</a>
        </div>
    </div>

    <script>
        // Overall Comparison Chart
        const overallCtx = document.getElementById('overallChart').getContext('2d');
        new Chart(overallCtx, {{
            type: 'bar',
            data: {{
                labels: {metric_names},
                datasets: [
                    {{
                        label: 'Baseline ({run_a.version_tag})',
                        data: {baseline_scores},
                        backgroundColor: 'rgba(102, 126, 234, 0.7)',
                        borderColor: 'rgba(102, 126, 234, 1)',
                        borderWidth: 2
                    }},
                    {{
                        label: 'Comparison ({run_b.version_tag})',
                        data: {comparison_scores},
                        backgroundColor: 'rgba(118, 75, 162, 0.7)',
                        borderColor: 'rgba(118, 75, 162, 1)',
                        borderWidth: 2
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1.0,
                        ticks: {{
                            callback: function(value) {{
                                return (value * 100).toFixed(0) + '%';
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Metric Scores Comparison',
                        font: {{
                            size: 16
                        }}
                    }},
                    legend: {{
                        display: true,
                        position: 'top'
                    }}
                }}
            }}
        }});

        // Improvement Chart
        const improvementCtx = document.getElementById('improvementChart').getContext('2d');
        new Chart(improvementCtx, {{
            type: 'bar',
            data: {{
                labels: {metric_names},
                datasets: [{{
                    label: 'Improvement',
                    data: {improvements},
                    backgroundColor: {improvements}.map(v =>
                        v > 0 ? 'rgba(40, 167, 69, 0.7)' :
                        v < 0 ? 'rgba(220, 53, 69, 0.7)' :
                        'rgba(108, 117, 125, 0.7)'
                    ),
                    borderColor: {improvements}.map(v =>
                        v > 0 ? 'rgba(40, 167, 69, 1)' :
                        v < 0 ? 'rgba(220, 53, 69, 1)' :
                        'rgba(108, 117, 125, 1)'
                    ),
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        ticks: {{
                            callback: function(value) {{
                                return (value >= 0 ? '+' : '') + (value * 100).toFixed(0) + '%';
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Performance Improvements',
                        font: {{
                            size: 16
                        }}
                    }},
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

        output_path.write_text(html)

    def _generate_single_markdown(
        self,
        run: EvaluationRun,
        stats: Dict[str, Any],
        output_path: Path,
    ):
        """Generate markdown report for single run."""
        lines = []

        # Header
        lines.append("# Evaluation Run Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Run details
        lines.append("## Run Details")
        lines.append("")
        lines.append(f"- **Version Tag:** {run.version_tag}")
        lines.append(f"- **Description:** {run.description or 'N/A'}")
        lines.append(f"- **Status:** {run.status}")
        lines.append(f"- **Created:** {run.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if run.completed_at:
            lines.append(f"- **Completed:** {run.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **Test Cases:** {len(run.test_cases)}")
        lines.append("")

        # Overall metrics
        lines.append("## Overall Metrics")
        lines.append("")
        lines.append("| Metric | Avg Score | Pass Rate | Total Tests |")
        lines.append("|--------|-----------|-----------|-------------|")

        for metric_name, data in stats["overall"].items():
            lines.append(
                f"| {metric_name} | {data['avg_score']:.3f} | "
                f"{data['pass_rate']:.1%} | {data['total_tests']} |"
            )

        lines.append("")

        # By category
        if stats["by_category"]:
            lines.append("## Metrics by Category")
            lines.append("")

            for category, metrics in stats["by_category"].items():
                lines.append(f"### {category}")
                lines.append("")
                lines.append("| Metric | Avg Score | Total Tests |")
                lines.append("|--------|-----------|-------------|")

                for metric_name, data in metrics.items():
                    lines.append(
                        f"| {metric_name} | {data['avg_score']:.3f} | {data['total_tests']} |"
                    )

                lines.append("")

        # By provider
        if stats["by_provider"]:
            lines.append("## Metrics by Provider")
            lines.append("")

            for provider, metrics in stats["by_provider"].items():
                lines.append(f"### {provider}")
                lines.append("")
                lines.append("| Metric | Avg Score | Total Tests |")
                lines.append("|--------|-----------|-------------|")

                for metric_name, data in metrics.items():
                    lines.append(
                        f"| {metric_name} | {data['avg_score']:.3f} | {data['total_tests']} |"
                    )

                lines.append("")

        # Detailed test cases
        lines.append("## Detailed Test Cases")
        lines.append("")

        for i, test_case in enumerate(stats["test_cases"], 1):
            lines.append(f"### Test Case {i}: {test_case['prompt_category']}")
            lines.append("")
            lines.append(f"**Prompt:** {test_case['prompt_text']}")
            lines.append("")

            for response in test_case["responses"]:
                lines.append(f"#### {response['provider']}/{response['model']}")
                lines.append("")

                if response["error"]:
                    lines.append(f"**Error:** {response['error']}")
                    lines.append("")
                    continue

                lines.append(f"**Response:** {response['response_text'][:500]}{'...' if len(response['response_text']) > 500 else ''}")
                lines.append("")
                lines.append(f"**Execution Time:** {response['execution_time_ms']}ms")
                lines.append("")

                lines.append("**Metrics:**")
                lines.append("")
                lines.append("| Metric | Score | Pass/Fail | Reasoning |")
                lines.append("|--------|-------|-----------|-----------|")

                for metric_name, metric_data in response["metrics"].items():
                    pass_fail = "✓ Pass" if metric_data["passed"] else "✗ Fail"
                    reasoning = metric_data["reasoning"][:100] if metric_data["reasoning"] else "N/A"
                    lines.append(
                        f"| {metric_name} | {metric_data['score']:.3f} | {pass_fail} | {reasoning} |"
                    )

                lines.append("")

        # Write to file
        output_path.write_text("\n".join(lines))

    def _generate_single_html(
        self,
        run: EvaluationRun,
        stats: Dict[str, Any],
        output_path: Path,
    ):
        """Generate HTML report for single run with detailed views."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TCLP Evaluation Report - {run.version_tag}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 50px;
        }}

        .section h2 {{
            color: #212529;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
        }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        .card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            border-left: 4px solid #667eea;
        }}

        .card h3 {{
            color: #495057;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }}

        .card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #212529;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        th, td {{
            padding: 15px;
            text-align: left;
        }}

        tbody tr:hover {{
            background: #f8f9fa;
        }}

        tbody tr:nth-child(even) {{
            background: #f8f9fa;
        }}

        .test-case {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 30px;
        }}

        .test-case h3 {{
            color: #495057;
            margin-bottom: 15px;
        }}

        .prompt {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
        }}

        .response {{
            background: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}

        .response h4 {{
            color: #667eea;
            margin-bottom: 10px;
        }}

        .response-text {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.6;
            max-height: 300px;
            overflow-y: auto;
        }}

        .metric-badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.9em;
            font-weight: bold;
            margin-right: 10px;
        }}

        .metric-badge.pass {{
            background: #d4edda;
            color: #155724;
        }}

        .metric-badge.fail {{
            background: #f8d7da;
            color: #721c24;
        }}

        .error {{
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #dc3545;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 TCLP Evaluation Report</h1>
            <p>{run.version_tag}</p>
            <p style="margin-top: 10px; font-size: 0.9em;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="content">
            <!-- Summary Cards -->
            <div class="summary-cards">
                <div class="card">
                    <h3>Test Cases</h3>
                    <div class="value">{len(run.test_cases)}</div>
                </div>
                <div class="card">
                    <h3>Metrics Tracked</h3>
                    <div class="value">{len(stats["overall"])}</div>
                </div>
                <div class="card">
                    <h3>Status</h3>
                    <div class="value">{run.status}</div>
                </div>
"""

        # Add completion time if available
        if run.completed_at:
            duration = (run.completed_at - run.created_at).total_seconds()
            html += f"""                <div class="card">
                    <h3>Duration</h3>
                    <div class="value">{int(duration)}s</div>
                </div>
"""

        html += """            </div>

            <!-- Overall Metrics -->
            <div class="section">
                <h2>📈 Overall Metrics</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Avg Score</th>
                            <th>Pass Rate</th>
                            <th>Total Tests</th>
                        </tr>
                    </thead>
                    <tbody>
"""

        for metric_name, data in stats["overall"].items():
            html += f"""                        <tr>
                            <td><strong>{metric_name}</strong></td>
                            <td>{data['avg_score']:.3f}</td>
                            <td>{data['pass_rate']:.1%}</td>
                            <td>{data['total_tests']}</td>
                        </tr>
"""

        html += """                    </tbody>
                </table>
            </div>
"""

        # Detailed test cases
        html += """            <div class="section">
                <h2>📋 Detailed Test Cases</h2>
"""

        for i, test_case in enumerate(stats["test_cases"], 1):
            html += f"""                <div class="test-case">
                    <h3>Test Case {i}: {test_case['prompt_category']}</h3>
                    <div class="prompt">
                        <strong>Prompt:</strong> {test_case['prompt_text']}
                    </div>
"""

            for response in test_case["responses"]:
                html += f"""                    <div class="response">
                        <h4>{response['provider']}/{response['model']}</h4>
"""

                if response["error"]:
                    html += f"""                        <div class="error">
                            <strong>Error:</strong> {response['error']}
                        </div>
"""
                else:
                    html += f"""                        <div class="response-text">{response['response_text']}</div>
                        <p><strong>Execution Time:</strong> {response['execution_time_ms']}ms</p>
                        <p style="margin-top: 15px;"><strong>Metrics:</strong></p>
"""

                    for metric_name, metric_data in response["metrics"].items():
                        badge_class = "pass" if metric_data["passed"] else "fail"
                        pass_fail = "✓ Pass" if metric_data["passed"] else "✗ Fail"
                        html += f"""                        <span class="metric-badge {badge_class}">
                            {metric_name}: {metric_data['score']:.3f} - {pass_fail}
                        </span>
"""

                html += """                    </div>
"""

            html += """                </div>
"""

        html += """            </div>
        </div>

        <div class="footer">
            Generated by TCLP Semantic Benchmark Tool |
            <a href="https://chancerylaneproject.org" style="color: #667eea; text-decoration: none;">The Chancery Lane Project</a>
        </div>
    </div>
</body>
</html>"""

        output_path.write_text(html)
