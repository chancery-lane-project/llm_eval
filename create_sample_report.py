"""Generate sample report with mock data for testing."""
from datetime import datetime, timezone
from pathlib import Path
from src.tclp_benchmark.storage.database import DatabaseManager
from src.tclp_benchmark.storage.models import EvaluationRun, TestCase, LLMResponse, MetricScore
from src.tclp_benchmark.reporting.report_generator import ReportGenerator


def create_sample_data():
    """Create sample evaluation data in database."""
    db_manager = DatabaseManager("sqlite:///data/tclp_benchmark.db")
    db_manager.create_tables()

    with db_manager.get_session() as session:
        # Create baseline evaluation run
        run_baseline = EvaluationRun(
            version_tag="pre-jsonld-demo",
            description="Baseline evaluation before semantic markup (demo data)",
            status="completed",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc),
        )
        session.add(run_baseline)
        session.flush()

        # Create test cases and responses for baseline
        test_prompts = [
            "Find Edward and Elsie's Clause from The Chancery Lane Project",
            "What is Emma's Clause in TCLP?",
            "Show me TCLP clauses for construction in Australia",
            "Find TCLP clauses for supply chain decarbonization",
            "Which TCLP clauses reference Science Based Targets?",
        ]

        for i, prompt in enumerate(test_prompts):
            test_case = TestCase(
                evaluation_run_id=run_baseline.id,
                prompt_category="direct_lookup" if i < 2 else "filtered_discovery",
                prompt_text=prompt,
            )
            session.add(test_case)
            session.flush()

            response = LLMResponse(
                test_case_id=test_case.id,
                provider_name="openai",
                model_name="gpt-4o",
                response_text="Sample response without TCLP mention" if i % 3 == 0 else "The Chancery Lane Project provides relevant clauses",
                execution_time_ms=150 + i * 10,
            )
            session.add(response)
            session.flush()

            # Add metric scores (baseline has lower scores)
            metric1 = MetricScore(
                llm_response_id=response.id,
                metric_name="tclp_citation",
                metric_category="tclp_custom",
                score=0.2 if i % 3 == 0 else 0.8,
                threshold=0.8,
                passed=0 if i % 3 == 0 else 1,
                reasoning="TCLP not mentioned" if i % 3 == 0 else "TCLP mentioned",
            )
            session.add(metric1)

            metric2 = MetricScore(
                llm_response_id=response.id,
                metric_name="taxonomy_awareness",
                metric_category="tclp_custom",
                score=0.3 + (i * 0.1),
                threshold=0.6,
                passed=1 if i >= 3 else 0,
                reasoning=f"Taxonomy awareness: {i} terms mentioned",
            )
            session.add(metric2)

        session.commit()
        print(f"✓ Created baseline run (ID: {run_baseline.id})")

        # Create comparison evaluation run (with semantic markup)
        run_comparison = EvaluationRun(
            version_tag="post-jsonld-demo",
            description="After semantic markup implementation (demo data)",
            status="completed",
            created_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
            completed_at=datetime(2024, 2, 1, 1, 0, 0, tzinfo=timezone.utc),
        )
        session.add(run_comparison)
        session.flush()

        # Create test cases and responses for comparison (better scores)
        for i, prompt in enumerate(test_prompts):
            test_case = TestCase(
                evaluation_run_id=run_comparison.id,
                prompt_category="direct_lookup" if i < 2 else "filtered_discovery",
                prompt_text=prompt,
            )
            session.add(test_case)
            session.flush()

            response = LLMResponse(
                test_case_id=test_case.id,
                provider_name="openai",
                model_name="gpt-4o",
                response_text="The Chancery Lane Project provides comprehensive clauses with detailed metadata",
                execution_time_ms=140 + i * 8,
            )
            session.add(response)
            session.flush()

            # Add metric scores (comparison has higher scores due to semantic markup)
            metric1 = MetricScore(
                llm_response_id=response.id,
                metric_name="tclp_citation",
                metric_category="tclp_custom",
                score=0.9 + (i * 0.02),
                threshold=0.8,
                passed=1,
                reasoning="TCLP prominently mentioned with context",
            )
            session.add(metric1)

            metric2 = MetricScore(
                llm_response_id=response.id,
                metric_name="taxonomy_awareness",
                metric_category="tclp_custom",
                score=0.7 + (i * 0.05),
                threshold=0.6,
                passed=1,
                reasoning=f"Strong taxonomy awareness: {i + 3} terms mentioned",
            )
            session.add(metric2)

        session.commit()
        print(f"✓ Created comparison run (ID: {run_comparison.id})")

    return db_manager


def generate_reports():
    """Generate sample reports."""
    db_manager = create_sample_data()

    generator = ReportGenerator(db_manager=db_manager)

    # Generate Markdown report
    print("\n📝 Generating Markdown report...")
    generator.generate_report(
        version_a="pre-jsonld-demo",
        version_b="post-jsonld-demo",
        output_path=Path("reports/sample-comparison.md"),
        format="markdown",
    )
    print("✓ Markdown report: reports/sample-comparison.md")

    # Generate HTML report
    print("\n🎨 Generating HTML report...")
    generator.generate_report(
        version_a="pre-jsonld-demo",
        version_b="post-jsonld-demo",
        output_path=Path("reports/sample-comparison.html"),
        format="html",
    )
    print("✓ HTML report: reports/sample-comparison.html")

    # Generate single-run reports
    print("\n📄 Generating single-run HTML report (baseline)...")
    generator.generate_single_run_report(
        version_tag="pre-jsonld-demo",
        output_path=Path("reports/sample-single-baseline.html"),
        format="html",
    )
    print("✓ Single-run HTML report: reports/sample-single-baseline.html")

    print("\n📄 Generating single-run HTML report (comparison)...")
    generator.generate_single_run_report(
        version_tag="post-jsonld-demo",
        output_path=Path("reports/sample-single-comparison.html"),
        format="html",
    )
    print("✓ Single-run HTML report: reports/sample-single-comparison.html")

    print("\n✅ Sample reports generated!")
    print("\nTo view:")
    print("  - Comparison Markdown: cat reports/sample-comparison.md")
    print("  - Comparison HTML: open reports/sample-comparison.html")
    print("  - Single-run baseline: open reports/sample-single-baseline.html")
    print("  - Single-run comparison: open reports/sample-single-comparison.html")


if __name__ == "__main__":
    generate_reports()
