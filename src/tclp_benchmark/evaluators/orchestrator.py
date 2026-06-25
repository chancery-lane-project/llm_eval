"""Evaluation orchestrator that coordinates the entire evaluation workflow."""
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from deepeval.test_case import LLMTestCase
from rich.console import Console
from rich.panel import Panel

from ..storage.database import DatabaseManager
from ..storage.models import EvaluationRun, TestCase, LLMResponse as DBLLMResponse, MetricScore
from ..config.models import PromptConfig, ProvidersConfig, MetricsConfig, TestPrompt
from ..llm_providers.base import BaseLLMProvider, LLMRequest
from ..llm_providers.openai_provider import OpenAIProvider
from ..llm_providers.anthropic_provider import AnthropicProvider
from ..llm_providers.gemini_provider import GeminiProvider
from ..llm_providers.openrouter_provider import OpenRouterProvider
from ..evaluators.custom_metrics import (
    TCLPCitationMetric,
    TaxonomyAwarenessMetric,
    EntityRetrievalAccuracyMetric,
    MetadataPrecisionMetric,
    RelationshipTraversalScoreMetric,
)

console = Console()


class EvaluationOrchestrator:
    """Orchestrates evaluation runs across LLM providers and metrics."""

    def __init__(self, db_manager: DatabaseManager, verbose: bool = True):
        """
        Initialise orchestrator.

        Args:
            db_manager: Database manager for storing results
            verbose: Whether to output detailed progress
        """
        self.db_manager = db_manager
        self.verbose = verbose

    async def run_evaluation(
        self,
        version_tag: str,
        prompts_config: PromptConfig,
        providers_config: ProvidersConfig,
        metrics_config: MetricsConfig,
        description: Optional[str] = None,
        category_filter: Optional[str] = None,
        provider_filter: Optional[List[str]] = None,
    ) -> int:
        """
        Run evaluation with given configuration.

        Args:
            version_tag: Version tag for this evaluation run
            prompts_config: Prompts configuration
            providers_config: Providers configuration
            metrics_config: Metrics configuration
            description: Optional description for this run
            category_filter: Optional category to filter prompts
            provider_filter: Optional list of providers to use

        Returns:
            Evaluation run ID
        """
        # Create evaluation run
        with self.db_manager.get_session() as session:
            evaluation_run = EvaluationRun(
                version_tag=version_tag,
                description=description,
                status="running",
            )
            session.add(evaluation_run)
            session.commit()
            session.refresh(evaluation_run)
            run_id = evaluation_run.id

        try:
            # Filter prompts by category if specified
            prompts_to_run = {}
            if category_filter:
                if category_filter in prompts_config.prompts:
                    prompts_to_run[category_filter] = prompts_config.prompts[category_filter]
            else:
                prompts_to_run = prompts_config.prompts

            # Execute prompts against each provider
            for category, prompts in prompts_to_run.items():
                for prompt in prompts:
                    await self._execute_prompt(
                        run_id=run_id,
                        category=category,
                        prompt=prompt,
                        providers_config=providers_config,
                        metrics_config=metrics_config,
                        provider_filter=provider_filter,
                    )

            # Mark run as completed
            with self.db_manager.get_session() as session:
                evaluation_run = session.query(EvaluationRun).filter_by(id=run_id).first()
                evaluation_run.status = "completed"
                evaluation_run.completed_at = datetime.now(timezone.utc)
                session.commit()

        except Exception as e:
            # Mark run as failed
            with self.db_manager.get_session() as session:
                evaluation_run = session.query(EvaluationRun).filter_by(id=run_id).first()
                evaluation_run.status = "failed"
                evaluation_run.completed_at = datetime.now(timezone.utc)
                session.commit()
            raise e

        return run_id

    async def _execute_prompt(
        self,
        run_id: int,
        category: str,
        prompt: TestPrompt,
        providers_config: ProvidersConfig,
        metrics_config: MetricsConfig,
        provider_filter: Optional[List[str]] = None,
    ):
        """
        Execute a single prompt against all enabled providers.

        Args:
            run_id: Evaluation run ID
            category: Prompt category
            prompt: Test prompt to execute
            providers_config: Providers configuration
            metrics_config: Metrics configuration
            provider_filter: Optional list of providers to use
        """
        # Show prompt being executed
        if self.verbose:
            console.print("\n")
            console.print(Panel(
                f"[bold cyan]{prompt.text}[/bold cyan]",
                title=f"[bold]📝 Prompt ({category})[/bold]",
                border_style="cyan",
            ))

        # Create test case
        with self.db_manager.get_session() as session:
            test_case = TestCase(
                evaluation_run_id=run_id,
                prompt_category=category,
                prompt_text=prompt.text,
                expected_outcome=str(prompt.expected_tclp_mention) if prompt.expected_tclp_mention else None,
            )
            session.add(test_case)
            session.commit()
            session.refresh(test_case)
            test_case_id = test_case.id

        # Execute against each enabled provider
        provider_configs = {
            "openai": providers_config.openai,
            "anthropic": providers_config.anthropic,
            "gemini": providers_config.gemini,
            "openrouter": providers_config.openrouter,
            "mcp": providers_config.mcp,
        }

        for provider_name, provider_config in provider_configs.items():
            # Skip if not enabled or filtered out
            if not provider_config or not provider_config.enabled:
                continue
            if provider_filter and provider_name not in provider_filter:
                continue

            # Handle OpenRouter's different config structure
            if provider_name == "openrouter":
                # OpenRouter has models as strings and max_tokens/temp at provider level
                from ..config.models import ModelConfig
                for model_name in provider_config.models:
                    model_config = ModelConfig(
                        name=model_name,
                        max_tokens=provider_config.max_tokens,
                        temperature=provider_config.temperature,
                    )
                    await self._execute_model(
                        test_case_id=test_case_id,
                        provider_name=provider_name,
                        provider_config=provider_config,
                        model_config=model_config,
                        prompt=prompt,
                        metrics_config=metrics_config,
                    )
            else:
                # Other providers have models as ModelConfig objects
                for model_config in provider_config.models:
                    await self._execute_model(
                        test_case_id=test_case_id,
                        provider_name=provider_name,
                        provider_config=provider_config,
                        model_config=model_config,
                        prompt=prompt,
                        metrics_config=metrics_config,
                    )

    async def _execute_model(
        self,
        test_case_id: int,
        provider_name: str,
        provider_config: Any,
        model_config: Any,
        prompt: TestPrompt,
        metrics_config: MetricsConfig,
    ):
        """
        Execute prompt against a specific model.

        Args:
            test_case_id: Test case ID
            provider_name: Provider name
            provider_config: Provider configuration
            model_config: Model configuration
            prompt: Test prompt
            metrics_config: Metrics configuration
        """
        # Show which model is being called
        if self.verbose:
            console.print(f"\n  [bold blue]→ {provider_name}/{model_config.name}[/bold blue]", end=" ")

        # Create provider
        provider = self._create_provider(provider_name, provider_config)

        # Create LLM request (no search context for now)
        request = LLMRequest(
            prompt=prompt.text,
            search_context=[],
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
            model_name=model_config.name,
        )

        # Generate response
        response = await provider.generate_response(request)

        # Show response or error
        if self.verbose:
            if response.error:
                console.print("[red]✗ Error[/red]")
                console.print(f"    [dim red]{response.error[:100]}...[/dim red]")
            else:
                console.print(f"[green]✓ {response.execution_time_ms}ms[/green]")
                # Show response in a panel
                response_preview = response.text[:300] + "..." if len(response.text) > 300 else response.text
                console.print(Panel(
                    f"[dim]{response_preview}[/dim]",
                    title="[bold green]Response[/bold green]",
                    border_style="green",
                    padding=(0, 2),
                ))

        # Store LLM response
        with self.db_manager.get_session() as session:
            db_response = DBLLMResponse(
                test_case_id=test_case_id,
                provider_name=response.provider_name,
                model_name=response.model_name,
                search_query=response.search_query,
                search_results=None,  # Will add search later
                response_text=response.text,
                response_metadata=response.metadata,
                execution_time_ms=response.execution_time_ms,
                error=response.error,
            )
            session.add(db_response)
            session.commit()
            session.refresh(db_response)
            response_id = db_response.id

        # Evaluate with metrics (only if response has text)
        if response.text:
            metric_scores = await self._evaluate_metrics(
                response_id=response_id,
                prompt=prompt,
                response_text=response.text,
                metrics_config=metrics_config,
            )

            # Show metrics
            if self.verbose and metric_scores:
                console.print("    [bold]Metrics:[/bold]")
                for metric_name, score_data in metric_scores.items():
                    status = "[green]✓ Pass[/green]" if score_data["passed"] else "[red]✗ Fail[/red]"
                    console.print(f"      • {metric_name}: {score_data['score']:.3f} {status}")

    async def _evaluate_metrics(
        self,
        response_id: int,
        prompt: TestPrompt,
        response_text: str,
        metrics_config: MetricsConfig,
    ) -> Dict[str, Any]:
        """
        Evaluate response with all enabled metrics.

        Args:
            response_id: LLM response ID
            prompt: Test prompt
            response_text: LLM response text
            metrics_config: Metrics configuration

        Returns:
            Dictionary of metric scores
        """
        scores = {}

        # Create DeepEval test case
        test_case = LLMTestCase(
            input=prompt.text,
            actual_output=response_text,
            expected_output="",
        )

        # Run custom TCLP metrics
        if metrics_config.tclp_custom:
            # TCLP Citation Metric
            if "tclp_citation" in metrics_config.tclp_custom:
                metric_config = metrics_config.tclp_custom["tclp_citation"]
                if metric_config.enabled:
                    metric = TCLPCitationMetric(threshold=metric_config.threshold)
                    score = metric.measure(test_case)

                    with self.db_manager.get_session() as session:
                        metric_score = MetricScore(
                            llm_response_id=response_id,
                            metric_name="tclp_citation",
                            metric_category="tclp_custom",
                            score=score,
                            threshold=metric_config.threshold,
                            passed=1 if metric.is_successful() else 0,
                            reasoning=metric.reason,
                        )
                        session.add(metric_score)
                        session.commit()

                    scores["tclp_citation"] = {
                        "score": score,
                        "passed": metric.is_successful(),
                        "threshold": metric_config.threshold,
                    }

            # Taxonomy Awareness Metric
            if "taxonomy_awareness" in metrics_config.tclp_custom:
                metric_config = metrics_config.tclp_custom["taxonomy_awareness"]
                if metric_config.enabled:
                    metric = TaxonomyAwarenessMetric(threshold=metric_config.threshold)
                    score = metric.measure(test_case)

                    with self.db_manager.get_session() as session:
                        metric_score = MetricScore(
                            llm_response_id=response_id,
                            metric_name="taxonomy_awareness",
                            metric_category="tclp_custom",
                            score=score,
                            threshold=metric_config.threshold,
                            passed=1 if metric.is_successful() else 0,
                            reasoning=metric.reason,
                        )
                        session.add(metric_score)
                        session.commit()

                    scores["taxonomy_awareness"] = {
                        "score": score,
                        "passed": metric.is_successful(),
                        "threshold": metric_config.threshold,
                    }

            # Entity Retrieval Accuracy Metric
            if "entity_retrieval_accuracy" in metrics_config.tclp_custom:
                metric_config = metrics_config.tclp_custom["entity_retrieval_accuracy"]
                if metric_config.enabled:
                    metric = EntityRetrievalAccuracyMetric(
                        threshold=metric_config.threshold,
                        expected_concepts=prompt.expected_concepts,
                    )
                    score = metric.measure(test_case)

                    with self.db_manager.get_session() as session:
                        metric_score = MetricScore(
                            llm_response_id=response_id,
                            metric_name="entity_retrieval_accuracy",
                            metric_category="tclp_custom",
                            score=score,
                            threshold=metric_config.threshold,
                            passed=1 if metric.is_successful() else 0,
                            reasoning=metric.reason,
                        )
                        session.add(metric_score)
                        session.commit()

                    scores["entity_retrieval_accuracy"] = {
                        "score": score,
                        "passed": metric.is_successful(),
                        "threshold": metric_config.threshold,
                    }

            # Metadata Precision Metric
            if "metadata_precision" in metrics_config.tclp_custom:
                metric_config = metrics_config.tclp_custom["metadata_precision"]
                if metric_config.enabled:
                    metric = MetadataPrecisionMetric(
                        threshold=metric_config.threshold,
                        expected_concepts=prompt.expected_concepts,
                    )
                    score = metric.measure(test_case)

                    with self.db_manager.get_session() as session:
                        metric_score = MetricScore(
                            llm_response_id=response_id,
                            metric_name="metadata_precision",
                            metric_category="tclp_custom",
                            score=score,
                            threshold=metric_config.threshold,
                            passed=1 if metric.is_successful() else 0,
                            reasoning=metric.reason,
                        )
                        session.add(metric_score)
                        session.commit()

                    scores["metadata_precision"] = {
                        "score": score,
                        "passed": metric.is_successful(),
                        "threshold": metric_config.threshold,
                    }

            # Relationship Traversal Score Metric
            if "relationship_traversal_score" in metrics_config.tclp_custom:
                metric_config = metrics_config.tclp_custom["relationship_traversal_score"]
                if metric_config.enabled:
                    metric = RelationshipTraversalScoreMetric(
                        threshold=metric_config.threshold,
                        expected_concepts=prompt.expected_concepts,
                    )
                    score = metric.measure(test_case)

                    with self.db_manager.get_session() as session:
                        metric_score = MetricScore(
                            llm_response_id=response_id,
                            metric_name="relationship_traversal_score",
                            metric_category="tclp_custom",
                            score=score,
                            threshold=metric_config.threshold,
                            passed=1 if metric.is_successful() else 0,
                            reasoning=metric.reason,
                        )
                        session.add(metric_score)
                        session.commit()

                    scores["relationship_traversal_score"] = {
                        "score": score,
                        "passed": metric.is_successful(),
                        "threshold": metric_config.threshold,
                    }

        return scores

    def _create_provider(self, provider_name: str, provider_config: Any) -> BaseLLMProvider:
        """
        Create LLM provider instance.

        Args:
            provider_name: Provider name
            provider_config: Provider configuration

        Returns:
            Provider instance
        """
        # Get API key from environment
        api_key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }

        api_key = os.getenv(api_key_map[provider_name])
        if not api_key:
            raise ValueError(f"API key not found for {provider_name}")

        # Create provider
        provider_map = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "gemini": GeminiProvider,
            "openrouter": OpenRouterProvider,
        }

        provider_class = provider_map[provider_name]
        # Convert Pydantic config to dict
        config_dict = provider_config.model_dump() if hasattr(provider_config, 'model_dump') else {}
        return provider_class(config=config_dict, api_key=api_key)
