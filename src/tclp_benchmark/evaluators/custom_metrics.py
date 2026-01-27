"""Custom TCLP-specific evaluation metrics."""
import re
from typing import Optional
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class TCLPCitationMetric(BaseMetric):
    """
    Custom metric: Does the response mention The Chancery Lane Project?

    Binary metric checking for explicit TCLP mentions in various forms.
    """

    def __init__(self, threshold: float = 0.8):
        """
        Initialise TCLP Citation metric.

        Args:
            threshold: Score threshold for success (0.0 to 1.0)
        """
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Calculate the metric score.

        Args:
            test_case: DeepEval test case

        Returns:
            Score (1.0 if mentioned, 0.0 if not)
        """
        response_text = test_case.actual_output.lower()

        # Check for various mentions of TCLP
        patterns = [
            r'\btclp\b',
            r'\bthe chancery lane project\b',
            r'\bchancery lane project\b',
            r'\bchancerylaneproject\.org\b',
        ]

        mentions_found = []
        for pattern in patterns:
            if re.search(pattern, response_text):
                mentions_found.append(pattern)

        # Binary score: 1.0 if mentioned, 0.0 if not
        self.score = 1.0 if mentions_found else 0.0

        if self.score == 1.0:
            self.reason = f"TCLP mentioned (patterns matched: {len(mentions_found)})"
        else:
            self.reason = "TCLP not mentioned in response"

        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async version of measure."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Check if metric passed threshold."""
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "TCLP Citation"


class TaxonomyAwarenessMetric(BaseMetric):
    """
    Custom metric: Does the response demonstrate awareness of TCLP taxonomies?

    Checks for mentions of climate impacts, jurisdictions, practice areas, etc.
    Scores based on number of taxonomy terms mentioned.
    """

    # TCLP taxonomy terms
    CLIMATE_IMPACTS = [
        "carbon reduction",
        "net zero",
        "biodiversity",
        "supply chain emissions",
        "scope 3",
        "renewable energy",
        "circular economy",
        "waste reduction",
    ]

    JURISDICTIONS = [
        "uk",
        "united kingdom",
        "england",
        "wales",
        "scotland",
        "eu",
        "european union",
        "us",
        "united states",
    ]

    PRACTICE_AREAS = [
        "commercial",
        "construction",
        "procurement",
        "energy",
        "real estate",
        "finance",
        "corporate",
    ]

    def __init__(self, threshold: float = 0.6):
        """
        Initialise Taxonomy Awareness metric.

        Args:
            threshold: Score threshold for success (0.0 to 1.0)
        """
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Calculate the metric score.

        Args:
            test_case: DeepEval test case

        Returns:
            Score based on taxonomy term mentions (0.0 to 1.0)
        """
        response_text = test_case.actual_output.lower()

        # Count mentions from each category using word boundaries
        climate_mentions = sum(
            1 for term in self.CLIMATE_IMPACTS
            if re.search(rf'\b{re.escape(term)}\b', response_text)
        )
        jurisdiction_mentions = sum(
            1 for term in self.JURISDICTIONS
            if re.search(rf'\b{re.escape(term)}\b', response_text)
        )
        practice_mentions = sum(
            1 for term in self.PRACTICE_AREAS
            if re.search(rf'\b{re.escape(term)}\b', response_text)
        )

        total_mentions = climate_mentions + jurisdiction_mentions + practice_mentions

        # Score based on number of taxonomy terms mentioned
        # 0 mentions = 0.0, 1 mention = 0.3, 2 mentions = 0.6, 3+ mentions = 1.0
        if total_mentions == 0:
            self.score = 0.0
        elif total_mentions == 1:
            self.score = 0.3
        elif total_mentions == 2:
            self.score = 0.6
        else:
            self.score = 1.0

        self.reason = (
            f"Taxonomy awareness: {total_mentions} terms mentioned "
            f"(climate: {climate_mentions}, jurisdiction: {jurisdiction_mentions}, "
            f"practice: {practice_mentions})"
        )

        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async version of measure."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Check if metric passed threshold."""
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Taxonomy Awareness"
