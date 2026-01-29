"""Custom TCLP-specific evaluation metrics."""
import re
from typing import Optional, List
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


class EntityRetrievalAccuracyMetric(BaseMetric):
    """
    Custom metric: Did the LLM find the exact named clause required?

    Binary metric (1.0 or 0.0) checking if specific named clauses from
    expected_concepts appear in the response.
    """

    def __init__(self, threshold: float = 1.0, expected_concepts: Optional[List[str]] = None):
        """
        Initialise Entity Retrieval Accuracy metric.

        Args:
            threshold: Score threshold for success (0.0 to 1.0)
            expected_concepts: List of expected clause names/concepts from prompt
        """
        self.threshold = threshold
        self.expected_concepts = expected_concepts or []
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Calculate the metric score.

        Args:
            test_case: DeepEval test case

        Returns:
            Score (1.0 if expected entities found, 0.0 otherwise)
        """
        if not self.expected_concepts:
            # No expected concepts specified, mark as successful
            self.score = 1.0
            self.reason = "No expected concepts specified"
            return self.score

        response_text = test_case.actual_output.lower()

        # Check if any expected concepts are mentioned
        # Look for clause names (often possessive like "Evie's Clause")
        found_concepts = []
        missing_concepts = []

        for concept in self.expected_concepts:
            concept_lower = concept.lower()
            # Check for exact match with word boundaries
            if re.search(rf'\b{re.escape(concept_lower)}\b', response_text):
                found_concepts.append(concept)
            else:
                missing_concepts.append(concept)

        # Binary score: 1.0 if at least one expected concept found
        self.score = 1.0 if found_concepts else 0.0

        if self.score == 1.0:
            self.reason = f"Found {len(found_concepts)}/{len(self.expected_concepts)} expected entities: {', '.join(found_concepts[:3])}"
        else:
            self.reason = f"No expected entities found. Expected: {', '.join(self.expected_concepts[:3])}"

        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async version of measure."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Check if metric passed threshold."""
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Entity Retrieval Accuracy"


class MetadataPrecisionMetric(BaseMetric):
    """
    Custom metric: Did the LLM correctly interpret metadata filters?

    Checks if the response correctly respects jurisdiction, status,
    and sector constraints from the expected concepts.
    """

    # Metadata keywords to look for
    MAINTENANCE_STATUS = [
        "actively maintained",
        "maintained",
        "not maintained",
        "archived",
    ]

    JURISDICTIONS = [
        "england & wales",
        "england and wales",
        "scotland",
        "ireland",
        "deutschland",
        "germany",
        "usa",
        "united states",
    ]

    def __init__(self, threshold: float = 0.9, expected_concepts: Optional[List[str]] = None):
        """
        Initialise Metadata Precision metric.

        Args:
            threshold: Score threshold for success (0.0 to 1.0)
            expected_concepts: List of expected metadata terms from prompt
        """
        self.threshold = threshold
        self.expected_concepts = expected_concepts or []
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Calculate the metric score.

        Args:
            test_case: DeepEval test case

        Returns:
            Score (0.0 to 1.0) based on metadata constraint adherence
        """
        if not self.expected_concepts:
            # No expected concepts specified, mark as successful
            self.score = 1.0
            self.reason = "No expected metadata specified"
            return self.score

        response_text = test_case.actual_output.lower()

        # Check for metadata terms in expected concepts
        expected_metadata = []
        found_metadata = []

        for concept in self.expected_concepts:
            concept_lower = concept.lower()

            # Check if this concept is a metadata term
            is_metadata = False

            # Check maintenance status
            for status in self.MAINTENANCE_STATUS:
                if status in concept_lower:
                    is_metadata = True
                    expected_metadata.append(concept)
                    if re.search(rf'\b{re.escape(status)}\b', response_text):
                        found_metadata.append(concept)
                    break

            if not is_metadata:
                # Check jurisdictions
                for jurisdiction in self.JURISDICTIONS:
                    if jurisdiction in concept_lower:
                        is_metadata = True
                        expected_metadata.append(concept)
                        if re.search(rf'\b{re.escape(jurisdiction)}\b', response_text):
                            found_metadata.append(concept)
                        break

        # If no metadata constraints found in expected concepts, check if response mentions any
        if not expected_metadata:
            # Score based on whether response shows metadata awareness
            metadata_mentions = 0
            for status in self.MAINTENANCE_STATUS:
                if re.search(rf'\b{re.escape(status)}\b', response_text):
                    metadata_mentions += 1
                    break
            for jurisdiction in self.JURISDICTIONS:
                if re.search(rf'\b{re.escape(jurisdiction)}\b', response_text):
                    metadata_mentions += 1
                    break

            self.score = min(1.0, metadata_mentions * 0.5)
            self.reason = f"No specific metadata constraints; response shows {metadata_mentions} metadata references"
        else:
            # Score based on proportion of expected metadata found
            self.score = len(found_metadata) / len(expected_metadata) if expected_metadata else 1.0
            self.reason = f"Found {len(found_metadata)}/{len(expected_metadata)} expected metadata terms"

        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async version of measure."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Check if metric passed threshold."""
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Metadata Precision"


class RelationshipTraversalScoreMetric(BaseMetric):
    """
    Custom metric: Did the LLM correctly identify relationships?

    Checks if the response correctly identifies parent/child or
    variant relationships between clauses when expected.
    """

    # Relationship keywords
    RELATIONSHIP_KEYWORDS = [
        "child clause",
        "parent",
        "handbook",
        "contains",
        "section",
        "derived from",
        "based on",
        "related to",
        "variant",
        "transposition",
        "consolidated",
        "annex",
    ]

    def __init__(self, threshold: float = 0.8, expected_concepts: Optional[List[str]] = None):
        """
        Initialise Relationship Traversal Score metric.

        Args:
            threshold: Score threshold for success (0.0 to 1.0)
            expected_concepts: List of expected concepts that may indicate relationships
        """
        self.threshold = threshold
        self.expected_concepts = expected_concepts or []
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Calculate the metric score.

        Args:
            test_case: DeepEval test case

        Returns:
            Score (0.0 to 1.0) based on relationship identification
        """
        response_text = test_case.actual_output.lower()

        # Count relationship keywords mentioned
        relationships_found = []
        for keyword in self.RELATIONSHIP_KEYWORDS:
            if re.search(rf'\b{re.escape(keyword)}\b', response_text):
                relationships_found.append(keyword)

        # Check if expected concepts mention relationships
        relationship_expected = False
        for concept in self.expected_concepts:
            concept_lower = concept.lower()
            for keyword in self.RELATIONSHIP_KEYWORDS:
                if keyword in concept_lower:
                    relationship_expected = True
                    break
            if relationship_expected:
                break

        # Score based on relationship mentions
        if relationship_expected:
            # If relationships expected, score on whether found
            self.score = 1.0 if relationships_found else 0.0
            if self.score == 1.0:
                self.reason = f"Relationships identified: {', '.join(relationships_found[:3])}"
            else:
                self.reason = "Expected relationships not identified"
        else:
            # If no relationships expected, score on relationship awareness anyway
            if relationships_found:
                self.score = 1.0
                self.reason = f"Relationships mentioned: {', '.join(relationships_found[:3])}"
            else:
                # No relationships expected or found - neutral score
                self.score = 0.5
                self.reason = "No relationship context in prompt or response"

        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async version of measure."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Check if metric passed threshold."""
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Relationship Traversal Score"
