"""SQLAlchemy ORM models for database schema."""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime, timezone

Base = declarative_base()


class EvaluationRun(Base):
    """Top-level evaluation run with version tagging."""

    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True)
    version_tag = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(20), default="running")  # running, completed, failed

    # Relationships
    test_cases = relationship(
        "TestCase", back_populates="evaluation_run", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<EvaluationRun(id={self.id}, version_tag='{self.version_tag}', status='{self.status}')>"


class TestCase(Base):
    """Individual test prompt execution."""

    __tablename__ = "test_cases"

    id = Column(Integer, primary_key=True)
    evaluation_run_id = Column(Integer, ForeignKey("evaluation_runs.id"), nullable=False, index=True)

    # Test prompt metadata
    prompt_category = Column(String(50), nullable=False, index=True)
    prompt_text = Column(Text, nullable=False)
    expected_outcome = Column(Text, nullable=True)

    # Execution metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    evaluation_run = relationship("EvaluationRun", back_populates="test_cases")
    llm_responses = relationship(
        "LLMResponse", back_populates="test_case", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<TestCase(id={self.id}, category='{self.prompt_category}')>"


class LLMResponse(Base):
    """LLM response to a test case."""

    __tablename__ = "llm_responses"

    id = Column(Integer, primary_key=True)
    test_case_id = Column(Integer, ForeignKey("test_cases.id"), nullable=False, index=True)

    # Provider details
    provider_name = Column(String(50), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)

    # Search context
    search_query = Column(Text, nullable=True)
    search_results = Column(JSON, nullable=True)

    # LLM output
    response_text = Column(Text, nullable=False)
    response_metadata = Column(JSON, nullable=True)

    # Execution metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    execution_time_ms = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)

    # Relationships
    test_case = relationship("TestCase", back_populates="llm_responses")
    metric_scores = relationship(
        "MetricScore", back_populates="llm_response", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<LLMResponse(id={self.id}, provider='{self.provider_name}', model='{self.model_name}')>"


class MetricScore(Base):
    """Individual metric evaluation score."""

    __tablename__ = "metric_scores"

    id = Column(Integer, primary_key=True)
    llm_response_id = Column(Integer, ForeignKey("llm_responses.id"), nullable=False, index=True)

    # Metric details
    metric_name = Column(String(100), nullable=False, index=True)
    metric_category = Column(String(50), nullable=False, index=True)

    # Score
    score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    passed = Column(Integer, nullable=False)  # Boolean: 1 if passed, 0 if failed

    # Detailed breakdown
    reasoning = Column(Text, nullable=True)
    metric_metadata = Column(JSON, nullable=True)

    # Execution metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    llm_response = relationship("LLMResponse", back_populates="metric_scores")

    def __repr__(self):
        return f"<MetricScore(id={self.id}, metric='{self.metric_name}', score={self.score:.3f})>"
