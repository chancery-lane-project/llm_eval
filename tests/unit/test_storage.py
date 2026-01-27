"""Tests for database storage layer."""
import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from tclp_benchmark.storage.models import Base, EvaluationRun, TestCase, LLMResponse, MetricScore
from tclp_benchmark.storage.database import DatabaseManager


@pytest.fixture
def in_memory_db():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(in_memory_db):
    """Create database session."""
    with Session(in_memory_db) as session:
        yield session


def test_evaluation_run_creation(db_session):
    """Test creating an evaluation run."""
    run = EvaluationRun(
        version_tag="pre-jsonld",
        description="Baseline evaluation before semantic markup",
        status="running",
    )
    db_session.add(run)
    db_session.commit()

    assert run.id is not None
    assert run.version_tag == "pre-jsonld"
    assert run.status == "running"
    assert isinstance(run.created_at, datetime)


def test_test_case_relationship(db_session):
    """Test EvaluationRun -> TestCase relationship."""
    run = EvaluationRun(version_tag="test-run", status="running")
    db_session.add(run)
    db_session.flush()

    test_case = TestCase(
        evaluation_run_id=run.id,
        prompt_category="direct_lookup",
        prompt_text="Find the net zero clause",
    )
    db_session.add(test_case)
    db_session.commit()

    assert len(run.test_cases) == 1
    assert run.test_cases[0].prompt_category == "direct_lookup"


def test_llm_response_relationship(db_session):
    """Test TestCase -> LLMResponse relationship."""
    run = EvaluationRun(version_tag="test-run", status="running")
    db_session.add(run)
    db_session.flush()

    test_case = TestCase(
        evaluation_run_id=run.id,
        prompt_category="direct_lookup",
        prompt_text="Test prompt",
    )
    db_session.add(test_case)
    db_session.flush()

    response = LLMResponse(
        test_case_id=test_case.id,
        provider_name="openai",
        model_name="gpt-4o",
        response_text="The Chancery Lane Project provides...",
    )
    db_session.add(response)
    db_session.commit()

    assert len(test_case.llm_responses) == 1
    assert test_case.llm_responses[0].provider_name == "openai"


def test_metric_score_relationship(db_session):
    """Test LLMResponse -> MetricScore relationship."""
    run = EvaluationRun(version_tag="test-run", status="running")
    db_session.add(run)
    db_session.flush()

    test_case = TestCase(
        evaluation_run_id=run.id,
        prompt_category="direct_lookup",
        prompt_text="Test",
    )
    db_session.add(test_case)
    db_session.flush()

    response = LLMResponse(
        test_case_id=test_case.id,
        provider_name="openai",
        model_name="gpt-4o",
        response_text="Test response",
    )
    db_session.add(response)
    db_session.flush()

    score = MetricScore(
        llm_response_id=response.id,
        metric_name="tclp_citation",
        metric_category="tclp_custom",
        score=1.0,
        threshold=0.8,
        passed=1,
    )
    db_session.add(score)
    db_session.commit()

    assert len(response.metric_scores) == 1
    assert response.metric_scores[0].metric_name == "tclp_citation"
    assert response.metric_scores[0].passed == 1


def test_cascade_delete(db_session):
    """Test cascade deletion of related records."""
    run = EvaluationRun(version_tag="test-run", status="running")
    db_session.add(run)
    db_session.flush()

    test_case = TestCase(
        evaluation_run_id=run.id,
        prompt_category="direct_lookup",
        prompt_text="Test",
    )
    db_session.add(test_case)
    db_session.commit()

    # Delete run should cascade to test_case
    db_session.delete(run)
    db_session.commit()

    assert db_session.query(TestCase).count() == 0


def test_database_manager_initialisation(tmp_path):
    """Test DatabaseManager creates database file."""
    db_path = tmp_path / "test.db"
    db_manager = DatabaseManager(f"sqlite:///{db_path}")
    db_manager.create_tables()

    assert db_path.exists()


def test_database_manager_get_session(tmp_path):
    """Test DatabaseManager provides working session."""
    db_path = tmp_path / "test.db"
    db_manager = DatabaseManager(f"sqlite:///{db_path}")
    db_manager.create_tables()

    with db_manager.get_session() as session:
        run = EvaluationRun(version_tag="test", status="running")
        session.add(run)
        session.commit()

        assert run.id is not None
