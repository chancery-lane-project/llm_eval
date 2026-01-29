"""Database connection and session management."""
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
from .models import Base


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, database_url: str):
        """
        Initialise database manager.

        Args:
            database_url: SQLAlchemy database URL (e.g., "sqlite:///data/benchmark.db")
        """
        self.database_url = database_url
        
        # For SQLite, ensure the directory exists
        if database_url.startswith("sqlite:///"):
            # Extract the file path
            # sqlite:///path/to/db.db -> path/to/db.db
            # sqlite:////absolute/path/db.db -> /absolute/path/db.db
            if database_url.startswith("sqlite:////"):
                # Absolute path (4 slashes)
                db_path = database_url[10:]  # Remove "sqlite:////"
            else:
                # Relative path (3 slashes)
                db_path = database_url[10:]  # Remove "sqlite:///"
            
            # Create parent directory if it doesn't exist
            db_file = Path(db_path)
            if db_file.parent != Path("."):
                db_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.engine = create_engine(
            database_url,
            echo=False,  # Set to True for SQL debugging
            future=True,  # Use SQLAlchemy 2.0 style
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get database session as context manager.

        Yields:
            SQLAlchemy session
        """
        session = self.SessionLocal()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
