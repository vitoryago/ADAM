"""
ADAM Database Module

Unified database engine setup with SQLAlchemy. Provides engine creation,
session management, and the Base declarative base for ORM models.

ORM model definitions live in adam.api.models (Task 7).
"""

import logging
from typing import Optional, AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Declarative base for all ORM models
Base = declarative_base()


class DatabaseEngine:
    """
    Unified database engine manager.
    Handles connection pooling, session management, and cleanup.
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database engine.

        Args:
            database_url: Database URL. If None, loads from config.
                          Defaults to sqlite:///./data/adam.db
        """
        if database_url is None:
            from adam.config import get_config
            config = get_config()
            database_url = config.database.url

        # Ensure async driver for SQLite
        if database_url.startswith('sqlite:///') and 'aiosqlite' not in database_url:
            database_url = database_url.replace('sqlite:///', 'sqlite+aiosqlite:///')

        self.database_url = database_url

        # Create async engine with optimized settings
        engine_kwargs = {
            'echo': False,
            'future': True,
        }

        # Add connection pool settings for non-SQLite databases
        if not database_url.startswith('sqlite'):
            engine_kwargs.update({
                'pool_size': 20,
                'max_overflow': 30,
                'pool_pre_ping': True,
                'pool_recycle': 3600,
            })

        self.engine = create_async_engine(database_url, **engine_kwargs)

        # Create session maker (compatible with SQLAlchemy 1.4+)
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def create_tables(self):
        """Create all tables in the database"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session"""
        async with self.async_session() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def health_check(self) -> bool:
        """Check if database is healthy"""
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def close(self):
        """Close the database engine"""
        await self.engine.dispose()
        logger.info("Database engine closed")

    def __str__(self):
        return f"DatabaseEngine({self.database_url})"


# Global engine instance
_engine: Optional[DatabaseEngine] = None


def get_engine(database_url: Optional[str] = None) -> DatabaseEngine:
    """
    Get the global database engine instance.

    Args:
        database_url: Database URL (only used on first call)

    Returns:
        DatabaseEngine instance
    """
    global _engine
    if _engine is None:
        _engine = DatabaseEngine(database_url)
    return _engine


async def create_tables(database_url: Optional[str] = None):
    """
    Create all database tables.

    Args:
        database_url: Database URL (optional)
    """
    engine = get_engine(database_url)
    await engine.create_tables()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async database session from the global engine.

    Returns:
        AsyncSession instance
    """
    engine = get_engine()
    async for session in engine.get_session():
        yield session


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session for FastAPI dependency injection.
    Commits on success, rolls back on error.
    """
    engine = get_engine()
    async with engine.async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database tables (convenience alias)."""
    await create_tables()


async def close_db():
    """Close database connections."""
    engine = get_engine()
    await engine.close()


class DatabaseSession:
    """Context manager for database sessions"""

    def __init__(self, engine: Optional[DatabaseEngine] = None):
        self.engine = engine or get_engine()

    async def __aenter__(self) -> AsyncSession:
        self.session_gen = self.engine.get_session()
        self.session = await self.session_gen.__anext__()
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                await self.session.rollback()
            await self.session_gen.aclose()
        except Exception as e:
            logger.error(f"Error closing database session: {e}")


def db_session(engine: Optional[DatabaseEngine] = None) -> DatabaseSession:
    """
    Create a database session context manager.

    Usage:
        async with db_session() as session:
            result = await session.execute(...)
    """
    return DatabaseSession(engine)
