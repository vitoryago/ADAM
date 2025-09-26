"""
Database Engine and Connection Management
"""

import asyncio
from typing import Optional, AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
import logging

from adam.config import get_config
from .models import Base

logger = logging.getLogger(__name__)


class DatabaseEngine:
    """
    Unified database engine manager
    Handles connection pooling, session management, and cleanup
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database engine

        Args:
            database_url: Database URL (if None, loads from config)
        """
        if database_url is None:
            config = get_config()
            database_url = config.database.url

        self.database_url = database_url

        # Create async engine with optimized settings
        engine_kwargs = {
            'echo': False,  # Set to True for SQL debugging
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

        # Create session maker
        self.async_session = async_sessionmaker(
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

    async def drop_tables(self):
        """Drop all tables (use with caution!)"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.warning("All database tables dropped")

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
    Get the global database engine instance

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
    Create all database tables

    Args:
        database_url: Database URL (optional)
    """
    engine = get_engine(database_url)
    await engine.create_tables()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async database session from the global engine

    Returns:
        AsyncSession instance
    """
    engine = get_engine()
    async for session in engine.get_session():
        yield session


# Context manager for easy session usage
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


# Convenience function
def db_session(engine: Optional[DatabaseEngine] = None) -> DatabaseSession:
    """
    Create a database session context manager

    Args:
        engine: Optional database engine

    Returns:
        DatabaseSession context manager

    Usage:
        async with db_session() as session:
            # Use session here
            result = await session.execute(...)
    """
    return DatabaseSession(engine)