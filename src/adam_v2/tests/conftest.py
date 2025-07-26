"""
Pytest configuration and shared fixtures for ADAM v2 tests
"""
import pytest
import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
import tempfile
import shutil
from pathlib import Path

# Import our models and app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import Base, get_db
from main import app
from models import Project, Conversation, Message


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def test_db():
    """Create a test database for each test function."""
    # Create temporary directory for test databases
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    
    # Create async engine
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        echo=False
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session factory
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    yield async_session
    
    # Cleanup
    await engine.dispose()
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
async def db_session(test_db) -> AsyncSession:
    """Get a database session for testing."""
    async with test_db() as session:
        yield session


@pytest.fixture(scope="function")
def test_client(test_db):
    """Create a test client with database override."""
    async def override_get_db():
        async with test_db() as session:
            yield session
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides.clear()


@pytest.fixture
async def sample_project(db_session):
    """Create a sample project for testing."""
    project = Project(
        name="Test Project",
        description="A test project for unit tests",
        settings={"model": "gpt-4", "temperature": 0.7}
    )
    db_session.add(project)
    await db_session.commit()
    await db_session.refresh(project)
    return project


@pytest.fixture
async def sample_conversation(db_session, sample_project):
    """Create a sample conversation for testing."""
    conversation = Conversation(
        project_id=sample_project.id,
        title="Test Conversation"
    )
    db_session.add(conversation)
    await db_session.commit()
    await db_session.refresh(conversation)
    return conversation


@pytest.fixture
async def sample_messages(db_session, sample_conversation):
    """Create sample messages for testing."""
    messages = [
        Message(
            conversation_id=sample_conversation.id,
            role="user",
            content="Hello, ADAM!"
        ),
        Message(
            conversation_id=sample_conversation.id,
            role="assistant",
            content="Hello! How can I help you today?",
            model="gpt-4",
            tokens_used=15,
            cost=0.0003
        ),
        Message(
            conversation_id=sample_conversation.id,
            role="user",
            content="Can you help me with Python?"
        ),
        Message(
            conversation_id=sample_conversation.id,
            role="assistant",
            content="Of course! I'd be happy to help you with Python. What specific topic or problem would you like to discuss?",
            model="gpt-4",
            tokens_used=25,
            cost=0.0005
        )
    ]
    
    for message in messages:
        db_session.add(message)
    
    await db_session.commit()
    return messages


@pytest.fixture
def mock_memory_manager(mocker):
    """Mock the ProjectMemoryManager for testing."""
    mock = mocker.Mock()
    mock.store_memory.return_value = "mem_12345"
    mock.search_memories.return_value = [
        {
            "id": "mem_001",
            "content": "Previous conversation about Python",
            "metadata": {"timestamp": "2024-01-01T12:00:00"},
            "distance": 0.1
        }
    ]
    mock.get_project_stats.return_value = {
        "total_memories": 42,
        "collection_name": "test_collection"
    }
    return mock


@pytest.fixture
def memory_temp_dir():
    """Create temporary directory for memory storage during tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)