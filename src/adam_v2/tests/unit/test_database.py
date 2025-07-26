"""
Unit tests for database configuration and session management
"""
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import init_db, get_db
from models import Project, Conversation, Message


class TestDatabaseConfiguration:
    """Test database setup and configuration"""
    
    @pytest.mark.asyncio
    async def test_database_initialization(self, test_db):
        """Test that database can be initialized"""
        # Create a session
        async with test_db() as session:
            assert isinstance(session, AsyncSession)
            
            # Verify we can execute a simple query
            result = await session.execute(select(1))
            assert result.scalar() == 1
    
    @pytest.mark.asyncio
    async def test_tables_created(self, db_session):
        """Test that all tables are created"""
        # Try to query each table
        tables_exist = True
        
        try:
            await db_session.execute(select(Project).limit(1))
            await db_session.execute(select(Conversation).limit(1))
            await db_session.execute(select(Message).limit(1))
        except Exception:
            tables_exist = False
        
        assert tables_exist
    
    @pytest.mark.asyncio
    async def test_get_db_dependency(self, test_db):
        """Test the get_db dependency function"""
        # Use the dependency
        async for session in get_db():
            assert isinstance(session, AsyncSession)
            
            # Create a test object
            project = Project(name="Test DB Project")
            session.add(project)
            await session.commit()
            
            # Verify it was saved
            result = await session.execute(
                select(Project).where(Project.name == "Test DB Project")
            )
            saved_project = result.scalar_one_or_none()
            assert saved_project is not None
            assert saved_project.name == "Test DB Project"
            break
    
    @pytest.mark.asyncio
    async def test_rollback_on_error(self, test_db):
        """Test that transactions rollback on error"""
        async with test_db() as session:
            # Create a project
            project = Project(name="Rollback Test")
            session.add(project)
            await session.flush()
            
            # Force an error before commit
            try:
                # This will cause an integrity error
                duplicate = Project(id=project.id, name="Duplicate")
                session.add(duplicate)
                await session.commit()
            except Exception:
                await session.rollback()
            
            # Verify nothing was saved
            result = await session.execute(
                select(Project).where(Project.name == "Rollback Test")
            )
            assert result.scalar_one_or_none() is None
    
    @pytest.mark.asyncio
    async def test_cascade_delete(self, db_session):
        """Test cascade delete relationships"""
        # Create project with conversations and messages
        project = Project(name="Cascade Test")
        db_session.add(project)
        await db_session.commit()
        
        conversation = Conversation(
            project_id=project.id,
            title="Test Conversation"
        )
        db_session.add(conversation)
        await db_session.commit()
        
        message = Message(
            conversation_id=conversation.id,
            role="user",
            content="Test message"
        )
        db_session.add(message)
        await db_session.commit()
        
        # Delete the project
        await db_session.delete(project)
        await db_session.commit()
        
        # Verify conversation and message were also deleted
        conv_result = await db_session.execute(
            select(Conversation).where(Conversation.id == conversation.id)
        )
        assert conv_result.scalar_one_or_none() is None
        
        msg_result = await db_session.execute(
            select(Message).where(Message.id == message.id)
        )
        assert msg_result.scalar_one_or_none() is None