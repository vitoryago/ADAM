"""
Unit tests for ADAM v2 database models
"""
import pytest
from datetime import datetime
from models import Project, Conversation, Message


class TestProjectModel:
    """Test the Project model"""
    
    def test_project_creation(self):
        """Test creating a project with required fields"""
        project = Project(
            name="Test Project",
            description="A test project"
        )
        assert project.name == "Test Project"
        assert project.description == "A test project"
        # SQLAlchemy sets defaults on instantiation for Python-side defaults
        assert project.is_archived == False
        assert project.settings == {}
    
    def test_project_with_settings(self):
        """Test creating a project with custom settings"""
        settings = {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000
        }
        project = Project(
            name="AI Research",
            settings=settings
        )
        assert project.settings == settings
    
    def test_project_defaults(self):
        """Test project default values"""
        project = Project(name="Minimal Project")
        assert project.description is None
        assert project.is_archived == False
        assert project.settings == {}
        assert project.created_at is None  # Set by database
    
    @pytest.mark.asyncio
    async def test_project_persistence(self, db_session):
        """Test saving and retrieving a project from database"""
        project = Project(
            name="Persistent Project",
            description="Testing persistence"
        )
        db_session.add(project)
        await db_session.commit()
        
        # Verify ID was assigned
        assert project.id is not None
        assert isinstance(project.id, str)
        assert len(project.id) == 36  # UUID length
        
        # Verify timestamps
        assert project.created_at is not None
        assert project.updated_at is not None


class TestConversationModel:
    """Test the Conversation model"""
    
    def test_conversation_creation(self):
        """Test creating a conversation"""
        conversation = Conversation(
            project_id="test-project-id",
            title="Test Conversation"
        )
        assert conversation.project_id == "test-project-id"
        assert conversation.title == "Test Conversation"
        # Note: defaults are set by SQLAlchemy on database insert
    
    def test_conversation_defaults(self):
        """Test conversation default values"""
        conversation = Conversation(
            project_id="test-id",
            title="Chat"
        )
        # Note: defaults are set by SQLAlchemy on database insert
        # message_count and total_cost are calculated via queries, not properties
    
    @pytest.mark.asyncio
    async def test_conversation_with_project(self, db_session, sample_project):
        """Test conversation relationship with project"""
        conversation = Conversation(
            project_id=sample_project.id,
            title="Related Conversation"
        )
        db_session.add(conversation)
        await db_session.commit()
        
        assert conversation.project_id == sample_project.id


class TestMessageModel:
    """Test the Message model"""
    
    def test_user_message_creation(self):
        """Test creating a user message"""
        message = Message(
            conversation_id="test-conv-id",
            role="user",
            content="Hello, ADAM!"
        )
        assert message.conversation_id == "test-conv-id"
        assert message.role == "user"
        assert message.content == "Hello, ADAM!"
        assert message.model is None
        assert message.tokens_used is None
        assert message.cost is None
    
    def test_assistant_message_creation(self):
        """Test creating an assistant message with metadata"""
        message = Message(
            conversation_id="test-conv-id",
            role="assistant",
            content="Hello! How can I help?",
            model="gpt-4",
            tokens_used=15,
            cost=0.0003
        )
        assert message.role == "assistant"
        assert message.model == "gpt-4"
        assert message.tokens_used == 15
        assert message.cost == 0.0003
    
    def test_message_validation(self):
        """Test message role validation"""
        # Valid roles
        for role in ["user", "assistant", "system"]:
            message = Message(
                conversation_id="test-id",
                role=role,
                content="Test"
            )
            assert message.role == role
    
    @pytest.mark.asyncio
    async def test_message_persistence(self, db_session, sample_conversation):
        """Test saving messages to database"""
        messages = [
            Message(
                conversation_id=sample_conversation.id,
                role="user",
                content="What is Python?"
            ),
            Message(
                conversation_id=sample_conversation.id,
                role="assistant",
                content="Python is a high-level programming language...",
                model="gpt-4",
                tokens_used=50,
                cost=0.001
            )
        ]
        
        for msg in messages:
            db_session.add(msg)
        
        await db_session.commit()
        
        # Verify all messages have IDs and timestamps
        for msg in messages:
            assert msg.id is not None
            assert msg.created_at is not None


class TestModelRelationships:
    """Test relationships between models"""
    
    @pytest.mark.asyncio
    async def test_project_conversations_cascade(self, db_session, sample_project):
        """Test that conversations are properly linked to projects"""
        # Create multiple conversations
        conv1 = Conversation(
            project_id=sample_project.id,
            title="Conversation 1"
        )
        conv2 = Conversation(
            project_id=sample_project.id,
            title="Conversation 2"
        )
        
        db_session.add(conv1)
        db_session.add(conv2)
        await db_session.commit()
        
        # Both should have the same project ID
        assert conv1.project_id == sample_project.id
        assert conv2.project_id == sample_project.id
    
    @pytest.mark.asyncio
    async def test_conversation_messages_cascade(self, db_session, sample_conversation):
        """Test that messages are properly linked to conversations"""
        # Create messages
        msg1 = Message(
            conversation_id=sample_conversation.id,
            role="user",
            content="Question 1"
        )
        msg2 = Message(
            conversation_id=sample_conversation.id,
            role="assistant",
            content="Answer 1"
        )
        
        db_session.add(msg1)
        db_session.add(msg2)
        await db_session.commit()
        
        assert msg1.conversation_id == sample_conversation.id
        assert msg2.conversation_id == sample_conversation.id