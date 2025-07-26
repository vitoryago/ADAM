"""
Unit tests for conversation management
"""
import pytest
from sqlalchemy import select
from models import Project, Conversation, Message


class TestConversationManagement:
    """Test conversation CRUD operations"""
    
    @pytest.mark.asyncio
    async def test_create_conversation(self, db_session):
        """Test creating a conversation in a project"""
        # Create project first
        project = Project(name="Test Project")
        db_session.add(project)
        await db_session.commit()
        
        # Create conversation
        conversation = Conversation(
            project_id=project.id,
            title="Test Conversation"
        )
        db_session.add(conversation)
        await db_session.commit()
        
        # Verify
        assert conversation.id is not None
        assert conversation.project_id == project.id
        assert conversation.title == "Test Conversation"
        assert conversation.is_pinned is False
    
    @pytest.mark.asyncio
    async def test_list_project_conversations(self, db_session, sample_project):
        """Test listing all conversations in a project"""
        # Create multiple conversations
        conversations = []
        for i in range(3):
            conv = Conversation(
                project_id=sample_project.id,
                title=f"Conversation {i}"
            )
            db_session.add(conv)
            conversations.append(conv)
        
        # Pin one conversation
        conversations[1].is_pinned = True
        
        await db_session.commit()
        
        # Query conversations
        result = await db_session.execute(
            select(Conversation)
            .where(Conversation.project_id == sample_project.id)
            .order_by(
                Conversation.is_pinned.desc(),
                Conversation.created_at.desc()
            )
        )
        
        queried_convs = result.scalars().all()
        
        # Verify
        assert len(queried_convs) == 3
        assert queried_convs[0].is_pinned is True  # Pinned first
        assert all(c.project_id == sample_project.id for c in queried_convs)
    
    @pytest.mark.asyncio
    async def test_update_conversation(self, db_session, sample_conversation):
        """Test updating conversation title and pin status"""
        # Update title
        sample_conversation.title = "Updated Title"
        sample_conversation.is_pinned = True
        
        await db_session.commit()
        await db_session.refresh(sample_conversation)
        
        # Verify
        assert sample_conversation.title == "Updated Title"
        assert sample_conversation.is_pinned is True
    
    @pytest.mark.asyncio
    async def test_delete_conversation_cascade(self, db_session, sample_conversation):
        """Test that deleting conversation deletes its messages"""
        # Add messages
        for i in range(3):
            message = Message(
                conversation_id=sample_conversation.id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}"
            )
            db_session.add(message)
        
        await db_session.commit()
        
        # Verify messages exist
        msg_result = await db_session.execute(
            select(Message).where(Message.conversation_id == sample_conversation.id)
        )
        assert len(msg_result.scalars().all()) == 3
        
        # Delete conversation
        await db_session.delete(sample_conversation)
        await db_session.commit()
        
        # Verify conversation deleted
        conv_result = await db_session.execute(
            select(Conversation).where(Conversation.id == sample_conversation.id)
        )
        assert conv_result.scalar_one_or_none() is None
        
        # Verify messages also deleted
        msg_result = await db_session.execute(
            select(Message).where(Message.conversation_id == sample_conversation.id)
        )
        assert len(msg_result.scalars().all()) == 0
    
    @pytest.mark.asyncio
    async def test_conversation_message_stats(self, db_session, sample_conversation):
        """Test calculating conversation statistics"""
        # Add messages with costs
        messages = [
            Message(
                conversation_id=sample_conversation.id,
                role="user",
                content="Question 1"
            ),
            Message(
                conversation_id=sample_conversation.id,
                role="assistant",
                content="Answer 1",
                model="gpt-4",
                tokens_used=100,
                cost=0.003
            ),
            Message(
                conversation_id=sample_conversation.id,
                role="user",
                content="Question 2"
            ),
            Message(
                conversation_id=sample_conversation.id,
                role="assistant",
                content="Answer 2",
                model="gpt-4",
                tokens_used=150,
                cost=0.0045
            )
        ]
        
        for msg in messages:
            db_session.add(msg)
        
        await db_session.commit()
        
        # Refresh to load relationships
        await db_session.refresh(sample_conversation)
        
        # Calculate stats using query instead of property
        from sqlalchemy import func
        stats_result = await db_session.execute(
            select(
                func.count(Message.id).label("count"),
                func.sum(Message.cost).label("total")
            ).where(Message.conversation_id == sample_conversation.id)
        )
        stats = stats_result.one()
        
        assert stats.count == 4
        assert float(stats.total or 0) == 0.0075