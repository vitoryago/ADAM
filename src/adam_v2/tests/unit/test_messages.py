"""
Unit tests for message management
"""
import pytest
from datetime import datetime
from models import Message, Conversation, Project
from services.llm_service import LLMService, LLMResponse, StreamChunk


class TestMessageModel:
    """Test the Message model"""
    
    def test_create_user_message(self):
        """Test creating a user message"""
        message = Message(
            conversation_id="test-conv-id",
            role="user",
            content="Hello ADAM!"
        )
        
        assert message.conversation_id == "test-conv-id"
        assert message.role == "user"
        assert message.content == "Hello ADAM!"
        assert message.model is None
        assert message.tokens_used is None
        assert message.cost is None
        assert message.has_image is False
    
    def test_create_assistant_message(self):
        """Test creating an assistant message with metadata"""
        message = Message(
            conversation_id="test-conv-id",
            role="assistant",
            content="Hello! I'm ADAM, how can I help you?",
            model="grok-3-mini-high",
            tokens_used=25,
            cost=0.00005
        )
        
        assert message.role == "assistant"
        assert message.model == "grok-3-mini-high"
        assert message.tokens_used == 25
        assert message.cost == 0.00005
    
    def test_message_with_image(self):
        """Test creating a message with image"""
        message = Message(
            conversation_id="test-conv-id",
            role="user",
            content="What's in this image?",
            has_image=True,
            image_url="data:image/jpeg;base64,/9j/4AAQ..."
        )
        
        assert message.has_image is True
        assert message.image_url.startswith("data:image")
    
    @pytest.mark.asyncio
    async def test_message_persistence(self, db_session, sample_conversation):
        """Test saving and retrieving messages"""
        # Create messages
        user_msg = Message(
            conversation_id=sample_conversation.id,
            role="user",
            content="What is Python?"
        )
        
        assistant_msg = Message(
            conversation_id=sample_conversation.id,
            role="assistant",
            content="Python is a high-level programming language...",
            model="grok-4",
            tokens_used=50,
            cost=0.001
        )
        
        db_session.add(user_msg)
        db_session.add(assistant_msg)
        await db_session.commit()
        
        # Verify IDs assigned
        assert user_msg.id is not None
        assert assistant_msg.id is not None
        
        # Verify timestamps
        assert user_msg.created_at is not None
        assert assistant_msg.created_at is not None


class TestLLMService:
    """Test the LLM service"""
    
    def test_llm_service_initialization(self):
        """Test LLM service initialization with project settings"""
        settings = {
            "model": "grok-4",
            "temperature": 0.5,
            "max_tokens": 1500
        }
        
        service = LLMService(project_settings=settings)
        
        assert service.default_model == "grok-4"
        assert service.temperature == 0.5
        assert service.max_tokens == 1500
    
    def test_llm_service_defaults(self):
        """Test LLM service with default settings"""
        service = LLMService()
        
        assert service.default_model is None
        assert service.temperature == 0.7
        assert service.max_tokens == 2000
    
    @pytest.mark.asyncio
    async def test_generate_response_mock(self):
        """Test generating response with mock LLM"""
        service = LLMService()
        
        # If ADAM LLM is not available, it should return mock
        if not service.llm_client:
            response = await service.generate_response(
                message="Hello",
                history=[],
                memory_context=""
            )
            
            assert isinstance(response, LLMResponse)
            assert response.model_used == "mock"
            assert response.content.startswith("This is a mock response")
    
    @pytest.mark.asyncio
    async def test_stream_response_mock(self):
        """Test streaming response with mock LLM"""
        service = LLMService()
        
        # If ADAM LLM is not available, it should stream mock
        if not service.llm_client:
            chunks = []
            async for chunk in service.stream_response(
                message="Hello",
                history=[],
                memory_context=""
            ):
                chunks.append(chunk)
            
            assert len(chunks) > 0
            assert all(isinstance(c, StreamChunk) for c in chunks)
            assert chunks[-1].is_final is True
    
    def test_estimate_cost(self):
        """Test cost estimation"""
        service = LLMService()
        
        # Test basic estimation
        cost = service.estimate_cost(
            message="This is a test message with about ten words or so",
            model="grok-3-mini-high"
        )
        
        assert cost >= 0
        
        # Test with image
        cost_with_image = service.estimate_cost(
            message="What's in this image?",
            model="grok-4",
            has_image=True
        )
        
        assert cost_with_image >= cost
    
    def test_model_selection_by_complexity(self):
        """Test model selection based on complexity"""
        service = LLMService()
        
        # Test without actual complexity analyzer
        # Should return default model
        model = service._select_model_by_complexity(None)
        assert model == "grok-3-mini-high"


class TestMessageHistory:
    """Test message history management"""
    
    @pytest.mark.asyncio
    async def test_conversation_history_order(self, db_session, sample_conversation):
        """Test that messages maintain correct order"""
        # Create messages with specific order
        messages = []
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            msg = Message(
                conversation_id=sample_conversation.id,
                role=role,
                content=f"Message {i}"
            )
            db_session.add(msg)
            messages.append(msg)
        
        await db_session.commit()
        
        # Refresh to get timestamps
        for msg in messages:
            await db_session.refresh(msg)
        
        # Verify order by creation time
        sorted_messages = sorted(messages, key=lambda m: m.created_at)
        for i, msg in enumerate(sorted_messages):
            assert msg.content == f"Message {i}"
    
    @pytest.mark.asyncio
    async def test_conversation_cost_tracking(self, db_session, sample_conversation):
        """Test tracking costs across conversation"""
        # Add messages with costs
        costs = [0.001, 0.002, 0.0015, 0.0025]
        total_cost = 0.0
        
        for i, cost in enumerate(costs):
            msg = Message(
                conversation_id=sample_conversation.id,
                role="assistant",
                content=f"Response {i}",
                model="grok-4",
                tokens_used=50 * (i + 1),
                cost=cost
            )
            db_session.add(msg)
            total_cost += cost
        
        await db_session.commit()
        
        # Calculate total cost
        from sqlalchemy import select, func
        result = await db_session.execute(
            select(func.sum(Message.cost))
            .where(Message.conversation_id == sample_conversation.id)
        )
        
        db_total = result.scalar() or 0.0
        assert abs(db_total - total_cost) < 0.0001  # Float comparison