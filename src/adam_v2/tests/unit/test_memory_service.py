"""
Unit tests for memory service
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import chromadb
from chromadb.config import Settings

from services.memory_service import (
    ProjectMemoryService,
    MemorySearchResult,
    MemoryType,
    ADAM_MEMORY_AVAILABLE
)


class TestProjectMemoryService:
    """Test the project memory service"""
    
    @pytest.fixture
    def mock_chroma_client(self):
        """Mock ChromaDB client"""
        with patch('chromadb.PersistentClient') as mock:
            yield mock
    
    @pytest.fixture
    def memory_service(self, mock_chroma_client):
        """Create memory service with mocked ChromaDB"""
        # Mock collection
        mock_collection = Mock()
        mock_collection.add = Mock()
        mock_collection.query = Mock()
        mock_collection.get = Mock()
        mock_collection.update = Mock()
        
        # Mock client methods
        mock_client_instance = Mock()
        mock_client_instance.get_collection = Mock(return_value=mock_collection)
        mock_client_instance.create_collection = Mock(return_value=mock_collection)
        mock_client_instance.delete_collection = Mock()
        
        mock_chroma_client.return_value = mock_client_instance
        
        # Create service
        service = ProjectMemoryService(
            project_id="test-project-123",
            project_name="Test Project"
        )
        service.collection = mock_collection
        
        return service
    
    @pytest.mark.asyncio
    async def test_store_memory(self, memory_service):
        """Test storing a memory"""
        memory_id = await memory_service.store_memory(
            content="Test memory content",
            memory_type=MemoryType.CONVERSATION,
            metadata={"test": "value"},
            conversation_id="conv-123",
            cost=0.01
        )
        
        assert memory_id is not None
        assert len(memory_id) == 16  # SHA256 truncated to 16 chars
        
        # Verify ChromaDB was called
        memory_service.collection.add.assert_called_once()
        call_args = memory_service.collection.add.call_args
        
        assert call_args[1]['documents'] == ["Test memory content"]
        assert call_args[1]['metadatas'][0]['memory_type'] == "conversation"
        assert call_args[1]['metadatas'][0]['conversation_id'] == "conv-123"
        assert call_args[1]['metadatas'][0]['generation_cost'] == 0.01
    
    @pytest.mark.asyncio
    async def test_search_memories(self, memory_service):
        """Test searching memories"""
        # Mock ChromaDB query response
        memory_service.collection.query.return_value = {
            'ids': [['mem1', 'mem2']],
            'documents': [['Memory 1 content', 'Memory 2 content']],
            'metadatas': [[
                {
                    'memory_type': 'conversation',
                    'timestamp': datetime.now().isoformat(),
                    'project_id': 'test-project-123'
                },
                {
                    'memory_type': 'code_pattern',
                    'timestamp': datetime.now().isoformat(),
                    'project_id': 'test-project-123'
                }
            ]],
            'distances': [[0.2, 0.5]]  # Lower is better
        }
        
        results = await memory_service.search_memories(
            query="test query",
            limit=5,
            memory_types=[MemoryType.CONVERSATION],
            min_relevance=0.5
        )
        
        assert len(results) == 2  # Both meet the 0.5 relevance threshold
        assert results[0].id == 'mem1'
        assert results[0].content == 'Memory 1 content'
        assert results[0].relevance_score == 0.9  # 1 - (0.2/2)
        assert results[1].relevance_score == 0.75  # 1 - (0.5/2)
    
    @pytest.mark.asyncio
    async def test_search_memories_by_conversation(self, memory_service):
        """Test searching memories filtered by conversation"""
        memory_service.collection.query.return_value = {
            'ids': [['mem1']],
            'documents': [['Conversation memory']],
            'metadatas': [[{
                'memory_type': 'conversation',
                'timestamp': datetime.now().isoformat(),
                'conversation_id': 'conv-123'
            }]],
            'distances': [[0.1]]
        }
        
        results = await memory_service.search_memories(
            query="test",
            conversation_id="conv-123"
        )
        
        # Verify where clause includes conversation_id
        call_args = memory_service.collection.query.call_args
        assert call_args[1]['where']['conversation_id'] == 'conv-123'
    
    @pytest.mark.asyncio
    async def test_get_memory_stats(self, memory_service):
        """Test getting memory statistics"""
        # Mock get all memories
        memory_service.collection.get.return_value = {
            'ids': ['mem1', 'mem2', 'mem3'],
            'metadatas': [
                {
                    'memory_type': 'conversation',
                    'generation_cost': 0.01,
                    'access_count': 5,
                    'timestamp': '2024-01-01T00:00:00'
                },
                {
                    'memory_type': 'code_pattern',
                    'generation_cost': 0.02,
                    'access_count': 3,
                    'timestamp': '2024-01-02T00:00:00'
                },
                {
                    'memory_type': 'conversation',
                    'generation_cost': 0.005,
                    'access_count': 1,
                    'timestamp': '2024-01-03T00:00:00'
                }
            ]
        }
        
        stats = await memory_service.get_memory_stats()
        
        assert stats['total_memories'] == 3
        assert stats['memory_types']['conversation'] == 2
        assert stats['memory_types']['code_pattern'] == 1
        assert stats['total_cost'] == 0.035
        assert stats['avg_access_count'] == 3.0
        assert stats['oldest_memory'] == '2024-01-01T00:00:00'
        assert stats['newest_memory'] == '2024-01-03T00:00:00'
    
    @pytest.mark.asyncio
    async def test_get_memory_stats_empty(self, memory_service):
        """Test getting stats when no memories exist"""
        memory_service.collection.get.return_value = {'ids': []}
        
        stats = await memory_service.get_memory_stats()
        
        assert stats['total_memories'] == 0
        assert stats['memory_types'] == {}
        assert stats['total_cost'] == 0.0
        assert stats['avg_access_count'] == 0.0
        assert stats['oldest_memory'] is None
        assert stats['newest_memory'] is None
    
    @pytest.mark.asyncio
    async def test_clear_memories(self, memory_service):
        """Test clearing all memories"""
        memory_service.collection.get.return_value = {
            'ids': ['mem1', 'mem2', 'mem3']
        }
        
        count = await memory_service.clear_memories()
        
        assert count == 3
        memory_service.chroma_client.delete_collection.assert_called_once_with(
            name=memory_service.collection_name
        )
    
    @pytest.mark.asyncio
    async def test_export_memories(self, memory_service):
        """Test exporting memories"""
        memory_service.collection.get.return_value = {
            'ids': ['mem1', 'mem2'],
            'documents': ['Memory 1', 'Memory 2'],
            'metadatas': [
                {'memory_type': 'conversation'},
                {'memory_type': 'code_pattern'}
            ]
        }
        
        export_data = await memory_service.export_memories()
        
        assert export_data['project_id'] == 'test-project-123'
        assert export_data['project_name'] == 'Test Project'
        assert export_data['memory_count'] == 2
        assert len(export_data['memories']) == 2
        assert export_data['memories'][0]['id'] == 'mem1'
        assert export_data['memories'][0]['content'] == 'Memory 1'
    
    @pytest.mark.asyncio
    async def test_import_memories(self, memory_service):
        """Test importing memories"""
        export_data = {
            'project_id': 'test-project-123',
            'project_name': 'Test Project',
            'memories': [
                {
                    'id': 'mem1',
                    'content': 'Memory 1',
                    'metadata': {'memory_type': 'conversation'}
                },
                {
                    'id': 'mem2',
                    'content': 'Memory 2',
                    'metadata': {'memory_type': 'code_pattern'}
                }
            ]
        }
        
        # Mock successful adds
        memory_service.collection.add.side_effect = [None, None]
        
        imported = await memory_service.import_memories(export_data)
        
        assert imported == 2
        assert memory_service.collection.add.call_count == 2
    
    @pytest.mark.asyncio
    async def test_import_memories_with_errors(self, memory_service):
        """Test importing memories with some failures"""
        export_data = {
            'project_id': 'test-project-123',
            'memories': [
                {'id': 'mem1', 'content': 'Memory 1', 'metadata': {}},
                {'id': 'mem2', 'content': 'Memory 2', 'metadata': {}}
            ]
        }
        
        # First succeeds, second fails
        memory_service.collection.add.side_effect = [None, Exception("Duplicate")]
        
        imported = await memory_service.import_memories(export_data)
        
        assert imported == 1  # Only one imported successfully
    
    def test_memory_id_generation(self, memory_service):
        """Test that memory IDs are unique"""
        id1 = memory_service._generate_memory_id("Same content")
        id2 = memory_service._generate_memory_id("Same content")
        
        assert id1 != id2  # Different due to timestamp
        assert len(id1) == 16
        assert len(id2) == 16
    
    def test_determine_memory_type(self):
        """Test memory type determination"""
        from services.llm_service import LLMService
        
        service = LLMService()
        
        # Test error detection
        assert service._determine_memory_type(
            "I'm getting an error", 
            "Here's the fix"
        ) == MemoryType.ERROR_SOLUTION
        
        # Test code pattern
        assert service._determine_memory_type(
            "Write a function",
            "```python\ndef hello():\n    pass\n```"
        ) == MemoryType.CODE_PATTERN
        
        # Test explanation
        assert service._determine_memory_type(
            "What is recursion?",
            "Recursion is..."
        ) == MemoryType.CONCEPT_EXPLANATION
        
        # Test screen analysis
        assert service._determine_memory_type(
            "What's in this image?",
            "I can see..."
        ) == MemoryType.SCREEN_ANALYSIS
        
        # Test default
        assert service._determine_memory_type(
            "Hello",
            "Hi there!"
        ) == MemoryType.CONVERSATION


class TestMemoryIntegration:
    """Test memory integration with LLM service"""
    
    @pytest.mark.asyncio
    async def test_llm_stores_valuable_responses(self):
        """Test that LLM service stores expensive responses"""
        from services.llm_service import LLMService
        
        with patch('services.memory_service.ProjectMemoryService') as mock_memory:
            mock_memory_instance = AsyncMock()
            mock_memory.return_value = mock_memory_instance
            
            # Mock LLM response
            mock_response = Mock()
            mock_response.content = "This is an expensive response"
            mock_response.model = "grok-4"
            mock_response.total_tokens = 1000
            mock_response.cost = 0.002  # Above threshold
            
            service = LLMService(project_id="test-123")
            service.llm_client = Mock()
            service.llm_client.query = Mock(return_value=mock_response)
            service.query_analyzer = Mock()
            service.query_analyzer.analyze_query = Mock(return_value=(Mock(value="MEDIUM"), None))
            
            # Generate response
            response = await service.generate_response(
                message="Explain quantum computing",
                history=[],
                memory_context=""
            )
            
            # Verify memory was stored
            mock_memory_instance.store_memory.assert_called_once()
            call_args = mock_memory_instance.store_memory.call_args
            
            assert "Explain quantum computing" in call_args[1]['content']
            assert "expensive response" in call_args[1]['content']
            assert call_args[1]['cost'] == 0.002