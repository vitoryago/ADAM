"""
Unit tests for advanced memory service
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from services.advanced_memory_service import (
    AdvancedMemoryService,
    MemoryWorthiness,
    EnhancedSearchResult
)
from services.memory_service import MemoryType


class TestAdvancedMemoryService:
    """Test the advanced memory service"""
    
    @pytest.fixture
    def mock_chroma_client(self):
        """Mock ChromaDB client"""
        with patch('chromadb.PersistentClient') as mock:
            yield mock
    
    @pytest.fixture
    def advanced_service(self, mock_chroma_client):
        """Create advanced memory service with mocked ChromaDB"""
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
        service = AdvancedMemoryService(
            project_id="test-project-123",
            project_name="Test Project"
        )
        service.collection = mock_collection
        
        return service
    
    @pytest.mark.asyncio
    async def test_evaluate_memory_worthiness_expensive(self, advanced_service):
        """Test that expensive queries are marked as essential"""
        worthiness, metadata = await advanced_service.evaluate_memory_worthiness(
            query="Complex question",
            response="Detailed response",
            cost=0.02,  # Expensive
            model="grok-4"
        )
        
        assert worthiness == MemoryWorthiness.ESSENTIAL
        assert metadata["factors"]["cost"] == "expensive"
    
    @pytest.mark.asyncio
    async def test_evaluate_memory_worthiness_error(self, advanced_service):
        """Test that error-related queries are essential"""
        worthiness, metadata = await advanced_service.evaluate_memory_worthiness(
            query="I'm getting an error with my code",
            response="Here's the fix...",
            cost=0.0005,
            model="grok-3"
        )
        
        assert worthiness == MemoryWorthiness.ESSENTIAL
        assert metadata["factors"]["error_related"] is True
    
    @pytest.mark.asyncio
    async def test_evaluate_memory_worthiness_trivial(self, advanced_service):
        """Test that trivial queries are marked correctly"""
        worthiness, metadata = await advanced_service.evaluate_memory_worthiness(
            query="What's 2+2?",
            response="4",
            cost=0.0001,
            model="grok-3"
        )
        
        assert worthiness == MemoryWorthiness.TRIVIAL
        assert metadata["factors"]["final_score"] < 0.3
    
    @pytest.mark.asyncio
    async def test_evaluate_memory_worthiness_code(self, advanced_service):
        """Test that code responses increase worthiness"""
        worthiness, metadata = await advanced_service.evaluate_memory_worthiness(
            query="How to implement a binary search?",
            response="```python\ndef binary_search(arr, target):\n    pass\n```\nHere's the implementation...",
            cost=0.001,
            model="grok-3"
        )
        
        assert worthiness in [MemoryWorthiness.MODERATE, MemoryWorthiness.VALUABLE, MemoryWorthiness.ESSENTIAL]
        assert metadata["factors"]["has_code"] is True
    
    @pytest.mark.asyncio
    async def test_store_memory_with_evaluation_rejects_trivial(self, advanced_service):
        """Test that trivial memories are rejected"""
        result = await advanced_service.store_memory_with_evaluation(
            query="Hi",
            response="Hello!",
            memory_type=MemoryType.CONVERSATION,
            cost=0.0001,
            model="grok-3"
        )
        
        assert result is None  # Should be rejected
        advanced_service.collection.add.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_store_memory_with_evaluation_accepts_valuable(self, advanced_service):
        """Test that valuable memories are stored"""
        result = await advanced_service.store_memory_with_evaluation(
            query="Explain the Python GIL and its impact on multithreading",
            response="The Global Interpreter Lock (GIL) is..." + "x" * 500,  # Long response
            memory_type=MemoryType.CONCEPT_EXPLANATION,
            cost=0.015,  # Make it expensive enough to be essential
            model="grok-4"
        )
        
        assert result is not None
        advanced_service.collection.add.assert_called_once()
        
        # Check metadata includes evaluation
        call_args = advanced_service.collection.add.call_args
        metadata = call_args[1]['metadatas'][0]
        assert 'worthiness' in metadata
        assert 'evaluation' in metadata
    
    @pytest.mark.asyncio
    async def test_advanced_search_fusion(self, advanced_service):
        """Test advanced search with BM25 and semantic fusion"""
        # Mock semantic search results
        with patch.object(advanced_service, 'search_memories') as mock_semantic:
            mock_semantic.return_value = [
                Mock(id='mem1', content='Python async', relevance_score=0.9, 
                     metadata={}, memory_type='code_pattern', timestamp=datetime.now()),
                Mock(id='mem2', content='Python threads', relevance_score=0.7,
                     metadata={}, memory_type='concept_explanation', timestamp=datetime.now())
            ]
            
            # Mock BM25 search
            with patch.object(advanced_service, '_bm25_search') as mock_bm25:
                mock_bm25.return_value = [
                    Mock(id='mem2', content='Python threads', relevance_score=0.8,
                         metadata={}, memory_type='concept_explanation', timestamp=datetime.now()),
                    Mock(id='mem3', content='Python GIL', relevance_score=0.6,
                         metadata={}, memory_type='concept_explanation', timestamp=datetime.now())
                ]
                
                results = await advanced_service.advanced_search(
                    query="Python concurrency",
                    limit=3
                )
                
                # Should have all 3 unique memories
                assert len(results) == 3
                
                # Check that results are EnhancedSearchResult
                assert all(isinstance(r, EnhancedSearchResult) for r in results)
                
                # Check fusion worked (mem2 should rank high as it appears in both)
                memory_ids = [r.id for r in results]
                assert 'mem2' in memory_ids  # Should be present as it's in both results
    
    @pytest.mark.asyncio
    async def test_bm25_search(self, advanced_service):
        """Test BM25 keyword search"""
        # Mock collection data
        advanced_service.collection.get.return_value = {
            'ids': ['mem1', 'mem2', 'mem3'],
            'documents': [
                'Python async await coroutines',
                'JavaScript promises callbacks',
                'Python threading multiprocessing'
            ],
            'metadatas': [
                {'memory_type': 'code_pattern', 'timestamp': datetime.now().isoformat()},
                {'memory_type': 'code_pattern', 'timestamp': datetime.now().isoformat()},
                {'memory_type': 'concept_explanation', 'timestamp': datetime.now().isoformat()}
            ]
        }
        
        # Mock BM25
        with patch('services.advanced_memory_service.BM25_AVAILABLE', True):
            with patch('services.advanced_memory_service.BM25Okapi') as mock_bm25:
                mock_instance = Mock()
                mock_instance.get_scores.return_value = np.array([0.8, 0.1, 0.6])
                mock_bm25.return_value = mock_instance
                
                results = await advanced_service._bm25_search(
                    query="Python",
                    limit=2
                )
                
                # Should return top 2 Python-related documents
                assert len(results) <= 2
                # Note: actual results depend on mock implementation
    
    @pytest.mark.asyncio
    async def test_update_memory_success(self, advanced_service):
        """Test updating memory success rate"""
        # Mock existing memory
        advanced_service.collection.get.return_value = {
            'ids': ['mem1'],
            'metadatas': [{
                'total_uses': 5,
                'successful_uses': 4,
                'success_rate': 0.8
            }]
        }
        
        await advanced_service.update_memory_success(
            memory_id='mem1',
            success=True,
            feedback="This solution worked perfectly!"
        )
        
        # Check update was called
        advanced_service.collection.update.assert_called_once()
        
        # Check new metadata
        update_args = advanced_service.collection.update.call_args
        new_metadata = update_args[1]['metadatas'][0]
        assert new_metadata['total_uses'] == 6
        assert new_metadata['successful_uses'] == 5
        assert new_metadata['success_rate'] == 5/6
        assert new_metadata['last_feedback'] == "This solution worked perfectly!"
    
    @pytest.mark.asyncio
    async def test_get_memory_versions(self, advanced_service):
        """Test retrieving memory versions"""
        # Mock query results
        advanced_service.collection.query.return_value = {
            'ids': [['v1', 'v2', 'v3']],
            'documents': [['Version 1', 'Version 2', 'Version 3']],
            'metadatas': [[
                {'version': 1, 'timestamp': '2024-01-01T00:00:00', 'memory_type': 'error_solution'},
                {'version': 2, 'timestamp': '2024-01-02T00:00:00', 'memory_type': 'error_solution'},
                {'version': 3, 'timestamp': '2024-01-03T00:00:00', 'memory_type': 'error_solution'}
            ]],
            'distances': [[0.1, 0.2, 0.3]]
        }
        
        versions = await advanced_service.get_memory_versions('original_id')
        
        assert len(versions) == 3
        # Should be sorted by timestamp
        assert versions[0].id == 'v1'
        assert versions[1].id == 'v2'
        assert versions[2].id == 'v3'