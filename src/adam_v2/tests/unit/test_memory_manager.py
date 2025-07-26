"""
Unit tests for the ProjectMemoryManager
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from memory_manager import ProjectMemoryManager, ProjectAwareMemorySystem


class TestProjectMemoryManager:
    """Test the ProjectMemoryManager class"""
    
    @patch('memory_manager.chromadb.PersistentClient')
    def test_initialization(self, mock_chromadb):
        """Test memory manager initialization"""
        # Setup mock
        mock_client = Mock()
        mock_chromadb.return_value = mock_client
        mock_client.get_collection.side_effect = Exception("Collection not found")
        
        # Create manager
        manager = ProjectMemoryManager("test-project-123")
        
        # Verify
        assert manager.project_id == "test-project-123"
        assert manager.collection_name == "adam_project_test-project-123"
        assert manager.collection is None
        mock_chromadb.assert_called_once()
    
    @patch('memory_manager.chromadb.PersistentClient')
    def test_initialize_collection(self, mock_chromadb):
        """Test creating a new collection"""
        # Setup mock
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.return_value = mock_client
        mock_client.create_collection.return_value = mock_collection
        
        # Create manager and initialize
        manager = ProjectMemoryManager("test-project")
        manager.collection = None
        manager.initialize_collection()
        
        # Verify collection creation
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert call_args[1]['name'] == "adam_project_test-project"
        assert call_args[1]['metadata']['project_id'] == "test-project"
        assert manager.collection == mock_collection
    
    @patch('memory_manager.chromadb.PersistentClient')
    def test_store_memory(self, mock_chromadb):
        """Test storing a memory"""
        # Setup mock
        mock_collection = Mock()
        mock_client = Mock()
        mock_chromadb.return_value = mock_client
        
        # Create manager
        manager = ProjectMemoryManager("test-project")
        manager.collection = mock_collection
        
        # Store memory
        memory_id = manager.store_memory(
            query="What is Python?",
            response="Python is a programming language",
            conversation_id="conv-123",
            memory_type="qa",
            metadata={"source": "test"}
        )
        
        # Verify
        assert memory_id.startswith("mem_test-project_")
        mock_collection.add.assert_called_once()
        
        # Check stored data
        call_args = mock_collection.add.call_args[1]
        assert len(call_args['ids']) == 1
        assert "Query: What is Python?" in call_args['documents'][0]
        assert call_args['metadatas'][0]['project_id'] == "test-project"
        assert call_args['metadatas'][0]['conversation_id'] == "conv-123"
        assert call_args['metadatas'][0]['memory_type'] == "qa"
        assert call_args['metadatas'][0]['source'] == "test"
    
    @patch('memory_manager.chromadb.PersistentClient')
    def test_search_memories(self, mock_chromadb):
        """Test searching memories"""
        # Setup mock
        mock_collection = Mock()
        mock_client = Mock()
        mock_chromadb.return_value = mock_client
        
        # Mock search results
        mock_collection.query.return_value = {
            'documents': [["Doc 1", "Doc 2"]],
            'ids': [["id1", "id2"]],
            'metadatas': [[{"meta": "1"}, {"meta": "2"}]],
            'distances': [[0.1, 0.2]]
        }
        
        # Create manager and search
        manager = ProjectMemoryManager("test-project")
        manager.collection = mock_collection
        
        results = manager.search_memories(
            query="Python tutorials",
            conversation_id="conv-123",
            n_results=5
        )
        
        # Verify search call
        mock_collection.query.assert_called_once_with(
            query_texts=["Python tutorials"],
            where={
                "project_id": "test-project",
                "conversation_id": "conv-123"
            },
            n_results=5
        )
        
        # Verify results
        assert len(results) == 2
        assert results[0]['id'] == "id1"
        assert results[0]['content'] == "Doc 1"
        assert results[0]['distance'] == 0.1
    
    @patch('memory_manager.chromadb.PersistentClient')
    def test_search_memories_no_conversation_filter(self, mock_chromadb):
        """Test searching without conversation filter"""
        # Setup mock
        mock_collection = Mock()
        mock_client = Mock()
        mock_chromadb.return_value = mock_client
        mock_collection.query.return_value = {
            'documents': [[]],
            'ids': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        # Create manager and search
        manager = ProjectMemoryManager("test-project")
        manager.collection = mock_collection
        
        results = manager.search_memories("test query")
        
        # Verify only project_id in where clause
        call_args = mock_collection.query.call_args[1]
        assert call_args['where'] == {"project_id": "test-project"}
        assert len(results) == 0
    
    @patch('memory_manager.chromadb.PersistentClient')
    def test_delete_collection(self, mock_chromadb):
        """Test deleting a collection"""
        # Setup mock
        mock_client = Mock()
        mock_chromadb.return_value = mock_client
        
        # Create manager and delete
        manager = ProjectMemoryManager("test-project")
        manager.delete_collection()
        
        # Verify
        mock_client.delete_collection.assert_called_once_with(
            "adam_project_test-project"
        )
    
    @patch('memory_manager.chromadb.PersistentClient')
    def test_get_project_stats(self, mock_chromadb):
        """Test getting project statistics"""
        # Setup mock
        mock_collection = Mock()
        mock_client = Mock()
        mock_chromadb.return_value = mock_client
        mock_collection.count.return_value = 42
        
        # Create manager
        manager = ProjectMemoryManager("test-project")
        manager.collection = mock_collection
        
        # Get stats
        stats = manager.get_project_stats()
        
        # Verify
        assert stats['total_memories'] == 42
        assert stats['collection_name'] == "adam_project_test-project"
        assert stats['project_id'] == "test-project"


class TestProjectAwareMemorySystem:
    """Test the ProjectAwareMemorySystem class"""
    
    @patch('memory_manager.ProjectMemoryManager')
    @patch('memory_manager.ADAMMemoryAdvanced.__init__')
    def test_initialization(self, mock_adam_init, mock_project_manager):
        """Test project-aware memory system initialization"""
        # Setup mocks
        mock_adam_init.return_value = None
        mock_manager_instance = Mock()
        mock_project_manager.return_value = mock_manager_instance
        
        # Create system
        system = ProjectAwareMemorySystem("test-project-789")
        
        # Verify
        assert system.project_id == "test-project-789"
        assert system.project_memory_manager == mock_manager_instance
        mock_adam_init.assert_called_once_with(
            persist_directory="./adam_memory_projects/test-project-789"
        )
    
    @patch('memory_manager.ProjectMemoryManager')
    def test_recall_with_context(self, mock_project_manager):
        """Test recalling memories with context"""
        # Setup mock
        mock_manager_instance = Mock()
        mock_project_manager.return_value = mock_manager_instance
        mock_manager_instance.search_memories.return_value = [
            {"id": "mem1", "content": "Test memory"}
        ]
        
        # Create system and recall
        with patch('memory_manager.ADAMMemoryAdvanced.__init__', return_value=None):
            system = ProjectAwareMemorySystem("test-project")
            results = system.recall_with_context(
                query="test query",
                conversation_id="conv-456",
                n_results=5
            )
        
        # Verify
        mock_manager_instance.search_memories.assert_called_once_with(
            query="test query",
            conversation_id="conv-456",
            n_results=5
        )
        assert results == [{"id": "mem1", "content": "Test memory"}]