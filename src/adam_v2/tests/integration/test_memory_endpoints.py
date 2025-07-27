"""
Integration tests for memory API endpoints
"""
import pytest
from fastapi import status
from unittest.mock import patch, Mock


class TestMemoryEndpoints:
    """Test memory-related API endpoints"""
    
    @pytest.fixture
    def project_setup(self, test_client):
        """Create project for testing"""
        # Create project
        project_response = test_client.post(
            "/api/projects",
            json={
                "name": "Memory Test Project",
                "settings": {
                    "model": "grok-3-mini-high",
                    "temperature": 0.7
                }
            }
        )
        project_id = project_response.json()["id"]
        
        return {"project_id": project_id}
    
    def test_search_memories_empty(self, test_client, project_setup):
        """Test searching memories when none exist"""
        response = test_client.post(
            f"/api/projects/{project_setup['project_id']}/memories/search",
            json={
                "query": "test search",
                "limit": 5
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        results = response.json()
        assert results == []
    
    @patch('services.memory_service.ADAM_MEMORY_AVAILABLE', True)
    @patch('services.memory_service.ProjectMemoryService')
    def test_search_memories_with_results(self, mock_memory_service, test_client, project_setup):
        """Test searching memories with results"""
        from services.memory_service import MemorySearchResult
        from datetime import datetime
        
        # Mock memory service
        mock_service = Mock()
        mock_memory_service.return_value = mock_service
        
        # Mock search results
        mock_service.search_memories.return_value = [
            MemorySearchResult(
                id="mem1",
                content="How to implement async in Python",
                metadata={"model": "grok-4"},
                relevance_score=0.95,
                memory_type="code_pattern",
                timestamp=datetime.now()
            ),
            MemorySearchResult(
                id="mem2",
                content="Python is a programming language",
                metadata={"model": "grok-3"},
                relevance_score=0.75,
                memory_type="concept_explanation",
                timestamp=datetime.now()
            )
        ]
        
        response = test_client.post(
            f"/api/projects/{project_setup['project_id']}/memories/search",
            json={
                "query": "python async",
                "limit": 10,
                "memory_types": ["code_pattern", "concept_explanation"],
                "min_relevance": 0.7
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        results = response.json()
        assert len(results) == 2
        assert results[0]["relevance_score"] == 0.95
        assert results[0]["memory_type"] == "code_pattern"
    
    @patch('services.memory_service.ADAM_MEMORY_AVAILABLE', True)
    @patch('services.memory_service.ProjectMemoryService')
    def test_store_memory(self, mock_memory_service, test_client, project_setup):
        """Test storing a new memory"""
        # Mock memory service
        mock_service = Mock()
        mock_memory_service.return_value = mock_service
        mock_service.store_memory.return_value = "mem123"
        
        response = test_client.post(
            f"/api/projects/{project_setup['project_id']}/memories",
            json={
                "content": "Important information about Python",
                "memory_type": "concept_explanation",
                "metadata": {"source": "manual"},
                "cost": 0.01
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert result["id"] == "mem123"
        assert result["content"] == "Important information about Python"
        assert result["memory_type"] == "CONCEPT_EXPLANATION"
    
    def test_store_memory_unavailable(self, test_client, project_setup):
        """Test storing memory when system unavailable"""
        with patch('services.memory_service.ADAM_MEMORY_AVAILABLE', False):
            response = test_client.post(
                f"/api/projects/{project_setup['project_id']}/memories",
                json={
                    "content": "Test content",
                    "memory_type": "conversation"
                }
            )
            
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    
    @patch('services.memory_service.ADAM_MEMORY_AVAILABLE', True)
    @patch('services.memory_service.ProjectMemoryService')
    def test_get_memory_stats(self, mock_memory_service, test_client, project_setup):
        """Test getting memory statistics"""
        # Mock memory service
        mock_service = Mock()
        mock_memory_service.return_value = mock_service
        mock_service.get_memory_stats.return_value = {
            "total_memories": 42,
            "memory_types": {
                "conversation": 20,
                "code_pattern": 15,
                "concept_explanation": 7
            },
            "total_cost": 0.125,
            "avg_access_count": 3.5,
            "oldest_memory": "2024-01-01T00:00:00",
            "newest_memory": "2024-01-20T00:00:00"
        }
        
        response = test_client.get(
            f"/api/projects/{project_setup['project_id']}/memories/stats"
        )
        
        assert response.status_code == status.HTTP_200_OK
        stats = response.json()
        assert stats["total_memories"] == 42
        assert stats["memory_types"]["conversation"] == 20
        assert stats["total_cost"] == 0.125
    
    @patch('services.memory_service.ADAM_MEMORY_AVAILABLE', True)
    @patch('services.memory_service.ProjectMemoryService')
    def test_clear_memories(self, mock_memory_service, test_client, project_setup):
        """Test clearing all memories"""
        # Mock memory service
        mock_service = Mock()
        mock_memory_service.return_value = mock_service
        mock_service.clear_memories.return_value = 10
        
        # Without confirmation
        response = test_client.delete(
            f"/api/projects/{project_setup['project_id']}/memories"
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        
        # With confirmation
        response = test_client.delete(
            f"/api/projects/{project_setup['project_id']}/memories?confirm=true"
        )
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert result["deleted"] == 10
        assert "Deleted 10 memories" in result["message"]
    
    @patch('services.memory_service.ADAM_MEMORY_AVAILABLE', True)
    @patch('services.memory_service.ProjectMemoryService')
    def test_export_memories(self, mock_memory_service, test_client, project_setup):
        """Test exporting memories"""
        # Mock memory service
        mock_service = Mock()
        mock_memory_service.return_value = mock_service
        mock_service.export_memories.return_value = {
            "project_id": project_setup['project_id'],
            "project_name": "Memory Test Project",
            "export_date": "2024-01-20T00:00:00",
            "memory_count": 5,
            "memories": [
                {
                    "id": "mem1",
                    "content": "Memory 1",
                    "metadata": {"type": "test"}
                }
            ]
        }
        
        response = test_client.get(
            f"/api/projects/{project_setup['project_id']}/memories/export"
        )
        
        assert response.status_code == status.HTTP_200_OK
        export_data = response.json()
        assert export_data["project_id"] == project_setup['project_id']
        assert export_data["memory_count"] == 5
        assert len(export_data["memories"]) == 1
    
    @patch('services.memory_service.ADAM_MEMORY_AVAILABLE', True)
    @patch('services.memory_service.ProjectMemoryService')
    def test_import_memories(self, mock_memory_service, test_client, project_setup):
        """Test importing memories"""
        # Mock memory service
        mock_service = Mock()
        mock_memory_service.return_value = mock_service
        mock_service.import_memories.return_value = 3
        
        export_data = {
            "project_id": project_setup['project_id'],
            "project_name": "Memory Test Project",
            "export_date": "2024-01-20T00:00:00",
            "memory_count": 3,
            "memories": []
        }
        
        response = test_client.post(
            f"/api/projects/{project_setup['project_id']}/memories/import",
            json=export_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert result["imported"] == 3
        assert "Imported 3 memories" in result["message"]
    
    def test_import_memories_wrong_project(self, test_client, project_setup):
        """Test importing memories from different project"""
        export_data = {
            "project_id": "different-project-id",
            "project_name": "Different Project",
            "export_date": "2024-01-20T00:00:00",
            "memory_count": 3,
            "memories": []
        }
        
        response = test_client.post(
            f"/api/projects/{project_setup['project_id']}/memories/import",
            json=export_data
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "different project" in response.json()["detail"]
    
    def test_get_memory_types(self, test_client):
        """Test getting available memory types"""
        response = test_client.get("/api/memory-types")
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        
        # Should indicate availability
        assert "available" in result
        assert "types" in result
        
        # If available, should have types
        if result["available"]:
            assert len(result["types"]) > 0
            assert all("value" in t and "name" in t for t in result["types"])
    
    def test_memory_search_with_conversation_filter(self, test_client, project_setup):
        """Test searching memories filtered by conversation"""
        response = test_client.post(
            f"/api/projects/{project_setup['project_id']}/memories/search",
            json={
                "query": "test",
                "conversation_id": "conv-123",
                "limit": 5
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        # Results would be filtered by conversation
    
    def test_project_not_found(self, test_client):
        """Test memory operations on non-existent project"""
        response = test_client.post(
            "/api/projects/non-existent/memories/search",
            json={"query": "test"}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Project non-existent not found" in response.json()["detail"]