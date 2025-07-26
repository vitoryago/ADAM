"""
Integration tests for conversation API endpoints
"""
import pytest
from fastapi import status


class TestConversationAPIEndpoints:
    """Test conversation-related API endpoints"""
    
    def test_create_conversation_in_project(self, test_client):
        """Test creating a conversation in a project"""
        # First create a project
        project_response = test_client.post(
            "/api/projects",
            json={"name": "Test Project for Conversations"}
        )
        assert project_response.status_code == status.HTTP_201_CREATED
        project_id = project_response.json()["id"]
        
        # Create conversation
        response = test_client.post(
            f"/api/projects/{project_id}/conversations",
            json={"title": "My First Conversation"}
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["title"] == "My First Conversation"
        assert data["project_id"] == project_id
        assert data["is_pinned"] is False
        assert "id" in data
    
    def test_create_conversation_invalid_project(self, test_client):
        """Test creating conversation in non-existent project"""
        response = test_client.post(
            "/api/projects/invalid-project-id/conversations",
            json={"title": "Test Conversation"}
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_list_project_conversations(self, test_client):
        """Test listing all conversations in a project"""
        # Create project
        project_response = test_client.post(
            "/api/projects",
            json={"name": "Project with Multiple Conversations"}
        )
        project_id = project_response.json()["id"]
        
        # Create multiple conversations
        conversation_titles = ["Conv 1", "Conv 2", "Pinned Conv", "Conv 4"]
        created_ids = []
        
        for title in conversation_titles:
            response = test_client.post(
                f"/api/projects/{project_id}/conversations",
                json={"title": title}
            )
            created_ids.append(response.json()["id"])
        
        # Pin one conversation
        test_client.post(f"/api/conversations/{created_ids[2]}/pin")
        
        # List conversations
        response = test_client.get(f"/api/projects/{project_id}/conversations")
        assert response.status_code == status.HTTP_200_OK
        
        conversations = response.json()
        assert len(conversations) == 4
        
        # Check pinned conversation is first
        assert conversations[0]["is_pinned"] is True
        assert conversations[0]["title"] == "Pinned Conv"
        
        # Others should not be pinned
        for conv in conversations[1:]:
            assert conv["is_pinned"] is False
    
    def test_get_conversation(self, test_client):
        """Test getting a specific conversation"""
        # Create project and conversation
        project_response = test_client.post(
            "/api/projects",
            json={"name": "Test Project"}
        )
        project_id = project_response.json()["id"]
        
        conv_response = test_client.post(
            f"/api/projects/{project_id}/conversations",
            json={"title": "Specific Conversation"}
        )
        conv_id = conv_response.json()["id"]
        
        # Get conversation
        response = test_client.get(f"/api/conversations/{conv_id}")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["id"] == conv_id
        assert data["title"] == "Specific Conversation"
        assert data["project_id"] == project_id
    
    def test_update_conversation_title(self, test_client):
        """Test renaming a conversation"""
        # Create project and conversation
        project_response = test_client.post(
            "/api/projects",
            json={"name": "Test Project"}
        )
        project_id = project_response.json()["id"]
        
        conv_response = test_client.post(
            f"/api/projects/{project_id}/conversations",
            json={"title": "Original Title"}
        )
        conv_id = conv_response.json()["id"]
        
        # Update title
        response = test_client.put(
            f"/api/conversations/{conv_id}",
            json={"title": "New Amazing Title"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["title"] == "New Amazing Title"
    
    def test_pin_unpin_conversation(self, test_client):
        """Test pinning and unpinning conversations"""
        # Create project and conversation
        project_response = test_client.post(
            "/api/projects",
            json={"name": "Test Project"}
        )
        project_id = project_response.json()["id"]
        
        conv_response = test_client.post(
            f"/api/projects/{project_id}/conversations",
            json={"title": "Conversation to Pin"}
        )
        conv_id = conv_response.json()["id"]
        
        # Pin conversation
        response = test_client.post(f"/api/conversations/{conv_id}/pin")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["is_pinned"] is True
        
        # Verify it's pinned
        get_response = test_client.get(f"/api/conversations/{conv_id}")
        assert get_response.json()["is_pinned"] is True
        
        # Unpin conversation
        response = test_client.post(f"/api/conversations/{conv_id}/unpin")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["is_pinned"] is False
    
    def test_delete_conversation(self, test_client):
        """Test deleting a conversation"""
        # Create project and conversation
        project_response = test_client.post(
            "/api/projects",
            json={"name": "Test Project"}
        )
        project_id = project_response.json()["id"]
        
        conv_response = test_client.post(
            f"/api/projects/{project_id}/conversations",
            json={"title": "Conversation to Delete"}
        )
        conv_id = conv_response.json()["id"]
        
        # Delete conversation
        response = test_client.delete(f"/api/conversations/{conv_id}")
        assert response.status_code == status.HTTP_204_NO_CONTENT
        
        # Verify it's deleted
        get_response = test_client.get(f"/api/conversations/{conv_id}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_conversation_stats(self, test_client):
        """Test getting conversation statistics"""
        # Create project and conversation
        project_response = test_client.post(
            "/api/projects",
            json={"name": "Test Project"}
        )
        project_id = project_response.json()["id"]
        
        conv_response = test_client.post(
            f"/api/projects/{project_id}/conversations",
            json={"title": "Conversation with Stats"}
        )
        conv_id = conv_response.json()["id"]
        
        # Get stats (should be empty initially)
        response = test_client.get(f"/api/conversations/{conv_id}/stats")
        assert response.status_code == status.HTTP_200_OK
        
        stats = response.json()
        assert stats["conversation_id"] == conv_id
        assert stats["title"] == "Conversation with Stats"
        assert stats["message_count"] == 0
        assert stats["total_tokens"] == 0
        assert stats["total_cost"] == 0.0
        assert stats["is_pinned"] is False
    
    def test_conversation_not_found(self, test_client):
        """Test accessing non-existent conversation"""
        # Try to get non-existent conversation
        response = test_client.get("/api/conversations/non-existent-id")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        # Try to update non-existent conversation
        response = test_client.put(
            "/api/conversations/non-existent-id",
            json={"title": "New Title"}
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        # Try to delete non-existent conversation
        response = test_client.delete("/api/conversations/non-existent-id")
        assert response.status_code == status.HTTP_404_NOT_FOUND