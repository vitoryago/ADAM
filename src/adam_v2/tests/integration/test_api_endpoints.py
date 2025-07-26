"""
Integration tests for ADAM v2 API endpoints
"""
import pytest
from fastapi import status
import json


class TestProjectEndpoints:
    """Test project-related API endpoints"""
    
    def test_create_project(self, test_client):
        """Test creating a new project"""
        response = test_client.post(
            "/api/projects",
            json={
                "name": "Test Project",
                "description": "A test project",
                "settings": {"model": "gpt-4"}
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == "Test Project"
        assert data["description"] == "A test project"
        assert "id" in data
        assert data["is_archived"] is False
    
    def test_list_projects(self, test_client):
        """Test listing all projects"""
        # Create some projects first
        for i in range(3):
            test_client.post(
                "/api/projects",
                json={"name": f"Project {i}"}
            )
        
        # List projects
        response = test_client.get("/api/projects")
        assert response.status_code == status.HTTP_200_OK
        
        projects = response.json()
        assert len(projects) >= 3
        assert all("name" in p for p in projects)
    
    def test_get_project(self, test_client):
        """Test getting a specific project"""
        # Create project
        create_response = test_client.post(
            "/api/projects",
            json={"name": "Specific Project"}
        )
        project_id = create_response.json()["id"]
        
        # Get project
        response = test_client.get(f"/api/projects/{project_id}")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["id"] == project_id
        assert data["name"] == "Specific Project"
    
    def test_update_project(self, test_client):
        """Test updating a project"""
        # Create project
        create_response = test_client.post(
            "/api/projects",
            json={"name": "Original Name"}
        )
        project_id = create_response.json()["id"]
        
        # Update project
        response = test_client.put(
            f"/api/projects/{project_id}",
            json={
                "name": "Updated Name",
                "description": "Now with description"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["description"] == "Now with description"
    
    def test_archive_project(self, test_client):
        """Test archiving a project"""
        # Create project
        create_response = test_client.post(
            "/api/projects",
            json={"name": "To Archive"}
        )
        project_id = create_response.json()["id"]
        
        # Archive project
        response = test_client.post(f"/api/projects/{project_id}/archive")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["is_archived"] is True
    
    def test_project_not_found(self, test_client):
        """Test accessing non-existent project"""
        response = test_client.get("/api/projects/non-existent-id")
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestConversationEndpoints:
    """Test conversation-related API endpoints"""
    
    @pytest.fixture
    def project_id(self, test_client):
        """Create a project and return its ID"""
        response = test_client.post(
            "/api/projects",
            json={"name": "Conversation Test Project"}
        )
        return response.json()["id"]
    
    def test_create_conversation(self, test_client, project_id):
        """Test creating a new conversation"""
        response = test_client.post(
            f"/api/projects/{project_id}/conversations",
            json={"title": "Test Conversation"}
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["title"] == "Test Conversation"
        assert data["project_id"] == project_id
        assert "id" in data
    
    def test_list_conversations(self, test_client, project_id):
        """Test listing conversations in a project"""
        # Create conversations
        for i in range(3):
            test_client.post(
                f"/api/projects/{project_id}/conversations",
                json={"title": f"Conversation {i}"}
            )
        
        # List conversations
        response = test_client.get(f"/api/projects/{project_id}/conversations")
        assert response.status_code == status.HTTP_200_OK
        
        conversations = response.json()
        assert len(conversations) == 3
        assert all(c["project_id"] == project_id for c in conversations)
    
    def test_rename_conversation(self, test_client, project_id):
        """Test renaming a conversation"""
        # Create conversation
        create_response = test_client.post(
            f"/api/projects/{project_id}/conversations",
            json={"title": "Original Title"}
        )
        conv_id = create_response.json()["id"]
        
        # Rename
        response = test_client.put(
            f"/api/conversations/{conv_id}",
            json={"title": "New Title"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["title"] == "New Title"
    
    def test_pin_conversation(self, test_client, project_id):
        """Test pinning/unpinning a conversation"""
        # Create conversation
        create_response = test_client.post(
            f"/api/projects/{project_id}/conversations",
            json={"title": "To Pin"}
        )
        conv_id = create_response.json()["id"]
        
        # Pin conversation
        response = test_client.post(f"/api/conversations/{conv_id}/pin")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["is_pinned"] is True
        
        # Unpin conversation
        response = test_client.post(f"/api/conversations/{conv_id}/unpin")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["is_pinned"] is False


class TestMessageEndpoints:
    """Test message-related API endpoints"""
    
    @pytest.fixture
    def conversation_id(self, test_client):
        """Create a project and conversation, return conversation ID"""
        # Create project
        project_response = test_client.post(
            "/api/projects",
            json={"name": "Message Test Project"}
        )
        project_id = project_response.json()["id"]
        
        # Create conversation
        conv_response = test_client.post(
            f"/api/projects/{project_id}/conversations",
            json={"title": "Message Test Conversation"}
        )
        return conv_response.json()["id"]
    
    def test_send_message(self, test_client, conversation_id):
        """Test sending a message"""
        response = test_client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={
                "content": "Hello, ADAM!",
                "use_memory": True,
                "model": "gpt-4"
            }
        )
        
        # Should return both user and assistant messages
        assert response.status_code == status.HTTP_200_OK
        messages = response.json()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, ADAM!"
        assert messages[1]["role"] == "assistant"
    
    def test_get_messages(self, test_client, conversation_id):
        """Test retrieving conversation messages"""
        # Send a message first
        test_client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"content": "Test message"}
        )
        
        # Get messages
        response = test_client.get(f"/api/conversations/{conversation_id}/messages")
        assert response.status_code == status.HTTP_200_OK
        
        messages = response.json()
        assert len(messages) >= 2  # User + assistant
        assert all("content" in m for m in messages)
        assert all("role" in m for m in messages)
    
    def test_message_with_image(self, test_client, conversation_id):
        """Test sending a message with an image attachment"""
        # This would require multipart form data
        # Simplified test for now
        response = test_client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={
                "content": "What's in this image?",
                "has_image": True
            }
        )
        
        assert response.status_code == status.HTTP_200_OK


class TestMemoryEndpoints:
    """Test memory-related API endpoints"""
    
    @pytest.fixture
    def project_with_memories(self, test_client):
        """Create a project with some memories"""
        # Create project
        project_response = test_client.post(
            "/api/projects",
            json={"name": "Memory Test Project"}
        )
        project_id = project_response.json()["id"]
        
        # Create conversation
        conv_response = test_client.post(
            f"/api/projects/{project_id}/conversations",
            json={"title": "Memory Conversation"}
        )
        conv_id = conv_response.json()["id"]
        
        # Send messages to create memories
        for i in range(3):
            test_client.post(
                f"/api/conversations/{conv_id}/messages",
                json={"content": f"Message {i} about Python"}
            )
        
        return project_id
    
    def test_search_project_memories(self, test_client, project_with_memories):
        """Test searching memories within a project"""
        response = test_client.post(
            f"/api/projects/{project_with_memories}/memories/search",
            json={"query": "Python"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        results = response.json()
        assert "memories" in results
        assert isinstance(results["memories"], list)
    
    def test_get_memory_stats(self, test_client, project_with_memories):
        """Test getting memory statistics"""
        response = test_client.get(
            f"/api/projects/{project_with_memories}/memories/stats"
        )
        
        assert response.status_code == status.HTTP_200_OK
        stats = response.json()
        assert "total_memories" in stats
        assert stats["total_memories"] >= 0