"""
Integration tests for message API endpoints
"""
import pytest
from fastapi import status
import json


class TestMessageEndpoints:
    """Test message-related API endpoints"""
    
    @pytest.fixture
    def conversation_setup(self, test_client):
        """Create project and conversation for testing"""
        # Create project
        project_response = test_client.post(
            "/api/projects",
            json={
                "name": "Message Test Project",
                "settings": {
                    "model": "grok-3-mini-high",
                    "temperature": 0.7
                }
            }
        )
        project_id = project_response.json()["id"]
        
        # Create conversation
        conv_response = test_client.post(
            f"/api/projects/{project_id}/conversations",
            json={"title": "Test Chat"}
        )
        conversation_id = conv_response.json()["id"]
        
        return {
            "project_id": project_id,
            "conversation_id": conversation_id
        }
    
    def test_send_message(self, test_client, conversation_setup):
        """Test sending a message and receiving response"""
        response = test_client.post(
            f"/api/conversations/{conversation_setup['conversation_id']}/messages",
            json={
                "content": "Hello ADAM!",
                "use_memory": False
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        messages = response.json()
        
        # Should return 2 messages (user + assistant)
        assert len(messages) == 2
        
        # First should be user message
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello ADAM!"
        
        # Second should be assistant message
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] is not None
        assert messages[1]["model"] is not None
        assert messages[1]["tokens_used"] is not None
        assert messages[1]["cost"] is not None
    
    def test_send_message_with_model_override(self, test_client, conversation_setup):
        """Test sending message with specific model"""
        response = test_client.post(
            f"/api/conversations/{conversation_setup['conversation_id']}/messages",
            json={
                "content": "Explain quantum computing",
                "model": "grok-4",
                "use_memory": True
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        messages = response.json()
        
        # Assistant should use specified model
        # Note: In mock mode, model might be "mock"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["model"] in ["grok-4", "mock"]
    
    def test_send_message_with_image(self, test_client, conversation_setup):
        """Test sending message with image"""
        response = test_client.post(
            f"/api/conversations/{conversation_setup['conversation_id']}/messages",
            json={
                "content": "What's in this image?",
                "has_image": True,
                "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
                "use_memory": False
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        messages = response.json()
        
        # User message should indicate image
        assert messages[0]["has_image"] is True
    
    def test_get_messages(self, test_client, conversation_setup):
        """Test retrieving conversation messages"""
        # Send some messages first
        for i in range(3):
            test_client.post(
                f"/api/conversations/{conversation_setup['conversation_id']}/messages",
                json={
                    "content": f"Message {i}",
                    "use_memory": False
                }
            )
        
        # Get messages
        response = test_client.get(
            f"/api/conversations/{conversation_setup['conversation_id']}/messages"
        )
        
        assert response.status_code == status.HTTP_200_OK
        messages = response.json()
        
        # Should have at least 6 messages (3 user + 3 assistant)
        assert len(messages) >= 6
        
        # Messages should alternate between user and assistant
        for i in range(0, len(messages), 2):
            if i < len(messages):
                assert messages[i]["role"] == "user"
            if i + 1 < len(messages):
                assert messages[i + 1]["role"] == "assistant"
    
    def test_get_messages_with_pagination(self, test_client, conversation_setup):
        """Test message pagination"""
        # Get limited messages
        response = test_client.get(
            f"/api/conversations/{conversation_setup['conversation_id']}/messages?limit=2&offset=0"
        )
        
        assert response.status_code == status.HTTP_200_OK
        messages = response.json()
        assert len(messages) <= 2
    
    def test_delete_message(self, test_client, conversation_setup):
        """Test deleting a message"""
        # Send a message first
        send_response = test_client.post(
            f"/api/conversations/{conversation_setup['conversation_id']}/messages",
            json={
                "content": "Message to delete",
                "use_memory": False
            }
        )
        
        messages = send_response.json()
        message_id = messages[0]["id"]  # User message ID
        
        # Delete the message
        response = test_client.delete(f"/api/messages/{message_id}")
        assert response.status_code == status.HTTP_204_NO_CONTENT
        
        # Verify it's deleted
        get_response = test_client.get(
            f"/api/conversations/{conversation_setup['conversation_id']}/messages"
        )
        remaining_messages = get_response.json()
        
        # Message should not be in the list
        message_ids = [msg["id"] for msg in remaining_messages]
        assert message_id not in message_ids
    
    def test_regenerate_response(self, test_client, conversation_setup):
        """Test regenerating last assistant response"""
        # Send initial message
        test_client.post(
            f"/api/conversations/{conversation_setup['conversation_id']}/messages",
            json={
                "content": "Tell me a joke",
                "use_memory": False
            }
        )
        
        # Regenerate with different model
        response = test_client.post(
            f"/api/conversations/{conversation_setup['conversation_id']}/regenerate",
            json={"model": "grok-4"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        messages = response.json()
        
        # Should return regenerated response
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Tell me a joke"
        assert messages[1]["role"] == "assistant"
    
    def test_conversation_not_found(self, test_client):
        """Test sending message to non-existent conversation"""
        response = test_client.post(
            "/api/conversations/non-existent-id/messages",
            json={
                "content": "Hello",
                "use_memory": False
            }
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_streaming_response(self, test_client, conversation_setup):
        """Test streaming message response"""
        # Note: Testing streaming with TestClient is tricky
        # This tests that the endpoint exists and returns proper headers
        with test_client as client:
            response = client.post(
                f"/api/conversations/{conversation_setup['conversation_id']}/messages/stream",
                json={
                    "content": "Stream this response",
                    "use_memory": False
                },
                stream=True
            )
            
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
            
            # Read some chunks
            chunks = []
            for line in response.iter_lines():
                if line:
                    chunks.append(line)
                if len(chunks) > 5:  # Just read a few
                    break
            
            assert len(chunks) > 0