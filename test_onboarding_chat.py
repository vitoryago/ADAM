#!/usr/bin/env python3
"""
Test onboarding integration in chat
"""

import requests
import json

API_BASE = "http://localhost:8000/api"

def test_onboarding_in_chat():
    """Test onboarding detection in regular chat"""
    
    # First, create a project and conversation
    # Get or create project
    projects_response = requests.get(f"{API_BASE}/projects")
    if projects_response.status_code == 200:
        projects = projects_response.json()
        if projects:
            project = projects[0]
            print(f"Using existing project: {project['name']}")
        else:
            # Create new project
            project_data = {
                "name": "Test Onboarding Project",
                "description": "Testing onboarding in chat"
            }
            create_response = requests.post(f"{API_BASE}/projects", json=project_data)
            if create_response.status_code == 200:
                project = create_response.json()
                print(f"Created new project: {project['name']}")
            else:
                print(f"Failed to create project: {create_response.text}")
                return
    else:
        print(f"Failed to get projects: {projects_response.text}")
        return
    
    # Create a conversation
    conv_data = {
        "title": "Onboarding Test Conversation"
    }
    conv_response = requests.post(f"{API_BASE}/projects/{project['id']}/conversations", json=conv_data)
    
    if conv_response.status_code in [200, 201]:
        conversation = conv_response.json()
        print(f"Created conversation: {conversation['id']}")
    else:
        print(f"Failed to create conversation: {conv_response.text}")
        return
    
    # Test onboarding requests
    test_messages = [
        "Can you help me onboard to the marketing analytics project?",
        "I'm new to this codebase, can you walk me through it?",
        "Show me an overview of the data pipeline",
        "What's next?",  # Should trigger progress check if in onboarding
        "I completed the first task"  # Should update progress
    ]
    
    for msg in test_messages:
        print(f"\n{'='*60}")
        print(f"Sending: {msg}")
        print('-'*60)
        
        message_data = {
            "content": msg,
            "use_memory": True,
            "use_search": False
        }
        
        response = requests.post(
            f"{API_BASE}/conversations/{conversation['id']}/messages",
            json=message_data
        )
        
        if response.status_code == 200:
            messages = response.json()
            # The response includes both user and assistant messages
            for message in messages:
                if message["role"] == "assistant":
                    print(f"ADAM Response (first 500 chars):")
                    print(message["content"][:500])
                    
                    if message.get("metadata"):
                        print(f"\nMetadata: {json.dumps(message['metadata'], indent=2)}")
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    print("Testing Onboarding Integration in Chat")
    print("="*60)
    test_onboarding_in_chat()