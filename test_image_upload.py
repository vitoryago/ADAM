#!/usr/bin/env python3
"""Test image upload to ADAM backend"""

import requests
import base64
import json
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8000/api"
TEST_IMAGE = "/Users/vitoryago/Downloads/test_image.png"  # Change this to your test image

def test_image_upload():
    # First, get a project ID
    response = requests.get(f"{API_BASE}/projects")
    if response.status_code != 200:
        print(f"Failed to get projects: {response.status_code}")
        print(response.text)
        return
    
    projects = response.json()
    if not projects:
        print("No projects found. Creating one...")
        response = requests.post(f"{API_BASE}/projects", json={"name": "Test Project"})
        if response.status_code != 200:
            print(f"Failed to create project: {response.status_code}")
            return
        project = response.json()
        project_id = project["id"]
    else:
        project_id = projects[0]["id"]
    
    print(f"Using project ID: {project_id}")
    
    # Get or create a conversation
    response = requests.get(f"{API_BASE}/projects/{project_id}/conversations")
    if response.status_code != 200:
        print(f"Failed to get conversations: {response.status_code}")
        return
    
    conversations = response.json()
    if not conversations:
        print("Creating conversation...")
        response = requests.post(f"{API_BASE}/projects/{project_id}/conversations", 
                               json={"title": "Test Image Upload"})
        if response.status_code != 200:
            print(f"Failed to create conversation: {response.status_code}")
            return
        conversation = response.json()
        conversation_id = conversation["id"]
    else:
        conversation_id = conversations[0]["id"]
    
    print(f"Using conversation ID: {conversation_id}")
    
    # Read and encode a small test image
    # Create a small test image if none exists
    if not Path(TEST_IMAGE).exists():
        print("Creating test image...")
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(TEST_IMAGE)
    
    with open(TEST_IMAGE, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    print(f"Image size (base64): {len(image_data)} chars")
    
    # Send message with image
    message_data = {
        "content": "What's in this image?",
        "use_memory": True,
        "model": "grok-2-vision-1212",  # Explicitly use vision model
        "has_image": True,
        "image_data": image_data
    }
    
    print("Sending message with image...")
    response = requests.post(
        f"{API_BASE}/conversations/{conversation_id}/messages",
        json=message_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Response status: {response.status_code}")
    if response.status_code != 200:
        print(f"Error: {response.text}")
    else:
        messages = response.json()
        print(f"Success! Got {len(messages)} messages")
        for msg in messages:
            print(f"- {msg['role']}: {msg['content'][:100]}...")

if __name__ == "__main__":
    test_image_upload()