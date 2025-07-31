#!/usr/bin/env python
"""Test memory persistence across conversations"""
import asyncio
import aiohttp
import json

BASE_URL = "http://localhost:8000"

async def create_project():
    """Create a test project"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/api/projects/",
            json={"name": "Memory Test Project", "description": "Testing memory across conversations"}
        ) as resp:
            project = await resp.json()
            print(f"Created project: {project['name']} (ID: {project['id']})")
            return project['id']

async def create_conversation(project_id, title):
    """Create a conversation"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/api/projects/{project_id}/conversations/",
            json={"title": title}
        ) as resp:
            conv = await resp.json()
            print(f"Created conversation: {conv['title']} (ID: {conv['id']})")
            return conv['id']

async def send_message(conversation_id, content, use_memory=True):
    """Send a message and get response"""
    async with aiohttp.ClientSession() as session:
        # Use the streaming endpoint
        async with session.post(
            f"{BASE_URL}/api/conversations/{conversation_id}/messages/stream",
            json={"content": content, "use_memory": use_memory}
        ) as resp:
            print(f"\nUser: {content}")
            print(f"Assistant: ", end="")
            
            full_response = ""
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if data['type'] == 'assistant_chunk':
                            chunk = data['content']
                            full_response += chunk
                            if len(full_response) <= 200:
                                print(chunk, end="", flush=True)
                            elif len(full_response) - len(chunk) < 200:
                                # Print the remaining part to reach 200 chars
                                remaining = 200 - (len(full_response) - len(chunk))
                                print(chunk[:remaining] + "...", end="", flush=True)
                    except json.JSONDecodeError:
                        pass
            
            print()  # New line after response
            return full_response

async def test_memory_persistence():
    """Test memory persistence across conversations"""
    # Create project
    project_id = await create_project()
    
    # Conversation 1: Store some information
    print("\n=== Conversation 1: Storing Information ===")
    conv1_id = await create_conversation(project_id, "Memory Storage Test")
    
    # Send messages with valuable information
    await send_message(conv1_id, "Remember this secret code: MEMORY-TEST-2024")
    await send_message(conv1_id, "Also, the project uses Python 3.11 with FastAPI and PostgreSQL")
    await send_message(conv1_id, """Here's an important configuration:
```python
DATABASE_URL = "postgresql://user:pass@localhost/adamdb"
REDIS_URL = "redis://localhost:6379"
SECRET_KEY = "development-key-123"
```
This is critical for the project setup.""")
    
    # Wait a bit
    await asyncio.sleep(2)
    
    # Conversation 2: Try to recall information
    print("\n=== Conversation 2: Testing Memory Recall ===")
    conv2_id = await create_conversation(project_id, "Memory Recall Test")
    
    # Ask about previously stored information
    await send_message(conv2_id, "What was the secret code we discussed?")
    await send_message(conv2_id, "What database and framework are we using for the project?")
    await send_message(conv2_id, "Can you show me the configuration we discussed earlier?")
    
    print("\n=== Memory Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_memory_persistence())