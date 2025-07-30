#!/usr/bin/env python3
"""Test ADAM's memory system"""

import asyncio
import requests
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test 1: Check if memory is being stored
async def test_memory_storage():
    print("=== Testing Memory Storage ===")
    
    # Get a conversation
    response = requests.get("http://localhost:8000/api/projects")
    projects = response.json()
    if not projects:
        print("No projects found")
        return
    
    project_id = projects[0]["id"]
    print(f"Using project: {projects[0]['name']} ({project_id})")
    
    # Send a memorable message
    response = requests.get(f"http://localhost:8000/api/projects/{project_id}/conversations")
    conversations = response.json()
    
    if conversations:
        conv_id = conversations[0]["id"]
        
        # Send a message with unique content
        unique_content = "The secret code for today is BANANA-SPLIT-2024. Remember this!"
        response = requests.post(
            f"http://localhost:8000/api/conversations/{conv_id}/messages",
            json={
                "content": unique_content,
                "use_memory": True,
                "model": "grok-3-mini-high"
            }
        )
        
        if response.status_code == 200:
            print("✓ Message sent successfully")
            messages = response.json()
            print(f"Assistant response: {messages[1]['content'][:100]}...")
            
            # Wait a bit for memory to be stored
            await asyncio.sleep(2)
            
            # Now test retrieval
            test_query = "What was the secret code I mentioned?"
            response = requests.post(
                f"http://localhost:8000/api/conversations/{conv_id}/messages",
                json={
                    "content": test_query,
                    "use_memory": True,
                    "model": "grok-3-mini-high"
                }
            )
            
            if response.status_code == 200:
                messages = response.json()
                assistant_response = messages[1]['content']
                print(f"\n✓ Memory test query sent")
                print(f"Assistant response: {assistant_response}")
                
                if "BANANA-SPLIT-2024" in assistant_response:
                    print("\n🎉 SUCCESS! ADAM remembered the secret code!")
                else:
                    print("\n⚠️  ADAM didn't recall the specific code")
        else:
            print(f"✗ Failed to send message: {response.status_code}")
            print(response.text)

# Test 2: Check memory service directly
async def test_memory_service():
    print("\n\n=== Testing Memory Service Directly ===")
    
    try:
        from adam_v2.services.memory_service import ProjectMemoryService
        from adam_v2.services.advanced_memory_service import AdvancedMemoryService
        
        project_id = "test-project"
        project_name = "Test Project"
        
        # Try advanced memory service
        try:
            memory_service = AdvancedMemoryService(project_id, project_name)
            print("✓ Using AdvancedMemoryService")
        except:
            memory_service = ProjectMemoryService(project_id, project_name)
            print("✓ Using ProjectMemoryService")
        
        # Store a test memory
        await memory_service.store_memory(
            content="Test memory: The password is WATERMELON",
            memory_type="conversation",
            metadata={"test": True},
            conversation_id="test-conv",
            cost=0.01
        )
        print("✓ Stored test memory")
        
        # Search for it
        memories = await memory_service.search_memories(
            query="What is the password?",
            limit=5
        )
        
        if memories:
            print(f"✓ Found {len(memories)} memories")
            for mem in memories:
                print(f"  - {mem.content[:100]}...")
                if "WATERMELON" in mem.content:
                    print("  🎉 Found our test memory!")
        else:
            print("✗ No memories found")
            
    except Exception as e:
        print(f"✗ Error testing memory service: {e}")
        import traceback
        traceback.print_exc()

# Test 3: Check if ChromaDB is running
def test_chromadb():
    print("\n\n=== Testing ChromaDB ===")
    
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./adam_memory_store")
        collections = client.list_collections()
        print(f"✓ ChromaDB is working. Found {len(collections)} collections")
        for col in collections:
            print(f"  - {col.name} ({col.count()} items)")
    except Exception as e:
        print(f"✗ ChromaDB error: {e}")

if __name__ == "__main__":
    print("Testing ADAM's Memory System...\n")
    
    # Test ChromaDB first
    test_chromadb()
    
    # Run async tests
    asyncio.run(test_memory_storage())
    asyncio.run(test_memory_service())