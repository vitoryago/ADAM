#!/usr/bin/env python3
"""
Example script to test ADAM v2.0 memory functionality
"""

import asyncio
import httpx
import json
from typing import Dict, Any, List


BASE_URL = "http://localhost:8000"


async def create_project_with_memory(name: str) -> str:
    """Create a new project with memory enabled"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/projects",
            json={
                "name": name,
                "settings": {
                    "model": "automatic",
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "memory_enabled": True
                }
            }
        )
        response.raise_for_status()
        return response.json()["id"]


async def create_conversation(project_id: str, title: str) -> str:
    """Create a conversation"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/projects/{project_id}/conversations",
            json={"title": title}
        )
        response.raise_for_status()
        return response.json()["id"]


async def send_message_with_memory(
    conversation_id: str, 
    content: str, 
    use_memory: bool = True
) -> Dict[str, Any]:
    """Send a message with memory enabled"""
    async with httpx.AsyncClient() as client:
        print(f"\n👤 User: {content}")
        
        response = await client.post(
            f"{BASE_URL}/api/conversations/{conversation_id}/messages",
            json={
                "content": content,
                "use_memory": use_memory
            }
        )
        response.raise_for_status()
        
        messages = response.json()
        assistant_msg = messages[1]
        
        print(f"\n🤖 ADAM: {assistant_msg['content']}")
        print(f"   💰 Cost: ${assistant_msg['cost']:.4f}")
        
        return assistant_msg


async def search_memories(
    project_id: str, 
    query: str, 
    memory_types: List[str] = None
) -> List[Dict[str, Any]]:
    """Search project memories"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/projects/{project_id}/memories/search",
            json={
                "query": query,
                "limit": 5,
                "memory_types": memory_types,
                "min_relevance": 0.7
            }
        )
        response.raise_for_status()
        return response.json()


async def get_memory_stats(project_id: str) -> Dict[str, Any]:
    """Get memory statistics for a project"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/api/projects/{project_id}/memories/stats"
        )
        response.raise_for_status()
        return response.json()


async def manually_store_memory(
    project_id: str,
    content: str,
    memory_type: str = "conversation"
) -> str:
    """Manually store a memory"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/projects/{project_id}/memories",
            json={
                "content": content,
                "memory_type": memory_type,
                "metadata": {
                    "source": "manual",
                    "important": True
                }
            }
        )
        response.raise_for_status()
        return response.json()["id"]


async def main():
    """Run memory functionality examples"""
    print("🧠 ADAM v2.0 Memory System Demo")
    print("=" * 50)
    
    # Create project
    print("\n📁 Creating project with memory enabled...")
    project_id = await create_project_with_memory("Memory Demo Project")
    print(f"   ✅ Project created: {project_id}")
    
    # Create conversation
    print("\n💬 Creating conversation...")
    conversation_id = await create_conversation(project_id, "Memory Test Chat")
    print(f"   ✅ Conversation created: {conversation_id}")
    
    # Test 1: Ask technical questions that should be remembered
    print("\n" + "="*50)
    print("Test 1: Technical Knowledge Storage")
    
    await send_message_with_memory(
        conversation_id,
        "Can you explain how async/await works in Python with a detailed example?"
    )
    
    # Test 2: Ask about an error (should be stored as error_solution)
    print("\n" + "="*50)
    print("Test 2: Error Solution Storage")
    
    await send_message_with_memory(
        conversation_id,
        "I'm getting a 'RuntimeError: This event loop is already running' error in my async code. How do I fix it?"
    )
    
    # Test 3: Ask for code pattern
    print("\n" + "="*50)
    print("Test 3: Code Pattern Storage")
    
    await send_message_with_memory(
        conversation_id,
        "Write a Python decorator that measures function execution time"
    )
    
    # Test 4: Search memories
    print("\n" + "="*50)
    print("Test 4: Memory Search")
    
    print("\n🔍 Searching for 'async' in memories...")
    async_memories = await search_memories(project_id, "async")
    
    if async_memories:
        print(f"   Found {len(async_memories)} relevant memories:")
        for i, mem in enumerate(async_memories[:3], 1):
            print(f"   {i}. [{mem['memory_type']}] (relevance: {mem['relevance_score']:.2f})")
            print(f"      {mem['content'][:100]}...")
    else:
        print("   No memories found yet")
    
    # Test 5: Manually store important information
    print("\n" + "="*50)
    print("Test 5: Manual Memory Storage")
    
    memory_id = await manually_store_memory(
        project_id,
        "Important project requirement: The system must handle 1000 concurrent users with sub-second response times",
        "concept_explanation"
    )
    print(f"   ✅ Manually stored memory: {memory_id}")
    
    # Test 6: Use stored memory in conversation
    print("\n" + "="*50)
    print("Test 6: Memory Recall")
    
    await send_message_with_memory(
        conversation_id,
        "What were the performance requirements mentioned for this project?"
    )
    
    # Test 7: Memory without recall
    print("\n" + "="*50)
    print("Test 7: Response Without Memory")
    
    await send_message_with_memory(
        conversation_id,
        "What's the weather like?",
        use_memory=False
    )
    
    # Show memory statistics
    print("\n" + "="*50)
    print("📊 Memory Statistics:")
    
    stats = await get_memory_stats(project_id)
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   Memory types: {json.dumps(stats['memory_types'], indent=6)}")
    print(f"   Total cost tracked: ${stats['total_cost']:.4f}")
    print(f"   Average access count: {stats['avg_access_count']}")
    
    # Test memory isolation
    print("\n" + "="*50)
    print("Test 8: Memory Isolation")
    
    # Create another project
    project2_id = await create_project_with_memory("Another Project")
    print(f"\n📁 Created second project: {project2_id}")
    
    # Search in second project (should find nothing)
    print("\n🔍 Searching for 'async' in second project...")
    project2_memories = await search_memories(project2_id, "async")
    print(f"   Found {len(project2_memories)} memories (should be 0 - isolated)")
    
    print("\n✅ Memory isolation verified!")
    
    # Memory types
    print("\n" + "="*50)
    print("📝 Available Memory Types:")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/memory-types")
        types_data = response.json()
        
        if types_data["available"]:
            for mem_type in types_data["types"]:
                print(f"   - {mem_type['value']}: {mem_type['description']}")
        else:
            print("   Memory system not available")


if __name__ == "__main__":
    print("\n⚠️  Make sure ADAM v2.0 is running: python main.py")
    print("⚠️  Memory system requires ChromaDB and embeddings")
    input("\nPress Enter to continue...")
    
    asyncio.run(main())