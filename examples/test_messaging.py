#!/usr/bin/env python3
"""
Example script to test ADAM v2.0 messaging functionality
"""

import asyncio
import httpx
import json
from typing import Dict, Any


BASE_URL = "http://localhost:8000"


async def create_project(name: str, settings: Dict[str, Any] = None) -> str:
    """Create a new project"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/projects",
            json={
                "name": name,
                "settings": settings or {
                    "model": "grok-3-mini-high",
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            }
        )
        response.raise_for_status()
        return response.json()["id"]


async def create_conversation(project_id: str, title: str) -> str:
    """Create a conversation in a project"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/projects/{project_id}/conversations",
            json={"title": title}
        )
        response.raise_for_status()
        return response.json()["id"]


async def send_message(conversation_id: str, content: str, use_memory: bool = True, model: str = None):
    """Send a message and print the response"""
    async with httpx.AsyncClient() as client:
        print(f"\n👤 User: {content}")
        
        response = await client.post(
            f"{BASE_URL}/api/conversations/{conversation_id}/messages",
            json={
                "content": content,
                "use_memory": use_memory,
                "model": model
            }
        )
        response.raise_for_status()
        
        messages = response.json()
        assistant_msg = messages[1]  # Second message is assistant response
        
        print(f"\n🤖 ADAM ({assistant_msg['model']}): {assistant_msg['content']}")
        print(f"   💰 Cost: ${assistant_msg['cost']:.4f} | 🔢 Tokens: {assistant_msg['tokens_used']}")
        
        return assistant_msg


async def stream_message(conversation_id: str, content: str):
    """Send a message and stream the response"""
    async with httpx.AsyncClient() as client:
        print(f"\n👤 User: {content}")
        print("\n🤖 ADAM: ", end="", flush=True)
        
        async with client.stream(
            "POST",
            f"{BASE_URL}/api/conversations/{conversation_id}/messages/stream",
            json={
                "content": content,
                "use_memory": True
            }
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        
                        if data["type"] == "assistant_chunk":
                            print(data["content"], end="", flush=True)
                        elif data["type"] == "complete":
                            print(f"\n   💰 Cost: ${data['cost']:.4f} | 🔢 Tokens: {data['tokens']} | 🎯 Model: {data['model']}")
                    except json.JSONDecodeError:
                        pass


async def get_conversation_messages(conversation_id: str):
    """Get all messages in a conversation"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/api/conversations/{conversation_id}/messages"
        )
        response.raise_for_status()
        return response.json()


async def main():
    """Run example messaging scenarios"""
    print("🚀 ADAM v2.0 Messaging Example")
    print("=" * 50)
    
    # Create a project
    print("\n📁 Creating project...")
    project_id = await create_project(
        "AI Assistant Demo",
        settings={
            "model": "automatic",  # Use intelligent routing
            "temperature": 0.7,
            "max_tokens": 2000
        }
    )
    print(f"   ✅ Project created: {project_id}")
    
    # Create a conversation
    print("\n💬 Creating conversation...")
    conversation_id = await create_conversation(project_id, "Demo Chat")
    print(f"   ✅ Conversation created: {conversation_id}")
    
    # Test 1: Simple message
    print("\n" + "="*50)
    print("Test 1: Simple Message")
    await send_message(conversation_id, "Hello ADAM! What are you capable of?")
    
    # Test 2: Technical query (should use better model)
    print("\n" + "="*50)
    print("Test 2: Technical Query")
    await send_message(
        conversation_id,
        "Can you explain how async/await works in Python with a code example?"
    )
    
    # Test 3: Use memory
    print("\n" + "="*50)
    print("Test 3: Memory Usage")
    await send_message(
        conversation_id,
        "What was my first question to you?",
        use_memory=True
    )
    
    # Test 4: Streaming response
    print("\n" + "="*50)
    print("Test 4: Streaming Response")
    await stream_message(
        conversation_id,
        "Write a short poem about artificial intelligence"
    )
    
    # Test 5: Model override
    print("\n" + "="*50)
    print("Test 5: Model Override")
    await send_message(
        conversation_id,
        "What's 2+2?",
        model="grok-3-mini-high"  # Force simple model
    )
    
    # Show conversation history
    print("\n" + "="*50)
    print("📜 Conversation History:")
    messages = await get_conversation_messages(conversation_id)
    print(f"   Total messages: {len(messages)}")
    
    total_cost = sum(msg.get("cost", 0) for msg in messages if msg.get("cost"))
    total_tokens = sum(msg.get("tokens_used", 0) for msg in messages if msg.get("tokens_used"))
    
    print(f"   💰 Total cost: ${total_cost:.4f}")
    print(f"   🔢 Total tokens: {total_tokens}")
    
    # Get project stats
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/projects/{project_id}/stats")
        stats = response.json()
        print(f"\n📊 Project Stats:")
        print(f"   Conversations: {stats['conversation_count']}")
        print(f"   Memory stats: {stats['memory_stats']}")


if __name__ == "__main__":
    print("\n⚠️  Make sure ADAM v2.0 is running: python main.py")
    print("⚠️  Set your API keys in .env file")
    input("\nPress Enter to continue...")
    
    asyncio.run(main())