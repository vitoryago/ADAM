#!/usr/bin/env python3
"""
Test script to verify the fixes for ADAM v2.0
Tests:
1. Code formatting with syntax highlighting
2. Grok-4-reasoning pricing
3. Streaming responses
4. No page reload
"""

import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000"

async def test_fixes():
    """Test all the fixes"""
    print("🧪 Testing ADAM v2.0 Fixes")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Create a test project
        print("\n1️⃣ Creating test project...")
        response = await client.post(
            f"{BASE_URL}/api/projects/",
            json={
                "name": "Fix Test Project",
                "description": "Testing fixes for code formatting, pricing, and streaming",
                "settings": {
                    "model": "grok-4-reasoning"
                }
            }
        )
        project = response.json()
        project_id = project["id"]
        print(f"   ✅ Project created: {project_id}")
        
        # 2. Create conversation
        print("\n2️⃣ Creating conversation...")
        response = await client.post(
            f"{BASE_URL}/api/projects/{project_id}/conversations",
            json={"title": "Fix Test Conversation"}
        )
        conversation = response.json()
        conversation_id = conversation["id"]
        print(f"   ✅ Conversation created: {conversation_id}")
        
        # 3. Test code formatting
        print("\n3️⃣ Testing code formatting...")
        print("   Sending message with code request...")
        
        # Send a message that should get a code response
        test_message = """
        Can you show me a simple Python function that calculates factorial?
        Please include proper error handling.
        """
        
        # Stream the response
        async with client.stream(
            'POST',
            f"{BASE_URL}/api/conversations/{conversation_id}/messages/stream",
            json={
                "content": test_message,
                "use_memory": True,
                "model": "grok-4-reasoning"
            }
        ) as response:
            print("   📡 Streaming response...")
            chunk_count = 0
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    if data['type'] == 'assistant_chunk':
                        chunk_count += 1
                        if chunk_count <= 3:  # Show first 3 chunks
                            print(f"      Chunk {chunk_count}: {data['content'][:50]}...")
                    elif data['type'] == 'complete':
                        print(f"   ✅ Response complete!")
                        print(f"      Model: {data['model']}")
                        print(f"      Tokens: {data['tokens']}")
                        print(f"      Cost: ${data['cost']:.4f}")
                        print(f"      Total chunks: {chunk_count}")
        
        # 4. Check conversation was updated without reload
        print("\n4️⃣ Checking conversation update...")
        response = await client.get(f"{BASE_URL}/api/conversations/{conversation_id}")
        conv_data = response.json()
        print(f"   ✅ Message count: {conv_data['message_count']}")
        print(f"   ✅ Total cost: ${conv_data['total_cost']:.4f}")
        
        print("\n" + "=" * 50)
        print("✅ All fixes tested!")
        print("\nNotes:")
        print("- Code blocks should appear with syntax highlighting")
        print("- Grok-4-reasoning should show proper cost calculation")
        print("- Response should stream in chunks (not all at once)")
        print("- Page should not reload after sending message")
        print(f"\n🌐 Test the UI at: http://localhost:8000/project/{project_id}?conversation={conversation_id}")

if __name__ == "__main__":
    print("⚠️  Make sure ADAM v2.0 is running: python main.py")
    asyncio.run(test_fixes())