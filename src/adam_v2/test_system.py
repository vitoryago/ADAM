#!/usr/bin/env python3
"""
Quick system test for ADAM v2.0
Tests all major functionality
"""

import asyncio
import httpx
import json
import sys
from datetime import datetime


BASE_URL = "http://localhost:8000"


async def test_system():
    """Run through all major features"""
    print("🧪 ADAM v2.0 System Test")
    print("=" * 50)
    
    async with httpx.AsyncClient() as client:
        # 1. Check health
        print("\n1️⃣ Checking health...")
        try:
            response = await client.get(f"{BASE_URL}/health")
            response.raise_for_status()
            print("   ✅ Server is healthy")
        except Exception as e:
            print(f"   ❌ Server not responding: {e}")
            print("   Make sure ADAM is running: python main.py")
            return
        
        # 2. Create a project
        print("\n2️⃣ Creating test project...")
        try:
            response = await client.post(
                f"{BASE_URL}/api/projects/",
                json={
                    "name": f"Test Project {datetime.now().strftime('%H:%M:%S')}",
                    "description": "Automated test project",
                    "settings": {
                        "model": "grok-3-mini-high",
                        "temperature": 0.7
                    }
                }
            )
            response.raise_for_status()
            project = response.json()
            project_id = project["id"]
            print(f"   ✅ Project created: {project['name']} (ID: {project_id})")
        except Exception as e:
            print(f"   ❌ Failed to create project: {e}")
            return
        
        # 3. Create a conversation
        print("\n3️⃣ Creating conversation...")
        try:
            response = await client.post(
                f"{BASE_URL}/api/projects/{project_id}/conversations",
                json={"title": "Test Conversation"}
            )
            response.raise_for_status()
            conversation = response.json()
            conversation_id = conversation["id"]
            print(f"   ✅ Conversation created: {conversation['title']}")
        except Exception as e:
            print(f"   ❌ Failed to create conversation: {e}")
            return
        
        # 4. Send a message (without streaming to simplify)
        print("\n4️⃣ Sending test message...")
        try:
            response = await client.post(
                f"{BASE_URL}/api/conversations/{conversation_id}/messages",
                json={
                    "content": "Hello ADAM! Can you tell me a short fun fact about Python?",
                    "use_memory": True
                }
            )
            response.raise_for_status()
            messages = response.json()
            if len(messages) >= 2:
                assistant_msg = messages[1]
                print(f"   ✅ Got response from {assistant_msg['model']}")
                print(f"      Cost: ${assistant_msg['cost']:.4f}")
                print(f"      Response: {assistant_msg['content'][:100]}...")
            else:
                print("   ⚠️  Unexpected response format")
        except Exception as e:
            print(f"   ❌ Failed to send message: {e}")
            print("      Check your API keys in .env file")
        
        # 5. Test memory stats
        print("\n5️⃣ Checking memory stats...")
        try:
            response = await client.get(
                f"{BASE_URL}/api/projects/{project_id}/memories/stats"
            )
            response.raise_for_status()
            stats = response.json()
            print(f"   ✅ Memory stats retrieved")
            print(f"      Total memories: {stats.get('total_memories', 0)}")
            print(f"      Total cost: ${stats.get('total_cost', 0):.4f}")
        except Exception as e:
            print(f"   ⚠️  Memory system not available: {e}")
        
        # 6. List projects
        print("\n6️⃣ Listing all projects...")
        try:
            response = await client.get(f"{BASE_URL}/api/projects/")
            response.raise_for_status()
            projects = response.json()
            print(f"   ✅ Found {len(projects)} projects")
            for p in projects[-3:]:  # Show last 3
                print(f"      - {p['name']} ({p.get('conversation_count', 0)} conversations)")
        except Exception as e:
            print(f"   ❌ Failed to list projects: {e}")
        
        print("\n" + "=" * 50)
        print("✅ System test complete!")
        print(f"\n🌐 Open http://localhost:8000 in your browser")
        print(f"📂 Your test project: http://localhost:8000/project/{project_id}")


if __name__ == "__main__":
    print("\n⚠️  Make sure ADAM v2.0 is running: python main.py")
    print("⚠️  Make sure you have API keys in .env file")
    print("\nStarting test...")
    
    asyncio.run(test_system())