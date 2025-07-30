#!/usr/bin/env python3
"""Debug memory storage and retrieval"""

import requests
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_recent_messages():
    """Check recent messages and their costs"""
    print("=== Checking Recent Messages ===")
    
    # Get projects
    response = requests.get("http://localhost:8000/api/projects")
    projects = response.json()
    
    if projects:
        project_id = projects[0]["id"]
        print(f"Project: {projects[0]['name']} ({project_id})")
        
        # Get conversations
        response = requests.get(f"http://localhost:8000/api/projects/{project_id}/conversations")
        conversations = response.json()
        
        if conversations:
            # Get messages from the most recent conversation
            conv_id = conversations[0]["id"]
            response = requests.get(f"http://localhost:8000/api/conversations/{conv_id}/messages")
            
            if response.status_code == 200:
                messages = response.json()
                print(f"\nFound {len(messages)} messages in conversation")
                
                for msg in messages[-10:]:  # Last 10 messages
                    print(f"\n{msg['role'].upper()}:")
                    print(f"  Content: {msg['content'][:100]}...")
                    if msg.get('model'):
                        print(f"  Model: {msg['model']}")
                    if msg.get('cost') is not None:
                        print(f"  Cost: ${msg['cost']:.6f}")
                    if msg.get('tokens_used'):
                        print(f"  Tokens: {msg['tokens_used']}")

async def check_memory_search():
    """Test memory search directly"""
    print("\n\n=== Testing Memory Search ===")
    
    try:
        from adam_v2.services.advanced_memory_service import AdvancedMemoryService
        
        # Use the first project
        response = requests.get("http://localhost:8000/api/projects")
        projects = response.json()
        
        if projects:
            project_id = projects[0]["id"]
            project_name = projects[0]["name"]
            
            memory_service = AdvancedMemoryService(project_id, project_name)
            
            # Search for our secret code
            query = "secret code BANANA-SPLIT"
            memories = await memory_service.advanced_search(
                query=query,
                limit=10,
                use_bm25=True,
                use_semantic=True
            )
            
            print(f"Searching for: '{query}'")
            print(f"Found {len(memories)} memories")
            
            for i, mem in enumerate(memories):
                print(f"\n{i+1}. Score: {mem.score:.3f}")
                print(f"   Content: {mem.content[:200]}...")
                if "BANANA" in mem.content:
                    print("   ✅ Contains BANANA!")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

async def test_cost_threshold():
    """Check if cost threshold is preventing storage"""
    print("\n\n=== Testing Cost Threshold ===")
    
    # Calculate approximate cost for grok-3-mini-high
    # From config: cost_per_1k_tokens = 0.004 (average of input/output)
    tokens = 500  # Typical response
    cost = (tokens / 1000) * 0.004
    print(f"Estimated cost for {tokens} tokens: ${cost:.6f}")
    print(f"Threshold for storage: $0.001000")
    print(f"Will it be stored? {'YES' if cost > 0.001 else 'NO'}")
    
if __name__ == "__main__":
    check_recent_messages()
    asyncio.run(check_memory_search())
    asyncio.run(test_cost_threshold())