#!/usr/bin/env python
"""Check what's stored in memory"""
import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adam_v2.services.memory_service import ProjectMemoryService

async def check_memory(project_id: str):
    """Check memories for a project"""
    memory_service = ProjectMemoryService(project_id, "Test Project")
    
    # Search for different queries
    queries = ["secret code", "configuration", "DATABASE_URL", "Python"]
    
    for query in queries:
        print(f"\n=== Searching for: {query} ===")
        memories = await memory_service.search_memories(
            query=query,
            conversation_id=None,  # Search across all conversations
            limit=5
        )
        
        for i, mem in enumerate(memories):
            print(f"\nMemory {i+1}:")
            print(f"Type: {mem.memory_type}")
            print(f"Score: {mem.relevance_score:.3f}")
            print(f"Content preview: {mem.content[:200]}...")
            if "configuration" in query.lower() or "database" in query.lower():
                # Show full content for configuration
                print(f"Full content:\n{mem.content}")

if __name__ == "__main__":
    # Use the project ID from the test
    project_id = "1bade603-ffa3-4155-9fa7-047fd32c69fb"
    asyncio.run(check_memory(project_id))