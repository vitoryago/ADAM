#!/usr/bin/env python3
"""
Simple test of intelligent routing
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.adam.llm.client import UnifiedLLMClient

async def test():
    client = UnifiedLLMClient()
    
    # Test 1: Simple query
    print("Test 1: Simple query")
    query1 = "What is Python?"
    analysis1 = client.analyze_query(query1)
    print(f"Query: {query1}")
    print(f"Complexity: {analysis1['complexity']}")
    print(f"Model: {analysis1['recommended_model']}")
    print()
    
    # Test 2: Complex query
    print("Test 2: Complex query")
    query2 = "Write a Python function to implement binary search"
    analysis2 = client.analyze_query(query2)
    print(f"Query: {query2}")
    print(f"Complexity: {analysis2['complexity']}")
    print(f"Model: {analysis2['recommended_model']}")
    print()
    
    # Test actual completion
    print("Testing actual completion...")
    response = await client.complete("List 3 benefits of Python", stream=False)
    print(f"Model used: {response.model}")
    print(f"Response: {response.content[:100]}...")

if __name__ == "__main__":
    asyncio.run(test())