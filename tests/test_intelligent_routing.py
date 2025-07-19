#!/usr/bin/env python3
"""
Test Intelligent Model Routing with grok-4-reasoning
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.adam.llm.client import UnifiedLLMClient
from src.adam.llm.config import LLMConfig

async def test_model_routing():
    """Test different queries to see model selection"""
    
    # Initialize client
    llm_config = LLMConfig()
    client = UnifiedLLMClient(llm_config)
    
    print("="*60)
    print("INTELLIGENT MODEL ROUTING TEST")
    print("="*60)
    print(f"\nAvailable models: {llm_config.get_available_models()}")
    print("\n")
    
    # Test queries of different complexities
    test_queries = [
        # Low complexity - should use grok-3-mini-high
        "What did we discuss in our last conversation?",
        "Give me a quick summary of BigQuery best practices",
        "List the main features of Python",
        
        # Medium complexity - should use grok-4
        "Analyze this BigQuery query for performance issues: SELECT * FROM large_table WHERE date > '2024-01-01'",
        "Explain how neural networks work",
        "What are the best practices for microservices architecture?",
        
        # High complexity - should use grok-4-reasoning
        "Write a Python class that implements a thread-safe LRU cache with TTL support and statistics tracking",
        "Design a scalable architecture for a real-time analytics system handling 1M events/second",
        "Debug this complex async Python code and explain why it's causing deadlocks: [imagine complex code here]",
        "Implement a BigQuery optimization framework that automatically detects and fixes common performance issues"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. Query: {query[:80]}...")
        
        # Analyze the query
        analysis = client.analyze_query(query)
        
        print(f"   Complexity: {analysis['complexity']}")
        print(f"   Recommended Model: {analysis['recommended_model']}")
        print(f"   Reasoning Effort: {analysis['reasoning_effort']}")
        print(f"   Confidence: {analysis['confidence']:.2f}")
        
        if analysis['reasoning']:
            print(f"   Why: {analysis['reasoning'][0]}")
        
        print()
    
    # Test actual completion with auto-routing
    print("\n" + "="*60)
    print("TESTING ACTUAL COMPLETIONS")
    print("="*60)
    
    # Simple query
    print("\n1. Simple Query Test:")
    query1 = "What are the main benefits of using BigQuery?"
    print(f"   Query: {query1}")
    
    response1 = await client.complete(query1, stream=False)
    print(f"   Model Used: {response1.model}")
    print(f"   Response: {response1.content[:200]}...")
    
    # Complex query
    print("\n2. Complex Query Test:")
    query2 = "Write a Python function that implements a distributed rate limiter using Redis with sliding window algorithm"
    print(f"   Query: {query2}")
    
    response2 = await client.complete(query2, stream=False)
    print(f"   Model Used: {response2.model}")
    print(f"   Response: {response2.content[:200]}...")
    
    # Show cost difference
    if response1.cost and response2.cost:
        print(f"\n   Cost Comparison:")
        print(f"   Simple query: ${response1.cost:.4f}")
        print(f"   Complex query: ${response2.cost:.4f}")

async def test_reasoning_modes():
    """Test different reasoning effort levels"""
    
    client = UnifiedLLMClient()
    
    print("\n" + "="*60)
    print("REASONING EFFORT LEVELS TEST")
    print("="*60)
    
    query = "Explain how to optimize a slow BigQuery query"
    
    # Test with different reasoning efforts
    for effort in ["low", "medium", "high"]:
        print(f"\nTesting with reasoning_effort='{effort}':")
        
        response = await client.complete(
            query, 
            reasoning_effort=effort,
            stream=False
        )
        
        print(f"Model: {response.model}")
        if response.reasoning_tokens:
            print(f"Reasoning tokens: {response.reasoning_tokens}")
        print(f"Response preview: {response.content[:150]}...")

async def main():
    """Run all tests"""
    await test_model_routing()
    await test_reasoning_modes()

if __name__ == "__main__":
    print("Starting Intelligent Model Routing Tests...")
    asyncio.run(main())