#!/usr/bin/env python3
"""
Test script for AI routing prototype
"""
import asyncio
import json
import sys
from pathlib import Path

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adam.llm.ai_router import AIRouter, SmartRoutingEngine

async def test_ai_router():
    """Test the AI routing with various queries"""
    
    # Initialize with GPT-5-mini for fast routing decisions
    router = AIRouter(routing_model="gpt-5-mini", enable_caching=True)
    
    test_queries = [
        "Hello, how are you?",
        "Write a Python function to implement quicksort with detailed comments",
        "Explain quantum computing in simple terms",
        "What's 2+2?",
        "Design a distributed system architecture for handling 1M requests per second with fault tolerance",
        "Debug this code: def factorial(n): return n * factorial(n-1)",
        "Translate 'Hello world' to Spanish",
        "Summarize the key concepts of machine learning",
        "Create a comprehensive test suite for a REST API",
        "What's the weather like?"
    ]
    
    print("=" * 80)
    print("Testing AI Router with GPT-5-mini as routing model")
    print("=" * 80)
    
    for query in test_queries:
        try:
            print(f"\n📝 Query: '{query[:60]}{'...' if len(query) > 60 else ''}'")
            
            # Route the query
            decision = await router.route(query)
            
            # Display results
            print(f"   ✓ Model: {decision.primary_model}")
            print(f"   ✓ Complexity: {decision.complexity}")
            print(f"   ✓ Task Type: {decision.task_type.value}")
            print(f"   ✓ Confidence: {decision.confidence:.2%}")
            print(f"   ✓ Est. Cost: ${decision.estimated_cost:.4f}")
            print(f"   ✓ Special Reqs: {decision.special_requirements or 'None'}")
            print(f"   ✓ Reasoning: {decision.reasoning[:100]}...")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    # Test caching
    print("\n" + "=" * 80)
    print("Testing Caching (repeating first query)")
    print("=" * 80)
    
    decision = await router.route("Hello, how are you?")
    print(f"Cached result: {decision.primary_model} (should be instant)")

async def test_smart_routing():
    """Test the SmartRoutingEngine with fallback"""
    
    print("\n" + "=" * 80)
    print("Testing SmartRoutingEngine (AI + Fallback)")
    print("=" * 80)
    
    # Test with AI routing enabled
    smart_router = SmartRoutingEngine(use_ai=True, routing_model="gpt-5-mini")
    
    test_query = "Implement a neural network from scratch in Python"
    
    print(f"\n📝 Query: '{test_query}'")
    
    result = await smart_router.route(test_query)
    
    print(f"   Method: {result['method']}")
    print(f"   Model: {result['model']}")
    print(f"   Confidence: {result.get('confidence', 'N/A')}")
    print(f"   Complexity: {result.get('complexity', 'N/A')}")
    print(f"   Reasoning: {result.get('reasoning', 'N/A')[:100]}...")
    
    # Test fallback when AI fails
    print("\n Testing fallback mechanism...")
    smart_router_fallback = SmartRoutingEngine(use_ai=False)
    
    result = await smart_router_fallback.route(test_query)
    
    print(f"   Method: {result['method']}")
    print(f"   Model: {result['model']}")
    print(f"   Complexity: {result.get('complexity', 'N/A')}")

async def test_vision_routing():
    """Test routing for vision tasks"""
    
    print("\n" + "=" * 80)
    print("Testing Vision Task Routing")
    print("=" * 80)
    
    router = AIRouter(routing_model="gpt-5-mini")
    
    vision_query = "Analyze this image and describe what you see in detail"
    
    # Simulate having image data
    context = {"has_image": True}
    
    decision = await router.route(vision_query, context=context)
    
    print(f"\n📝 Query: '{vision_query}'")
    print(f"   Context: {context}")
    print(f"   ✓ Model: {decision.primary_model}")
    print(f"   ✓ Special Reqs: {decision.special_requirements}")
    print(f"   ✓ Reasoning: {decision.reasoning[:100]}...")

if __name__ == "__main__":
    # Run all tests
    asyncio.run(test_ai_router())
    asyncio.run(test_smart_routing())
    asyncio.run(test_vision_routing())
    
    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)