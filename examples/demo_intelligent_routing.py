#!/usr/bin/env python3
"""
Demo: Intelligent Model Routing with grok-4-reasoning
Shows how ADAM automatically selects the right model for each query
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.adam.llm.client import UnifiedLLMClient
from src.adam.llm.config import LLMConfig

async def demo_routing():
    """Demonstrate intelligent model routing"""
    
    print("="*70)
    print("ADAM INTELLIGENT MODEL ROUTING DEMO")
    print("="*70)
    print("\nADAM now automatically selects the best model for your query:")
    print("- grok-4-reasoning: Complex tasks requiring deep thinking")
    print("- grok-4: Standard analysis and queries")
    print("- grok-3-mini-high: Simple questions and memory tasks")
    print("\n" + "="*70 + "\n")
    
    # Initialize client
    client = UnifiedLLMClient()
    
    # Demo queries
    demos = [
        {
            "title": "Simple Memory Query",
            "query": "Give me a quick summary of what we discussed earlier",
            "expected": "grok-3-mini-high"
        },
        {
            "title": "Standard BigQuery Analysis",
            "query": "How can I optimize a BigQuery query that's scanning too much data?",
            "expected": "grok-4"
        },
        {
            "title": "Complex Code Generation",
            "query": "Write a Python class that implements a distributed cache with Redis backend, TTL support, connection pooling, and async operations",
            "expected": "grok-4-reasoning"
        }
    ]
    
    for demo in demos:
        print(f"📝 {demo['title']}")
        print(f"Query: '{demo['query']}'\n")
        
        # Analyze the query
        analysis = client.analyze_query(demo['query'])
        
        print(f"Analysis Results:")
        print(f"  - Complexity: {analysis['complexity']}")
        print(f"  - Selected Model: {analysis['recommended_model']}")
        print(f"  - Reasoning Effort: {analysis['reasoning_effort']}")
        print(f"  - Confidence: {analysis['confidence']:.2%}")
        
        if analysis['reasoning']:
            print(f"  - Why: {analysis['reasoning'][0]}")
        
        # Get actual response
        print(f"\nGetting response from {analysis['recommended_model']}...")
        response = await client.complete(demo['query'], stream=False)
        
        print(f"Response preview: {response.content[:150]}...")
        print(f"Model used: {response.model}")
        if response.reasoning_tokens:
            print(f"Reasoning tokens: {response.reasoning_tokens}")
        print(f"Cost: ${response.cost:.4f}")
        
        print("\n" + "-"*70 + "\n")
    
    # Show cost comparison
    print("💰 COST EFFICIENCY DEMONSTRATION")
    print("-"*70)
    
    complex_query = "Write a complete implementation of a B+ tree in Python"
    simple_query = "What is a B+ tree?"
    
    # Complex with grok-4-reasoning
    print(f"\nComplex Query: '{complex_query}'")
    resp1 = await client.complete(complex_query, stream=False)
    print(f"  Model: {resp1.model}")
    print(f"  Tokens: {resp1.total_tokens}")
    print(f"  Cost: ${resp1.cost:.4f}")
    
    # Simple with auto-selection
    print(f"\nSimple Query: '{simple_query}'")
    resp2 = await client.complete(simple_query, stream=False)
    print(f"  Model: {resp2.model}")
    print(f"  Tokens: {resp2.total_tokens}")
    print(f"  Cost: ${resp2.cost:.4f}")
    
    if resp1.cost > 0 and resp2.cost > 0:
        savings = (1 - resp2.cost/resp1.cost) * 100
        print(f"\n✅ Cost savings by using appropriate models: {savings:.1f}%")

async def main():
    """Run the demo"""
    try:
        await demo_routing()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have set your XAI_API_KEY environment variable")

if __name__ == "__main__":
    print("Starting Intelligent Model Routing Demo...\n")
    asyncio.run(main())