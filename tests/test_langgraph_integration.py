#!/usr/bin/env python3
"""
Test the LangGraph conversation state machine integration
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.adam.langgraph_conversation import (
    LangGraphConversationSystem,
    ConversationState,
    QueryComplexityAnalyzer,
    MemoryConfidenceScorer
)


def test_query_complexity_analyzer():
    """Test query complexity analysis"""
    print("=== Testing Query Complexity Analyzer ===\n")
    
    analyzer = QueryComplexityAnalyzer()
    
    test_cases = [
        ("What is SQL?", "simple"),
        ("How do I create a Python list?", "simple"),
        ("Debug this race condition in my async distributed system with microservices", "complex"),
        ("Implement a scalable architecture for real-time data processing", "complex"),
        ("Explain how to use git branches", "moderate"),
        ("Can you help me optimize this query?", "moderate"),
    ]
    
    for query, expected in test_cases:
        complexity, score = analyzer.analyze(query)
        status = "✓" if complexity == expected else "✗"
        print(f"{status} Query: {query[:50]}...")
        print(f"  Expected: {expected}, Got: {complexity} (confidence: {score:.2f})")
        print()


def test_memory_confidence_scorer():
    """Test memory confidence scoring"""
    print("\n=== Testing Memory Confidence Scorer ===\n")
    
    scorer = MemoryConfidenceScorer()
    
    test_cases = [
        # (query, memory_query, memory_response, similarity, age_days, expected_range)
        ("How to optimize SQL?", "SQL optimization tips", "Use indexes...", 0.9, 5, (0.8, 1.0)),
        ("Debug Python error", "Python debugging", "Check stack trace", 0.7, 30, (0.5, 0.7)),
        ("Old memory test", "Some old query", "Old response", 0.8, 180, (0.3, 0.5)),
    ]
    
    for query, mem_query, mem_response, similarity, age, expected_range in test_cases:
        confidence = scorer.calculate_confidence(
            query, mem_query, mem_response, similarity, age
        )
        in_range = expected_range[0] <= confidence <= expected_range[1]
        status = "✓" if in_range else "✗"
        
        print(f"{status} Query: {query}")
        print(f"  Memory: {mem_query} (age: {age} days)")
        print(f"  Confidence: {confidence:.2f} (expected: {expected_range})")
        print()


async def test_full_flow():
    """Test the full LangGraph flow"""
    print("\n=== Testing Full LangGraph Flow ===\n")
    
    # Create system
    lg_system = LangGraphConversationSystem()
    
    # Test different query types
    test_queries = [
        {
            "query": "What is a Python list?",
            "expected_complexity": "simple",
            "expected_model": ["grok-3-mini-reasoning-high"]
        },
        {
            "query": "Debug this complex race condition in my distributed async system with Redis pub/sub",
            "expected_complexity": "complex",
            "expected_model": ["o3"]
        },
        {
            "query": "How do I optimize a slow database query with multiple joins?",
            "expected_complexity": "moderate",
            "expected_model": ["grok-3-mini-reasoning-high"]
        },
        {
            "query": "Implement a function to parse and validate JSON with error handling",
            "expected_complexity": "complex",
            "expected_model": ["claude-opus-4"]
        }
    ]
    
    for test_case in test_queries:
        query = test_case["query"]
        print(f"Query: {query}")
        
        result = await lg_system.process_query(query, "test_session")
        
        # Check complexity
        complexity_match = result["complexity"] == test_case["expected_complexity"]
        print(f"  {'✓' if complexity_match else '✗'} Complexity: {result['complexity']} "
              f"(expected: {test_case['expected_complexity']})")
        
        # Check model selection
        model_match = result["model_used"] in test_case["expected_model"]
        print(f"  {'✓' if model_match else '✗'} Model: {result['model_used']} "
              f"(expected one of: {test_case['expected_model']})")
        
        print(f"  Cost: ${result['total_cost']:.4f}")
        print(f"  Memory used: {result['memory_used']}")
        print(f"  Memory confidence: {result['memory_confidence']:.2f}")
        print()


async def test_state_machine_flow():
    """Test individual state transitions"""
    print("\n=== Testing State Machine Transitions ===\n")
    
    from src.adam.langgraph_conversation import (
        analyze_query_node,
        check_memory_node,
        verify_memory_freshness_node,
        route_to_llm_node
    )
    
    # Create initial state
    state = ConversationState(
        query="How to optimize a complex SQL query with window functions?",
        complexity="simple",
        complexity_score=0.0,
        memory_found=False,
        memory_confidence=0.0,
        memory_ids=[],
        memory_content=None,
        memory_age_days=None,
        should_verify=False,
        should_use_memory=False,
        selected_model="mistral",
        response=None,
        total_cost=0.0,
        retry_count=0,
        error_message=None,
        conversation_id="test",
        timestamp=asyncio.get_event_loop().time()
    )
    
    print("1. Initial state")
    print(f"   Query: {state['query'][:50]}...")
    
    # Test analyze query node
    state = analyze_query_node(state)
    print(f"\n2. After query analysis")
    print(f"   Complexity: {state['complexity']} (score: {state['complexity_score']:.2f})")
    
    # Test memory check node
    state = check_memory_node(state)
    print(f"\n3. After memory check")
    print(f"   Memory found: {state['memory_found']}")
    print(f"   Memory confidence: {state['memory_confidence']:.2f}")
    print(f"   Should verify: {state['should_verify']}")
    
    # Test freshness verification
    if state['should_verify']:
        state = verify_memory_freshness_node(state)
        print(f"\n4. After freshness check")
        print(f"   Should use memory: {state['should_use_memory']}")
    
    # Test LLM routing
    state = route_to_llm_node(state)
    print(f"\n5. After LLM routing")
    print(f"   Selected model: {state['selected_model']}")
    print(f"   Total cost: ${state['total_cost']:.4f}")


async def test_cost_optimization():
    """Test cost optimization strategies"""
    print("\n=== Testing Cost Optimization ===\n")
    
    lg_system = LangGraphConversationSystem()
    
    # Simulate 100 queries
    query_types = [
        ("What is {}?", "simple", 0.7),  # 70% simple queries
        ("How to implement {}?", "moderate", 0.2),  # 20% moderate
        ("Debug complex {} in distributed system", "complex", 0.1)  # 10% complex
    ]
    
    topics = ["SQL", "Python", "API", "database", "caching", "async", "threading"]
    total_cost = 0.0
    model_usage = {"grok-3-mini-reasoning-high": 0, "o3": 0, "claude-opus-4": 0}
    
    print("Simulating 100 queries...")
    for i in range(100):
        # Select query type based on distribution
        rand = asyncio.get_event_loop().time() % 1
        if rand < 0.7:
            template, _, _ = query_types[0]
        elif rand < 0.9:
            template, _, _ = query_types[1]
        else:
            template, _, _ = query_types[2]
        
        topic = topics[i % len(topics)]
        query = template.format(topic)
        
        result = await lg_system.process_query(query, f"session_{i}")
        total_cost += result["total_cost"]
        model_usage[result["model_used"]] += 1
    
    print(f"\nTotal cost for 100 queries: ${total_cost:.2f}")
    print(f"Average cost per query: ${total_cost/100:.4f}")
    print("\nModel usage distribution:")
    for model, count in model_usage.items():
        print(f"  {model}: {count} queries ({count}%)")
    
    # Calculate monthly projection
    queries_per_day = 50  # Assumption
    monthly_cost = (total_cost / 100) * queries_per_day * 30
    print(f"\nProjected monthly cost at {queries_per_day} queries/day: ${monthly_cost:.2f}")


async def main():
    """Run all tests"""
    test_query_complexity_analyzer()
    test_memory_confidence_scorer()
    await test_full_flow()
    await test_state_machine_flow()
    await test_cost_optimization()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())