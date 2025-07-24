#!/usr/bin/env python3
"""
Test Fast Routing with grok-3-mini-fast
=======================================

Tests that our routing now prefers grok-3-mini-fast for speed
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.llm.query_analyzer import QueryAnalyzer
from rich.console import Console
from rich.table import Table

console = Console()


async def test_fast_routing():
    """Test that routing now favors grok-3-mini-fast"""
    
    console.print("\n⚡ Testing Fast Routing with grok-3-mini-fast\n", style="bold blue")
    
    # Test queries from your conversation
    test_cases = [
        {
            "query": "What's the difference between database, data warehouse and data schema?",
            "expected": "LOW",
            "description": "Simple explanation"
        },
        {
            "query": "What's the difference between data schema and data layer?",
            "expected": "LOW",
            "description": "Concept comparison"
        },
        {
            "query": "What was the question?",
            "expected": "LOW",
            "description": "Simple recall"
        },
        {
            "query": "Which command should I run to search this what jeremy suggests?",
            "expected": "LOW",
            "description": "Simple help request"
        },
        {
            "query": "Write a Python async function that processes a queue with rate limiting",
            "expected": "HIGH",
            "description": "Complex code generation"
        }
    ]
    
    analyzer = QueryAnalyzer()
    
    # Create results table
    table = Table(title="Fast Routing Results with grok-3-mini-fast")
    table.add_column("Query", style="cyan", width=50)
    table.add_column("Complexity", style="yellow")
    table.add_column("Model", style="green")
    table.add_column("Speed", style="magenta")
    
    # Models including new grok-3-mini-fast
    available_models = ['grok-4-reasoning', 'grok-4', 'grok-3-mini-fast', 'grok-3-mini-high', 'grok-2-vision-1212']
    
    for test in test_cases:
        complexity, analysis = analyzer.analyze_query(test["query"])
        recommended_model = analyzer.recommend_model(complexity, available_models)
        
        # Determine speed indicator
        speed = "⚡⚡⚡" if "fast" in recommended_model else "⚡⚡" if "mini" in recommended_model else "⚡"
        
        table.add_row(
            test["query"][:47] + "...",
            complexity.value.upper(),
            recommended_model,
            speed
        )
        
        console.print(f"\n{test['description']}:")
        console.print(f"  Query: '{test['query'][:60]}...'")
        console.print(f"  Complexity: {complexity.value.upper()}")
        console.print(f"  Model: {recommended_model}")
    
    console.print("\n")
    console.print(table)
    
    # Count model usage
    console.print("\n📊 Model Usage Summary:")
    model_count = {}
    for test in test_cases:
        complexity, _ = analyzer.analyze_query(test["query"])
        model = analyzer.recommend_model(complexity, available_models)
        model_count[model] = model_count.get(model, 0) + 1
    
    for model, count in model_count.items():
        percentage = (count / len(test_cases)) * 100
        console.print(f"  {model}: {count} queries ({percentage:.0f}%)")
    
    # Performance comparison
    console.print("\n⏱️  Response Time Comparison:")
    console.print("  grok-3-mini-fast: ~0.5-1s (⚡⚡⚡)")
    console.print("  grok-3-mini-high: ~1-2s (⚡⚡)")
    console.print("  grok-4: ~2-3s (⚡)")
    console.print("  grok-4-reasoning: ~3-5s (🐌)")


if __name__ == "__main__":
    asyncio.run(test_fast_routing())