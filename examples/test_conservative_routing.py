#!/usr/bin/env python3
"""
Test Conservative Routing
========================

Tests that our routing is now more conservative and prefers grok-3-mini-high
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.llm.query_analyzer import QueryAnalyzer
from rich.console import Console
from rich.table import Table

console = Console()


async def test_conservative_routing():
    """Test that routing now favors cheaper models"""
    
    console.print("\n🧪 Testing Conservative Routing\n", style="bold blue")
    
    # Test queries from your conversation
    test_cases = [
        {
            "query": "What's the difference between database, data warehouse and data schema?",
            "expected": "LOW",  # Should be LOW now
            "description": "Simple explanation"
        },
        {
            "query": "explain this image",
            "expected": "LOW",
            "description": "Image explanation"
        },
        {
            "query": "how can I do that?",
            "expected": "LOW",
            "description": "Simple follow-up"
        },
        {
            "query": "can you give an example with one of those models from the image?",
            "expected": "LOW",
            "description": "Simple example request"
        },
        {
            "query": "what's the difference between semantic layer and transformation layer?",
            "expected": "LOW",
            "description": "Concept comparison"
        },
        {
            "query": "Write a Python async function that processes a queue with rate limiting and error handling",
            "expected": "HIGH",  # Only this should be HIGH
            "description": "Complex code generation"
        }
    ]
    
    analyzer = QueryAnalyzer()
    
    # Create results table
    table = Table(title="Conservative Routing Results")
    table.add_column("Query", style="cyan", width=50)
    table.add_column("Expected", style="yellow")
    table.add_column("Actual", style="green")
    table.add_column("Score", style="magenta")
    table.add_column("Model", style="blue")
    
    available_models = ['grok-4-reasoning', 'grok-4', 'grok-3-mini-high', 'grok-2-vision-1212']
    
    for test in test_cases:
        complexity, analysis = analyzer.analyze_query(test["query"])
        recommended_model = analyzer.recommend_model(complexity, available_models)
        
        actual = complexity.value.upper()
        expected = test["expected"]
        
        table.add_row(
            test["query"][:47] + "...",
            expected,
            actual,
            f"C:{analysis['scores']['complex']} M:{analysis['scores']['medium']} L:{analysis['scores']['low']}",
            recommended_model
        )
        
        console.print(f"\n{test['description']}:")
        console.print(f"  Query: '{test['query'][:60]}...'")
        console.print(f"  Complexity: {actual} (expected: {expected})")
        console.print(f"  Model: {recommended_model}")
        console.print(f"  Scores - Complex: {analysis['scores']['complex']}, Medium: {analysis['scores']['medium']}, Low: {analysis['scores']['low']}")
    
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


if __name__ == "__main__":
    asyncio.run(test_conservative_routing())