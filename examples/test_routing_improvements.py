#!/usr/bin/env python3
"""
Test Improved Routing Logic
===========================

Tests that our routing improvements correctly assign simpler models
for queries that don't need reasoning.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.llm.client import UnifiedLLMClient
from src.adam.llm.query_analyzer import QueryAnalyzer
from rich.console import Console
from rich.table import Table

console = Console()


async def test_routing_improvements():
    """Test that routing is more balanced now"""
    
    console.print("\n🧪 Testing Improved Routing Logic\n", style="bold blue")
    
    # Test queries that were over-classified before
    test_cases = [
        {
            "query": "What's the difference between database, data warehouse, and data schema?",
            "expected": "MEDIUM",  # Should be medium, not high
            "description": "Explanation question"
        },
        {
            "query": "How can we do that?", 
            "expected": "LOW",     # Simple follow-up
            "description": "Simple how-to"
        },
        {
            "query": "Explain neural networks",
            "expected": "MEDIUM",  # Not high complexity
            "description": "Technical explanation"
        },
        {
            "query": "Write a Python async function that processes a queue with rate limiting and error handling",
            "expected": "HIGH",    # This should still be high
            "description": "Complex code generation"
        },
        {
            "query": "What is machine learning?",
            "expected": "MEDIUM",  # Basic explanation
            "description": "Concept explanation"
        },
        {
            "query": "Compare SQL and NoSQL databases",
            "expected": "MEDIUM",  # Comparison, not high complexity
            "description": "Technology comparison"
        }
    ]
    
    analyzer = QueryAnalyzer()
    client = UnifiedLLMClient()
    
    # Create results table
    table = Table(title="Routing Logic Test Results")
    table.add_column("Query", style="cyan", width=40)
    table.add_column("Expected", style="yellow")
    table.add_column("Actual", style="green")
    table.add_column("Model", style="magenta")
    table.add_column("Status", style="white")
    
    for test in test_cases:
        complexity, analysis = analyzer.analyze_query(test["query"])
        
        # Get recommended model
        available_models = client.config.get_available_models()
        recommended_model = analyzer.recommend_model(complexity, available_models)
        
        # Check if complexity matches expectation
        actual = complexity.value.upper()
        expected = test["expected"]
        status = "✅" if actual == expected else "❌"
        
        table.add_row(
            test["query"][:37] + "...",
            expected,
            actual,
            recommended_model,
            status
        )
        
        console.print(f"\n{test['description']}:")
        console.print(f"  Query: '{test['query'][:60]}...'")
        console.print(f"  Complexity: {actual} (expected: {expected})")
        console.print(f"  Model: {recommended_model}")
        console.print(f"  Confidence: {analysis['confidence']:.1%}")
    
    console.print(table)
    
    # Test actual routing with automatic model
    console.print("\n\n🤖 Testing Automatic Model Selection\n")
    
    for i, test in enumerate(test_cases[:3]):  # Test first 3
        console.print(f"\nTest {i+1}: {test['description']}")
        
        try:
            response = await client.complete(
                prompt=test["query"],
                model="automatic",
                max_tokens=50
            )
            
            if hasattr(response, 'raw_response') and response.raw_response:
                routing = response.raw_response.get('routing_decision')
                if routing:
                    console.print(f"  Auto-selected: {routing['selected_model']}")
                    console.print(f"  Complexity: {routing['complexity']}")
                    
                    # Check if it's using expensive models unnecessarily
                    if routing['selected_model'] == 'grok-4-reasoning' and test['expected'] != 'HIGH':
                        console.print("  ⚠️ Using expensive model for non-complex query", style="yellow")
                    else:
                        console.print("  ✅ Appropriate model selection", style="green")
        except Exception as e:
            console.print(f"  Error: {str(e)}", style="red")


if __name__ == "__main__":
    asyncio.run(test_routing_improvements())