#!/usr/bin/env python3
"""
Test Automatic Model Routing
============================

This script tests the new "automatic" model selection feature,
ensuring it routes queries to the most appropriate models.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.llm.client import UnifiedLLMClient
from src.adam.llm.config import LLMConfig
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


async def test_automatic_routing():
    """Test automatic model routing with various query types"""
    
    console.print("\n🤖 Testing Automatic Model Routing\n", style="bold blue")
    
    # Initialize client
    config = LLMConfig()
    client = UnifiedLLMClient(config)
    
    # Test queries of different complexities
    test_queries = [
        {
            "query": "What time is it?",
            "expected_complexity": "LOW",
            "description": "Simple question"
        },
        {
            "query": "List my recent memories",
            "expected_complexity": "LOW", 
            "description": "Memory retrieval"
        },
        {
            "query": "Explain the differences between BM25 and vector search",
            "expected_complexity": "MEDIUM",
            "description": "Technical explanation"
        },
        {
            "query": "Implement a Python function that uses async/await to process a queue of tasks with rate limiting and error handling",
            "expected_complexity": "HIGH",
            "description": "Complex code generation"
        },
        {
            "query": "Analyze this complex distributed system architecture and suggest optimizations for handling 1M+ concurrent users",
            "expected_complexity": "HIGH", 
            "description": "System design analysis"
        },
        {
            "query": "How are you today?",
            "expected_complexity": "LOW",
            "description": "Casual greeting"
        }
    ]
    
    # Create results table
    table = Table(title="Automatic Routing Test Results")
    table.add_column("Query", style="cyan", width=40)
    table.add_column("Expected", style="yellow")
    table.add_column("Actual Model", style="green")
    table.add_column("Complexity", style="magenta")
    table.add_column("Confidence", style="blue")
    table.add_column("Status", style="white")
    
    results = []
    
    for test_case in test_queries:
        console.print(f"\n🧪 Testing: {test_case['description']}")
        
        try:
            # Test with automatic model
            response = await client.complete(
                prompt=test_case["query"],
                model="automatic",
                max_tokens=100
            )
            
            # Extract routing decision
            routing_info = response.raw_response.get('routing_decision') if response.raw_response else None
            
            if routing_info:
                actual_model = routing_info['selected_model']
                complexity = routing_info['complexity']
                confidence = routing_info['confidence']
                
                # Check if routing matches expectation
                status = "✅ Correct" if complexity == test_case['expected_complexity'] else "⚠️ Different"
                
                table.add_row(
                    test_case['query'][:35] + "..." if len(test_case['query']) > 35 else test_case['query'],
                    test_case['expected_complexity'],
                    actual_model,
                    complexity,
                    f"{confidence:.1%}",
                    status
                )
                
                results.append({
                    'test_case': test_case,
                    'routing_info': routing_info,
                    'response': response,
                    'correct': complexity == test_case['expected_complexity']
                })
                
                console.print(f"  → Selected: {actual_model} (complexity: {complexity}, confidence: {confidence:.1%})")
                
            else:
                table.add_row(
                    test_case['query'][:35] + "...",
                    test_case['expected_complexity'],
                    "ERROR",
                    "N/A",
                    "N/A",
                    "❌ No routing info"
                )
                console.print("  → ERROR: No routing information returned", style="red")
                
        except Exception as e:
            console.print(f"  → ERROR: {str(e)}", style="red")
            table.add_row(
                test_case['query'][:35] + "...",
                test_case['expected_complexity'],
                "ERROR",
                "N/A", 
                "N/A",
                f"❌ {str(e)[:20]}..."
            )
    
    # Display results
    console.print(table)
    
    # Calculate success rate
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        success_rate = correct_count / len(results)
        
        console.print(f"\n📊 Success Rate: {success_rate:.1%} ({correct_count}/{len(results)})")
        
        # Show detailed routing decisions
        console.print("\n🔍 Detailed Routing Decisions:")
        for result in results:
            routing = result['routing_info']
            correct = "✅" if result['correct'] else "⚠️"
            
            console.print(f"\n{correct} {result['test_case']['description']}")
            console.print(f"  Model: {routing['selected_model']}")
            console.print(f"  Complexity: {routing['complexity']} (expected: {result['test_case']['expected_complexity']})")
            console.print(f"  Confidence: {routing['confidence']:.1%}")
            if routing.get('indicators'):
                console.print(f"  Indicators: {', '.join(routing['indicators'])}")
            if routing.get('reasoning'):
                console.print(f"  Reasoning: {', '.join(routing['reasoning'])}")
    
    return results


async def test_cost_optimization():
    """Test that automatic routing actually saves costs"""
    
    console.print("\n💰 Testing Cost Optimization\n", style="bold green")
    
    client = UnifiedLLMClient()
    
    # Simple queries that should use cheap models
    simple_queries = [
        "Hello",
        "What's 2+2?", 
        "List files",
        "Clear screen"
    ]
    
    total_auto_cost = 0
    total_expensive_cost = 0
    
    for query in simple_queries:
        # Test with automatic
        auto_response = await client.complete(query, model="automatic", max_tokens=50)
        auto_cost = auto_response.cost
        
        # Test with expensive model  
        expensive_response = await client.complete(query, model="grok-4-reasoning", max_tokens=50)
        expensive_cost = expensive_response.cost
        
        total_auto_cost += auto_cost
        total_expensive_cost += expensive_cost
        
        console.print(f"Query: '{query}'")
        console.print(f"  Auto: ${auto_cost:.5f} | Expensive: ${expensive_cost:.5f}")
    
    savings = (total_expensive_cost - total_auto_cost) / total_expensive_cost
    console.print(f"\n💡 Total Savings: {savings:.1%}")
    console.print(f"   Auto total: ${total_auto_cost:.5f}")
    console.print(f"   Expensive total: ${total_expensive_cost:.5f}")


async def test_vision_routing():
    """Test that automatic routing handles vision queries"""
    
    console.print("\n🖼️ Testing Vision Routing\n", style="bold cyan")
    
    client = UnifiedLLMClient()
    
    # Create fake image data
    fake_image = b"fake_image_data"
    
    try:
        response = await client.complete(
            prompt="What's in this image?",
            model="automatic",
            image_data=fake_image,
            max_tokens=50
        )
        
        routing_info = response.raw_response.get('routing_decision') if response.raw_response else None
        
        if routing_info:
            selected_model = routing_info['selected_model']
            console.print(f"✅ Vision query routed to: {selected_model}")
            
            # Check if selected model supports vision
            config = client.config.get_model_config(selected_model)
            if config and config.supports_vision:
                console.print("✅ Selected model supports vision")
            else:
                console.print("❌ Selected model doesn't support vision")
        else:
            console.print("❌ No routing information available")
            
    except Exception as e:
        console.print(f"⚠️ Vision routing test failed: {str(e)}")


if __name__ == "__main__":
    async def main():
        console.print(Panel.fit(
            "[bold]Automatic Model Routing Test Suite[/bold]\n\n"
            "This script validates that:\n"
            "• Simple queries route to fast/cheap models\n" 
            "• Complex queries route to powerful models\n"
            "• Vision queries route to vision-capable models\n"
            "• Cost optimization works as expected",
            border_style="blue"
        ))
        
        # Run all tests
        await test_automatic_routing()
        await test_cost_optimization() 
        await test_vision_routing()
        
        console.print("\n🎉 Automatic routing tests completed!", style="bold green")
    
    asyncio.run(main())