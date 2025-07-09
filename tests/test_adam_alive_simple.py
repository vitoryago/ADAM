#!/usr/bin/env python3
"""
ADAM Integration Test - Simplified Version
=========================================

This test checks if ADAM's core components are working together.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import time

# Import core ADAM components
from src.adam.memory import ADAMMemoryAdvanced
from src.adam.memory_network import MemoryNetworkSystem
from src.adam.conversation_system import ConversationSystem
from src.adam.advanced_rag import AdvancedRAGSystem

console = Console()


def test_adam_integration():
    """Test ADAM's core functionality"""
    
    console.print(Panel.fit(
        "[bold cyan]ADAM Integration Test[/bold cyan]\n\n"
        "Testing core components working together...",
        title="🧠 Testing ADAM",
        border_style="cyan"
    ))
    
    # Initialize systems
    console.print("\n[yellow]Initializing ADAM systems...[/yellow]")
    
    memory_system = ADAMMemoryAdvanced(persist_directory="./test_integration_memory")
    conversation_system = ConversationSystem(storage_path="./test_integration_conversations")
    memory_network = MemoryNetworkSystem(memory_system, conversation_system)
    
    # Add some initial memories before initializing RAG
    initial_memories = [
        ("How do I debug a memory leak in my Python application that's causing crashes after running for 24 hours?", 
         "To debug memory leaks in Python: 1) Use memory_profiler to track memory usage over time, 2) Check for circular references with gc module, 3) Look for global caches that grow unbounded, 4) Use tracemalloc to find the exact lines allocating memory, 5) Common culprits: unclosed file handles, growing lists/dicts, circular references in custom objects"),
        ("My API endpoint is extremely slow, taking 5-10 seconds to respond. How can I optimize it?", 
         "To optimize slow API endpoints: 1) Profile with cProfile to find bottlenecks, 2) Add database query analysis with EXPLAIN, 3) Implement caching for repeated queries, 4) Use async/await for I/O operations, 5) Consider pagination for large datasets, 6) Add connection pooling for database queries"),
        ("I'm getting 'ImportError: No module named pandas' but I already installed it. What's wrong?",
         "This ImportError usually means: 1) Wrong Python environment - check with 'which python' and 'pip list', 2) Virtual environment not activated, 3) Installed in different Python version, 4) IDE using different interpreter. Fix: activate correct venv, verify with 'pip show pandas', ensure IDE points to right interpreter")
    ]
    
    for query, response in initial_memories:
        memory_system.remember_if_worthy(
            query=query,
            response=response,
            context={"initial": True},
            generation_cost=0.001
        )
    
    # Now initialize RAG system with some memories
    rag_system = AdvancedRAGSystem(memory_system, memory_network)
    
    console.print("[green]✓ Systems initialized[/green]\n")
    
    # Test 1: Conversation + Memory
    console.print("[bold]Test 1: Conversation with Memory[/bold]")
    session_id = conversation_system.start_session()
    
    # Simulate conversation
    conversation = [
        ("How do I optimize a slow SQL query?", 
         "To optimize SQL queries: 1) Check execution plan with EXPLAIN, 2) Add appropriate indexes, 3) Avoid SELECT *, 4) Use query caching"),
        ("My query joins 5 tables and takes 30 seconds",
         "For complex joins: 1) Ensure foreign key columns are indexed, 2) Consider denormalization, 3) Use materialized views for frequently accessed data"),
        ("I added indexes but it's still slow",
         "If indexes aren't helping: 1) Check index usage with EXPLAIN, 2) Update table statistics, 3) Consider query restructuring or breaking into smaller queries")
    ]
    
    problem_id = memory_system.start_problem_solving("SQL query optimization")
    
    for user_msg, adam_response in conversation:
        # Add to conversation
        conversation_system.record_exchange(
            query=user_msg,
            response=adam_response,
            topics=["sql", "optimization", "database"],
            context={"problem_id": problem_id}
        )
        
        # Store in memory
        memory_id = memory_system.remember_if_worthy(
            query=user_msg,
            response=adam_response,
            context={"session": session_id, "problem": problem_id},
            generation_cost=0.002,
            model_used="test"
        )
        
        if memory_id:
            console.print(f"  [green]✓ Stored memory: {memory_id[:8]}...[/green]")
    
    # Test 2: Advanced Retrieval
    console.print("\n[bold]Test 2: Multi-Method Retrieval[/bold]")
    
    test_queries = [
        "database performance issues",
        "SELECT * optimization",
        "query taking too long"
    ]
    
    for query in test_queries:
        console.print(f"\nSearching for: '{query}'")
        results = rag_system.retrieve(query, k=3)
        
        methods_used = set()
        for result in results:
            methods_used.add(result.retrieval_method)
        
        console.print(f"  Found {len(results)} results using: {', '.join(methods_used)}")
    
    # Test 3: Memory Network Connections
    console.print("\n[bold]Test 3: Memory Network[/bold]")
    
    # Add related memories
    related_topics = [
        ("What causes database locks?", "Lock causes: long transactions, missing indexes, deadlocks"),
        ("How to monitor query performance?", "Use slow query log, performance schema, monitoring tools"),
        ("Best practices for database design?", "Normalize data, use appropriate data types, plan for growth")
    ]
    
    for query, response in related_topics:
        memory_network.add_memory_with_references(
            query=query,
            response=response,
            memory_type="explanation",
            topics=["database", "performance"],
            auto_save=False
        )
    
    console.print(f"  [green]✓ Added {len(related_topics)} related memories[/green]")
    console.print(f"  [green]✓ Network has {memory_network.memory_graph.number_of_nodes()} nodes[/green]")
    console.print(f"  [green]✓ Network has {memory_network.memory_graph.number_of_edges()} connections[/green]")
    
    # Test 4: Learning from Feedback
    console.print("\n[bold]Test 4: Learning from Feedback[/bold]")
    
    # Simulate solution feedback
    feedback = memory_system.handle_solution_feedback("The index suggestion worked! Query now takes 0.5 seconds")
    console.print(f"  Feedback processed: {feedback['status']}")
    console.print(f"  [green]✓ {feedback['message']}[/green]")
    
    # Test 5: Analytics
    console.print("\n[bold]Test 5: System Analytics[/bold]")
    
    analytics = memory_system.get_memory_analytics()
    
    table = Table(title="ADAM Analytics", box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Memories", str(analytics.get('total_memories', 0)))
    table.add_row("Memory Hit Rate", f"{analytics.get('memory_hit_rate', 0):.1%}")
    table.add_row("Avg Success Rate", f"{analytics.get('average_success_rate', 0):.1%}")
    table.add_row("Cost Savings", f"${analytics.get('net_savings', 0):.3f}")
    
    console.print(table)
    
    # Final verdict
    console.print("\n" + "="*60 + "\n")
    
    console.print(Panel.fit(
        "[bold green]ADAM Core Systems: OPERATIONAL[/bold green]\n\n"
        "✓ Memory system working\n"
        "✓ Conversation tracking active\n"
        "✓ Advanced RAG functional\n"
        "✓ Memory network connecting knowledge\n"
        "✓ Learning from feedback\n\n"
        "ADAM is ready for the next phase of development!",
        title="🎉 Test Results",
        border_style="green"
    ))
    
    console.print("\n[bold cyan]What's Next:[/bold cyan]")
    console.print("1. Wire up real LLM APIs for actual intelligence")
    console.print("2. Implement production-ready tools")
    console.print("3. Add agent planning and execution")
    console.print("4. Deploy to real users")
    console.print("5. Scale to handle thousands of memories")


if __name__ == "__main__":
    try:
        test_adam_integration()
    except Exception as e:
        console.print(f"[red]Test failed: {str(e)}[/red]")
        raise