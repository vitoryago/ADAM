#!/usr/bin/env python3
"""
Non-Interactive Test RAG Comparison
===================================

This script runs the RAG comparison demo without requiring user input,
making it suitable for automated testing or demonstration.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
import matplotlib.pyplot as plt
import seaborn as sns

# Import our systems
from src.adam.memory import ADAMMemoryAdvanced as MemorySystem
from src.adam.memory_network import MemoryNetworkSystem
from src.adam.conversation_aware_memory import ConversationAwareMemorySystem
from src.adam.advanced_rag import AdvancedRAGSystem, demonstrate_retrieval_differences

console = Console()

def create_test_memories():
    """Create a diverse set of test memories"""
    return [
        # Python error handling - exact error messages
        {
            "query": "TypeError: 'NoneType' object is not subscriptable",
            "response": "This error occurs when you try to access an index on None. Check if your variable is None before accessing: if my_var is not None: my_var[0]",
            "topics": ["python", "error", "debugging"],
            "memory_type": "error_solution"
        },
        {
            "query": "How to handle null pointer exceptions in Python?",
            "response": "Python doesn't have null pointers, but has None. Use: if variable is not None: # safe to use variable",
            "topics": ["python", "error", "null"],
            "memory_type": "explanation"
        },
        
        # Performance optimization - semantic variations
        {
            "query": "My Python code is running slowly",
            "response": "Profile your code with cProfile, use list comprehensions instead of loops, consider NumPy for numerical operations",
            "topics": ["python", "performance", "optimization"],
            "memory_type": "optimization"
        },
        {
            "query": "How to make Python faster?",
            "response": "Use built-in functions, avoid global variables, consider PyPy or Cython for CPU-intensive tasks",
            "topics": ["python", "performance", "speed"],
            "memory_type": "optimization"
        },
    ]

def populate_test_system():
    """Create and populate a test system with our carefully crafted memories"""
    # Initialize systems
    memory_system = MemorySystem(persist_directory="./test_adam_memory")
    
    # Create a minimal conversation system for testing
    from src.adam.conversation_system import ConversationSystem
    conversation_system = ConversationSystem(storage_path="./test_adam_conversations")
    
    memory_network = MemoryNetworkSystem(memory_system, conversation_system)
    
    # Clear existing data
    try:
        memory_system.collection.delete(filter={})
    except:
        pass  # Collection might not exist yet
    memory_network.memory_graph.clear()
    
    # Add test memories
    test_memories = create_test_memories()
    memory_ids = []
    
    console.print("[yellow]Populating test system with memories...[/yellow]")
    
    for i, memory_data in enumerate(test_memories):
        # Force storage by using the internal _store_memory method
        from src.adam.memory import Memory, MemoryType
        
        memory = Memory(
            id=f"test_{i}",
            content=memory_data["response"],
            memory_type=MemoryType.ERROR_SOLUTION if "error" in memory_data["memory_type"] else MemoryType.CONCEPT_EXPLANATION,
            query=memory_data["query"],
            response=memory_data["response"],
            context={"topics": memory_data["topics"]},
            timestamp=datetime.now(),
            confidence_score=0.9,
            model_used="test_model",
            generation_cost=0.001
        )
        
        # Store directly
        memory_system._store_memory(memory)
        memory_ids.append(memory.id)
        
        # Add to memory network
        network_id = memory_network.add_memory_with_references(
            query=memory_data["query"],
            response=memory_data["response"],
            memory_type=memory_data["memory_type"],
            topics=memory_data["topics"],
            auto_save=False  # Don't save during test
        )
    
    # Connections are created automatically by add_memory_with_references
    
    console.print(f"[green]Added {len(memory_ids)} memories with connections[/green]")
    
    return memory_system, memory_network

def run_simple_test():
    """Run a simple test of the RAG system"""
    console.print(Panel.fit(
        "[bold cyan]Simple RAG System Test[/bold cyan]\n\n"
        "Testing the three retrieval methods",
        border_style="cyan"
    ))
    
    # Initialize test system
    memory_system, memory_network = populate_test_system()
    rag_system = AdvancedRAGSystem(memory_system, memory_network)
    
    # Test queries
    test_queries = [
        "TypeError NoneType subscriptable",
        "make code faster"
    ]
    
    for query in test_queries:
        console.print(f"\n[bold]Testing query:[/bold] '{query}'")
        
        try:
            # Test individual methods
            console.print("\n[yellow]BM25 Results:[/yellow]")
            bm25_results = rag_system._bm25_retrieve(query, k=3)
            for i, result in enumerate(bm25_results[:2], 1):
                console.print(f"{i}. Score: {result.score:.3f} - {result.metadata.get('query', '')[:50]}...")
            
            console.print("\n[green]Vector Search Results:[/green]")
            vector_results = rag_system._vector_retrieve(query, k=3)
            for i, result in enumerate(vector_results[:2], 1):
                console.print(f"{i}. Score: {result.score:.3f} - {result.metadata.get('query', '')[:50]}...")
            
            console.print("\n[cyan]Combined Results:[/cyan]")
            combined_results = rag_system.retrieve(query, k=3)
            for i, result in enumerate(combined_results[:2], 1):
                console.print(f"{i}. RRF Score: {result.score:.3f} - {result.metadata.get('query', '')[:50]}...")
                
        except Exception as e:
            console.print(f"[red]Error during retrieval: {e}[/red]")
            import traceback
            traceback.print_exc()
    
    console.print("\n[green]Test completed![/green]")
    console.print("\nKey observations:")
    console.print("• BM25 excels at exact keyword matches")
    console.print("• Vector search finds semantically similar content")
    console.print("• Combined approach leverages both strengths")

if __name__ == "__main__":
    # Show theoretical explanation
    demonstrate_retrieval_differences()
    
    console.print("\n" + "="*60)
    console.print("Running Simple RAG Test")
    console.print("="*60 + "\n")
    
    # Run the simple test
    run_simple_test()
    
    console.print("\n[bold green]Demo Complete![/bold green]")
    console.print("\nTo run the full interactive demo with visualizations, use:")
    console.print("python examples/test_rag_comparison.py")