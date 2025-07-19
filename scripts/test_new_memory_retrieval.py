#!/usr/bin/env python3
"""
Test if the newly saved DAG memory can be retrieved
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from src.adam.memory_search_enhanced import MemorySearchEnhancer
from rich.console import Console

console = Console()

def test_new_memory():
    """Test retrieval of the newly saved memory"""
    
    # Initialize components
    memory = ADAMMemoryAdvanced()
    enhancer = MemorySearchEnhancer()
    
    # Search specifically for the new memory
    console.print("\n[yellow]Testing retrieval of newly saved DAG memory[/yellow]")
    console.print("Memory ID: cba5f19ac7bc")
    console.print("=" * 60)
    
    # Try various queries
    queries = [
        "new_fee_repricing_user",
        "MARKETING_ANALYTICS DAG", 
        "last dag we created together",
        "DbtOperator with new_fee_repricing",
        "the dag code I gave you for fee repricing"
    ]
    
    for query in queries:
        console.print(f"\n[cyan]Query: '{query}'[/cyan]")
        
        # Get memories
        raw_memories = memory.recall_with_context(query=query, n_results=5)
        
        # Check if our memory is in the results
        found_new_memory = False
        for i, mem in enumerate(raw_memories):
            mem_id = mem.get('memory_id', '')
            if 'cba5f19' in mem_id:
                found_new_memory = True
                console.print(f"[green]✓ Found new memory at position {i+1}![/green]")
                console.print(f"Similarity score: {mem.get('similarity', 0):.3f}")
                
                # Check content
                content = mem.get('content', '')
                if 'MARKETING_ANALYTICS' in content:
                    console.print("[green]✓ Contains MARKETING_ANALYTICS[/green]")
                if 'new_fee_repricing_user' in content:
                    console.print("[green]✓ Contains new_fee_repricing_user[/green]")
                break
        
        if not found_new_memory:
            console.print("[red]✗ New memory not in top 5 results[/red]")
            
            # Check all memories for debugging
            all_memories = memory.recall_with_context(query=query, n_results=20)
            for i, mem in enumerate(all_memories):
                if 'cba5f19' in mem.get('memory_id', ''):
                    console.print(f"[yellow]Found at position {i+1} (outside top 5)[/yellow]")
                    break
    
    # Now test with enhanced search
    console.print("\n" + "=" * 60)
    console.print("[yellow]Testing with enhanced memory search[/yellow]")
    
    query = "bring back the last dag we created with new_fee_repricing_user"
    raw_memories = memory.recall_with_context(query=query, n_results=10)
    
    if raw_memories:
        # Create mock conversation history  
        conversation_history = [
            {"role": "user", "content": "I need the dag code for fee repricing"},
            {"role": "assistant", "content": "I'll help you with that DAG"}
        ]
        
        enhanced_memories, context = enhancer.enhance_memory_search(
            query=query,
            conversation_history=conversation_history,
            raw_memories=raw_memories
        )
        
        console.print(f"\nSearch context: intent={context.user_intent}, terms={context.technical_terms}")
        
        # Check top result
        if enhanced_memories:
            top = enhanced_memories[0]
            console.print(f"\nTop result:")
            console.print(f"Memory ID: {top.get('memory_id', 'unknown')}")
            console.print(f"Relevance score: {top.get('relevance_score', 0):.3f}")
            
            content = top.get('content', '')
            if 'cba5f19' in top.get('memory_id', ''):
                console.print("[green]✓ This is our newly saved DAG memory![/green]")
                
                # Extract the DAG code
                if 'from airflow_constants.pods import MARKETING_ANALYTICS' in content:
                    console.print("\n[green]✓ PERFECT! Found the exact DAG code with MARKETING_ANALYTICS[/green]")

if __name__ == "__main__":
    test_new_memory()