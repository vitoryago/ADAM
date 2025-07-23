#!/usr/bin/env python3
"""
Test the enhanced DAG retrieval with query improvements
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from src.adam.memory_search_enhanced import MemorySearchEnhancer, format_memory_for_prompt
from rich.console import Console

console = Console()

def test_enhanced_retrieval():
    """Test if enhanced search helps retrieve today's DAG"""
    
    console.print("\n[yellow]Testing Enhanced DAG Retrieval[/yellow]")
    console.print("=" * 60)
    
    memory = ADAMMemoryAdvanced()
    enhancer = MemorySearchEnhancer()
    
    # Test the problematic queries
    test_queries = [
        "can you bring me back any DAG we have done?",
        "we were talking about some DAG",
        "Hi ADAM, can you bring me back any DAG we have done?"
    ]
    
    for query in test_queries:
        console.print(f"\n[cyan]Testing: '{query}'[/cyan]")
        
        # Get memories with enhancement
        raw_memories = memory.recall_with_context(query=query, n_results=15)
        
        if raw_memories:
            # Apply enhancement
            enhanced_memories, context = enhancer.enhance_memory_search(
                query=query,
                conversation_history=[],
                raw_memories=raw_memories
            )
            
            # Show the enhanced query
            enhanced_query = enhancer.build_enhanced_query(context)
            console.print(f"Enhanced query: '{enhanced_query}'")
            console.print(f"Intent: {context.user_intent}")
            
            # Check top 3 results
            found_todays_dag = False
            for i, mem in enumerate(enhanced_memories[:3]):
                content = mem.get('content', '')
                metadata = mem.get('metadata', {})
                timestamp = metadata.get('timestamp', '')
                score = mem.get('relevance_score', 0)
                
                # Check if this is today's DAG
                if '2025-07-23' in timestamp and 'MARKETING_ANALYTICS' in content:
                    found_todays_dag = True
                    console.print(f"\n[green]✓ Found today's DAG at position {i+1}![/green]")
                    console.print(f"Score: {score:.3f}")
                    console.print(f"Timestamp: {timestamp}")
                    
                    # Show DAG preview
                    if 'from airflow_constants.pods import MARKETING_ANALYTICS' in content:
                        console.print("[green]✓ Contains correct MARKETING_ANALYTICS import[/green]")
                    break
            
            if not found_todays_dag:
                console.print("[red]✗ Today's DAG still not in top 3[/red]")
                
                # Check if it's anywhere in top 10
                for i, mem in enumerate(enhanced_memories[3:10], 4):
                    content = mem.get('content', '')
                    metadata = mem.get('metadata', {})
                    timestamp = metadata.get('timestamp', '')
                    
                    if '2025-07-23' in timestamp and 'MARKETING_ANALYTICS' in content:
                        console.print(f"[yellow]Found at position {i} (outside top 3)[/yellow]")
                        break
    
    # Try a specific query with manual recency hint
    console.print("\n" + "=" * 60)
    console.print("[cyan]Testing with manual recency hint:[/cyan]")
    
    specific_query = "bring me the most recent DAG we created"
    raw_memories = memory.recall_with_context(query=specific_query, n_results=5)
    
    if raw_memories:
        enhanced_memories, context = enhancer.enhance_memory_search(
            query=specific_query,
            conversation_history=[],
            raw_memories=raw_memories
        )
        
        # Check if top result is today's DAG
        if enhanced_memories:
            top = enhanced_memories[0]
            content = top.get('content', '')
            metadata = top.get('metadata', {})
            timestamp = metadata.get('timestamp', '')
            
            if '2025-07-23' in timestamp and 'MARKETING_ANALYTICS' in content:
                console.print("\n[bold green]✓ SUCCESS! Manual recency hint retrieves today's DAG![/bold green]")
                console.print("This proves the issue is with generic queries.")

if __name__ == "__main__":
    test_enhanced_retrieval()