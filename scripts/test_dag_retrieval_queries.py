#!/usr/bin/env python3
"""
Test different queries to see which ones retrieve today's DAG
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from src.adam.memory_search_enhanced import MemorySearchEnhancer, format_memory_for_prompt
from rich.console import Console
from rich.table import Table

console = Console()

def test_retrieval_queries():
    """Test various queries to see which retrieve today's DAG"""
    
    console.print("\n[yellow]Testing DAG Retrieval with Various Queries[/yellow]")
    console.print("Target: DAG from 2025-07-23 with MARKETING_ANALYTICS")
    console.print("=" * 80)
    
    # Initialize
    memory = ADAMMemoryAdvanced()
    enhancer = MemorySearchEnhancer()
    
    # Test queries - from generic to specific
    test_queries = [
        # Generic queries (what user is asking)
        "can you bring me back any DAG we have done?",
        "we were talking about some DAG",
        "bring back the last dag we created together",
        "Hi ADAM, can you bring me back any DAG we have done?",
        
        # More specific queries
        "the DAG with MARKETING_ANALYTICS",
        "the new_fee_repricing_user DAG",
        "DAG we created today",
        "the most recent DAG we discussed",
        
        # Very specific queries
        "new_fee_repricing_user MARKETING_ANALYTICS DAG",
        "sf__dbt_marketing_private__new_fee_repricing_user"
    ]
    
    results = []
    
    for query in test_queries:
        console.print(f"\n[cyan]Query: '{query}'[/cyan]")
        
        # Get raw memories
        raw_memories = memory.recall_with_context(query=query, n_results=10)
        
        if raw_memories:
            # Enhance with context
            enhanced_memories, context = enhancer.enhance_memory_search(
                query=query,
                conversation_history=[],
                raw_memories=raw_memories
            )
            
            # Check if we found today's DAG
            found_todays_dag = False
            position = -1
            
            for i, mem in enumerate(enhanced_memories[:5]):
                content = mem.get('content', '')
                metadata = mem.get('metadata', {})
                timestamp = metadata.get('timestamp', '')
                
                # Check if this is today's DAG
                if '2025-07-23' in timestamp and 'MARKETING_ANALYTICS' in content:
                    found_todays_dag = True
                    position = i + 1
                    break
            
            if found_todays_dag:
                console.print(f"[green]✓ Found today's DAG at position {position}![/green]")
            else:
                console.print(f"[red]✗ Today's DAG not in top 5 results[/red]")
                
                # Show what was returned instead
                top_result = enhanced_memories[0] if enhanced_memories else None
                if top_result:
                    content = top_result.get('content', '')
                    if 'Response:' in content:
                        resp_start = content.find('Response:')
                        preview = content[resp_start:resp_start+200]
                        console.print(f"Top result: {preview}...")
            
            results.append({
                'query': query,
                'found': found_todays_dag,
                'position': position if found_todays_dag else 'Not found',
                'intent': context.user_intent
            })
    
    # Summary table
    console.print("\n" + "=" * 80)
    table = Table(title="Query Retrieval Results")
    table.add_column("Query", style="cyan")
    table.add_column("Found?", style="green")
    table.add_column("Position", style="yellow")
    table.add_column("Intent", style="magenta")
    
    for result in results:
        table.add_row(
            result['query'][:40] + "..." if len(result['query']) > 40 else result['query'],
            "✓" if result['found'] else "✗",
            str(result['position']),
            result['intent']
        )
    
    console.print(table)
    
    # Analysis
    console.print("\n[yellow]Analysis:[/yellow]")
    generic_found = any(r['found'] for r in results[:4])
    specific_found = any(r['found'] for r in results[4:])
    
    if not generic_found and specific_found:
        console.print("[red]✗ Generic queries FAIL to retrieve today's DAG[/red]")
        console.print("[green]✓ Specific queries SUCCEED in retrieving today's DAG[/green]")
        console.print("\nThis explains why ADAM fails when users ask generic questions!")
    elif generic_found:
        console.print("[green]✓ Some generic queries successfully retrieve today's DAG[/green]")

if __name__ == "__main__":
    test_retrieval_queries()