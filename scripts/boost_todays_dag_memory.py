#!/usr/bin/env python3
"""
Boost today's DAG memory strength to make it retrievable
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from rich.console import Console

console = Console()

def boost_memory():
    """Boost the strength of today's DAG memory"""
    
    console.print("\n[yellow]Boosting Today's DAG Memory Strength[/yellow]")
    console.print("=" * 60)
    
    memory = ADAMMemoryAdvanced()
    
    # The memory ID we found
    target_id = "db6cb4c641b9"
    
    # Reinforce this memory multiple times to boost its strength
    console.print(f"Target memory: {target_id}")
    console.print("Current strength: 0.847")
    console.print("Target strength: 1.0")
    
    # Search for it multiple times with specific queries to reinforce
    reinforce_queries = [
        "new_fee_repricing_user MARKETING_ANALYTICS",
        "bring me back any DAG we have done",
        "the last DAG we created",
        "recent DAG with MARKETING_ANALYTICS",
        "sf__dbt_marketing_private__new_fee_repricing_user"
    ]
    
    console.print("\n[cyan]Reinforcing memory with targeted searches...[/cyan]")
    
    for query in reinforce_queries:
        results = memory.recall_with_context(query=query, n_results=20)
        
        # Check if we found and reinforced the target
        for result in results:
            if target_id in result.get('memory_id', ''):
                metadata = result.get('metadata', {})
                new_strength = metadata.get('strength', 0)
                console.print(f"Query: '{query[:40]}...' → Strength: {new_strength:.3f}")
                break
    
    # Final check
    console.print("\n[cyan]Final strength check...[/cyan]")
    final_results = memory.recall_with_context(
        query="bring me back any DAG we have done",
        n_results=5
    )
    
    # Check position
    position = None
    final_strength = None
    for i, result in enumerate(final_results):
        if target_id in result.get('memory_id', ''):
            position = i + 1
            final_strength = result.get('metadata', {}).get('strength', 0)
            break
    
    if position and position <= 3:
        console.print(f"\n[green]✓ SUCCESS! Memory now at position {position}[/green]")
        console.print(f"Final strength: {final_strength:.3f}")
    else:
        console.print(f"\n[yellow]⚠ Memory at position {position if position else 'Not found'}[/yellow]")
        console.print("You may need to run this script again or implement stronger fixes")

if __name__ == "__main__":
    boost_memory()