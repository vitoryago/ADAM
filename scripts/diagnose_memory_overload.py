#!/usr/bin/env python3
"""
Diagnose if too many memories are being retrieved and drowning out relevant ones
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from src.adam.memory_search_enhanced import MemorySearchEnhancer
from rich.console import Console
from rich.table import Table

console = Console()

def diagnose_memory_retrieval():
    """Check what memories are being retrieved for the user's queries"""
    
    console.print("\n[bold yellow]Memory Retrieval Diagnosis[/bold yellow]")
    console.print("=" * 80)
    
    memory = ADAMMemoryAdvanced()
    enhancer = MemorySearchEnhancer()
    
    # Test the actual queries from the conversation
    test_queries = [
        "Recently we were talking about some code",
        "something related to orchestration", 
        "Can you bring the DAG we have created?",
        "private and new user fee",
        "new_fee_repricing_user"
    ]
    
    target_content = "new_fee_repricing_user"
    target_memory_id = "db6cb4c641b9"
    
    for query in test_queries:
        console.print(f"\n[cyan]Query: '{query}'[/cyan]")
        
        # Get raw memories
        raw_memories = memory.recall_with_context(query=query, n_results=20)
        
        # Find position of target memory
        target_position = None
        target_found = False
        
        for i, mem in enumerate(raw_memories):
            memory_id = mem.get('memory_id', '')
            content = mem.get('content', '')
            
            if target_memory_id in memory_id or target_content in content:
                target_position = i + 1
                target_found = True
                break
        
        if target_found:
            console.print(f"[green]✓ Target memory found at position {target_position}[/green]")
        else:
            console.print(f"[red]✗ Target memory NOT found in top 20[/red]")
        
        # Show what's being retrieved instead
        console.print("\nTop 5 memories retrieved:")
        table = Table()
        table.add_column("Pos", style="cyan")
        table.add_column("Date", style="green") 
        table.add_column("Strength", style="yellow")
        table.add_column("Content Preview", style="white", width=50)
        
        for i, mem in enumerate(raw_memories[:5]):
            metadata = mem.get('metadata', {})
            timestamp = metadata.get('timestamp', 'unknown')[:10]
            strength = metadata.get('strength', 0)
            content = mem.get('content', '')
            
            # Extract meaningful preview
            if 'Response:' in content:
                preview = content.split('Response:')[1][:50] + "..."
            else:
                preview = content[:50] + "..."
            
            table.add_row(
                str(i+1),
                timestamp,
                f"{strength:.2f}",
                preview
            )
        
        console.print(table)
        
        # Apply enhancement to see if it helps
        if raw_memories:
            enhanced_memories, context = enhancer.enhance_memory_search(
                query=query,
                conversation_history=[],
                raw_memories=raw_memories
            )
            
            # Check if enhancement helps
            for i, mem in enumerate(enhanced_memories[:5]):
                content = mem.get('content', '')
                if target_content in content:
                    console.print(f"\n[green]After enhancement: Target at position {i+1}[/green]")
                    break
    
    # Diagnosis
    console.print("\n" + "=" * 80)
    console.print("[yellow]DIAGNOSIS:[/yellow]")
    console.print("1. The target memory exists but is being outranked by older, stronger memories")
    console.print("2. Generic queries match too many unrelated memories")
    console.print("3. The user needs more specific queries or better memory filtering")

if __name__ == "__main__":
    diagnose_memory_retrieval()