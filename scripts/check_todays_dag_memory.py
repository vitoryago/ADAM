#!/usr/bin/env python3
"""
Check if today's DAG conversation was saved to memory
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from datetime import datetime
from rich.console import Console
from rich.table import Table

console = Console()

def check_todays_dag():
    """Check for DAG memories created today"""
    
    console.print("\n[yellow]Checking for today's DAG conversation in memory...[/yellow]")
    console.print("Today's date: 2025-07-23")
    console.print("=" * 60)
    
    # Initialize memory
    memory = ADAMMemoryAdvanced()
    
    # Search for DAG memories
    search_queries = [
        "new_fee_repricing_user MARKETING_ANALYTICS",
        "sf__dbt_marketing_private__new_fee_repricing_user",
        "rsaki new_fee_repricing",
        "cashadvance_private_table",
        "InstaCash repricing"
    ]
    
    found_todays_dag = False
    all_dag_memories = set()
    
    for query in search_queries:
        console.print(f"\n[cyan]Searching for: '{query}'[/cyan]")
        results = memory.recall_with_context(query=query, n_results=10)
        
        for result in results:
            memory_id = result.get('memory_id', 'unknown')
            metadata = result.get('metadata', {})
            timestamp_str = metadata.get('timestamp', '')
            content = result.get('content', '')
            
            # Check if this is from today
            if '2025-07-23' in timestamp_str:
                console.print(f"[green]✓ Found memory from TODAY: {memory_id}[/green]")
                console.print(f"  Timestamp: {timestamp_str}")
                
                # Check if it has the correct DAG
                if 'MARKETING_ANALYTICS' in content and 'new_fee_repricing_user' in content:
                    found_todays_dag = True
                    console.print("[bold green]✓ This is TODAY's DAG with MARKETING_ANALYTICS![/bold green]")
                    
                    # Show content preview
                    if 'from airflow_constants.pods import MARKETING_ANALYTICS' in content:
                        console.print("\n[yellow]Content preview:[/yellow]")
                        start = content.find('from pathlib')
                        if start != -1:
                            console.print(content[start:start+500] + "...")
            
            all_dag_memories.add((memory_id, timestamp_str))
    
    if not found_todays_dag:
        console.print("\n[red]✗ Today's DAG conversation was NOT found in memory![/red]")
        console.print("This explains why ADAM can't retrieve it.")
        
        # Show what DAG memories exist
        console.print("\n[yellow]All DAG-related memories found:[/yellow]")
        table = Table(title="DAG Memories")
        table.add_column("Memory ID", style="cyan")
        table.add_column("Timestamp", style="green")
        
        for mem_id, timestamp in sorted(all_dag_memories, key=lambda x: x[1], reverse=True)[:10]:
            table.add_row(mem_id[:12], timestamp)
        
        console.print(table)
    
    return found_todays_dag

if __name__ == "__main__":
    check_todays_dag()