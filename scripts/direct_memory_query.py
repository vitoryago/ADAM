#!/usr/bin/env python3
"""
Query ChromaDB directly to find the DAG memory
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from rich.console import Console
import chromadb

console = Console()

def direct_query():
    """Query ChromaDB directly"""
    
    console.print("\n[yellow]Direct ChromaDB Query[/yellow]")
    console.print("=" * 60)
    
    # Initialize memory to get the collection
    memory = ADAMMemoryAdvanced()
    
    # Get all data from collection
    try:
        all_data = memory.collection.get()
        
        console.print(f"Total memories in collection: {len(all_data['ids'])}")
        
        # Find the specific memory
        found = False
        for i, mem_id in enumerate(all_data['ids']):
            if 'cba5f19' in mem_id:
                found = True
                console.print(f"\n[green]Found memory at index {i}![/green]")
                console.print(f"ID: {mem_id}")
                
                # Get metadata and document
                metadata = all_data['metadatas'][i]
                document = all_data['documents'][i]
                
                console.print(f"\nMetadata:")
                for key, value in metadata.items():
                    console.print(f"  {key}: {value}")
                
                console.print(f"\nDocument length: {len(document)} characters")
                
                # Check content
                if 'MARKETING_ANALYTICS' in document:
                    console.print("[green]✓ Contains MARKETING_ANALYTICS[/green]")
                if 'new_fee_repricing_user' in document:
                    console.print("[green]✓ Contains new_fee_repricing_user[/green]")
                if 'from airflow_constants.pods import MARKETING_ANALYTICS' in document:
                    console.print("[green]✓ Contains the exact import statement![/green]")
                
                # Show a snippet of the response
                if 'Response:' in document:
                    resp_start = document.find('Response:')
                    resp_text = document[resp_start+9:resp_start+500]
                    console.print(f"\n[cyan]Response preview:[/cyan]\n{resp_text}...")
                
                break
        
        if not found:
            console.print("\n[red]Memory cba5f19* not found in collection![/red]")
            
            # Check if there are any recent memories
            console.print("\n[yellow]Checking for recent memories (last 10):[/yellow]")
            for i in range(max(0, len(all_data['ids'])-10), len(all_data['ids'])):
                if i < len(all_data['ids']):
                    mem_id = all_data['ids'][i]
                    timestamp = all_data['metadatas'][i].get('timestamp', 'unknown')
                    console.print(f"  {mem_id}: {timestamp}")
                    
    except Exception as e:
        console.print(f"[red]Error querying ChromaDB: {e}[/red]")

if __name__ == "__main__":
    direct_query()