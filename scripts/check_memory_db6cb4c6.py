#!/usr/bin/env python3
"""
Check the specific memory db6cb4c6 that appears in searches
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from rich.console import Console

console = Console()

def check_memory():
    """Check memory db6cb4c6"""
    
    memory = ADAMMemoryAdvanced()
    
    # Get all memories
    all_data = memory.collection.get()
    
    found = False
    for i, mem_id in enumerate(all_data['ids']):
        if 'db6cb4c6' in mem_id:
            found = True
            console.print(f"\n[green]Found memory: {mem_id}[/green]")
            
            metadata = all_data['metadatas'][i]
            document = all_data['documents'][i]
            
            console.print(f"\nTimestamp: {metadata.get('timestamp', 'unknown')}")
            console.print(f"Strength: {metadata.get('strength', 0)}")
            console.print(f"Memory type: {metadata.get('memory_type', 'unknown')}")
            
            # Check if this is today's DAG
            if '2025-07-23' in metadata.get('timestamp', ''):
                console.print("[bold green]✓ This is from TODAY![/bold green]")
            
            # Check content
            if 'MARKETING_ANALYTICS' in document and 'new_fee_repricing_user' in document:
                console.print("[green]✓ Contains MARKETING_ANALYTICS and new_fee_repricing_user[/green]")
                
                # Show the actual DAG code
                if 'from airflow_constants.pods import MARKETING_ANALYTICS' in document:
                    console.print("\n[yellow]This IS the correct DAG![/yellow]")
                    
                    # Extract response
                    if 'Response:' in document:
                        resp_start = document.find('Response:')
                        response = document[resp_start+9:]
                        
                        # Find the code block
                        if '```python' in response:
                            code_start = response.find('```python')
                            code_end = response.find('```', code_start + 9)
                            if code_start != -1 and code_end != -1:
                                console.print("\n[cyan]DAG Code:[/cyan]")
                                console.print(response[code_start:code_end+3][:500] + "...")
            
            break
    
    if not found:
        console.print("[red]Memory db6cb4c6 not found[/red]")

if __name__ == "__main__":
    check_memory()