#!/usr/bin/env python3
"""
Extract the full response from the DAG memory
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from rich.console import Console

console = Console()

def extract_response():
    """Extract the full response from memory cba5f19ac7bc"""
    
    # Initialize memory
    memory = ADAMMemoryAdvanced()
    
    # Get all data
    all_data = memory.collection.get()
    
    # Find the memory
    for i, mem_id in enumerate(all_data['ids']):
        if mem_id == 'cba5f19ac7bc':
            document = all_data['documents'][i]
            
            # Extract response
            if 'Response:' in document:
                resp_start = document.find('Response:')
                response = document[resp_start+9:]
                
                # Check for MARKETING_ANALYTICS
                if 'MARKETING_ANALYTICS' in response:
                    console.print("[green]✓ Response contains MARKETING_ANALYTICS![/green]")
                    
                    # Find the DAG code
                    if '```python' in response:
                        code_start = response.find('```python')
                        code_end = response.find('```', code_start + 9)
                        if code_start != -1 and code_end != -1:
                            dag_code = response[code_start+9:code_end]
                            
                            # Check if this is the correct DAG
                            if 'from airflow_constants.pods import MARKETING_ANALYTICS' in dag_code:
                                console.print("\n[bold green]✓ FOUND THE CORRECT DAG![/bold green]")
                                console.print("\n[yellow]Full DAG code:[/yellow]")
                                console.print("```python")
                                console.print(dag_code)
                                console.print("```")
                            else:
                                console.print("\n[red]✗ DAG code doesn't have MARKETING_ANALYTICS import[/red]")
                                console.print("First 500 chars of DAG code:")
                                console.print(dag_code[:500])
                else:
                    console.print("[red]✗ Response does NOT contain MARKETING_ANALYTICS[/red]")
                    console.print("\nFirst 1000 chars of response:")
                    console.print(response[:1000])
            
            break

if __name__ == "__main__":
    extract_response()