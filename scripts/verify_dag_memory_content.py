#!/usr/bin/env python3
"""
Verify the content of the newly saved DAG memory
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from rich.console import Console

console = Console()

def verify_dag_memory():
    """Check the exact content of memory cba5f19a"""
    
    console.print("\n[yellow]Verifying DAG Memory Content[/yellow]")
    console.print("=" * 60)
    
    # Initialize memory
    memory = ADAMMemoryAdvanced()
    
    # Search for memories containing the ID pattern
    all_memories = memory.recall_with_context("new_fee_repricing", n_results=30)
    
    found = False
    for mem in all_memories:
        mem_id = mem.get('memory_id', '')
        if 'cba5f19' in mem_id:
            found = True
            console.print(f"\n[green]Found memory: {mem_id}[/green]")
            console.print(f"Similarity score: {mem.get('similarity', 0):.3f}")
            console.print(f"Strength: {mem.get('metadata', {}).get('strength', 0):.3f}")
            
            content = mem.get('content', '')
            console.print(f"\n[cyan]Content length: {len(content)} characters[/cyan]")
            
            # Check for key components
            checks = {
                'new_fee_repricing_user': 'new_fee_repricing_user' in content,
                'MARKETING_ANALYTICS': 'MARKETING_ANALYTICS' in content,
                'DbtOperator': 'DbtOperator' in content,
                'from airflow_constants.pods': 'from airflow_constants.pods' in content,
                'rsaki (owner)': "'rsaki'" in content or '"rsaki"' in content
            }
            
            console.print("\n[yellow]Component checks:[/yellow]")
            for component, present in checks.items():
                console.print(f"  {component}: {'✓' if present else '✗'}")
            
            # Show query and response structure
            if 'Query:' in content and 'Response:' in content:
                query_start = content.find('Query:')
                query_end = content.find('Response:')
                response_start = query_end
                
                query_text = content[query_start+6:query_end].strip()[:200]
                console.print(f"\n[cyan]Query preview:[/cyan]\n{query_text}...")
                
                # Check if response contains the correct DAG
                response_text = content[response_start+9:]
                if 'from airflow_constants.pods import MARKETING_ANALYTICS' in response_text:
                    console.print("\n[green]✓ Response contains the CORRECT DAG with MARKETING_ANALYTICS![/green]")
                    
                    # Extract DAG code
                    code_start = response_text.find('```python')
                    if code_start != -1:
                        code_end = response_text.find('```', code_start + 9)
                        if code_end != -1:
                            dag_code = response_text[code_start+9:code_end]
                            console.print("\n[yellow]DAG code snippet:[/yellow]")
                            console.print(dag_code[:300] + "...")
                else:
                    console.print("\n[red]✗ Response does NOT contain MARKETING_ANALYTICS imports[/red]")
            
            break
    
    if not found:
        console.print("\n[red]Memory cba5f19a not found in search results![/red]")
        console.print("This memory might have been deleted or not indexed properly.")

if __name__ == "__main__":
    verify_dag_memory()