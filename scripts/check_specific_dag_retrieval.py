#!/usr/bin/env python3
"""
Check if we can retrieve the specific new_fee_repricing_user DAG
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from src.adam.memory_search_enhanced import MemorySearchEnhancer
from rich.console import Console

console = Console()

def check_specific_dag_retrieval():
    """Check if we can retrieve the specific DAG"""
    
    console.print("\n[yellow]Checking Specific DAG Retrieval[/yellow]")
    console.print("=" * 60)
    
    # Initialize components
    memory = ADAMMemoryAdvanced()
    enhancer = MemorySearchEnhancer()
    
    # Try different queries
    test_queries = [
        "bring back the last dag we created together",
        "show me the new_fee_repricing_user dag",
        "get the dag with MARKETING_ANALYTICS constants",
        "the dag we created for fee repricing with DbtOperator"
    ]
    
    found_correct_dag = False
    
    for query in test_queries:
        console.print(f"\n[cyan]Query: '{query}'[/cyan]")
        
        # Search
        raw_memories = memory.recall_with_context(query=query, n_results=10)
        
        if raw_memories:
            # Apply enhancement
            enhanced_memories, context = enhancer.enhance_memory_search(
                query=query,
                conversation_history=[],
                raw_memories=raw_memories
            )
            
            # Check top result
            top_memory = enhanced_memories[0] if enhanced_memories else None
            if top_memory:
                content = top_memory.get('content', '')
                
                # Check for specific DAG markers
                has_new_fee = 'new_fee_repricing_user' in content
                has_marketing_analytics = 'MARKETING_ANALYTICS' in content
                has_dbt_operator = 'DbtOperator' in content
                
                console.print(f"Top result score: {top_memory.get('relevance_score', 0):.3f}")
                console.print(f"Contains new_fee_repricing_user: {'✓' if has_new_fee else '✗'}")
                console.print(f"Contains MARKETING_ANALYTICS: {'✓' if has_marketing_analytics else '✗'}")
                console.print(f"Contains DbtOperator: {'✓' if has_dbt_operator else '✗'}")
                
                if has_new_fee and has_marketing_analytics and has_dbt_operator:
                    found_correct_dag = True
                    console.print("\n[green]✓ FOUND THE CORRECT DAG![/green]")
                    
                    # Extract the DAG code
                    if '```python' in content:
                        start_idx = content.find('```python')
                        end_idx = content.find('```', start_idx + 9)
                        if start_idx != -1 and end_idx != -1:
                            dag_code = content[start_idx:end_idx+3]
                            console.print("\n[cyan]DAG Code:[/cyan]")
                            console.print(dag_code)
                    break
                else:
                    # Show what we got instead
                    console.print("\n[yellow]Content preview:[/yellow]")
                    if 'Response:' in content:
                        response_start = content.find('Response:')
                        preview = content[response_start:response_start+300]
                    else:
                        preview = content[:300]
                    console.print(preview + "...")
    
    if not found_correct_dag:
        console.print("\n[red]✗ Could not find the specific DAG in any query[/red]")
        console.print("\nThis suggests the DAG conversation might not be in memory or")
        console.print("the search/scoring needs further optimization.")
    
    return found_correct_dag

if __name__ == "__main__":
    check_specific_dag_retrieval()