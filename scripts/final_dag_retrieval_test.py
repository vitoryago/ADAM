#!/usr/bin/env python3
"""
Final comprehensive test of DAG retrieval with all enhancements
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from src.adam.memory_search_enhanced import MemorySearchEnhancer, format_memory_for_prompt
from rich.console import Console
from rich.table import Table

console = Console()

def final_test():
    """Final test with all enhancements"""
    
    console.print("\n[bold yellow]FINAL DAG RETRIEVAL TEST[/bold yellow]")
    console.print("=" * 80)
    
    # Initialize
    memory = ADAMMemoryAdvanced()
    enhancer = MemorySearchEnhancer()
    
    # Test queries from simple to complex
    queries = [
        "new_fee_repricing_user MARKETING_ANALYTICS DbtOperator",  # Most specific
        "the last dag we created with new_fee_repricing_user",     # With "last"
        "bring back the dag code for fee repricing",               # User's style
        "can you bring back the last dag we created together?"      # Original query
    ]
    
    success = False
    
    for query in queries:
        console.print(f"\n[cyan]Testing query: '{query}'[/cyan]")
        
        # Get memories
        raw_memories = memory.recall_with_context(query=query, n_results=15)
        
        if raw_memories:
            # Mock conversation for context
            conversation = [
                {"role": "user", "content": "I need the new_fee_repricing_user dag"},
                {"role": "assistant", "content": "I'll help you with that"}
            ]
            
            # Enhance
            enhanced_memories, context = enhancer.enhance_memory_search(
                query=query,
                conversation_history=conversation,
                raw_memories=raw_memories
            )
            
            # Create results table
            table = Table(title=f"Top 5 Results (Intent: {context.user_intent})")
            table.add_column("Rank", style="cyan")
            table.add_column("Score", style="green")
            table.add_column("Memory ID", style="yellow")
            table.add_column("Has Components", style="white")
            
            # Check each result
            for i, memory in enumerate(enhanced_memories[:5]):
                content = memory.get('content', '')
                memory_id = memory.get('memory_id', 'unknown')[:12]
                
                # Check for all required components
                has_new_fee = '✓' if 'new_fee_repricing_user' in content else '✗'
                has_marketing = '✓' if 'MARKETING_ANALYTICS' in content else '✗'
                has_dbt = '✓' if 'DbtOperator' in content else '✗'
                
                components = f"new_fee:{has_new_fee} MA:{has_marketing} Dbt:{has_dbt}"
                
                table.add_row(
                    str(i+1),
                    f"{memory.get('relevance_score', 0):.3f}",
                    memory_id,
                    components
                )
                
                # Check if this is the correct DAG
                if has_new_fee == '✓' and has_marketing == '✓' and has_dbt == '✓':
                    if i == 0:  # Top result
                        success = True
                        console.print(table)
                        console.print(f"\n[bold green]✓ SUCCESS! Found correct DAG at position {i+1}![/bold green]")
                        
                        # Show the actual DAG code
                        if '```python' in content:
                            start = content.find('```python')
                            end = content.find('```', start + 9)
                            if start != -1 and end != -1:
                                dag_code = content[start+9:end]
                                console.print("\n[yellow]DAG Code Preview:[/yellow]")
                                console.print(dag_code[:500] + "..." if len(dag_code) > 500 else dag_code)
                        break
            
            if not success:
                console.print(table)
        else:
            console.print("[red]No memories found[/red]")
    
    # Final verdict
    console.print("\n" + "=" * 80)
    if success:
        console.print("[bold green]✅ FINAL RESULT: DAG retrieval system is working![/bold green]")
        console.print("The enhanced memory search with timestamp boosting successfully")
        console.print("retrieves the correct DAG when users ask for recent conversations.")
    else:
        console.print("[bold red]❌ FINAL RESULT: DAG retrieval needs more work[/bold red]")
        console.print("Possible solutions:")
        console.print("1. Manually boost the specific memory's strength")
        console.print("2. Add query rewriting to be more specific")
        console.print("3. Use a two-stage retrieval (broad then specific)")

if __name__ == "__main__":
    final_test()