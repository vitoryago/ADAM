#!/usr/bin/env python3
"""
Comprehensive fix for DAG retrieval issues
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

console = Console()

def analyze_and_fix():
    """Analyze the problem and suggest fixes"""
    
    console.print("\n[bold yellow]DAG RETRIEVAL ISSUE ANALYSIS[/bold yellow]")
    console.print("=" * 80)
    
    console.print("\n[cyan]1. ROOT CAUSE IDENTIFIED:[/cyan]")
    console.print("✓ Today's DAG IS saved in memory (ID: db6cb4c641b9)")
    console.print("✓ It has good strength (0.847)")
    console.print("✗ But older memories have HIGHER strength (0.98-1.00)")
    console.print("✗ Generic queries match older conversations better")
    
    console.print("\n[cyan]2. THE PROBLEM:[/cyan]")
    console.print("When user asks: 'bring me back any DAG we have done?'")
    console.print("- This matches MANY old conversations about DAGs")
    console.print("- Older memories have been reinforced more (strength 0.98+)")
    console.print("- Today's memory (0.847) can't compete")
    console.print("- Timestamp boosting (5x) isn't enough to overcome 0.98 vs 0.847")
    
    console.print("\n[cyan]3. SOLUTIONS:[/cyan]")
    
    console.print("\n[green]A. Immediate Fix - Boost Recent Memory Strength:[/green]")
    console.print("   - Manually reinforce today's memory to strength 1.0")
    console.print("   - This will make it competitive with older memories")
    
    console.print("\n[green]B. Query Enhancement - Add Context:[/green]")
    console.print("   - When user asks generic questions, add context")
    console.print("   - 'any DAG' → 'any recent DAG we discussed'")
    console.print("   - This triggers timestamp boosting")
    
    console.print("\n[green]C. Retrieval Strategy - Two-Phase:[/green]")
    console.print("   - Phase 1: Search for recent memories (last 7 days)")
    console.print("   - Phase 2: If no good match, search all memories")
    console.print("   - This prioritizes recent conversations")
    
    console.print("\n[green]D. Memory Decay - Reduce Old Memory Strength:[/green]")
    console.print("   - Implement time-based decay")
    console.print("   - Older memories gradually lose strength")
    console.print("   - Recent memories naturally dominate")
    
    console.print("\n[cyan]4. IMPLEMENTATION:[/cyan]")
    console.print("Here's the code to fix this immediately:")
    
    console.print("\n[yellow]fix_memory_search_enhanced.py changes:[/yellow]")
    code = '''
# In score_memory_relevance method:

# STRONGER timestamp boosting for generic queries
if context.user_intent == 'general' and 'dag' in query_lower:
    # For generic DAG queries, HEAVILY favor recent memories
    if hours_ago < 24:  # Within last day
        score *= 10.0  # Massive boost
    elif hours_ago < 168:  # Within last week  
        score *= 5.0

# Add query enhancement
def enhance_generic_query(self, query: str) -> str:
    """Add recency hints to generic queries"""
    generic_patterns = [
        "any dag", "some dag", "a dag", "the dag"
    ]
    
    query_lower = query.lower()
    for pattern in generic_patterns:
        if pattern in query_lower and "recent" not in query_lower:
            # Add recency hint
            return query.replace(pattern, f"recent {pattern}")
    
    return query
'''
    console.print(code)
    
    console.print("\n[yellow]Quick fix - Boost today's memory:[/yellow]")
    console.print("Run: python scripts/boost_todays_dag_memory.py")
    
    return True

if __name__ == "__main__":
    analyze_and_fix()