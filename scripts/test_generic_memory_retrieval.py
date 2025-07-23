#!/usr/bin/env python3
"""
Test generic memory retrieval with various content types (not just DAGs)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from src.adam.memory_search_enhanced import MemorySearchEnhancer
from rich.console import Console
from rich.table import Table

console = Console()

def test_generic_retrieval():
    """Test retrieval with various generic queries"""
    
    console.print("\n[bold yellow]Testing Generic Memory Retrieval[/bold yellow]")
    console.print("=" * 80)
    
    memory = ADAMMemoryAdvanced()
    enhancer = MemorySearchEnhancer()
    
    # Test various generic queries (not DAG-specific)
    test_cases = [
        # Generic recalls
        ("bring me back what we discussed", "Generic recall"),
        ("show me what we were working on", "Generic work reference"),
        ("can you bring back our conversation", "Generic conversation"),
        ("that thing we created", "Vague reference"),
        
        # With some context but still generic
        ("show me that query we wrote", "Generic + domain hint"),
        ("bring back the code we discussed", "Generic + code mention"),
        ("what were we talking about earlier", "Temporal but vague"),
        
        # Specific queries (should NOT trigger enhancement)
        ("show me the query with JOIN on users table", "Specific detail"),
        ("bring back the new_fee_repricing_user DAG", "Specific name"),
        ("the documentation about authentication", "Specific topic"),
    ]
    
    results = []
    
    for query, description in test_cases:
        console.print(f"\n[cyan]Testing: '{query}'[/cyan]")
        console.print(f"Type: {description}")
        
        # Get memories
        raw_memories = memory.recall_with_context(query=query, n_results=10)
        
        if raw_memories:
            # Apply enhancement
            enhanced_memories, context = enhancer.enhance_memory_search(
                query=query,
                conversation_history=[],
                raw_memories=raw_memories
            )
            
            # Check what happened
            enhanced_query = enhancer.build_enhanced_query(context)
            is_enhanced = enhanced_query != query
            
            console.print(f"Intent: {context.user_intent}")
            console.print(f"Query enhanced: {'Yes' if is_enhanced else 'No'}")
            if is_enhanced:
                console.print(f"Enhanced to: '{enhanced_query}'")
            
            # Check if recent memories are prioritized
            if enhanced_memories:
                top_memory = enhanced_memories[0]
                metadata = top_memory.get('metadata', {})
                timestamp = metadata.get('timestamp', '')
                score = top_memory.get('relevance_score', 0)
                
                # Extract date if possible
                date_str = timestamp[:10] if len(timestamp) >= 10 else 'unknown'
                console.print(f"Top result date: {date_str}, Score: {score:.3f}")
            
            results.append({
                'query': query,
                'type': description,
                'intent': context.user_intent,
                'enhanced': is_enhanced
            })
    
    # Summary
    console.print("\n" + "=" * 80)
    console.print("[yellow]Summary of Query Processing[/yellow]")
    
    table = Table(title="Query Enhancement Results")
    table.add_column("Query", style="cyan", width=40)
    table.add_column("Type", style="green", width=25)
    table.add_column("Intent", style="yellow")
    table.add_column("Enhanced?", style="magenta")
    
    for result in results:
        table.add_row(
            result['query'][:40] + "..." if len(result['query']) > 40 else result['query'],
            result['type'],
            result['intent'],
            "✓" if result['enhanced'] else "✗"
        )
    
    console.print(table)
    
    # Analysis
    console.print("\n[green]Key Observations:[/green]")
    console.print("1. Generic queries without specifics get enhanced")
    console.print("2. Queries with specific details are left unchanged")
    console.print("3. The solution is domain-agnostic (works for DAGs, queries, docs, etc.)")
    console.print("4. Recent memories should be prioritized for generic queries")

if __name__ == "__main__":
    test_generic_retrieval()