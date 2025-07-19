#!/usr/bin/env python3
"""
Test the timestamp-based boosting in memory search
Verifies that recent memories get boosted when queries contain "last", "latest", etc.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from src.adam.memory_search_enhanced import MemorySearchEnhancer, SearchContext
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
import json

console = Console()

def test_timestamp_boosting():
    """Test that timestamp boosting works correctly"""
    
    console.print("\n[yellow]Testing Timestamp-Based Memory Boosting[/yellow]")
    console.print("=" * 60)
    
    # Initialize components
    memory = ADAMMemoryAdvanced()
    enhancer = MemorySearchEnhancer()
    
    # Test queries that should trigger timestamp boosting
    test_queries = [
        "bring back the last dag we created together",
        "show me the latest dag code",
        "what was the most recent dag we built?",
        "get the newest fee repricing dag",
        "the last dag we created"
    ]
    
    for query in test_queries:
        console.print(f"\n[cyan]Query: '{query}'[/cyan]")
        
        # Search for memories
        memories_result = memory.recall_with_context(query=query, n_results=10)
        
        # Handle the result format (could be dict or list)
        if isinstance(memories_result, dict):
            raw_memories = memories_result.get('memories', [])
        else:
            raw_memories = memories_result if memories_result else []
        
        if raw_memories:
            # Apply enhancement
            enhanced_memories, context = enhancer.enhance_memory_search(
                query=query,
                conversation_history=[],
                raw_memories=raw_memories
            )
            
            # Display context analysis
            console.print(f"Intent detected: [green]{context.user_intent}[/green]")
            console.print(f"Technical terms: [yellow]{context.technical_terms}[/yellow]")
            
            # Show top memories with scores
            table = Table(title="Top 3 Memories (with relevance scores)")
            table.add_column("Score", style="cyan")
            table.add_column("Timestamp", style="green")
            table.add_column("Age", style="yellow")
            table.add_column("Content Preview", style="white")
            
            for memory in enhanced_memories[:3]:
                score = memory.get('relevance_score', 0)
                metadata = memory.get('metadata', {})
                timestamp_str = metadata.get('timestamp', 'unknown')
                
                # Calculate age
                age_str = "unknown"
                if timestamp_str != 'unknown':
                    try:
                        memory_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        now = datetime.now(memory_time.tzinfo) if memory_time.tzinfo else datetime.now()
                        hours_ago = (now - memory_time).total_seconds() / 3600
                        
                        if hours_ago < 1:
                            age_str = f"{int(hours_ago * 60)} minutes ago"
                        elif hours_ago < 24:
                            age_str = f"{int(hours_ago)} hours ago"
                        else:
                            age_str = f"{int(hours_ago / 24)} days ago"
                    except:
                        pass
                
                # Get content preview
                content = memory.get('content', '')
                if 'new_fee_repricing_user' in content:
                    preview = "✓ Contains new_fee_repricing_user DAG"
                elif 'DbtOperator' in content:
                    preview = "✓ Contains DbtOperator code"
                else:
                    preview = content[:50] + "..."
                
                table.add_row(
                    f"{score:.3f}",
                    timestamp_str[:16] if timestamp_str != 'unknown' else 'unknown',
                    age_str,
                    preview
                )
            
            console.print(table)
            
            # Check if boosting worked
            if any(word in query.lower() for word in ['last', 'latest', 'recent', 'newest']):
                console.print("\n[green]✓ Timestamp boosting ACTIVATED for this query[/green]")
                
                # Find the most recent memory
                most_recent = None
                most_recent_time = None
                
                for memory in raw_memories:
                    metadata = memory.get('metadata', {})
                    timestamp_str = metadata.get('timestamp', '')
                    if timestamp_str:
                        try:
                            memory_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if most_recent_time is None or memory_time > most_recent_time:
                                most_recent_time = memory_time
                                most_recent = memory
                        except:
                            pass
                
                if most_recent:
                    # Check if it's in top results
                    top_memory_ids = [m.get('memory_id') for m in enhanced_memories[:3]]
                    if most_recent.get('memory_id') in top_memory_ids:
                        console.print("[green]✓ Most recent memory is in TOP 3 results![/green]")
                    else:
                        console.print("[red]✗ Most recent memory NOT in top 3 results[/red]")
                        console.print(f"Most recent memory ID: {most_recent.get('memory_id')}")
        else:
            console.print("[red]No memories found for this query[/red]")
    
    # Test specific DAG query
    console.print("\n" + "=" * 60)
    console.print("[yellow]Testing specific DAG retrieval:[/yellow]")
    
    specific_query = "bring back the last dag we created with new_fee_repricing_user"
    memories_result = memory.recall_with_context(query=specific_query, n_results=10)
    
    # Handle the result format
    if isinstance(memories_result, dict):
        raw_memories = memories_result.get('memories', [])
    else:
        raw_memories = memories_result if memories_result else []
    
    if raw_memories:
        enhanced_memories, context = enhancer.enhance_memory_search(
            query=specific_query,
            conversation_history=[],
            raw_memories=raw_memories
        )
        
        # Check if we found the right DAG
        found_correct_dag = False
        for memory in enhanced_memories[:1]:  # Check top result
            content = memory.get('content', '')
            if 'new_fee_repricing_user' in content and 'MARKETING_ANALYTICS' in content:
                found_correct_dag = True
                console.print("\n[green]✓ Found the correct DAG in top result![/green]")
                console.print(f"Memory ID: {memory.get('memory_id')}")
                console.print(f"Score: {memory.get('relevance_score', 0):.3f}")
                
                # Show the actual DAG code portion
                if 'DbtOperator' in content:
                    start_idx = content.find('```python')
                    if start_idx != -1:
                        end_idx = content.find('```', start_idx + 9)
                        if end_idx != -1:
                            dag_code = content[start_idx:end_idx+3]
                            console.print("\n[cyan]DAG Code Preview:[/cyan]")
                            console.print(dag_code[:500] + "..." if len(dag_code) > 500 else dag_code)
        
        if not found_correct_dag:
            console.print("\n[red]✗ Did not find the specific DAG in top results[/red]")
            console.print("Top result contains:")
            console.print(enhanced_memories[0].get('content', '')[:200] + "...")

if __name__ == "__main__":
    test_timestamp_boosting()