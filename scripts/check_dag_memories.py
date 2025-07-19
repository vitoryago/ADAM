#!/usr/bin/env python3
"""
Check all DAG-related memories and their timestamps
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from datetime import datetime
from rich.console import Console
from rich.table import Table

console = Console()

def check_dag_memories():
    """Check all DAG memories and sort by timestamp"""
    
    # Initialize memory system
    memory = ADAMMemoryAdvanced()
    
    # Search for all DAG-related memories
    console.print("\n[yellow]Searching for all DAG-related memories...[/yellow]")
    
    queries = ["dag", "DbtOperator", "new_fee_repricing_user", "MARKETING_ANALYTICS"]
    all_dag_memories = {}
    
    for query in queries:
        results = memory.recall_with_context(query=query, n_results=20)
        for result in results:
            memory_id = result.get('memory_id', 'unknown')
            if memory_id not in all_dag_memories:
                content = result.get('content', '')
                metadata = result.get('metadata', {})
                
                # Check if this is really a DAG conversation
                if any(term in content for term in ['DbtOperator', 'dag', 'DAG', 'airflow']):
                    all_dag_memories[memory_id] = {
                        'content': content,
                        'timestamp': metadata.get('timestamp', 'unknown'),
                        'query': metadata.get('query_text', '')[:100],
                        'has_marketing_analytics': 'MARKETING_ANALYTICS' in content,
                        'has_new_fee': 'new_fee_repricing_user' in content,
                        'memory_id': memory_id
                    }
    
    # Sort by timestamp
    sorted_memories = sorted(
        all_dag_memories.items(),
        key=lambda x: x[1]['timestamp'] if x[1]['timestamp'] != 'unknown' else '',
        reverse=True
    )
    
    # Display results
    table = Table(title="DAG Memories (Most Recent First)")
    table.add_column("Memory ID", style="cyan")
    table.add_column("Timestamp", style="green")
    table.add_column("Has MA Constants", style="yellow")
    table.add_column("Has new_fee", style="magenta")
    table.add_column("Query Preview", style="white")
    
    for memory_id, data in sorted_memories[:10]:  # Show top 10
        timestamp_str = data['timestamp']
        if timestamp_str != 'unknown':
            try:
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                timestamp_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass
        
        table.add_row(
            memory_id[:12],
            timestamp_str,
            "✓" if data['has_marketing_analytics'] else "✗",
            "✓" if data['has_new_fee'] else "✗",
            data['query'][:50] + "..."
        )
    
    console.print(table)
    
    # Show the most recent DAG with MoneyLion pattern
    console.print("\n[yellow]Most recent DAG with MoneyLion pattern:[/yellow]")
    for memory_id, data in sorted_memories:
        if data['has_marketing_analytics']:
            console.print(f"\nMemory ID: {memory_id}")
            console.print(f"Timestamp: {data['timestamp']}")
            console.print(f"Query: {data['query']}")
            console.print(f"Content preview: {data['content'][:500]}...")
            break
    
    # Show the most recent new_fee_repricing DAG
    console.print("\n[yellow]Most recent new_fee_repricing_user DAG:[/yellow]")
    for memory_id, data in sorted_memories:
        if data['has_new_fee']:
            console.print(f"\nMemory ID: {memory_id}")
            console.print(f"Timestamp: {data['timestamp']}")
            console.print(f"Content preview: {data['content'][:500]}...")
            break

if __name__ == "__main__":
    check_dag_memories()