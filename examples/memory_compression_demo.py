#!/usr/bin/env python3
"""
Memory Compression Demo
Shows how ADAM's intelligent compression preserves value while reducing storage
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory_compressor import MemoryCompressor
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

console = Console()

async def main():
    """Demo the memory compression system"""
    console.print(Panel.fit(
        "[bold cyan]ADAM Memory Compression Demo[/bold cyan]\n\n"
        "This demo shows how ADAM intelligently compresses memories\n"
        "while preserving the most valuable information.",
        title="🗜️ Memory Compression"
    ))
    
    compressor = MemoryCompressor()
    
    # Example: A typical SQL optimization memory
    original_memory = """
Query: I have a complex Snowflake query joining 5 tables that's taking over 10 minutes. How can I make it faster?

Response: I'll help you optimize your complex Snowflake query. Here's a comprehensive approach:

1. **Analyze the Query Profile**
   First, run your query with the Snowflake Query Profile to identify bottlenecks:
   - Look for full table scans
   - Check data spillage to disk
   - Identify the most expensive operations

2. **Optimize JOIN Order**
   Place smaller tables first and filter early:
   ```sql
   -- Better: Filter before joining
   WITH filtered_orders AS (
     SELECT * FROM orders 
     WHERE order_date >= '2024-01-01'
   )
   SELECT ...
   FROM filtered_orders
   JOIN customers ON ...
   ```

3. **Use Clustering Keys**
   For large fact tables, clustering dramatically improves performance:
   ```sql
   ALTER TABLE large_fact_table 
   CLUSTER BY (date_column, frequently_filtered_column);
   ```

4. **Leverage Materialized Views**
   For complex aggregations that run frequently:
   ```sql
   CREATE MATERIALIZED VIEW mv_daily_summary AS
   SELECT date, customer_id, SUM(amount) as total
   FROM large_fact_table
   GROUP BY date, customer_id;
   ```

5. **Consider Query Rewrite**
   Sometimes restructuring the query helps:
   - Use CTEs to break complex logic
   - Replace NOT IN with NOT EXISTS
   - Use QUALIFY for window function filtering

The most impactful optimization is usually proper clustering keys on your largest tables. This can reduce query time from minutes to seconds.

Would you like me to review your specific query for targeted optimizations?
"""
    
    metadata = {
        'memory_type': 'code_pattern',
        'query_text': "I have a complex Snowflake query joining 5 tables that's taking over 10 minutes",
        'topics': ['snowflake', 'optimization', 'performance', 'sql']
    }
    
    # Show original
    console.print("\n[bold yellow]Original Memory[/bold yellow]")
    console.print(f"Size: {len(original_memory):,} characters")
    console.print(Panel(original_memory[:500] + "...\n\n[dim](truncated for display)[/dim]", 
                       border_style="yellow"))
    
    # Compress at different levels
    console.print("\n[bold green]Applying Intelligent Compression...[/bold green]\n")
    
    results = {}
    for level in ['moderate', 'high', 'ultra']:
        result = await compressor.compress_memory(original_memory, metadata, level)
        results[level] = result
    
    # Display results
    table = Table(title="Compression Results")
    table.add_column("Level", style="cyan", width=12)
    table.add_column("Size", style="yellow", width=15)
    table.add_column("Reduction", style="green", width=12)
    table.add_column("What's Preserved", style="magenta", width=40)
    
    for level, result in results.items():
        preserved = ", ".join(result.preserved_elements) if result.preserved_elements else "essence"
        table.add_row(
            level.upper(),
            f"{len(result.compressed_content)} chars",
            f"{result.compression_ratio:.1%}",
            preserved
        )
    
    console.print(table)
    
    # Show each compression level
    console.print("\n[bold]Compression Examples:[/bold]\n")
    
    # Moderate
    console.print("[cyan]MODERATE Compression (for 7-30 day old memories):[/cyan]")
    console.print(Panel(
        results['moderate'].compressed_content[:400] + "...",
        title=f"Reduced to {len(results['moderate'].compressed_content)} chars ({results['moderate'].compression_ratio:.0%} smaller)",
        border_style="cyan"
    ))
    
    # High
    console.print("\n[yellow]HIGH Compression (for 30-90 day old memories):[/yellow]")
    console.print(Panel(
        results['high'].compressed_content,
        title=f"Reduced to {len(results['high'].compressed_content)} chars ({results['high'].compression_ratio:.0%} smaller)",
        border_style="yellow"
    ))
    
    # Ultra
    console.print("\n[red]ULTRA Compression (for 90+ day old memories):[/red]")
    console.print(Panel(
        results['ultra'].compressed_content,
        title=f"Reduced to {len(results['ultra'].compressed_content)} chars ({results['ultra'].compression_ratio:.0%} smaller)",
        border_style="red"
    ))
    
    # Cost savings
    console.print("\n[bold green]💰 Cost Savings Example:[/bold green]")
    console.print(f"""
If you have 10,000 memories averaging {len(original_memory):,} characters each:
- Original storage: {len(original_memory) * 10000 / 1024 / 1024:.1f} MB
- After compression: ~{len(original_memory) * 10000 * 0.42 / 1024 / 1024:.1f} MB
- Storage saved: ~{len(original_memory) * 10000 * 0.58 / 1024 / 1024:.1f} MB (58%)
- Tokens saved: ~{results['moderate'].tokens_saved * 10000:,} tokens
""")
    
    console.print("[dim]Note: Actual savings depend on memory age distribution[/dim]")

if __name__ == "__main__":
    asyncio.run(main())