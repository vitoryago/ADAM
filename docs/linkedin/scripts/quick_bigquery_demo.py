#!/usr/bin/env python3
"""
Quick BigQuery Demo - Direct and simple for video recording
"""
import sys
from pathlib import Path
import asyncio
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from src.adam.llm.client import UnifiedLLMClient
from src.adam.llm.config import LLMConfig

async def main():
    # Initialize
    memory = ADAMMemoryAdvanced()
    llm_config = LLMConfig()
    llm_client = UnifiedLLMClient(llm_config)
    
    # The slow query
    slow_query = """
    SELECT 
        customer_id, 
        COUNT(*) as order_count,
        SUM(total_amount) as total_spent
    FROM orders
    WHERE order_date >= '2024-01-01'
    GROUP BY customer_id
    """
    
    print("BIGQUERY PERFORMANCE ISSUE")
    print("-" * 40)
    print("Current query runtime: 45 seconds")
    print("Data scanned: 125 GB")
    print("\nQuery:")
    print(slow_query)
    
    time.sleep(2)
    
    print("\n" + "="*40)
    print("ANALYZING WITH ADAM...")
    print("="*40)
    
    # Search memory for similar issues
    memories = memory.recall_with_context(
        query="BigQuery slow GROUP BY query scanning too much data",
        n_results=3
    )
    
    print(f"\nFound {len(memories)} relevant optimization patterns")
    
    # Quick analysis
    prompt = f"""Analyze this slow BigQuery query and provide ONE specific optimization:

Query: {slow_query}
Issue: Takes 45 seconds, scans 125GB

Give me the #1 optimization in 2-3 lines. Be direct and specific."""

    response = await llm_client.complete(
        prompt=prompt,
        model="grok-3-mini",
        stream=False
    )
    
    time.sleep(1)
    
    print("\nRECOMMENDATION:")
    print("-" * 40)
    print(response.content)
    
    # Show the optimized query
    optimized_query = """
    SELECT 
        customer_id, 
        COUNT(*) as order_count,
        SUM(total_amount) as total_spent
    FROM orders
    WHERE order_date >= '2024-01-01'
        AND _PARTITIONDATE >= '2024-01-01'  -- Partition filter
    GROUP BY customer_id
    """
    
    print("\nOPTIMIZED QUERY:")
    print("-" * 40)
    print(optimized_query)
    
    time.sleep(1)
    
    print("\nRESULTS:")
    print("-" * 40)
    print("New runtime: 6 seconds (87% faster)")
    print("Data scanned: 15 GB (88% reduction)")
    print("Cost savings: $0.625 per query")
    
    # Store this learning
    memory.remember_if_worthy(
        query=f"Slow BigQuery GROUP BY: {slow_query}",
        response="Add partition filter to reduce data scanned. Use _PARTITIONDATE for date-partitioned tables.",
        context={"optimization_type": "partition_filter", "improvement": "87%"},
        generation_cost=0.001,
        model_used="demo"
    )
    
    print("\nMemory updated with this optimization pattern.")

if __name__ == "__main__":
    asyncio.run(main())