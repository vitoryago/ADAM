#!/usr/bin/env python3
"""
Example analytics queries using ADAM's LLM system
Shows how to use Grok models for SQL and dbt help
"""
import asyncio
from dotenv import load_dotenv
import sys
from pathlib import Path

# Load environment and add src to path
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adam.llm.client import UnifiedLLMClient, quick_complete, reasoning_complete

async def test_analytics_queries():
    """Test various analytics engineering queries"""
    print("🧠 ADAM Analytics Assistant Examples\n")
    
    # Initialize client
    client = UnifiedLLMClient()
    
    # Example 1: SQL Optimization
    print("1️⃣ SQL Optimization Query")
    print("-" * 50)
    query = """
    My Snowflake query is slow. Here's the SQL:
    
    SELECT 
        customer_id,
        order_date,
        SUM(amount) OVER (PARTITION BY customer_id ORDER BY order_date) as running_total
    FROM orders
    WHERE order_date >= '2024-01-01'
    
    How can I optimize this?
    """
    
    response = await client.complete(query, model="grok-4")
    print(f"ADAM: {response.content[:500]}...")
    print(f"[Model: {response.model} | Tokens: {response.total_tokens}]\n")
    
    # Example 2: dbt Error Debugging
    print("2️⃣ dbt Error Debugging")
    print("-" * 50)
    error_query = """
    My dbt model is failing with this error:
    Database Error in model stg_orders (models/staging/stg_orders.sql)
    001003 (42000): SQL compilation error:
    syntax error line 3 at position 7 unexpected 'FROM'.
    
    What's wrong?
    """
    
    response = await client.complete(error_query, reasoning_effort="high")
    print(f"ADAM: {response.content[:500]}...")
    print(f"[Used reasoning: {'Yes' if response.reasoning_content else 'No'}]\n")
    
    # Example 3: Data Quality Check
    print("3️⃣ Data Quality Issue")
    print("-" * 50)
    quality_query = """
    The revenue numbers in my dashboard don't match between two reports.
    Report A shows $1.2M for January, Report B shows $1.5M.
    How should I debug this discrepancy?
    """
    
    response = await reasoning_complete(quality_query, effort="high")
    print(f"ADAM's Answer: {response['answer'][:500]}...")
    print(f"Reasoning tokens: {response['tokens']['reasoning']}")
    print(f"Total tokens: {response['tokens']['total']}\n")
    
    # Example 4: Quick SQL Pattern
    print("4️⃣ Quick SQL Pattern")
    print("-" * 50)
    result = await quick_complete("How do I write an incremental dbt model?")
    print(f"ADAM: {result[:300]}...\n")
    
    # Example 5: Auto Model Selection
    print("5️⃣ Auto Model Selection Demo")
    print("-" * 50)
    
    queries = [
        ("What is CTE?", "Simple query"),
        ("Debug why my 50-line SQL query with 10 CTEs is slow", "Complex SQL"),
        ("Explain step-by-step how Snowflake clustering works", "Reasoning task")
    ]
    
    for query, query_type in queries:
        response = await client.complete(query)  # Let ADAM choose
        print(f"{query_type}: Used {response.model}")

if __name__ == "__main__":
    print("Testing ADAM's Analytics Engineering capabilities...\n")
    asyncio.run(test_analytics_queries())