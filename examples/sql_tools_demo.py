#!/usr/bin/env python3
"""
SQL Tools Demo - Shows how ADAM helps analytics engineers
"""
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Setup
load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adam.tools.sql_tools import SQLAnalyzer, SQLFormatter, SQLOptimizer, analyze_sql
from adam.llm.client import UnifiedLLMClient


async def main():
    print("🔧 ADAM SQL Tools Demo\n")
    print("=" * 60)
    
    # Example 1: Analyze a problematic query
    print("\n📊 Example 1: Analyzing a Slow Snowflake Query")
    print("-" * 60)
    
    slow_query = """
    SELECT DISTINCT *
    FROM orders o, customers c
    WHERE o.customer_id = c.id
    AND o.order_date BETWEEN '2024-01-01' AND '2024-12-31'
    AND o.status NOT IN (SELECT status FROM cancelled_orders)
    AND o.amount > 100
    ORDER BY o.order_date DESC
    """
    
    analyzer = SQLAnalyzer("snowflake")
    issues, metrics = analyzer.analyze_query(slow_query)
    
    print(f"Query Metrics:")
    print(f"  - Lines: {metrics.line_count}")
    print(f"  - Complexity Score: {metrics.complexity_score}/10")
    print(f"  - Joins: {metrics.join_count}")
    print(f"  - Subqueries: {metrics.subquery_count}")
    
    print(f"\n🚨 Issues Found ({len(issues)}):")
    for i, issue in enumerate(issues, 1):
        print(f"\n{i}. [{issue.level.value.upper()}] {issue.message}")
        if issue.suggestion:
            print(f"   💡 Suggestion: {issue.suggestion}")
        if issue.estimated_impact:
            print(f"   📈 Impact: {issue.estimated_impact}")
    
    # Example 2: Format SQL according to dbt standards
    print("\n\n📝 Example 2: Formatting SQL (dbt style)")
    print("-" * 60)
    
    messy_query = "SELECT customer_id,COUNT(*) as order_count,SUM(amount) as total FROM orders WHERE status='completed' GROUP BY customer_id HAVING COUNT(*)>5"
    
    formatter = SQLFormatter("dbt")
    formatted = formatter.format_query(messy_query)
    
    print("Before:")
    print(messy_query)
    print("\nAfter:")
    print(formatted)
    
    # Example 3: Get AI-powered optimization suggestions
    print("\n\n🤖 Example 3: AI-Powered Query Optimization")
    print("-" * 60)
    
    complex_query = """
    SELECT 
        c.customer_id,
        c.customer_name,
        (SELECT COUNT(*) FROM orders WHERE customer_id = c.id) as order_count,
        (SELECT SUM(amount) FROM orders WHERE customer_id = c.id) as total_spent,
        (SELECT MAX(order_date) FROM orders WHERE customer_id = c.id) as last_order
    FROM customers c
    WHERE c.active = true
    AND EXISTS (SELECT 1 FROM orders WHERE customer_id = c.id AND order_date > '2024-01-01')
    """
    
    optimizer = SQLOptimizer("snowflake")
    
    print("Analyzing and optimizing query...")
    result = await optimizer.optimize_query(complex_query)
    
    print(f"\n📊 Analysis Results:")
    print(f"  - Issues found: {len(result['issues'])}")
    print(f"  - Estimated improvement: {result['estimated_improvement']}")
    
    print(f"\n💡 Recommendations:")
    for rec in result['recommendations']:
        print(f"  {rec}")
    
    print(f"\n✨ Optimized Query:")
    print(result['optimized_query'])
    
    # Example 4: Quick analysis function
    print("\n\n⚡ Example 4: Quick Analysis")
    print("-" * 60)
    
    quick_query = """
    SELECT * FROM sales 
    WHERE region LIKE '%west%' 
    AND date >= CURRENT_DATE - 30
    """
    
    print("Using quick analyze function...")
    analysis = await analyze_sql(quick_query, "snowflake")
    
    print(f"Issues: {len(analysis['issues'])}")
    for issue in analysis['issues'][:3]:  # Show first 3
        print(f"  - {issue.message}")
    
    # Example 5: Real-world scenario
    print("\n\n🌟 Example 5: Real-World Analytics Scenario")
    print("-" * 60)
    
    print("Scenario: Monthly revenue report is timing out")
    
    revenue_query = """
    WITH monthly_orders AS (
        SELECT * FROM orders 
        WHERE order_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '12 months')
    ),
    customer_metrics AS (
        SELECT 
            customer_id,
            COUNT(DISTINCT order_id) as orders,
            SUM(amount) as revenue
        FROM monthly_orders
        GROUP BY customer_id
    )
    SELECT DISTINCT
        DATE_TRUNC('month', mo.order_date) as month,
        c.customer_segment,
        COUNT(DISTINCT mo.customer_id) as customers,
        SUM(mo.amount) as revenue
    FROM monthly_orders mo
    JOIN customers c ON mo.customer_id = c.id
    WHERE c.customer_segment IN ('Enterprise', 'SMB', 'Startup')
    GROUP BY 1, 2
    ORDER BY 1 DESC, 2
    """
    
    analyzer = SQLAnalyzer("snowflake")
    issues, metrics = analyzer.analyze_query(revenue_query)
    
    print(f"\nComplexity Score: {metrics.complexity_score}/10")
    print(f"CTEs: {metrics.cte_count}")
    
    # Show Snowflake-specific recommendations
    snowflake_issues = [i for i in issues if "cluster" in i.message.lower() or "snowflake" in i.message.lower()]
    if snowflake_issues:
        print("\n🏔️  Snowflake-Specific Optimizations:")
        for issue in snowflake_issues:
            print(f"  - {issue.message}")
    
    print("\n✅ Done! ADAM can help you optimize SQL queries in seconds.")


if __name__ == "__main__":
    print("Starting ADAM SQL Tools Demo...")
    asyncio.run(main())