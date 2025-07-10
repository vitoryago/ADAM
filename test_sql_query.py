#!/usr/bin/env python3
"""
Test SQL Query Analysis with ADAM
"""
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Setup
load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adam.llm.client import UnifiedLLMClient
from adam.tools.sql_tools import SQLAnalyzer, SQLOptimizer


async def analyze_your_query():
    """Analyze the Apple Search Ads query"""
    
    # Your SQL query for Apple Search Ads last 7 days
    query = """
    SELECT 
        date,
        campaign_name,
        keyword_name,
        keyword_cost,
        keyword_clicks,
        keyword_impressions,
        installs,
        su_lite,
        customer,
        -- Calculate metrics
        CASE WHEN keyword_clicks > 0 THEN keyword_cost / keyword_clicks ELSE 0 END AS cpc,
        CASE WHEN installs > 0 THEN keyword_cost / installs ELSE 0 END AS cpi,
        CASE WHEN keyword_impressions > 0 THEN keyword_clicks / keyword_impressions ELSE 0 END AS ctr,
        CASE WHEN keyword_clicks > 0 THEN installs / keyword_clicks ELSE 0 END AS install_rate
    FROM 
        fct_daily_performance_marketing_keyword_level
    WHERE 
        ad_platform = 'apple'
        AND date >= CURRENT_DATE - 7
        AND keyword_cost > 0
    ORDER BY 
        keyword_cost DESC
    LIMIT 100
    """
    
    print("🍎 Analyzing Apple Search Ads Query for Last 7 Days\n")
    print("=" * 60)
    
    # Initialize tools
    sql_analyzer = SQLAnalyzer("snowflake")
    sql_optimizer = SQLOptimizer("snowflake")
    llm = UnifiedLLMClient()
    
    # Analyze query
    print("\n📊 SQL Analysis:")
    issues, metrics = sql_analyzer.analyze_query(query)
    print(f"- Complexity Score: {metrics.complexity_score}/10")
    print(f"- Line Count: {metrics.line_count}")
    print(f"- Join Count: {metrics.join_count}")
    print(f"- Issues Found: {len(issues)}")
    
    if issues:
        print("\n⚠️  Issues:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue.message}")
            if issue.suggestion:
                print(f"   💡 {issue.suggestion}")
    
    # Optimize query
    print("\n🚀 Optimizing Query...")
    optimized = await sql_optimizer.optimize_query(query)
    
    print("\n✨ Optimized Query:")
    print(optimized['optimized_query'])
    
    # Get LLM insights
    print("\n🤖 Getting Additional Insights...")
    
    prompt = f"""As a data analytics expert, analyze this query that filters Apple Search Ads data for the last 7 days and finds best performing keywords.

Query: {query}

Provide:
1. What metrics this query calculates
2. Suggestions for additional useful metrics
3. Best practices for analyzing paid search performance

Be concise and technical."""

    response = await llm.complete(prompt, temperature=0.3)
    print("\n📝 Expert Analysis:")
    print(response.content)
    print(f"\n[Model: {response.model}]")


if __name__ == "__main__":
    asyncio.run(analyze_your_query())