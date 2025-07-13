#!/usr/bin/env python3
"""
Seed ADAM's memory with BigQuery optimization scenarios for LinkedIn demo
"""
import sys
import json
from pathlib import Path
import asyncio
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.adam.memory import ADAMMemoryAdvanced

def load_scenarios():
    """Load BigQuery scenarios from JSON"""
    scenarios_path = Path(__file__).parent.parent / "data" / "bigquery_scenarios.json"
    with open(scenarios_path, 'r') as f:
        return json.load(f)

async def seed_bigquery_patterns():
    """Seed memory with BigQuery optimization patterns"""
    print("🧠 Seeding ADAM's memory with BigQuery optimization patterns...")
    
    memory = ADAMMemoryAdvanced()
    scenarios = load_scenarios()
    
    # Seed optimization patterns first
    print("\n📚 Learning optimization patterns...")
    for pattern, description in scenarios['optimization_patterns'].items():
        query = f"What is {pattern} in BigQuery optimization?"
        response = f"{pattern.replace('_', ' ').title()}: {description}"
        
        memory.remember_if_worthy(
            query=query,
            response=response,
            context={
                "domain": "bigquery",
                "type": "optimization_pattern",
                "pattern": pattern
            },
            generation_cost=0.001,
            model_used="demo"
        )
        print(f"  ✅ Learned about {pattern}")
    
    # Seed specific scenarios with solutions
    print("\n🔧 Learning from real-world scenarios...")
    optimizations = {
        "bq_001": """
To optimize this JOIN query:
1. **Partition the orders table** by order_date to reduce scan size
2. **Cluster both tables** by customer_id for efficient joins
3. **Create a materialized view** for customer order summaries
4. **Use APPROX_COUNT_DISTINCT** if exact counts aren't required

Optimized query:
```sql
-- Use partitioned table
SELECT c.customer_id, c.name, 
       COUNT(o.order_id) as order_count,
       SUM(o.total_amount) as total_spent
FROM `project.dataset.customers_clustered` c
LEFT JOIN `project.dataset.orders_partitioned` o
  ON c.customer_id = o.customer_id
WHERE o.order_date >= '2023-01-01'
  AND _PARTITIONDATE >= '2023-01-01'  -- Partition filter
GROUP BY c.customer_id, c.name
```
Expected improvement: 70% reduction in runtime, 80% less data scanned.""",
        
        "bq_002": """
To optimize window functions:
1. **Pre-filter data** before applying window functions
2. **Use clustering** on user_id and event_time
3. **Consider materialized intermediate results**
4. **Limit window frame size** where possible

Optimized approach:
```sql
-- Create temporary table with filtered data
CREATE TEMP TABLE recent_events AS
SELECT user_id, event_time, event_type
FROM `project.dataset.user_events`
WHERE DATE(event_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY);

-- Apply window functions on smaller dataset
SELECT user_id, event_time, event_type,
       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_time DESC) as rn,
       LEAD(event_type) OVER (PARTITION BY user_id ORDER BY event_time) as next_event
FROM recent_events
```
Expected improvement: 60% faster, 50% less memory usage.""",
        
        "bq_003": """
To optimize CTEs:
1. **Materialize CTEs** that are used multiple times
2. **Use CREATE TEMP TABLE** for complex intermediate results
3. **Push filters down** to the earliest possible stage

Optimized version:
```sql
-- Materialize the daily stats
CREATE TEMP TABLE daily_stats AS
SELECT DATE(timestamp) as date, 
       COUNT(*) as events
FROM `project.dataset.events`
WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH)
GROUP BY date;

-- Now compute weekly stats
SELECT DATE_TRUNC(date, WEEK) as week,
       SUM(events) as weekly_events
FROM daily_stats
GROUP BY week
```
Expected improvement: 40% faster execution.""",
        
        "bq_004": """
To optimize aggregations:
1. **Pre-aggregate at lower granularity** using scheduled queries
2. **Use partitioning and clustering** on group-by columns
3. **Consider APPROX functions** for distinct counts
4. **Create aggregation tables** for common patterns

Optimized strategy:
```sql
-- Use pre-aggregated daily table
SELECT product_category,
       DATE_TRUNC(day, MONTH) as month,
       SUM(unique_customers_daily) as unique_customers,
       SUM(revenue_daily) as revenue
FROM `project.dataset.daily_product_metrics`
WHERE day >= DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH)
GROUP BY product_category, month
ORDER BY month DESC, revenue DESC
```
Expected improvement: 85% faster, 90% less data processed.""",
        
        "bq_005": """
To optimize string operations:
1. **Pre-compute derived fields** during ingestion
2. **Use persisted computed columns**
3. **Create lookup tables** for domain mappings
4. **Index frequently searched patterns**

Optimized approach:
```sql
-- Use pre-computed columns
SELECT user_id, email, 
       email_domain,  -- Pre-computed column
       email_provider -- Pre-computed column
FROM `project.dataset.users_enriched`
WHERE email_valid = TRUE  -- Pre-validated flag
```
Expected improvement: 70% faster, cleaner code."""
    }
    
    for scenario in scenarios['scenarios']:
        scenario_id = scenario['id']
        
        # Create problem query
        problem_query = f"My BigQuery query is slow: {scenario['query'][:100]}... Issue: {scenario['issue']}"
        
        # Store problem with solution
        if scenario_id in optimizations:
            memory.remember_if_worthy(
                query=problem_query,
                response=optimizations[scenario_id],
                context={
                    "domain": "bigquery",
                    "type": "performance_optimization",
                    "scenario_id": scenario_id,
                    "original_runtime": scenario['metrics']['original_runtime'],
                    "data_processed_gb": scenario['metrics']['data_processed_gb']
                },
                generation_cost=0.002,
                model_used="demo"
            )
            print(f"  ✅ Learned optimization for: {scenario['title']}")
    
    # Add general BigQuery best practices
    print("\n📋 Adding best practices...")
    best_practices = [
        {
            "query": "What are BigQuery best practices for performance?",
            "response": """
Key BigQuery performance best practices:

1. **Partitioning & Clustering**
   - Partition by date/timestamp columns
   - Cluster by high-cardinality filter columns
   - Always include partition filters

2. **Query Optimization**
   - Filter early and aggressively
   - Avoid SELECT * - specify only needed columns
   - Use APPROX functions for large-scale analytics

3. **Join Optimization**
   - Put smaller tables on the left side of JOINs
   - Use clustering keys that match join conditions
   - Consider denormalization for frequent joins

4. **Resource Management**
   - Use slots effectively with reservation
   - Monitor and control costs with quotas
   - Schedule heavy queries during off-peak hours

5. **Data Organization**
   - Use appropriate data types (especially for timestamps)
   - Compress data with optimal formats
   - Archive old data to reduce scan costs
"""
        },
        {
            "query": "How do I debug slow BigQuery queries?",
            "response": """
BigQuery query debugging steps:

1. **Check Query Execution Details**
   - View query plan in BigQuery UI
   - Look for stages with high slot time
   - Identify data skew issues

2. **Analyze Key Metrics**
   - Bytes processed vs bytes billed
   - Slot utilization and wait time
   - Cache hit rate

3. **Common Issues to Check**
   - Missing partition filters
   - Cartesian joins (missing join conditions)
   - Repeated subqueries that could be CTEs
   - Window functions on large partitions

4. **Tools to Use**
   - INFORMATION_SCHEMA views
   - Query execution timeline
   - Slot utilization graphs
   - Cost breakdown analysis

5. **Quick Wins**
   - Add LIMIT during development
   - Use table samples for testing
   - Preview data before full queries
"""
        }
    ]
    
    for practice in best_practices:
        memory.remember_if_worthy(
            query=practice['query'],
            response=practice['response'],
            context={
                "domain": "bigquery",
                "type": "best_practice",
                "category": "general"
            },
            generation_cost=0.001,
            model_used="demo"
        )
        print(f"  ✅ Added best practice knowledge")
    
    # Get memory stats
    stats = memory.get_memory_analytics()
    print(f"\n📊 Memory seeding complete!")
    print(f"  Total memories: {stats.get('total_memories', 0)}")
    print(f"  BigQuery domain memories: ~{len(scenarios['scenarios']) + len(scenarios['optimization_patterns']) + len(best_practices)}")
    
    return memory

if __name__ == "__main__":
    print("="*60)
    print("🚀 BigQuery Memory Seeding for LinkedIn Demo")
    print("="*60)
    
    asyncio.run(seed_bigquery_patterns())
    
    print("\n✅ Memory seeded successfully!")
    print("\nNext steps:")
    print("1. Run the BigQuery demo: python docs/linkedin/scripts/run_bigquery_demo.py")
    print("2. Visualize the memory: python docs/linkedin/scripts/visualize_memory_network.py")