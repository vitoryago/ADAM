#!/usr/bin/env python3
"""
Batch Optimization Demo - Multiple queries, rapid results
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.adam.memory import ADAMMemoryAdvanced

def main():
    memory = ADAMMemoryAdvanced()
    
    queries = [
        {
            "name": "Customer Analytics Query",
            "before": {"runtime": 85, "cost": 2.10},
            "after": {"runtime": 5, "cost": 0.12},
            "fix": "Added clustering on customer_id"
        },
        {
            "name": "Product Performance Report", 
            "before": {"runtime": 120, "cost": 3.50},
            "after": {"runtime": 8, "cost": 0.23},
            "fix": "Materialized view for aggregations"
        },
        {
            "name": "Daily Revenue Dashboard",
            "before": {"runtime": 45, "cost": 1.35},
            "after": {"runtime": 3, "cost": 0.09},
            "fix": "Partition pruning on date column"
        }
    ]
    
    print("ADAM BATCH OPTIMIZATION RESULTS")
    print("="*70)
    print()
    
    total_before_time = 0
    total_after_time = 0
    total_before_cost = 0
    total_after_cost = 0
    
    for query in queries:
        print(f"{query['name']}:")
        print(f"  Before: {query['before']['runtime']}s (${query['before']['cost']})")
        print(f"  After:  {query['after']['runtime']}s (${query['after']['cost']})")
        print(f"  Fix:    {query['fix']}")
        print()
        
        total_before_time += query['before']['runtime']
        total_after_time += query['after']['runtime']
        total_before_cost += query['before']['cost']
        total_after_cost += query['after']['cost']
    
    print("-"*70)
    print("TOTAL IMPACT:")
    print(f"  Runtime: {total_before_time}s → {total_after_time}s (↓{int((1-total_after_time/total_before_time)*100)}%)")
    print(f"  Cost:    ${total_before_cost:.2f} → ${total_after_cost:.2f} (↓{int((1-total_after_cost/total_before_cost)*100)}%)")
    print()
    print(f"Memories consulted: {len(memory.recall_with_context('BigQuery', n_results=10))}")

if __name__ == "__main__":
    main()