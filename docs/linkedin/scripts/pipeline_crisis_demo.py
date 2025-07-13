#!/usr/bin/env python3
"""
Data Pipeline Crisis - ADAM saves the day
"""
import sys
import os
import time
import random
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import io
from contextlib import redirect_stdout, redirect_stderr

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    from src.adam.memory import ADAMMemoryAdvanced
    memory = ADAMMemoryAdvanced()

def print_progress(message, steps=3):
    """Show progress dots"""
    print(message, end="", flush=True)
    for _ in range(steps):
        time.sleep(0.3)
        print(".", end="", flush=True)
    print(" ✓")

def main():
    print("🚨 CRITICAL: Data Pipeline Failure")
    print("="*60)
    print("System: Revenue Analytics Pipeline")
    print("Status: Failed 3 times in last hour")
    print("Error: 'Resources Exceeded during query execution'")
    print("Impact: $2.3M daily revenue reporting blocked")
    print()
    
    time.sleep(2)
    
    print("LAST SUCCESSFUL RUN: Yesterday 11:45 PM")
    print("FIRST FAILURE: Today 12:15 AM")
    print("Change: Data volume increased 10x overnight(?)")
    print()
    
    time.sleep(1.5)
    
    print_progress("Checking data volumes")
    print("Found: Black Friday sale started at midnight!")
    print("Order volume: 850K → 8.5M records/hour")
    print()
    
    time.sleep(1)
    
    print("FAILING QUERY:")
    print("-"*60)
    print("INSERT INTO revenue_summary")
    print("SELECT ")
    print("  DATE(order_time) as date,")
    print("  product_category,")
    print("  country,")
    print("  COUNT(*) as orders,")
    print("  SUM(amount) as revenue,")
    print("  AVG(amount) as avg_order_value")
    print("FROM orders o")
    print("JOIN products p ON o.product_id = p.id")
    print("JOIN customers c ON o.customer_id = c.id")
    print("WHERE order_time >= DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 24 HOUR)")
    print("GROUP BY date, product_category, country")
    print()
    
    time.sleep(2)
    
    print("ADAM ANALYZING FAILURE PATTERN...")
    print("-"*60)
    
    with redirect_stdout(io.StringIO()):
        memories = memory.recall_with_context(
            query="BigQuery resources exceeded Black Friday scale spike",
            n_results=7
        )
    
    print(f"Found {len(memories)} similar incidents in memory")
    print()
    
    print_progress("Analyzing query execution plan")
    print_progress("Checking slot utilization")
    print_progress("Identifying bottlenecks")
    print()
    
    print("ROOT CAUSE IDENTIFIED:")
    print("- JOIN explosion: 8.5M × 500K × 2M records")
    print("- No partition filter on massive tables")
    print("- Scanning entire customer base for 24h window")
    print()
    
    time.sleep(1.5)
    
    print("ADAM'S EMERGENCY FIX:")
    print("-"*60)
    print("1. Add partition filters to all tables")
    print("2. Process in hourly micro-batches")
    print("3. Use temporary tables for large JOINs")
    print("4. Enable query result caching")
    print()
    
    print("IMPLEMENTING FIX...")
    time.sleep(1)
    
    print("\nMODIFIED QUERY (Batch 1 of 24):")
    print("-"*60)
    print("-- Process hour by hour to avoid resource limits")
    print("DECLARE batch_hour DATETIME DEFAULT '2024-11-24 00:00:00';")
    print()
    print("CREATE TEMP TABLE hourly_orders AS")
    print("SELECT * FROM orders")
    print("WHERE order_time >= batch_hour")
    print("  AND order_time < DATETIME_ADD(batch_hour, INTERVAL 1 HOUR)")
    print("  AND _PARTITIONTIME = TIMESTAMP(DATE(batch_hour));")
    print()
    print("-- Continue with filtered dataset...")
    print()
    
    time.sleep(2)
    
    print("TESTING FIRST BATCH...")
    print("-"*60)
    print_progress("Executing")
    print("Batch 1/24: SUCCESS (12 seconds)")
    print("Resources used: 4% of limit")
    print()
    
    print("DEPLOYING FULL SOLUTION...")
    print_progress("Processing 24 hourly batches", steps=5)
    print()
    
    print("✅ PIPELINE RESTORED")
    print("-"*60)
    print("Total execution time: 4.8 minutes")
    print("Cost: $45 (was $450 before optimization)")
    print("All revenue data now available")
    print()
    
    print("LESSONS LEARNED (stored in ADAM):")
    print("- Holiday spikes need different query patterns")
    print("- Micro-batching prevents resource exhaustion")
    print("- Partition filters are critical at scale")

if __name__ == "__main__":
    main()