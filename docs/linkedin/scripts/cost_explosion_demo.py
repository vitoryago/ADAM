#!/usr/bin/env python3
"""
BigQuery Cost Explosion - ADAM identifies the culprit
"""
import sys
import os
import time
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import io
from contextlib import redirect_stdout, redirect_stderr

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    from src.adam.memory import ADAMMemoryAdvanced
    memory = ADAMMemoryAdvanced()

def main():
    print("💸 URGENT: BigQuery Costs Out of Control")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    print("Alert: Daily spend exceeded $10,000 threshold")
    print("Normal daily spend: $800-1,200")
    print("Current trajectory: $312,000/month (!)")
    print()
    
    time.sleep(2)
    
    print("COST BREAKDOWN (Last 24 hours):")
    print("-"*60)
    print("09:00 AM - $823 ✓")
    print("10:00 AM - $891 ✓")
    print("11:00 AM - $8,234 ⚠️")
    print("12:00 PM - $9,123 ⚠️")
    print("01:00 PM - $8,956 ⚠️")
    print("Current: $7,234/hour")
    print()
    
    time.sleep(1.5)
    
    print("INVESTIGATING WITH ADAM...")
    print("-"*60)
    
    # Search for cost-related issues
    with redirect_stdout(io.StringIO()):
        memories = memory.recall_with_context(
            query="BigQuery sudden cost spike expensive queries",
            n_results=5
        )
    
    print("ADAM: Let me analyze your query logs...")
    time.sleep(1)
    print(f"Found {len(memories)} similar cost explosion patterns")
    print()
    
    print("Scanning INFORMATION_SCHEMA.JOBS...")
    time.sleep(1.5)
    
    print("\n🎯 CULPRIT FOUND!")
    print("-"*60)
    print("Query ID: analytics_dashboard_refresh_v2")
    print("Frequency: Every 5 minutes (was daily!)")
    print("Cost per run: $45")
    print("Daily cost: $12,960")
    print()
    
    print("THE PROBLEMATIC QUERY:")
    print("-"*60)
    print("-- Someone removed the date filter!")
    print("SELECT ")
    print("  customer_id,")
    print("  product_id,")
    print("  SUM(quantity * price) as revenue")
    print("FROM `project.sales.transactions`  -- 5 years of data!")
    print("-- WHERE transaction_date >= DATE_SUB(CURRENT_DATE(), 30)")
    print("GROUP BY customer_id, product_id")
    print()
    
    time.sleep(2)
    
    print("ADAM'S FINDINGS:")
    print("-"*60)
    print("1. Date filter was commented out 3 hours ago")
    print("2. Query now scans 5 years instead of 30 days")
    print("3. Scheduled to run every 5 min (288x/day)")
    print("4. Each run processes 4.5TB of data")
    print()
    
    time.sleep(1)
    
    print("IMMEDIATE ACTIONS:")
    print("-"*60)
    print("✓ Killing active query instances...")
    time.sleep(0.5)
    print("✓ Updating schedule to hourly...")
    time.sleep(0.5)
    print("✓ Restoring date filter...")
    time.sleep(0.5)
    print("✓ Adding cost controls...")
    print()
    
    print("OPTIMIZED QUERY:")
    print("-"*60)
    print("SELECT ")
    print("  customer_id,")
    print("  product_id,")
    print("  SUM(quantity * price) as revenue")
    print("FROM `project.sales.transactions`")
    print("WHERE transaction_date >= DATE_SUB(CURRENT_DATE(), 30)")
    print("  AND _PARTITIONDATE >= DATE_SUB(CURRENT_DATE(), 30)")
    print("GROUP BY customer_id, product_id")
    print()
    
    time.sleep(1.5)
    
    print("COST IMPACT:")
    print("-"*60)
    print("Before: $45 per run × 288 runs = $12,960/day")
    print("After:  $0.52 per run × 24 runs = $12.48/day")
    print("Savings: $12,947.52/day (99.9% reduction)")
    print("Annual savings: $4.7M")
    print()
    
    print("PREVENTIVE MEASURES IMPLEMENTED:")
    print("-"*60)
    print("1. Query cost alerts if > $10/run")
    print("2. Require approval for queries > 1TB")
    print("3. Automated partition filter validation")
    print("4. Daily cost anomaly detection")
    print()
    
    print("✅ Crisis averted. Costs returning to normal.")
    
    # Store this incident
    with redirect_stdout(io.StringIO()):
        memory.remember_if_worthy(
            query="BigQuery cost explosion missing date filter",
            response="Check for removed WHERE clauses and partition filters. Verify query scheduling frequency.",
            context={"incident_type": "cost_explosion", "root_cause": "missing_filter"},
            generation_cost=0.001,
            model_used="demo"
        )

if __name__ == "__main__":
    main()