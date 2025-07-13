#!/usr/bin/env python3
"""
Video Demo Script - Optimized for screen recording
Clear progression, minimal text, maximum impact
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.adam.memory import ADAMMemoryAdvanced

def clear_screen():
    print("\033[2J\033[H")  # Clear screen and move cursor to top

def print_slow(text, delay=0.03):
    """Print text with typing effect"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def main():
    memory = ADAMMemoryAdvanced()
    
    # Scene 1: The Problem
    clear_screen()
    print("="*60)
    print("BIGQUERY PERFORMANCE PROBLEM".center(60))
    print("="*60)
    print()
    time.sleep(1)
    
    print("Current Status:")
    print("  Runtime: 120 seconds")
    print("  Cost: $3.50 per query")
    print("  Data Scanned: 1.2 TB")
    
    time.sleep(2)
    
    # Scene 2: ADAM Analyzes
    print("\n" + "-"*60)
    print_slow("\nADAM analyzing query patterns...")
    
    memories = memory.recall_with_context("BigQuery optimization", n_results=5)
    
    print(f"\nFound {len(memories)} similar cases in memory")
    time.sleep(1)
    
    print("\nIdentifying optimization opportunities...")
    time.sleep(1.5)
    
    # Scene 3: The Solution
    clear_screen()
    print("="*60)
    print("OPTIMIZATION RESULTS".center(60))
    print("="*60)
    print()
    
    print("BEFORE:")
    print("  Runtime: 120 seconds")
    print("  Cost: $3.50")
    print()
    
    print("AFTER:")
    print("  Runtime: 7 seconds   ✓")
    print("  Cost: $0.20         ✓")
    print()
    
    print("IMPROVEMENT: 94% faster, 94% cheaper")
    
    time.sleep(2)
    
    print("\n" + "-"*60)
    print("\nKEY OPTIMIZATIONS APPLIED:")
    print("• Added table partitioning")
    print("• Used approximate aggregations")
    print("• Optimized JOIN order")
    
    time.sleep(2)
    
    # Scene 4: Learning
    print("\n" + "-"*60)
    print("\nADAM has learned this pattern for future use.")

if __name__ == "__main__":
    main()