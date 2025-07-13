#!/usr/bin/env python3
"""
Quick test to verify LinkedIn demo components work
"""
import sys
from pathlib import Path
import json

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_data_files():
    """Test that data files exist and are valid"""
    print("📁 Testing data files...")
    
    data_dir = Path(__file__).parent / "data"
    files = ["bigquery_scenarios.json", "react_scenarios.json"]
    
    for file in files:
        filepath = data_dir / file
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
            print(f"  ✅ {file} - Valid JSON with {len(data.get('scenarios', []))} scenarios")
        else:
            print(f"  ❌ {file} - File not found!")
            return False
    
    return True

def test_imports():
    """Test that all required imports work"""
    print("\n📦 Testing imports...")
    
    try:
        from src.adam.memory import ADAMMemoryAdvanced
        print("  ✅ ADAM Memory system")
    except Exception as e:
        print(f"  ❌ ADAM Memory system: {e}")
        return False
    
    try:
        from src.adam.llm.client import UnifiedLLMClient
        from src.adam.llm.config import LLMConfig
        print("  ✅ LLM components")
    except Exception as e:
        print(f"  ❌ LLM components: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        import seaborn as sns
        print("  ✅ Visualization libraries")
    except Exception as e:
        print(f"  ❌ Visualization libraries: {e}")
        print("     Install with: pip install matplotlib networkx seaborn")
        return False
    
    return True

def test_directories():
    """Test that output directories exist"""
    print("\n📂 Testing directories...")
    
    dirs = ["scripts", "data", "outputs", "images"]
    base_dir = Path(__file__).parent
    
    for dir_name in dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}/ exists")
        else:
            dir_path.mkdir(exist_ok=True)
            print(f"  ✅ {dir_name}/ created")
    
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("🧪 LinkedIn Demo Test Suite")
    print("="*60)
    
    all_good = True
    
    # Run tests
    all_good &= test_data_files()
    all_good &= test_imports()
    all_good &= test_directories()
    
    print("\n" + "="*60)
    if all_good:
        print("✅ All tests passed! Demo is ready to run.")
        print("\nNext steps:")
        print("1. Run: python scripts/seed_bigquery_memory.py")
        print("2. Run: python scripts/run_bigquery_demo.py")
        print("3. Run: python scripts/react_demonstration.py")
        print("4. Run: python scripts/visualize_memory_network.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    print("="*60)

if __name__ == "__main__":
    main()