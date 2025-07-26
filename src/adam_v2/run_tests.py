#!/usr/bin/env python3
"""
Test runner for ADAM v2
Run all tests or specific test categories
"""
import sys
import subprocess
from pathlib import Path

def run_tests(test_type="all"):
    """Run tests based on type"""
    base_dir = Path(__file__).parent
    
    # Build pytest command
    cmd = ["pytest", "-v"]
    
    if test_type == "unit":
        cmd.extend(["-m", "unit", "tests/unit/"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration", "tests/integration/"])
    elif test_type == "coverage":
        cmd.extend(["--cov", "--cov-report=html", "--cov-report=term"])
    elif test_type != "all":
        print(f"Unknown test type: {test_type}")
        print("Valid options: all, unit, integration, coverage")
        return 1
    
    # Run tests
    print(f"Running {test_type} tests...")
    result = subprocess.run(cmd, cwd=base_dir)
    
    return result.returncode

if __name__ == "__main__":
    test_type = sys.argv[1] if len(sys.argv) > 1 else "all"
    exit_code = run_tests(test_type)
    sys.exit(exit_code)