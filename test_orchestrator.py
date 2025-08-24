#!/usr/bin/env python3
"""
Test the orchestrator with different folder requests to ensure no hardcoding
"""

import asyncio
import sys
import os
sys.path.insert(0, '/Users/vitoryago/ADAM/src')

from adam_v2.agents.orchestrator import get_orchestrator

async def test_folder_request(folder_path: str, description: str):
    """Test orchestrator with a specific folder request"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Folder: {folder_path}")
    print('='*60)
    
    orchestrator = get_orchestrator()
    
    # Create a request that mentions the folder
    request = f"Show me what's in the {folder_path} folder"
    
    print(f"Request: {request}")
    print("Processing...")
    
    result = await orchestrator.process_request(
        user_message=request,
        workspace_path=folder_path
    )
    
    print(f"\nStatus: {result['status']}")
    print(f"Execution time: {result['execution_time']:.2f} seconds")
    print(f"Tasks executed: {result.get('tasks_executed', 0)}")
    
    if result['status'] == 'success':
        print("\n✅ Success! Response:")
        print(result['response'][:500])  # First 500 chars
    else:
        print(f"\n❌ Failed: {result.get('error', 'Unknown error')}")
    
    return result['status'] == 'success'

async def main():
    """Run tests with different folders"""
    test_cases = [
        ("/Users/vitoryago/ADAM/src", "ADAM source folder"),
        ("/Users/vitoryago/ADAM/vscode-extension", "VSCode extension folder"),
        ("/Users/vitoryago", "Home directory"),
    ]
    
    results = []
    for folder, description in test_cases:
        if os.path.exists(folder):
            success = await test_folder_request(folder, description)
            results.append((description, success))
        else:
            print(f"⚠️ Skipping {description} - folder doesn't exist")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    for desc, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{desc}: {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 All tests passed! No hardcoding detected.")
    else:
        print("\n⚠️ Some tests failed. Check the implementation.")

if __name__ == "__main__":
    asyncio.run(main())