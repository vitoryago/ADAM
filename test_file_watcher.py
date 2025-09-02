#!/usr/bin/env python3
"""
Test script for ADAM file watcher functionality
"""

import requests
import json
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
PROJECT_ID = "b2e0d55b-6a13-4dcf-839f-a96d88bff792"  # VSCode project ID
TEST_DIR = "/Users/vitoryago/ADAM/test_watch_dir"

def main():
    print("🧪 Testing ADAM File Watcher")
    print("=" * 50)
    
    # 1. Create test directory
    test_path = Path(TEST_DIR)
    test_path.mkdir(exist_ok=True)
    print(f"✅ Created test directory: {TEST_DIR}")
    
    # 2. Start watching
    print(f"\n📡 Starting file watcher for project {PROJECT_ID[:8]}...")
    response = requests.post(
        f"{BASE_URL}/api/projects/{PROJECT_ID}/watch",
        json={
            "directory": TEST_DIR,
            "ignored_patterns": [".git", "__pycache__", "*.pyc"]
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Watcher started: {result['message']}")
        print(f"   Status: {json.dumps(result['status'], indent=2)}")
    else:
        print(f"❌ Failed to start watcher: {response.status_code}")
        print(f"   {response.text}")
        return
    
    # 3. Create test files
    print("\n📝 Creating test files...")
    
    # Create Python file
    test_py = test_path / "example.py"
    test_py.write_text("""
def hello_world():
    '''Simple hello world function'''
    print("Hello from ADAM file watcher!")
    return "Hello, World!"

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process(self, item):
        self.data.append(item)
        return len(self.data)
""")
    print(f"✅ Created: {test_py}")
    
    # Create JavaScript file
    test_js = test_path / "app.js"
    test_js.write_text("""
function calculateSum(numbers) {
    return numbers.reduce((a, b) => a + b, 0);
}

class UserManager {
    constructor() {
        this.users = [];
    }
    
    addUser(user) {
        this.users.push(user);
        return this.users.length;
    }
}

export { calculateSum, UserManager };
""")
    print(f"✅ Created: {test_js}")
    
    # Wait for processing
    print("\n⏳ Waiting for files to be indexed...")
    time.sleep(3)
    
    # 4. Check status
    print("\n📊 Checking watcher status...")
    response = requests.get(f"{BASE_URL}/api/projects/{PROJECT_ID}/watch/status")
    if response.status_code == 200:
        status = response.json()
        print(f"✅ Watcher status:")
        print(f"   Watching: {status.get('watching', False)}")
        print(f"   Files tracked: {status.get('files_tracked', 0)}")
        print(f"   Pending changes: {status.get('pending_changes', 0)}")
    
    # 5. Modify a file
    print("\n✏️ Modifying example.py...")
    test_py.write_text(test_py.read_text() + """

def new_feature():
    '''Added by file watcher test'''
    return "This function was added during testing"
""")
    print("✅ File modified")
    
    # Wait for processing
    time.sleep(3)
    
    # 6. Check status again
    print("\n📊 Checking updated status...")
    response = requests.get(f"{BASE_URL}/api/projects/{PROJECT_ID}/watch/status")
    if response.status_code == 200:
        status = response.json()
        print(f"✅ Updated status:")
        print(f"   Files tracked: {status.get('files_tracked', 0)}")
    
    # 7. Stop watching
    print("\n🛑 Stopping file watcher...")
    response = requests.delete(f"{BASE_URL}/api/projects/{PROJECT_ID}/watch")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ {result['message']}")
    
    print("\n✨ File watcher test complete!")

if __name__ == "__main__":
    main()