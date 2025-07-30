#!/usr/bin/env python3
"""
Fix common backend issues for ADAM v2
"""
import os
import sys
from pathlib import Path

def check_and_fix_backend():
    print("🔧 Checking ADAM v2 Backend...")
    
    # 1. Check environment variables
    print("\n1. Checking API Keys...")
    env_file = Path(".env")
    if not env_file.exists():
        print("   ❌ .env file not found")
        if Path(".env.example").exists():
            print("   📝 Creating .env from .env.example")
            import shutil
            shutil.copy(".env.example", ".env")
            print("   ⚠️  Please edit .env and add your API keys")
    else:
        print("   ✅ .env file exists")
    
    # Load env vars
    from dotenv import load_dotenv
    load_dotenv()
    
    xai_key = os.getenv("XAI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not xai_key or xai_key == "your-xai-api-key":
        print("   ⚠️  XAI_API_KEY not set - Grok models won't work")
    else:
        print("   ✅ XAI_API_KEY is set")
        
    if not openai_key or openai_key == "your-openai-api-key":
        print("   ⚠️  OPENAI_API_KEY not set - OpenAI models won't work")
    else:
        print("   ✅ OPENAI_API_KEY is set")
    
    # 2. Check database directory
    print("\n2. Checking Database...")
    data_dir = Path("./data")
    if not data_dir.exists():
        print("   📁 Creating data directory")
        data_dir.mkdir(exist_ok=True)
    print("   ✅ Data directory exists")
    
    # 3. Check Python path for imports
    print("\n3. Checking Python Path...")
    parent_dir = str(Path(__file__).parent.parent.parent)
    if parent_dir not in sys.path:
        print(f"   📝 Adding {parent_dir} to Python path")
        sys.path.insert(0, parent_dir)
    print("   ✅ Python path configured")
    
    # 4. Check dependencies
    print("\n4. Checking Dependencies...")
    missing_deps = []
    
    deps_to_check = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("sqlalchemy", "SQLAlchemy"),
        ("aiosqlite", "Async SQLite"),
        ("chromadb", "ChromaDB (optional)"),
    ]
    
    for module, name in deps_to_check:
        try:
            __import__(module)
            print(f"   ✅ {name} installed")
        except ImportError:
            print(f"   ❌ {name} not installed")
            missing_deps.append(module)
    
    if missing_deps:
        print(f"\n   ⚠️  Install missing dependencies with:")
        print(f"   pip install {' '.join(missing_deps)}")
    
    # 5. Test ADAM imports
    print("\n5. Testing ADAM Module Imports...")
    try:
        from adam.llm.client import UnifiedLLMClient
        print("   ✅ ADAM LLM client can be imported")
    except ImportError as e:
        print(f"   ❌ Cannot import ADAM LLM client: {e}")
        print("   ⚠️  Memory and LLM features will be limited")
    
    # 6. Create test script
    print("\n6. Creating test script...")
    test_script = '''#!/usr/bin/env python3
"""Test ADAM v2 API endpoints"""
import requests
import json

base_url = "http://localhost:8000"

# Test health endpoint
print("Testing health endpoint...")
response = requests.get(f"{base_url}/api/health")
print(f"Health check: {response.json()}")

# Test projects endpoint
print("\\nTesting projects endpoint...")
response = requests.get(f"{base_url}/api/projects/")
print(f"Projects: {len(response.json())} found")
'''
    
    with open("test_api.py", "w") as f:
        f.write(test_script)
    os.chmod("test_api.py", 0o755)
    print("   ✅ Created test_api.py")
    
    print("\n✅ Backend check complete!")
    print("\nTo start the server:")
    print("  python main.py")
    print("\nTo test the API:")
    print("  python test_api.py")

if __name__ == "__main__":
    check_and_fix_backend()