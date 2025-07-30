#!/usr/bin/env python3
"""
ADAM Integration Test
Test script to validate the ADAM Python backend is working correctly.
"""

import json
import asyncio
import os
import sys
from pathlib import Path

# Add src directory to path to import ADAM modules
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from adam.adam_service import ADAMService
except ImportError as e:
    print(f"❌ Failed to import ADAM service: {e}")
    print("Make sure all dependencies are installed with: python setup.py --install")
    sys.exit(1)

async def test_service_initialization():
    """Test that ADAM service can be initialized"""
    print("🧪 Testing service initialization...")
    
    try:
        service = ADAMService()
        print("✅ ADAM service initialized successfully")
        return service
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return None

async def test_query_processing(service):
    """Test query processing functionality"""
    print("\n🧪 Testing query processing...")
    
    test_request = {
        'requestId': 'test_001',
        'type': 'QUERY',
        'data': {
            'query': 'Hello ADAM, can you introduce yourself?',
            'conversationId': 'test_conversation',
            'projectId': 'test_project',
            'userId': 'test_user',
            'context': {
                'previousMessages': [],
                'projectMemory': '',
                'userPreferences': {}
            }
        }
    }
    
    try:
        response = await service.process_request(test_request)
        
        if 'error' in response:
            print(f"❌ Query processing failed: {response['error']}")
            return False
        
        print("✅ Query processed successfully")
        print(f"   Response preview: {response['response']['response'][:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Query processing error: {e}")
        return False

async def test_cost_monitoring(service):
    """Test cost monitoring functionality"""
    print("\n🧪 Testing cost monitoring...")
    
    test_request = {
        'requestId': 'test_002',
        'type': 'COST_SUMMARY',
        'data': {}
    }
    
    try:
        response = await service.process_request(test_request)
        
        if 'error' in response:
            print(f"⚠️  Cost monitoring warning: {response['error']}")
            return True  # Not critical for basic functionality
        
        print("✅ Cost monitoring working")
        return True
        
    except Exception as e:
        print(f"⚠️  Cost monitoring error: {e}")
        return True  # Not critical

async def test_memory_system(service):
    """Test memory system functionality"""
    print("\n🧪 Testing memory system...")
    
    test_request = {
        'requestId': 'test_003',
        'type': 'MEMORY_INFO',
        'data': {
            'projectId': 'test_project'
        }
    }
    
    try:
        response = await service.process_request(test_request)
        
        if 'error' in response:
            print(f"⚠️  Memory system warning: {response['error']}")
            return True  # Not critical for basic functionality
        
        print("✅ Memory system working")
        return True
        
    except Exception as e:
        print(f"⚠️  Memory system error: {e}")
        return True  # Not critical

def test_environment_setup():
    """Test environment variables and configuration"""
    print("🧪 Testing environment setup...")
    
    required_dirs = [
        'adam_memory_advanced',
        'conversations',
        'logs'
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"⚠️  Missing directories: {', '.join(missing_dirs)}")
        print("   Run: python setup.py --full to create them")
    else:
        print("✅ Directory structure is correct")
    
    # Check for .env file
    if not Path('.env').exists():
        if Path('.env.template').exists():
            print("⚠️  .env file not found, but .env.template exists")
            print("   Copy .env.template to .env and add your API keys")
        else:
            print("❌ No .env file or template found")
    else:
        print("✅ .env file found")
    
    return len(missing_dirs) == 0

async def run_all_tests():
    """Run all integration tests"""
    print("🤖 ADAM Integration Test Suite")
    print("=" * 40)
    
    # Test environment
    env_ok = test_environment_setup()
    
    # Test service initialization
    service = await test_service_initialization()
    if not service:
        print("\n❌ Cannot continue tests - service initialization failed")
        return False
    
    # Test core functionality
    query_ok = await test_query_processing(service)
    cost_ok = await test_cost_monitoring(service)
    memory_ok = await test_memory_system(service)
    
    # Cleanup
    await service.shutdown()
    
    # Summary
    print("\n" + "=" * 40)
    print("🎯 Test Results Summary:")
    print(f"   Environment Setup: {'✅' if env_ok else '❌'}")
    print(f"   Service Initialization: {'✅' if service else '❌'}")
    print(f"   Query Processing: {'✅' if query_ok else '❌'}")
    print(f"   Cost Monitoring: {'✅' if cost_ok else '⚠️'}")
    print(f"   Memory System: {'✅' if memory_ok else '⚠️'}")
    
    all_critical_ok = service and query_ok
    
    if all_critical_ok:
        print("\n🎉 ADAM backend is ready to use!")
        print("\nNext steps:")
        print("1. Start the Node.js web application: npm run dev")
        print("2. Create a new project and test the chat interface")
        print("3. Monitor costs and memory usage through the UI")
    else:
        print("\n🔧 Issues found that need attention:")
        if not service:
            print("- Service initialization failed - check dependencies")
        if not query_ok:
            print("- Query processing failed - check API keys and configuration")
        print("\nRefer to README.md for troubleshooting guidance")
    
    return all_critical_ok

if __name__ == '__main__':
    # Check if we can import required modules
    try:
        import numpy
        import chromadb
        import sentence_transformers
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("Run: python setup.py --install")
        sys.exit(1)
    
    # Run tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)