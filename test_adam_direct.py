#!/usr/bin/env python3
"""Test ADAM functionality without langgraph"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path but don't import adam package directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing ADAM Direct Imports...")
print("=" * 50)

# Test 1: Environment
print("\n1. Environment:")
print(f"   OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Not set'}")
print(f"   XAI_API_KEY: {'✅ Set' if os.getenv('XAI_API_KEY') else '❌ Not set'}")

# Test 2: Import specific modules directly
print("\n2. Direct Module Imports:")
try:
    # Import without going through __init__.py
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/adam'))
    
    # Import config directly
    import llm.config as llm_config
    config = llm_config.LLMConfig()
    print(f"   ✅ LLM Config loaded")
    print(f"   ✅ Available models: {config.get_available_models()}")
except Exception as e:
    print(f"   ❌ Config error: {e}")

# Test 3: Test LLM Client
print("\n3. Testing LLM Client:")
try:
    import llm.client as llm_client
    client = llm_client.UnifiedLLMClient()
    print(f"   ✅ LLM Client created")
    
    # Test with Grok
    result = client.complete_sync(
        prompt="Say 'Hello, ADAM is working!'",
        model="grok-3-mini-fast",
        max_tokens=20
    )
    print(f"   ✅ Response: {result.content}")
except Exception as e:
    print(f"   ❌ LLM Client error: {e}")

# Test 4: Memory System
print("\n4. Testing Memory System:")
try:
    # Import Memory directly, avoiding langgraph
    import memory_config
    import memory
    
    mem_config = memory_config.MemoryConfig()
    print(f"   ✅ Memory config loaded")
    print(f"   ✅ Embedding model: {mem_config.embedding_model}")
    
    # Test memory instance
    mem = memory.Memory()
    print(f"   ✅ Memory instance created")
    print(f"   ✅ Collection name: {mem.collection_name}")
except Exception as e:
    print(f"   ❌ Memory error: {e}")

print("\n" + "=" * 50)
print("ADAM Direct Test Complete!")