#!/usr/bin/env python3
"""Test core ADAM functionality"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing ADAM Core Components...")
print("=" * 50)

# Test 1: Environment Variables
print("\n1. Environment Variables:")
print(f"   OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Not set'}")
print(f"   XAI_API_KEY: {'✅ Set' if os.getenv('XAI_API_KEY') else '❌ Not set'}")

# Test 2: Import core modules (avoiding langgraph)
print("\n2. Testing Core Imports:")
try:
    from adam.llm.config import LLMConfig
    print("   ✅ LLM Config imported")
    
    config = LLMConfig()
    print(f"   ✅ Available models: {len(config.get_available_models())}")
except Exception as e:
    print(f"   ❌ LLM Config error: {e}")

try:
    from adam.llm.client import UnifiedLLMClient
    print("   ✅ LLM Client imported")
except Exception as e:
    print(f"   ❌ LLM Client error: {e}")

try:
    from adam.memory import Memory
    print("   ✅ Memory system imported")
except Exception as e:
    print(f"   ❌ Memory error: {e}")

try:
    from adam.llm.query_analyzer import QueryAnalyzer
    print("   ✅ Query Analyzer imported")
except Exception as e:
    print(f"   ❌ Query Analyzer error: {e}")

# Test 3: Test basic functionality
print("\n3. Testing Basic Functionality:")
try:
    from adam.llm.client import UnifiedLLMClient
    client = UnifiedLLMClient()
    print("   ✅ LLM Client initialized")
    
    # Test a simple completion
    result = client.complete_sync(
        prompt="Say 'Hello ADAM test'",
        model="grok-3-mini-fast",
        max_tokens=10
    )
    print(f"   ✅ Test completion: {result.content[:50]}...")
except Exception as e:
    print(f"   ❌ Functionality test error: {e}")

print("\n" + "=" * 50)
print("ADAM Core Test Complete!")