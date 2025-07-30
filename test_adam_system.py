#!/usr/bin/env python3
"""
Simple test script to check ADAM system functionality
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing ADAM System Components...")
print("="*50)

# Test 1: Environment variables
print("\n1. Checking environment variables:")
from dotenv import load_dotenv
load_dotenv()

keys_found = []
if os.getenv("XAI_API_KEY"):
    keys_found.append("✅ XAI_API_KEY found")
else:
    keys_found.append("❌ XAI_API_KEY missing")
    
if os.getenv("OPENAI_API_KEY"):
    keys_found.append("✅ OPENAI_API_KEY found")
else:
    keys_found.append("❌ OPENAI_API_KEY missing")

for key in keys_found:
    print(f"   {key}")

# Test 2: LLM Config
print("\n2. Testing LLM Configuration:")
try:
    from src.adam.llm.config import LLMConfig
    config = LLMConfig()
    available_models = config.get_available_models()
    print(f"   ✅ LLM Config loaded successfully")
    print(f"   Available models: {len(available_models)}")
    print(f"   Models: {', '.join(available_models[:3])}...")
except Exception as e:
    print(f"   ❌ Error loading LLM Config: {e}")

# Test 3: LLM Client (without full ADAM imports)
print("\n3. Testing LLM Client:")
try:
    # Direct test without importing through ADAM module
    os.chdir("src/adam/llm")
    import client
    llm_client = client.UnifiedLLMClient()
    print(f"   ✅ LLM Client initialized successfully")
    
    # Test a simple API call
    print("\n4. Testing API call to Grok:")
    import asyncio
    
    async def test_api():
        try:
            response = await llm_client.complete(
                prompt="Say 'Hello from ADAM!' in exactly 5 words.",
                model="grok-3-mini-fast",
                max_tokens=20
            )
            return response
        except Exception as e:
            return f"Error: {e}"
    
    result = asyncio.run(test_api())
    if hasattr(result, 'content'):
        print(f"   ✅ API Response: {result.content}")
    else:
        print(f"   ❌ API Error: {result}")
        
except Exception as e:
    print(f"   ❌ Error with LLM Client: {e}")

print("\n" + "="*50)
print("ADAM System Test Complete")