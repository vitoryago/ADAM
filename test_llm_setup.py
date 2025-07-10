#!/usr/bin/env python3
"""
Test script to verify LLM configuration and API keys are working
Run this after setting up your environment variables
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file with override
load_dotenv(override=True)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adam.llm.config import LLMConfig, SETUP_INSTRUCTIONS
from adam.llm.client import UnifiedLLMClient, quick_complete, reasoning_complete

async def test_llm_setup():
    """Test the LLM configuration and connections"""
    print("🚀 Testing ADAM LLM Configuration...\n")
    
    # Check configuration
    config = LLMConfig()
    
    print("📋 Configured Models:")
    for model_name, model_config in config.models.items():
        print(f"  - {model_name}: {model_config.provider.value} ({model_config.api_name})")
    
    print("\n🔑 API Key Status:")
    print(f"  - XAI_API_KEY: {'✅ Set' if config.get_api_key(config.models['grok-4'].provider) else '❌ Not set'}")
    print(f"  - OPENAI_API_KEY: {'✅ Set' if config.get_api_key(config.models['o4-mini-high'].provider) else '❌ Not set'}")
    
    available_models = config.get_available_models()
    print(f"\n📡 Available Models (with API keys): {available_models}")
    
    if not available_models:
        print("\n❌ No API keys found!")
        print(SETUP_INSTRUCTIONS)
        return
    
    # Test each available model
    print("\n🧪 Testing Each Available Model...")
    
    client = UnifiedLLMClient(config)
    
    for model in available_models:
        print(f"\n📝 Testing {model}...")
        try:
            # Simple test
            response = await client.complete(
                prompt="What is 2+2? Answer with just the number.",
                model=model,
                temperature=0
            )
            print(f"  ✅ Response: {response.content}")
            print(f"  📊 Tokens: {response.total_tokens} (cost: ${response.cost:.4f})")
            
            # Test reasoning if supported
            model_config = config.get_model_config(model)
            if model_config.supports_reasoning:
                print(f"  🧠 Testing reasoning capabilities...")
                response = await client.complete(
                    prompt="Why is the sky blue? Think step by step.",
                    model=model,
                    reasoning_effort="low" if model == "grok-3-mini" else "medium"
                )
                print(f"  ✅ Reasoning test passed")
                if response.reasoning_content:
                    print(f"  💭 Has reasoning content: {len(response.reasoning_content)} chars")
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    # Test auto-selection
    print("\n🤖 Testing Auto Model Selection...")
    
    test_queries = [
        ("What's the weather?", "fast query"),
        ("Explain quantum computing in detail", "complex query"),
        ("SELECT * FROM users WHERE age > 18", "SQL query"),
        ("Debug this error: undefined is not a function", "debugging query")
    ]
    
    for query, query_type in test_queries:
        try:
            response = await client.complete(query, temperature=0)
            print(f"  {query_type}: Selected {response.model}")
        except Exception as e:
            print(f"  {query_type}: Failed - {str(e)}")
    
    # Test convenience functions
    print("\n⚡ Testing Convenience Functions...")
    
    try:
        # Quick complete
        result = await quick_complete("What is the capital of France?")
        print(f"  ✅ quick_complete: {result[:50]}...")
        
        # Reasoning complete (if available)
        if any("o4-mini" in m or "grok-3-mini" in m for m in available_models):
            result = await reasoning_complete("Why do objects fall?", effort="low")
            print(f"  ✅ reasoning_complete: Got answer with {result['tokens']['total']} tokens")
    except Exception as e:
        print(f"  ❌ Convenience functions: {str(e)}")
    
    print("\n✨ LLM Configuration Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_llm_setup())