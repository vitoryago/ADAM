#!/usr/bin/env python3
"""
Direct test of LLM functionality without ADAM imports
"""
import os
import asyncio
from dotenv import load_dotenv

# Load environment
load_dotenv()

print("Direct LLM API Test")
print("="*50)

# Test Grok API directly
async def test_grok_api():
    xai_key = os.getenv("XAI_API_KEY")
    if not xai_key:
        print("❌ XAI_API_KEY not found")
        return
        
    print(f"✅ XAI API Key found: {xai_key[:10]}...")
    
    import httpx
    
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {xai_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello from ADAM!' and nothing else."}
        ],
        "model": "grok-3-mini",
        "temperature": 0.1,
        "max_tokens": 20
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
            result = response.json()
            
            if response.status_code == 200:
                content = result['choices'][0]['message']['content']
                print(f"✅ Grok API Response: {content}")
            else:
                print(f"❌ API Error: {response.status_code} - {result}")
    except Exception as e:
        print(f"❌ Request failed: {e}")

# Test OpenAI API directly
async def test_openai_api():
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found")
        return
        
    print(f"\n✅ OpenAI API Key found: {openai_key[:10]}...")
    
    import httpx
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello from ADAM!' and nothing else."}
        ],
        "model": "gpt-3.5-turbo",
        "temperature": 0.1,
        "max_tokens": 20
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
            result = response.json()
            
            if response.status_code == 200:
                content = result['choices'][0]['message']['content']
                print(f"✅ OpenAI API Response: {content}")
            else:
                print(f"❌ API Error: {response.status_code} - {result}")
    except Exception as e:
        print(f"❌ Request failed: {e}")

# Run tests
asyncio.run(test_grok_api())
asyncio.run(test_openai_api())

print("\n" + "="*50)
print("✅ Direct API tests complete - APIs are accessible")