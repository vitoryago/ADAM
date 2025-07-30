#!/usr/bin/env python3
"""
Simple test to verify API keys and configuration
"""
import os
from dotenv import load_dotenv

# Load environment from .env file
load_dotenv()

def test_api_keys():
    """Test if API keys are properly loaded"""
    print("🧪 Testing API Key Configuration...")
    
    # Check XAI (Grok) API key
    xai_key = os.getenv('XAI_API_KEY')
    if xai_key:
        print(f"✅ XAI API Key: Found ({xai_key[:10]}...{xai_key[-4:]})")
    else:
        print("❌ XAI API Key: Not found")
    
    # Check OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print(f"✅ OpenAI API Key: Found ({openai_key[:10]}...{openai_key[-4:]})")
    else:
        print("❌ OpenAI API Key: Not found")
    
    # Check ADAM configuration
    adam_name = os.getenv('ADAM_NAME', 'ADAM')
    adam_model = os.getenv('ADAM_EMBEDDING_MODEL', 'all-mpnet-base-v2')
    adam_language = os.getenv('ADAM_LANGUAGE', 'en')
    
    print(f"✅ ADAM Name: {adam_name}")
    print(f"✅ ADAM Embedding Model: {adam_model}")
    print(f"✅ ADAM Language: {adam_language}")
    
    return bool(xai_key and openai_key)

def test_basic_imports():
    """Test if basic modules can be imported"""
    print("\n🧪 Testing Basic Imports...")
    
    try:
        import json
        print("✅ json module: OK")
        
        import sys
        print("✅ sys module: OK")
        
        import os
        print("✅ os module: OK")
        
        from pathlib import Path
        print("✅ pathlib module: OK")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("ADAM Backend Configuration Test")
    print("=" * 50)
    
    api_test = test_api_keys()
    import_test = test_basic_imports()
    
    print("\n" + "=" * 50)
    if api_test and import_test:
        print("✅ All tests passed! Backend ready to use.")
    else:
        print("❌ Some tests failed. Check configuration.")
    print("=" * 50)