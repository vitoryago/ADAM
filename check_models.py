#!/usr/bin/env python3
"""
Quick diagnostic script to check which models are available in ADAM
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adam.llm.config import LLMConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("ADAM MODEL CONFIGURATION CHECK")
print("=" * 60)

# Check environment variables
print("\n1. ENVIRONMENT VARIABLES:")
print("-" * 40)
openai_key = os.getenv("OPENAI_API_KEY")
xai_key = os.getenv("XAI_API_KEY")
grok_key = os.getenv("GROK_API_KEY")

print(f"OPENAI_API_KEY: {'✓ Set' if openai_key else '✗ Not set'}")
print(f"XAI_API_KEY: {'✓ Set' if xai_key else '✗ Not set'}")
print(f"GROK_API_KEY: {'✓ Set' if grok_key else '✗ Not set'}")

if openai_key:
    print(f"  -> OpenAI key starts with: {openai_key[:7]}...")
if xai_key:
    print(f"  -> XAI key starts with: {xai_key[:7]}...")
if grok_key:
    print(f"  -> GROK key starts with: {grok_key[:7]}...")

# Initialize config
config = LLMConfig()

print("\n2. CONFIGURED MODELS:")
print("-" * 40)
all_models = list(config.models.keys())
print(f"Total models configured: {len(all_models)}")
for model in all_models:
    model_config = config.models[model]
    print(f"  - {model}: {model_config.provider.value}")

print("\n3. AVAILABLE MODELS (with API keys):")
print("-" * 40)
available = config.get_available_models()
if available:
    print(f"Models ready to use: {len(available)}")
    for model in available:
        model_config = config.models[model]
        print(f"  ✓ {model} ({model_config.provider.value})")
else:
    print("⚠️  No models available! Please set API keys.")

print("\n4. MODEL CATEGORIES:")
print("-" * 40)
gpt5_models = [m for m in available if 'gpt-5' in m]
grok_models = [m for m in available if 'grok' in m]
other_models = [m for m in available if m not in gpt5_models and m not in grok_models and m != 'automatic']

print(f"GPT-5 models: {gpt5_models if gpt5_models else 'None (need OPENAI_API_KEY)'}")
print(f"Grok models: {grok_models if grok_models else 'None (need XAI_API_KEY or GROK_API_KEY)'}")
print(f"Other models: {other_models if other_models else 'None'}")

print("\n5. RECOMMENDATIONS:")
print("-" * 40)
if not openai_key:
    print("⚠️  To use GPT-5 models, add to your .env file:")
    print("    OPENAI_API_KEY=your-openai-api-key-here")
if not xai_key and not grok_key:
    print("⚠️  To use Grok models, add to your .env file:")
    print("    XAI_API_KEY=your-xai-api-key-here")
if openai_key and (xai_key or grok_key):
    print("✓ All API keys configured! All models should be available.")

print("\n" + "=" * 60)