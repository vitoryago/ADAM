#!/usr/bin/env python3
"""
Test Vision Model Support in ADAM
=================================

This script verifies that vision models are properly configured
and can handle image inputs.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.llm.client import UnifiedLLMClient
from src.adam.llm.config import LLMConfig
from rich.console import Console
from rich.table import Table

console = Console()


async def test_vision_models():
    """Test which models support vision"""
    
    console.print("\n🔍 Testing Vision Model Support in ADAM\n", style="bold blue")
    
    # Initialize config and client
    config = LLMConfig()
    client = UnifiedLLMClient(config)
    
    # Create a table for results
    table = Table(title="Vision Model Support")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="yellow")
    table.add_column("Vision Support", style="green")
    table.add_column("Status", style="magenta")
    
    # Test each available model
    available_models = config.get_available_models()
    
    for model_name in available_models:
        model_config = config.get_model_config(model_name)
        
        if model_config:
            provider = model_config.provider.value
            supports_vision = "✅ Yes" if model_config.supports_vision else "❌ No"
            
            # Check if model is actually available
            if model_name in available_models:
                status = "🟢 Available"
            else:
                status = "🔴 Not Available"
            
            table.add_row(model_name, provider, supports_vision, status)
    
    console.print(table)
    
    # Test a simple query with a vision model
    vision_models = [m for m in available_models 
                    if config.get_model_config(m).supports_vision]
    
    if vision_models:
        console.print(f"\n🖼️  Found {len(vision_models)} vision-capable models: {', '.join(vision_models)}")
        
        # Test with fake image data
        test_model = vision_models[0]
        console.print(f"\n🧪 Testing {test_model} with simulated image...")
        
        try:
            # Create fake image data (just for testing config)
            fake_image = b"fake_image_data"
            
            response = await client.complete(
                prompt="What's in this image?",
                model=test_model,
                image_data=fake_image,
                max_tokens=50
            )
            
            if hasattr(response, 'content'):
                console.print(f"✅ {test_model} accepted image input (though actual processing may not work with fake data)")
            else:
                console.print(f"✅ {test_model} configuration supports images")
                
        except Exception as e:
            console.print(f"⚠️  Error testing {test_model}: {str(e)}")
    else:
        console.print("\n❌ No vision-capable models found in available models")
    
    # Show configuration summary
    console.print("\n📊 Configuration Summary:")
    console.print(f"- Total models configured: {len(config.models)}")
    console.print(f"- Available models (with API keys): {len(available_models)}")
    console.print(f"- Vision-capable models: {len(vision_models)}")


if __name__ == "__main__":
    asyncio.run(test_vision_models())