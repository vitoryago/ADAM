#!/usr/bin/env python3
"""
Test Grok Vision Model Implementation
=====================================

This script tests the Grok vision model implementation with actual image data.
"""
import asyncio
import sys
import os
from pathlib import Path
import base64
from PIL import Image
import io

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.llm.client import UnifiedLLMClient
from src.adam.llm.config import LLMConfig
from rich.console import Console

console = Console()


def create_test_image():
    """Create a simple test image"""
    # Create a simple image with text
    img = Image.new('RGB', (400, 200), color='white')
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    # Draw some text
    text = "ADAM Vision Test"
    # Use default font
    draw.text((100, 80), text, fill='black')
    
    # Draw a simple shape
    draw.rectangle([50, 50, 150, 150], outline='red', width=3)
    draw.ellipse([250, 50, 350, 150], outline='blue', width=3)
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return img_byte_arr


async def test_grok_vision():
    """Test Grok vision models with an image"""
    
    console.print("\n🖼️  Testing Grok Vision Implementation\n", style="bold blue")
    
    # Initialize client
    config = LLMConfig()
    client = UnifiedLLMClient(config)
    
    # Get available vision models
    vision_models = [
        m for m in config.get_available_models()
        if config.get_model_config(m).supports_vision and "grok" in m
    ]
    
    if not vision_models:
        console.print("❌ No Grok vision models available", style="red")
        return
    
    console.print(f"Found {len(vision_models)} Grok vision models: {', '.join(vision_models)}")
    
    # Create test image
    console.print("\n📸 Creating test image...")
    image_data = create_test_image()
    console.print(f"Image size: {len(image_data)} bytes")
    
    # Test with each vision model
    for model in vision_models[:1]:  # Test with first model only
        console.print(f"\n🧪 Testing {model}...")
        
        try:
            response = await client.complete(
                prompt="Describe what you see in this image. What shapes and text are visible?",
                model=model,
                image_data=image_data,
                max_tokens=200
            )
            
            console.print(f"\n✅ {model} Response:")
            console.print(response.content, style="green")
            
            # Show token usage if available
            if hasattr(response, 'raw_response') and response.raw_response:
                image_tokens = response.raw_response.get('prompt_image_tokens', 0)
                if image_tokens > 0:
                    console.print(f"\n📊 Image tokens used: {image_tokens}")
            
            console.print(f"💰 Cost: ${response.cost:.4f}")
            
        except Exception as e:
            console.print(f"❌ Error with {model}: {str(e)}", style="red")
            import traceback
            traceback.print_exc()
    
    # Test with a URL (using NASA image from docs)
    console.print("\n\n🌐 Testing with web URL...")
    try:
        # Note: This won't work with our current implementation
        # as we only support base64 images, not URLs
        console.print("ℹ️  URL support would require additional implementation")
    except Exception as e:
        console.print(f"Expected: {str(e)}", style="yellow")


async def test_image_encoding():
    """Test the image encoding matches Grok's format"""
    console.print("\n🔧 Testing Image Encoding Format\n", style="bold blue")
    
    # Create test image
    image_data = create_test_image()
    
    # Encode as base64 (matching our implementation)
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    # Show format
    console.print("Expected format for Grok:")
    console.print(f"data:image/jpeg;base64,{base64_image[:50]}...")
    
    console.print(f"\nBase64 length: {len(base64_image)} characters")
    console.print(f"Estimated tokens: ~{len(image_data) // 1000 * 256} tokens")


if __name__ == "__main__":
    # Check if PIL is installed
    try:
        import PIL
    except ImportError:
        console.print("❌ Please install Pillow: pip install Pillow", style="red")
        sys.exit(1)
    
    asyncio.run(test_grok_vision())
    asyncio.run(test_image_encoding())