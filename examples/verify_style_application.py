#!/usr/bin/env python3
"""
Verify that styles are being applied correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.adam_v2.services.response_style_service import ResponseStyleService, ResponseStyle

def test_style_prompts():
    """Test that style prompts are different"""
    service = ResponseStyleService()
    
    print("Verifying Style System Prompts")
    print("="*60)
    
    styles = [ResponseStyle.FRIENDLY, ResponseStyle.CREATIVE, ResponseStyle.FORMAL]
    
    for style in styles:
        config = service.get_style_config(style)
        print(f"\n{style.value.upper()}:")
        print(f"Temperature: {config.temperature}")
        print(f"System Prompt (first 150 chars):")
        print(f"  {config.system_prompt[:150]}...")
        
        # Test enhancement
        test_prompt = "Hey ADAM"
        enhanced = service.enhance_prompt_for_style(test_prompt, style)
        print(f"Enhanced prompt: {enhanced}")

if __name__ == "__main__":
    test_style_prompts()