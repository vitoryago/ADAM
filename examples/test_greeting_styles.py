#!/usr/bin/env python3
"""
Test greeting responses with different styles
"""

import requests
import json

PROJECT_ID = "95c0982e-2d4f-4f01-9bbc-eab38ef2523e"
BASE_URL = "http://localhost:8000"

def test_greeting_style(style: str, query: str):
    """Test a greeting with specific style"""
    # Create conversation
    conv_resp = requests.post(
        f"{BASE_URL}/api/projects/{PROJECT_ID}/conversations",
        json={"title": f"Greeting - {style}"}
    )
    conv_id = conv_resp.json()["id"]
    
    # Send message
    response = requests.post(
        f"{BASE_URL}/api/conversations/{conv_id}/messages",
        json={
            "content": query,
            "use_memory": False,
            "model": "gpt-4.1-mini-2025-04-14",
            "response_style": style
        }
    )
    
    messages = response.json()
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""

def main():
    print("Testing Greeting Style Responses")
    print("="*60)
    
    queries = ["Hey ADAM", "Hey ADAM, how have you been doing?"]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-"*40)
        
        styles = ["normal", "friendly", "creative", "formal", "concise"]
        responses = {}
        
        for style in styles:
            response = test_greeting_style(style, query)
            responses[style] = response
            print(f"\n{style.upper()}:")
            print(response[:200] + ("..." if len(response) > 200 else ""))
        
        # Check for identical responses
        print("\n" + "-"*40)
        print("Duplicate Check:")
        found_duplicates = False
        for i, style1 in enumerate(styles):
            for style2 in styles[i+1:]:
                if responses[style1] == responses[style2]:
                    print(f"⚠️  {style1} and {style2} are IDENTICAL")
                    found_duplicates = True
        
        if not found_duplicates:
            print("✓ All styles produced unique responses")

if __name__ == "__main__":
    main()