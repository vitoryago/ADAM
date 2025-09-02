#!/usr/bin/env python3
"""
Test to verify response style differences
"""

import requests
import json

PROJECT_ID = "95c0982e-2d4f-4f01-9bbc-eab38ef2523e"
BASE_URL = "http://localhost:8000"

def test_style(style: str, query: str):
    """Test a specific style"""
    # Create conversation
    conv_resp = requests.post(
        f"{BASE_URL}/api/projects/{PROJECT_ID}/conversations",
        json={"title": f"Style Test - {style}"}
    )
    conv_id = conv_resp.json()["id"]
    
    # Send message with specific style using streaming endpoint
    response = requests.post(
        f"{BASE_URL}/api/conversations/{conv_id}/messages/stream",
        json={
            "content": query,
            "use_memory": False,
            "model": "gpt-4.1-mini-2025-04-14",
            "response_style": style
        },
        stream=True
    )
    
    print(f"\n{'='*60}")
    print(f"Style: {style.upper()}")
    print(f"{'='*60}")
    
    full_response = ""
    for line in response.iter_lines():
        if line:
            line_text = line.decode('utf-8')
            if line_text.startswith('data: '):
                try:
                    data = json.loads(line_text[6:])
                    if data.get('type') == 'assistant_chunk':
                        chunk = data.get('content', '')
                        full_response += chunk
                except:
                    pass
    
    # Show first 300 chars
    print(full_response[:300] + ("..." if len(full_response) > 300 else ""))
    return full_response

def main():
    # Test with a prompt that should elicit different styles
    query = "Tell me about pizza"
    
    print("Testing Response Style Differences")
    print("Query:", query)
    
    # Test each style
    styles = ["friendly", "creative", "formal", "concise"]
    responses = {}
    
    for style in styles:
        response = test_style(style, query)
        responses[style] = response
    
    # Check for duplicates
    print("\n" + "="*60)
    print("Checking for identical responses:")
    print("="*60)
    
    for i, style1 in enumerate(styles):
        for style2 in styles[i+1:]:
            if responses[style1] == responses[style2]:
                print(f"⚠️  {style1} and {style2} gave IDENTICAL responses!")
            else:
                print(f"✓ {style1} and {style2} are different")

if __name__ == "__main__":
    main()