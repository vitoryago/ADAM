#!/usr/bin/env python3
"""
Test updated style responses
"""

import requests
import json
import time

PROJECT_ID = "95c0982e-2d4f-4f01-9bbc-eab38ef2523e"
BASE_URL = "http://localhost:8000"

def test_style(style, query):
    """Test a specific style"""
    # Create a new conversation
    conv_resp = requests.post(
        f"{BASE_URL}/api/projects/{PROJECT_ID}/conversations",
        json={"title": f"Style Test - {style} - {time.time()}"}
    )
    
    if conv_resp.status_code not in [200, 201]:
        print(f"Failed to create conversation: {conv_resp.status_code}")
        return None
        
    conv_id = conv_resp.json()["id"]
    
    # Send message with specific style
    msg_resp = requests.post(
        f"{BASE_URL}/api/conversations/{conv_id}/messages",
        json={
            "content": query,
            "use_memory": False,
            "model": "gpt-4.1-mini-2025-04-14",
            "response_style": style
        }
    )
    
    if msg_resp.status_code == 200:
        messages = msg_resp.json()
        for msg in messages:
            if msg.get("role") == "assistant":
                return msg.get("content", "")
    return None

def main():
    print("Testing Updated Response Styles")
    print("="*60)
    
    query = "Hey ADAM"
    print(f"Query: '{query}'\n")
    
    styles = ["concise", "normal", "friendly", "creative"]
    
    for style in styles:
        print(f"\n{style.upper()}:")
        print("-"*40)
        response = test_style(style, query)
        if response:
            print(response)
        else:
            print("(No response)")
    
    # Test a longer query too
    query2 = "What's your favorite thing about helping people?"
    print(f"\n{'='*60}")
    print(f"Query: '{query2}'\n")
    
    for style in styles:
        print(f"\n{style.upper()}:")
        print("-"*40)
        response = test_style(style, query2)
        if response:
            print(response[:200] + ("..." if len(response) > 200 else ""))

if __name__ == "__main__":
    main()