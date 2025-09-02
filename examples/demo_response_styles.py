#!/usr/bin/env python3
"""
Demonstrate ADAM's Response Style System
"""

import requests
import json

# Use an existing project
PROJECT_ID = "95c0982e-2d4f-4f01-9bbc-eab38ef2523e"  # PDT to DBT project
BASE_URL = "http://localhost:8000"

def test_response_style(style: str, query: str):
    """Test a specific response style"""
    print(f"\n{'='*60}")
    print(f"Style: {style.upper()}")
    print(f"{'='*60}")
    
    # Create a new conversation for this test
    conv_response = requests.post(
        f"{BASE_URL}/api/projects/{PROJECT_ID}/conversations",
        json={"title": f"Style Test - {style}"}
    )
    
    if conv_response.status_code not in [200, 201]:
        print(f"Failed to create conversation: {conv_response.text}")
        return
        
    conv_id = conv_response.json()["id"]
    
    # Send message with specific style
    msg_response = requests.post(
        f"{BASE_URL}/api/conversations/{conv_id}/messages",
        json={
            "content": query,
            "use_memory": False,
            "model": "gpt-4.1-mini-2025-04-14",
            "response_style": style
        }
    )
    
    if msg_response.status_code != 200:
        print(f"Failed to send message: {msg_response.text}")
        return
    
    messages = msg_response.json()
    
    # Find and display assistant response
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            word_count = len(content.split())
            print(f"\nResponse ({word_count} words):")
            print("-" * 40)
            # Show first 500 chars
            if len(content) > 500:
                print(content[:500] + "...")
            else:
                print(content)
            break

def main():
    print("ADAM Response Style Demonstration")
    print("=" * 60)
    
    # Check available styles
    styles_response = requests.get(f"{BASE_URL}/api/styles")
    if styles_response.status_code == 200:
        styles = styles_response.json()
        print("\nAvailable Styles:")
        for style_name, info in styles.items():
            print(f"  • {style_name}: {info['description']} (temp: {info['temperature']})")
    
    # Test query
    query = "What is machine learning?"
    print(f"\nTest Query: '{query}'")
    
    # Test different styles
    styles_to_test = ["concise", "normal", "explanatory", "friendly"]
    
    for style in styles_to_test:
        test_response_style(style, query)
    
    print(f"\n{'='*60}")
    print("Demonstration Complete!")

if __name__ == "__main__":
    main()