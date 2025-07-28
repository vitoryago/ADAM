#!/usr/bin/env python3
"""
Test markdown rendering functionality
"""

import asyncio
import httpx
import json

async def test_markdown_rendering():
    """Test that markdown is properly rendered in responses"""
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        # First, get projects
        response = await client.get(f"{base_url}/api/projects")
        projects = response.json()
        
        if not projects:
            print("No projects found. Please create a project first.")
            return
        
        project = projects[0]
        print(f"Using project: {project['name']}")
        
        # Get conversations
        response = await client.get(f"{base_url}/api/projects/{project['id']}/conversations")
        conversations = response.json()
        
        if not conversations:
            print("No conversations found. Please create a conversation first.")
            return
        
        conversation = conversations[0]
        print(f"Using conversation: {conversation['title']}")
        
        # Send a test message
        test_message = {
            "content": "Please show me an example of converting DBT date from CET to PDT in SQL with proper formatting",
            "use_memory": True,
            "model": None
        }
        
        print("\nSending test message...")
        
        # Use streaming endpoint
        response = await client.post(
            f"{base_url}/api/conversations/{conversation['id']}/messages/stream",
            json=test_message,
            headers={"Accept": "text/event-stream"}
        )
        
        print("\nReceiving streamed response:")
        full_content = ""
        
        # Read the SSE stream
        async for line in response.aiter_lines():
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])
                    if data['type'] == 'assistant_chunk':
                        full_content += data['content']
                        print(data['content'], end='', flush=True)
                    elif data['type'] == 'complete':
                        print(f"\n\nModel: {data['model']}")
                        print(f"Tokens: {data['tokens']}")
                        print(f"Cost: ${data['cost']:.4f}")
                except json.JSONDecodeError:
                    pass
        
        print("\n\nNow checking the rendered HTML...")
        
        # Get the messages as HTML
        response = await client.get(
            f"{base_url}/conversations/{conversation['id']}/messages/html"
        )
        
        html_content = response.text
        
        # Check if markdown was rendered properly
        if '<pre' in html_content and 'language-sql' in html_content:
            print("✅ Markdown rendering is working! Code blocks are properly formatted.")
        else:
            print("❌ Markdown rendering issue detected. HTML output:")
            print(html_content[-500:])  # Show last 500 chars

if __name__ == "__main__":
    asyncio.run(test_markdown_rendering())