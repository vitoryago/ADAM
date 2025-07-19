#!/usr/bin/env python3
"""
Force save a session's conversations to memory
Useful when memory storage hasn't happened yet
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime
from src.adam.memory import ADAMMemoryAdvanced
from src.adam.conversation_system import ConversationSystem
from rich.console import Console

console = Console()

def force_save_session_to_memory(session_id: str = None):
    """Force save session conversations to memory"""
    
    # Load sessions from file
    sessions_file = Path("data/web_sessions.json")
    if not sessions_file.exists():
        console.print("[red]No sessions file found![/red]")
        return
    
    with open(sessions_file, 'r') as f:
        sessions = json.load(f)
    
    if not sessions:
        console.print("[red]No sessions found![/red]")
        return
    
    # If no session_id provided, show available sessions
    if not session_id:
        console.print("[yellow]Available sessions:[/yellow]")
        for sid, session_data in sessions.items():
            msg_count = len(session_data.get('messages', []))
            console.print(f"  {sid} - {msg_count} messages")
        
        # Use the most recent session with the DAG conversation
        # Looking for the one with the reference DAG code
        for sid, session_data in sessions.items():
            messages = session_data.get('messages', [])
            for msg in messages:
                if 'DbtOperator' in msg.get('content', '') and 'MARKETING_ANALYTICS' in msg.get('content', ''):
                    session_id = sid
                    console.print(f"\n[green]Found DAG session: {session_id}[/green]")
                    break
            if session_id:
                break
    
    if not session_id:
        console.print("[red]Could not find session with DAG code![/red]")
        return
    
    # Get the session data
    session_data = sessions.get(session_id)
    if not session_data:
        console.print(f"[red]Session {session_id} not found![/red]")
        return
    
    # Initialize memory system
    console.print("\n[yellow]Initializing memory system...[/yellow]")
    memory = ADAMMemoryAdvanced()
    
    # Process messages and save to memory
    messages = session_data.get('messages', [])
    saved_count = 0
    
    for i in range(0, len(messages) - 1, 2):  # Process in pairs (user, assistant)
        if i + 1 < len(messages):
            user_msg = messages[i]
            assistant_msg = messages[i + 1]
            
            if user_msg['role'] == 'user' and assistant_msg['role'] == 'assistant':
                query = user_msg['content']
                response = assistant_msg['content']
                
                # Check if this is the DAG creation conversation
                if 'DbtOperator' in response and 'new_fee_repricing_user' in response:
                    console.print(f"\n[cyan]Found DAG conversation![/cyan]")
                    console.print(f"Query preview: {query[:100]}...")
                    console.print(f"Response preview: {response[:100]}...")
                    
                    # Force save to memory
                    console.print("\n[yellow]Forcing save to memory...[/yellow]")
                    
                    # Call remember_if_worthy with high cost to ensure it's saved
                    memory_id = memory.remember_if_worthy(
                        query=query,
                        response=response,
                        context={
                            "session_id": session_id,
                            "forced_save": True,
                            "source": "force_save_script"
                        },
                        generation_cost=0.1,  # High cost to ensure worthiness
                        model_used=assistant_msg.get('metadata', {}).get('model', 'unknown')
                    )
                    
                    if memory_id:
                        console.print(f"[green]✓ Saved to memory with ID: {memory_id}[/green]")
                        saved_count += 1
                    else:
                        console.print(f"[red]✗ Failed to save to memory[/red]")
    
    console.print(f"\n[green]Saved {saved_count} conversations to memory[/green]")
    
    # Test retrieval
    console.print("\n[yellow]Testing retrieval...[/yellow]")
    test_queries = [
        "new_fee_repricing_user dag",
        "last dag we created",
        "bring back the dag code"
    ]
    
    for query in test_queries:
        results = memory.recall_with_context(query=query, n_results=3)
        console.print(f"\nQuery: '{query}' - Found {len(results)} results")
        for result in results[:1]:
            content = result.get('content', '')
            if 'new_fee_repricing_user' in content:
                console.print("[green]✓ Found the DAG memory![/green]")

if __name__ == "__main__":
    import sys
    session_id = sys.argv[1] if len(sys.argv) > 1 else None
    force_save_session_to_memory(session_id)