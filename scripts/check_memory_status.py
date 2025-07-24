#!/usr/bin/env python3
"""
Check ADAM Memory Status
=======================

Shows current memory usage before cleaning
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def check_memory_status():
    """Check and display current memory status"""
    
    memory_dir = Path("./adam_memory_advanced")
    
    console.print(Panel.fit(
        "🧠 ADAM Memory Status Check",
        title="Memory Analysis"
    ))
    
    if not memory_dir.exists():
        console.print("[yellow]No memory directory found. ADAM has no stored memories.[/yellow]")
        return
    
    # Create status table
    table = Table(title="Memory Statistics")
    table.add_column("Component", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Size", style="green")
    
    total_size = 0
    
    # 1. Check ChromaDB collections
    try:
        client = chromadb.PersistentClient(path=str(memory_dir))
        collections = client.list_collections()
        
        total_memories = 0
        for collection in collections:
            count = collection.count()
            total_memories += count
            table.add_row(f"Collection: {collection.name}", str(count), "-")
        
        table.add_row("Total Memories in ChromaDB", str(total_memories), "-")
    except Exception as e:
        table.add_row("ChromaDB Collections", "Error", str(e))
    
    # 2. Check conversations
    conversations_dir = memory_dir / "conversations"
    if conversations_dir.exists():
        conv_files = list(conversations_dir.glob("*.json"))
        conv_size = sum(f.stat().st_size for f in conv_files) / 1024 / 1024  # MB
        total_size += conv_size
        table.add_row("Conversation Sessions", str(len(conv_files)), f"{conv_size:.2f} MB")
    
    # 3. Check memory network
    network_dir = memory_dir / "memory_network"
    if network_dir.exists():
        network_files = list(network_dir.glob("*"))
        network_size = sum(f.stat().st_size for f in network_files if f.is_file()) / 1024 / 1024
        total_size += network_size
        table.add_row("Memory Network Files", str(len(network_files)), f"{network_size:.2f} MB")
    
    # 4. Check total directory size
    def get_dir_size(path):
        total = 0
        for entry in Path(path).rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
        return total / 1024 / 1024  # MB
    
    total_size = get_dir_size(memory_dir)
    
    console.print(table)
    
    console.print(f"\n[bold]Total Memory Storage: {total_size:.2f} MB[/bold]")
    
    # Show recent memories
    console.print("\n[yellow]Recent Memory Activity:[/yellow]")
    
    activity_file = memory_dir / "activity_log.json"
    if activity_file.exists():
        import json
        with open(activity_file, 'r') as f:
            activity = json.load(f)
            console.print(f"Last activity: {activity.get('last_activity', 'Unknown')}")
            console.print(f"Total interactions: {activity.get('total_interactions', 0)}")

if __name__ == "__main__":
    check_memory_status()