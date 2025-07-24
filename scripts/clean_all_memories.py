#!/usr/bin/env python3
"""
Clean All ADAM Memories - Fresh Start Script
==========================================

This script completely cleans ADAM's memory system:
1. Deletes all ChromaDB collections
2. Removes all conversation history
3. Clears the memory network
4. Resets cost tracking and metadata

USE WITH CAUTION: This will permanently delete all stored memories!
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel

console = Console()

def clean_adam_memory():
    """Clean all ADAM memory data"""
    
    console.print(Panel.fit(
        "[bold red]⚠️  WARNING: COMPLETE MEMORY WIPE ⚠️[/bold red]\n\n"
        "This will permanently delete:\n"
        "• All stored memories in ChromaDB\n"
        "• All conversation history\n"
        "• Memory network graph\n"
        "• Cost tracking data\n"
        "• Activity logs\n\n"
        "[bold]This action cannot be undone![/bold]",
        title="🧠 ADAM Memory Clean"
    ))
    
    if not Confirm.ask("\n[bold yellow]Are you sure you want to delete ALL memories?[/bold yellow]"):
        console.print("[green]Operation cancelled.[/green]")
        return
    
    # Create backup first
    backup_dir = Path(f"./adam_memory_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    memory_dir = Path("./adam_memory_advanced")
    
    if memory_dir.exists():
        console.print(f"\n[cyan]Creating backup at: {backup_dir}[/cyan]")
        shutil.copytree(memory_dir, backup_dir)
        console.print("[green]✓ Backup created[/green]")
    
    try:
        # 1. Clean ChromaDB collections
        console.print("\n[yellow]1. Cleaning ChromaDB collections...[/yellow]")
        client = chromadb.PersistentClient(path=str(memory_dir))
        
        collections = client.list_collections()
        for collection in collections:
            console.print(f"   Deleting collection: {collection.name}")
            client.delete_collection(collection.name)
        
        console.print("[green]✓ ChromaDB collections deleted[/green]")
        
        # 2. Clean conversation history
        console.print("\n[yellow]2. Cleaning conversation history...[/yellow]")
        conversations_dir = memory_dir / "conversations"
        if conversations_dir.exists():
            file_count = len(list(conversations_dir.glob("*.json")))
            shutil.rmtree(conversations_dir)
            conversations_dir.mkdir(exist_ok=True)
            console.print(f"[green]✓ Deleted {file_count} conversation files[/green]")
        
        # 3. Clean memory network
        console.print("\n[yellow]3. Cleaning memory network...[/yellow]")
        network_dir = memory_dir / "memory_network"
        if network_dir.exists():
            shutil.rmtree(network_dir)
            network_dir.mkdir(exist_ok=True)
            console.print("[green]✓ Memory network cleared[/green]")
        
        # 4. Clean metadata files
        console.print("\n[yellow]4. Cleaning metadata files...[/yellow]")
        metadata_files = [
            "access_patterns.json",
            "cost_savings.json",
            "activity_log.json"
        ]
        
        for filename in metadata_files:
            filepath = memory_dir / filename
            if filepath.exists():
                filepath.unlink()
                console.print(f"   Deleted: {filename}")
        
        console.print("[green]✓ Metadata files cleaned[/green]")
        
        # 5. Clean ChromaDB internal files
        console.print("\n[yellow]5. Cleaning ChromaDB internal files...[/yellow]")
        chroma_files = ["chroma.sqlite3"]
        for filename in chroma_files:
            filepath = memory_dir / filename
            if filepath.exists():
                filepath.unlink()
                console.print(f"   Deleted: {filename}")
        
        # Clean UUID directories (ChromaDB collections)
        for item in memory_dir.iterdir():
            if item.is_dir() and len(item.name) == 36 and '-' in item.name:  # UUID pattern
                shutil.rmtree(item)
                console.print(f"   Deleted collection directory: {item.name}")
        
        console.print("[green]✓ ChromaDB files cleaned[/green]")
        
        # Final summary
        console.print(Panel.fit(
            "[bold green]✅ Memory Clean Complete![/bold green]\n\n"
            f"• Backup saved to: {backup_dir}\n"
            "• All memories have been deleted\n"
            "• ADAM will start fresh on next run\n\n"
            "[dim]To restore from backup, rename the backup directory to 'adam_memory_advanced'[/dim]",
            title="🧠 Clean Complete"
        ))
        
    except Exception as e:
        console.print(f"\n[red]❌ Error during cleanup: {str(e)}[/red]")
        console.print("[yellow]Some files may not have been deleted. Please check manually.[/yellow]")
        raise


if __name__ == "__main__":
    clean_adam_memory()