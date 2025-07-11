#!/usr/bin/env python3
"""
Configure ADAM's Embedding Model
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory_config import MemoryConfig
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

console = Console()

def show_embedding_options():
    """Display available embedding models"""
    console.print("\n[bold cyan]Available Embedding Models for ADAM[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model Name", style="cyan", width=25)
    table.add_column("Provider", style="yellow")
    table.add_column("Dimension", style="green")
    table.add_column("Description", style="white")
    
    for name, config in MemoryConfig.EMBEDDING_MODELS.items():
        table.add_row(
            name,
            config.provider.value,
            str(config.dimension),
            config.description
        )
    
    console.print(table)
    
    # Show current configuration
    config = MemoryConfig()
    console.print(f"\n[bold green]Current Configuration:[/bold green]")
    console.print(f"Model: [cyan]{config.embedding_model_name}[/cyan]")
    console.print(f"Provider: [yellow]{config.embedding_config.provider.value}[/yellow]")
    console.print(f"Dimension: [green]{config.embedding_config.dimension}[/green]")

def set_embedding_model():
    """Set the embedding model"""
    console.print("\n[bold]Choose an embedding model:[/bold]")
    
    # Local models (free)
    console.print("\n[yellow]Local Models (Free, No API Key Required):[/yellow]")
    console.print("1. all-MiniLM-L6-v2 - Fast, lightweight")
    console.print("2. all-mpnet-base-v2 - Higher quality (recommended)")
    console.print("3. all-MiniLM-L12-v2 - Balanced")
    
    # OpenAI models
    console.print("\n[cyan]OpenAI Models (Requires API Key):[/cyan]")
    console.print("4. text-embedding-3-small - Good balance")
    console.print("5. text-embedding-3-large - Best quality")
    console.print("6. text-embedding-ada-002 - Previous gen")
    
    choice = Prompt.ask("\nSelect model (1-6)", default="2")
    
    model_map = {
        "1": "all-MiniLM-L6-v2",
        "2": "all-mpnet-base-v2",
        "3": "all-MiniLM-L12-v2",
        "4": "text-embedding-3-small",
        "5": "text-embedding-3-large",
        "6": "text-embedding-ada-002"
    }
    
    if choice in model_map:
        model_name = model_map[choice]
        
        # Update .env file
        env_path = Path(__file__).parent.parent / ".env"
        
        # Read existing .env
        env_content = ""
        if env_path.exists():
            env_content = env_path.read_text()
        
        # Update or add ADAM_EMBEDDING_MODEL
        import re
        if "ADAM_EMBEDDING_MODEL" in env_content:
            env_content = re.sub(
                r'ADAM_EMBEDDING_MODEL=.*', 
                f'ADAM_EMBEDDING_MODEL={model_name}', 
                env_content
            )
        else:
            env_content += f"\n# ADAM Embedding Model\nADAM_EMBEDDING_MODEL={model_name}\n"
        
        # Write back
        env_path.write_text(env_content)
        
        console.print(f"\n[green]✅ Updated embedding model to: {model_name}[/green]")
        console.print("\n[yellow]Note: You'll need to restart ADAM for changes to take effect.[/yellow]")
        console.print("[yellow]The memory will be rebuilt with the new embeddings.[/yellow]")
    else:
        console.print("[red]Invalid choice[/red]")

if __name__ == "__main__":
    show_embedding_options()
    
    if Prompt.ask("\nDo you want to change the embedding model?", choices=["y", "n"], default="n") == "y":
        set_embedding_model()