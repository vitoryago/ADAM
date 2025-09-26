#!/usr/bin/env python3
"""
ADAM Complete CLI - Full-featured interface with transparency
"""

import asyncio
from pathlib import Path
from typing import Optional
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from adam import ADAMSystem, ADAMMemoryAdvanced
from adam.config.unified import get_config
from adam.llm.async_client import AsyncLLMClient


def main():
    """Main entry point for adam-complete command"""
    console = Console()

    console.print(Panel.fit(
        "[bold blue]ADAM Complete Interface[/bold blue]\n"
        "[dim]Full transparency mode with memory insights[/dim]",
        border_style="blue"
    ))

    # Initialize configuration
    config = get_config()

    # Initialize ADAM system with memory
    memory_system = ADAMMemoryAdvanced(persist_directory=config.memory.path)
    adam = ADAMSystem(config=config)

    # Initialize LLM client
    llm_client = AsyncLLMClient()

    console.print("\n[dim]Commands: 'exit' to quit, 'clear' to clear screen, 'memory' to show memory stats[/dim]\n")

    while True:
        try:
            # Get user input
            user_input = console.input("\n[bold green]You:[/bold green] ").strip()

            if user_input.lower() == 'exit':
                console.print("[yellow]Goodbye![/yellow]")
                break
            elif user_input.lower() == 'clear':
                os.system('clear' if os.name != 'nt' else 'cls')
                continue
            elif user_input.lower() == 'memory':
                # Show memory statistics
                stats = memory_system.get_statistics()
                console.print(Panel(
                    f"Total Memories: {stats.get('total_memories', 0)}\n"
                    f"Collections: {stats.get('collections', 0)}\n"
                    f"Last Updated: {stats.get('last_updated', 'N/A')}",
                    title="Memory Statistics",
                    border_style="cyan"
                ))
                continue
            elif not user_input:
                continue

            # Search memory for context
            console.print("\n[dim]Searching memory...[/dim]", end="")
            memory_results = memory_system.search(user_input, k=3)

            if memory_results:
                console.print(" [green]✓[/green]")
                console.print(f"[dim]Found {len(memory_results)} relevant memories[/dim]")
            else:
                console.print(" [dim]No relevant memories found[/dim]")

            # Process with ADAM
            console.print("\n[bold cyan]ADAM:[/bold cyan] ", end="")

            # Build context with memory
            context = ""
            if memory_results:
                context = "\n".join([m.get('content', '') for m in memory_results])

            # Get response from LLM
            response = asyncio.run(llm_client.complete(user_input, system_prompt=context if context else None))

            # Display the response content
            if hasattr(response, 'content'):
                console.print(f"[cyan]{response.content}[/cyan]")
            else:
                console.print(response)

            # Store in memory if significant
            if len(user_input.split()) > 5:
                response_text = response.content if hasattr(response, 'content') else str(response)
                memory_system.add_memory(
                    content=f"Q: {user_input}\nA: {response_text}",
                    metadata={
                        'type': 'conversation',
                        'source': 'cli_complete'
                    }
                )
                console.print("[dim]→ Saved to memory[/dim]")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            console.print("\n[yellow]Troubleshooting:[/yellow]")
            console.print("1. Create a .env file in your project root with:")
            console.print("   [dim]XAI_API_KEY=your_xai_api_key_here[/dim]")
            console.print("   [dim]OPENAI_API_KEY=your_openai_api_key_here[/dim]")
            console.print("2. Install missing dependencies: [dim]pip install -e . --force-reinstall[/dim]")
            console.print("3. Check that your API keys are valid and have credits")
            continue


if __name__ == "__main__":
    main()