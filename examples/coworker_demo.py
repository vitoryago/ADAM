#!/usr/bin/env python3
"""
ADAM Coworker Demo - Shows how ADAM can act as an AI coworker
with project-based memory and screen awareness
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adam.project_aware_memory import ProjectAwareMemory
from adam.project_manager import ProjectManager
from adam.llm.client import UnifiedLLMClient
from adam.llm.config import LLMConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def main():
    """Demonstrate ADAM as an AI coworker"""
    
    console.print(Panel.fit(
        "[bold cyan]ADAM AI Coworker Demo[/bold cyan]\n"
        "Project-based memory with screen awareness",
        border_style="cyan"
    ))
    
    # Initialize components
    console.print("\n🔧 Initializing ADAM components...")
    
    # Project manager
    project_manager = ProjectManager()
    
    # Project-aware memory
    memory = ProjectAwareMemory(project_manager=project_manager)
    
    # LLM client
    llm_config = LLMConfig()
    llm_client = UnifiedLLMClient(config=llm_config)
    
    # Show current projects
    console.print("\n📁 [bold]Available Projects:[/bold]")
    projects = memory.list_projects()
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("Memories")
    
    for project in projects:
        summary = memory.get_project_summary(project.id)
        table.add_row(
            project.id[:8],
            project.name,
            project.description,
            str(summary['total_memories'])
        )
    
    console.print(table)
    
    # Create or select project
    console.print("\n🆕 Creating demo project...")
    demo_project = memory.create_project(
        "Python Development",
        "Working on Python scripts and debugging"
    )
    console.print(f"✅ Created project: [green]{demo_project.name}[/green]")
    
    # Demonstrate screen capture
    console.print("\n📸 [bold]Screen Capture Demo:[/bold]")
    
    # Capture current screen
    console.print("Capturing current screen...")
    screen_data = memory.screen_capture.capture_screen()
    
    if screen_data:
        console.print("✅ Screen captured successfully!")
        
        # Extract text from screen
        screen_text = memory.screen_capture.extract_text_from_image(screen_data)
        if screen_text:
            console.print(f"\n📝 Extracted text preview:\n[dim]{screen_text[:200]}...[/dim]")
        
        # Analyze with vision model
        console.print("\n🤖 Analyzing screen with AI...")
        
        # Prepare prompt based on screen content
        prompt = "Please analyze this screenshot and tell me what you see. Focus on any code, errors, or important UI elements."
        
        # Get AI analysis
        response = await llm_client.complete(
            prompt=prompt,
            model="grok-2-vision-1212",  # Use vision model
            image_data=screen_data,
            temperature=0.7
        )
        
        console.print(Panel(
            response.content,
            title="AI Analysis",
            border_style="green"
        ))
        
        # Store in project memory
        memory_id = memory.remember_with_screen(
            query=prompt,
            response=response.content,
            screen_capture=screen_data,
            generation_cost=response.cost,
            model_used=response.model
        )
        
        if memory_id:
            console.print(f"\n💾 Stored in project memory: [cyan]{memory_id}[/cyan]")
    
    else:
        console.print("❌ Screen capture not available. Install mss: pip install mss")
    
    # Demonstrate project context
    console.print("\n🔍 [bold]Testing Project Context:[/bold]")
    
    # Add some test memories
    test_queries = [
        ("How do I handle exceptions in Python?", "Python Development"),
        ("What's the syntax for list comprehensions?", "Python Development"),
        ("How do I debug JavaScript?", "Web Development")
    ]
    
    # Create another project
    web_project = memory.create_project("Web Development", "Frontend and backend web work")
    
    for query, project_name in test_queries:
        # Switch to appropriate project
        if project_name == "Web Development":
            memory.switch_project(web_project.id)
        else:
            memory.switch_project(demo_project.id)
        
        # Get response
        console.print(f"\n❓ Query in {project_name}: [yellow]{query}[/yellow]")
        
        response = await llm_client.complete(
            prompt=query,
            model="grok-3-mini-fast",
            temperature=0.7
        )
        
        # Store in project memory
        memory.remember_if_worthy(
            query=query,
            response=response.content,
            generation_cost=response.cost,
            model_used=response.model
        )
        
        console.print(f"✅ Stored in {project_name} project")
    
    # Search within projects
    console.print("\n🔎 [bold]Project-Specific Search:[/bold]")
    
    # Search Python project
    console.print(f"\nSearching in Python Development project:")
    python_results = memory.search_project_memories(
        "exception handling",
        project_id=demo_project.id,
        n_results=3
    )
    
    for result in python_results:
        console.print(f"  • {result.get('query', 'N/A')[:50]}... (score: {result.get('combined_score', 0):.2f})")
    
    # Search Web project
    console.print(f"\nSearching in Web Development project:")
    web_results = memory.search_project_memories(
        "debug",
        project_id=web_project.id,
        n_results=3
    )
    
    for result in web_results:
        console.print(f"  • {result.get('query', 'N/A')[:50]}... (score: {result.get('combined_score', 0):.2f})")
    
    # Show project summaries
    console.print("\n📊 [bold]Project Summaries:[/bold]")
    
    for project in [demo_project, web_project]:
        summary = memory.get_project_summary(project.id)
        
        console.print(f"\n[bold]{project.name}:[/bold]")
        console.print(f"  Total memories: {summary['total_memories']}")
        console.print(f"  With screenshots: {summary['with_screen_captures']}")
        console.print(f"  Total cost: ${summary['total_cost']:.4f}")
        
        if summary['memory_types']:
            console.print("  Memory types:")
            for mem_type, count in summary['memory_types'].items():
                console.print(f"    - {mem_type}: {count}")
    
    # Demonstrate monitoring (optional)
    console.print("\n👁️ [bold]Screen Monitoring:[/bold]")
    console.print("Starting screen monitoring (captures every 30 seconds)...")
    console.print("[dim]Press Ctrl+C to stop monitoring and exit[/dim]\n")
    
    # Start monitoring
    memory.start_screen_monitoring(interval=30)
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        console.print("\n\n🛑 Stopping screen monitoring...")
        memory.stop_screen_monitoring()
        console.print("✅ Demo completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!")