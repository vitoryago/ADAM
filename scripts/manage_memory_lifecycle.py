#!/usr/bin/env python3
"""
Memory Lifecycle Management Script
Manually trigger decay cycles and view memory health
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.memory import ADAMMemoryAdvanced
from src.adam.memory_lifecycle import MemoryLifecycleManager
from src.adam.activity_tracker import ActivityTracker
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()

async def apply_decay(memory_path: str):
    """Apply decay to all memories"""
    console.print("\n[yellow]🔄 Applying decay to all memories...[/yellow]")
    
    memory = ADAMMemoryAdvanced(memory_path)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing memories...", total=None)
        
        # Apply decay
        compress_candidates = await memory.lifecycle_manager.apply_decay_to_all_memories()
        
        progress.update(task, completed=True)
    
    console.print(f"\n[green]✅ Decay applied successfully![/green]")
    
    if compress_candidates:
        console.print(f"\n[cyan]Found {len(compress_candidates)} memories ready for compression:[/cyan]")
        for mem_id, tier in compress_candidates[:5]:  # Show first 5
            console.print(f"  • {mem_id[:12]}... → {tier}")
        if len(compress_candidates) > 5:
            console.print(f"  ... and {len(compress_candidates) - 5} more")

def show_memory_health(memory_path: str):
    """Display memory system health and statistics"""
    console.print("\n[bold cyan]Memory System Health Report[/bold cyan]\n")
    
    memory = ADAMMemoryAdvanced(memory_path)
    
    # Get all memories
    all_data = memory.collection.get()
    
    if not all_data['ids']:
        console.print("[yellow]No memories found in the system.[/yellow]")
        return
    
    # Analyze memory health
    health_data = []
    strength_distribution = []
    tier_counts = {'active': 0, 'archive': 0, 'landmark': 0, 'compressed': 0}
    
    for i, (mem_id, metadata) in enumerate(zip(all_data['ids'], all_data['metadatas'])):
        strength = memory.lifecycle_manager.get_memory_strength(mem_id, metadata)
        current_strength = strength.calculate_decayed_strength()
        strength_distribution.append(current_strength)
        
        # Get tier
        tier = memory.lifecycle_manager.classify_memory_tier(mem_id, metadata)
        if tier.startswith('compress'):
            tier_counts['compressed'] += 1
        elif tier in tier_counts:
            tier_counts[tier] += 1
        
        # Add to health data (show top 10)
        if i < 10:
            health_data.append({
                'id': mem_id[:8] + '...',
                'type': metadata.get('memory_type', 'unknown'),
                'strength': current_strength,
                'access_count': metadata.get('access_count', 0),
                'tier': tier,
                'age_days': (datetime.now() - datetime.fromisoformat(
                    metadata.get('timestamp', datetime.now().isoformat())
                )).days
            })
    
    # Display summary stats
    summary_table = Table(title="Memory System Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Total Memories", str(len(all_data['ids'])))
    summary_table.add_row("Average Strength", f"{sum(strength_distribution) / len(strength_distribution):.3f}")
    summary_table.add_row("Min Strength", f"{min(strength_distribution):.3f}")
    summary_table.add_row("Max Strength", f"{max(strength_distribution):.3f}")
    
    console.print(summary_table)
    
    # Display tier distribution
    tier_table = Table(title="Memory Tier Distribution")
    tier_table.add_column("Tier", style="cyan")
    tier_table.add_column("Count", style="green")
    tier_table.add_column("Percentage", style="yellow")
    
    total = sum(tier_counts.values())
    for tier, count in tier_counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        tier_table.add_row(tier, str(count), f"{percentage:.1f}%")
    
    console.print("\n")
    console.print(tier_table)
    
    # Display top memories
    if health_data:
        memory_table = Table(title="Top 10 Memories by Access")
        memory_table.add_column("ID", style="cyan")
        memory_table.add_column("Type", style="magenta")
        memory_table.add_column("Strength", style="green")
        memory_table.add_column("Accesses", style="yellow")
        memory_table.add_column("Age (days)", style="blue")
        memory_table.add_column("Tier", style="red")
        
        # Sort by access count
        health_data.sort(key=lambda x: x['access_count'], reverse=True)
        
        for mem in health_data:
            memory_table.add_row(
                mem['id'],
                mem['type'],
                f"{mem['strength']:.3f}",
                str(mem['access_count']),
                str(mem['age_days']),
                mem['tier']
            )
        
        console.print("\n")
        console.print(memory_table)

def mark_landmark(memory_path: str, memory_id: str):
    """Mark a memory as landmark (never compress)"""
    memory = ADAMMemoryAdvanced(memory_path)
    
    # Get the memory
    result = memory.collection.get(ids=[memory_id])
    if not result['ids']:
        console.print(f"[red]Memory {memory_id} not found![/red]")
        return
    
    # Update metadata
    metadata = result['metadatas'][0]
    metadata['landmark'] = True
    
    # Update in database
    memory.collection.update(
        ids=[memory_id],
        metadatas=[metadata]
    )
    
    console.print(f"[green]✅ Memory {memory_id} marked as landmark![/green]")

def show_activity_report(memory_path: str):
    """Show activity patterns and usage statistics"""
    console.print("\n[bold cyan]ADAM Activity Report[/bold cyan]\n")
    
    # Initialize activity tracker
    tracker = ActivityTracker(memory_path)
    summary = tracker.get_activity_summary()
    
    if summary.get("status") == "No activity recorded":
        console.print("[yellow]No activity has been recorded yet.[/yellow]")
        return
    
    # Display summary panel
    summary_text = f"""[bold]Activity Overview[/bold]
    
First Activity: {summary.get('first_activity', 'N/A')}
Last Activity: {summary.get('last_activity', 'N/A')}
Days Since Last Activity: {summary.get('days_since_last_activity', 0)} calendar days

[bold]Usage Statistics[/bold]
Total Active Days: {summary.get('total_active_days', 0)}
Total Interactions: {summary.get('total_interactions', 0)}
Average per Active Day: {summary.get('avg_interactions_per_day', 0)}

[bold]Peak Usage[/bold]
Most Active Day: {summary.get('most_active_day', 'N/A')}
Interactions: {summary.get('most_active_day_count', 0)}

[bold]Memory Age Calculation[/bold]
Current Active Day Index: {summary.get('current_active_day_index', 0)}
(Memories age based on active days, not calendar days)"""
    
    console.print(Panel(summary_text, title="📊 Activity Summary", border_style="blue"))
    
    # Show recent activity pattern
    recent_pattern = tracker.get_activity_pattern(last_n_days=14)
    if recent_pattern:
        activity_table = Table(title="Last 14 Active Days")
        activity_table.add_column("Date", style="cyan")
        activity_table.add_column("Interactions", style="green")
        activity_table.add_column("Activity Level", style="yellow")
        
        for date, count in recent_pattern.items():
            # Visual activity level
            if count >= 20:
                level = "████████ High"
            elif count >= 10:
                level = "█████ Medium"
            elif count >= 5:
                level = "███ Low"
            else:
                level = "█ Minimal"
            
            activity_table.add_row(date, str(count), level)
        
        console.print("\n")
        console.print(activity_table)

def main():
    parser = argparse.ArgumentParser(description="ADAM Memory Lifecycle Management")
    parser.add_argument(
        "--path", 
        default="./adam_memory_advanced",
        help="Path to memory directory (default: ./adam_memory_advanced)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Decay command
    decay_parser = subparsers.add_parser("decay", help="Apply decay to all memories")
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Show memory system health")
    
    # Landmark command
    landmark_parser = subparsers.add_parser("landmark", help="Mark memory as landmark")
    landmark_parser.add_argument("memory_id", help="Memory ID to mark as landmark")
    
    # Activity command
    activity_parser = subparsers.add_parser("activity", help="Show activity report")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "decay":
            asyncio.run(apply_decay(args.path))
        elif args.command == "health":
            show_memory_health(args.path)
        elif args.command == "landmark":
            mark_landmark(args.path, args.memory_id)
        elif args.command == "activity":
            show_activity_report(args.path)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()