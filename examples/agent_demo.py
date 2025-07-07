#!/usr/bin/env python3
"""
ADAM Agent System Demonstration
===============================

This demo showcases ADAM's transformation from a reactive Q&A system
to a proactive, goal-directed agent capable of complex task execution.

The demo illustrates:
1. Goal decomposition - Breaking "write a market analysis report" into subtasks
2. Planning with dependencies - Research before analysis before writing
3. Execution with monitoring - Real-time progress tracking
4. Error recovery - Handling API failures gracefully
5. Reflection and learning - Improving approach based on outcomes

This shows the fundamental shift from:
"What's the weather?" → "It's sunny"

To:
"Create a competitive analysis of AI assistants" → 
  - Plans research strategy
  - Gathers data from multiple sources
  - Analyzes findings
  - Creates comprehensive report
  - Learns from the process
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
import time

# Import our agent system
from src.adam.agent_system import (
    ADAMAgentSystem, Goal, TaskPriority, AgentMode,
    AgentMonitoringDashboard
)
from src.adam.memory_network import MemoryNetworkSystem
from src.adam.memory import MemorySystem
from src.adam.conversation_aware_memory import ConversationAwareMemorySystem

console = Console()


async def demonstrate_goal_decomposition():
    """
    Demonstrate how ADAM breaks down complex goals into manageable tasks
    
    This shows why explicit planning beats chained LLM calls:
    - Dependencies are identified upfront
    - Resource allocation is optimized
    - Progress can be tracked
    - Failures can be isolated
    """
    console.print(Panel.fit(
        "[bold cyan]Goal Decomposition Demo[/bold cyan]\n\n"
        "Watch how ADAM transforms a high-level goal into an executable plan",
        border_style="cyan"
    ))
    
    # Create a complex goal
    goal = Goal(
        description="Create a comprehensive market analysis report on AI assistants comparing features, pricing, and user satisfaction",
        success_criteria=[
            "Compare at least 5 major AI assistants",
            "Include pricing information",
            "Analyze user reviews and satisfaction scores",
            "Create visual comparisons",
            "Provide strategic recommendations"
        ],
        constraints=[
            "Use only publicly available information",
            "Complete within 30 minutes",
            "Focus on business users"
        ],
        deadline=datetime.now() + timedelta(minutes=30),
        priority=TaskPriority.HIGH
    )
    
    # Show the goal
    console.print("\n[yellow]Goal:[/yellow]")
    console.print(goal.description)
    console.print("\n[yellow]Success Criteria:[/yellow]")
    for criterion in goal.success_criteria:
        console.print(f"  • {criterion}")
    
    # Simulate decomposition (in real system, this would use LLM)
    await asyncio.sleep(2)  # Simulate thinking
    
    console.print("\n[green]Decomposed into tasks:[/green]")
    
    tasks = [
        {
            "name": "Research AI Assistants",
            "description": "Identify and gather information on major AI assistants",
            "dependencies": [],
            "estimated_time": "5 minutes"
        },
        {
            "name": "Collect Pricing Data",
            "description": "Find pricing tiers and plans for each assistant",
            "dependencies": ["Research AI Assistants"],
            "estimated_time": "5 minutes"
        },
        {
            "name": "Analyze User Reviews",
            "description": "Gather and analyze user satisfaction data",
            "dependencies": ["Research AI Assistants"],
            "estimated_time": "10 minutes"
        },
        {
            "name": "Create Comparison Matrix",
            "description": "Build feature and pricing comparison tables",
            "dependencies": ["Collect Pricing Data", "Analyze User Reviews"],
            "estimated_time": "5 minutes"
        },
        {
            "name": "Generate Visualizations",
            "description": "Create charts and graphs for key insights",
            "dependencies": ["Create Comparison Matrix"],
            "estimated_time": "3 minutes"
        },
        {
            "name": "Write Report",
            "description": "Compile findings into comprehensive report",
            "dependencies": ["Generate Visualizations"],
            "estimated_time": "5 minutes"
        },
        {
            "name": "Review and Polish",
            "description": "Ensure quality and completeness",
            "dependencies": ["Write Report"],
            "estimated_time": "2 minutes"
        }
    ]
    
    # Display task breakdown
    for i, task in enumerate(tasks, 1):
        deps = f" (depends on: {', '.join(task['dependencies'])})" if task['dependencies'] else ""
        console.print(f"{i}. [cyan]{task['name']}[/cyan]{deps}")
        console.print(f"   {task['description']} [{task['estimated_time']}]")
    
    console.print("\n[dim]This decomposition enables parallel execution where possible[/dim]")
    console.print("[dim]and clear progress tracking throughout[/dim]")


async def demonstrate_agent_architectures():
    """
    Show the different agent architectures in action
    
    This demonstrates why different approaches suit different tasks:
    - ReAct: Flexible exploration
    - Plan-and-Execute: Efficient execution
    - Reflexion: Continuous improvement
    """
    console.print(Panel.fit(
        "[bold cyan]Agent Architecture Comparison[/bold cyan]\n\n"
        "Same goal, different approaches",
        border_style="cyan"
    ))
    
    goal = "Debug why the API integration is failing"
    
    # ReAct Approach
    console.print("\n[yellow]1. ReAct Agent (Reasoning + Acting)[/yellow]")
    console.print("Best for: Exploratory debugging where the issue is unclear\n")
    
    react_steps = [
        ("Think", "The API is failing. I should check the error logs first."),
        ("Act", "search_tool: Check recent error logs"),
        ("Observe", "Found 'Authentication failed' errors"),
        ("Think", "Authentication issue. Let me check the API credentials."),
        ("Act", "file_operation_tool: Read API config file"),
        ("Observe", "API key is present but might be expired"),
        ("Think", "I should verify the API key validity"),
        ("Act", "web_interact_tool: Test API key validity"),
        ("Observe", "API key expired yesterday"),
        ("Think", "Found the issue! The API key needs renewal.")
    ]
    
    for step_type, content in react_steps:
        icon = "🤔" if step_type == "Think" else "⚡" if step_type == "Act" else "👀"
        console.print(f"{icon} [cyan]{step_type}:[/cyan] {content}")
        await asyncio.sleep(0.5)
    
    # Plan-and-Execute Approach
    console.print("\n[yellow]2. Plan-and-Execute Agent[/yellow]")
    console.print("Best for: Systematic debugging with clear steps\n")
    
    plan_execute_steps = [
        "📋 Created plan with 5 tasks:",
        "  1. Check service status",
        "  2. Review error logs", 
        "  3. Validate configuration",
        "  4. Test API connectivity",
        "  5. Generate diagnostic report",
        "",
        "⚡ Executing plan...",
        "✅ Task 1 complete: Service is running",
        "✅ Task 2 complete: Found authentication errors",
        "✅ Task 3 complete: Configuration valid",
        "❌ Task 4 failed: API key expired",
        "✅ Task 5 complete: Report generated"
    ]
    
    for step in plan_execute_steps:
        console.print(step)
        await asyncio.sleep(0.3)
    
    # Reflexion Approach
    console.print("\n[yellow]3. Reflexion Agent[/yellow]")
    console.print("Best for: Learning from similar past issues\n")
    
    reflexion_steps = [
        "🔍 Checking memory for similar past issues...",
        "💡 Found: Previous API failures were often auth-related",
        "⚡ Applying learned strategy: Check auth first",
        "✅ Confirmed: API key expired (as predicted)",
        "📝 Reflection: Auth issues are common, should add monitoring",
        "💾 Stored strategy for future API debugging"
    ]
    
    for step in reflexion_steps:
        console.print(step)
        await asyncio.sleep(0.5)
    
    console.print("\n[green]Each approach found the issue, but:[/green]")
    console.print("• ReAct explored flexibly (good for unknowns)")
    console.print("• Plan-and-Execute was systematic (good for procedures)")
    console.print("• Reflexion leveraged experience (good for patterns)")


async def demonstrate_complex_execution():
    """
    Demonstrate ADAM executing a complex multi-step task with error recovery
    
    This shows:
    - Real-time execution monitoring
    - Handling of failures
    - Dynamic replanning
    - Learning from the experience
    """
    console.print(Panel.fit(
        "[bold cyan]Complex Task Execution Demo[/bold cyan]\n\n"
        "Goal: Analyze and optimize a slow database query",
        border_style="cyan"
    ))
    
    # Initialize systems (mock for demo)
    memory_system = MemorySystem(base_dir="./demo_memory")
    memory_network = MemoryNetworkSystem(memory_system)
    conversation_memory = ConversationAwareMemorySystem(
        conversation_system=None,
        memory_network=memory_network
    )
    
    # Create agent system
    agent_system = ADAMAgentSystem(
        memory_system=memory_network,
        conversation_memory=conversation_memory,
        mode=AgentMode.PLAN_EXECUTE
    )
    
    # Create the goal
    goal = Goal(
        description="Analyze and optimize the slow user_analytics query that's causing timeouts",
        success_criteria=[
            "Identify why the query is slow",
            "Propose optimization strategies",
            "Implement the best optimization",
            "Verify performance improvement"
        ],
        priority=TaskPriority.HIGH
    )
    
    console.print(f"\n[yellow]Goal:[/yellow] {goal.description}")
    
    # Simulate execution with progress
    console.print("\n[cyan]Execution Progress:[/cyan]")
    
    execution_steps = [
        ("🔍 Analyzing query execution plan", True, "Found full table scan on users table"),
        ("📊 Checking table statistics", True, "Table has 5M rows, no indexes on join columns"),
        ("💡 Identifying optimization opportunities", True, "Missing indexes on user_id and created_at"),
        ("🔧 Creating optimization plan", True, "Will add composite index on (user_id, created_at)"),
        ("⚠️  Testing optimization in dev", False, "Dev database is down"),
        ("🔄 Switching to staging environment", True, "Connected to staging successfully"),
        ("✨ Applying optimization", True, "Index created successfully"),
        ("📈 Measuring performance", True, "Query time reduced from 45s to 0.3s"),
        ("✅ Documenting changes", True, "Created migration script and docs")
    ]
    
    failed_tasks = []
    successful_tasks = []
    
    for step, success, result in execution_steps:
        console.print(f"\n{step}")
        
        # Simulate execution time
        for _ in track(range(20), description="Executing..."):
            await asyncio.sleep(0.1)
        
        if success:
            console.print(f"[green]✓ Success:[/green] {result}")
            successful_tasks.append(step)
        else:
            console.print(f"[red]✗ Failed:[/red] {result}")
            failed_tasks.append(step)
            console.print("[yellow]→ Attempting recovery...[/yellow]")
    
    # Show execution summary
    console.print(Panel.fit(
        f"[bold]Execution Complete[/bold]\n\n"
        f"✅ Successful tasks: {len(successful_tasks)}\n"
        f"❌ Failed tasks: {len(failed_tasks)}\n"
        f"🔄 Recovered from: {len(failed_tasks)} failures\n\n"
        f"[green]Result: Query optimization successful![/green]\n"
        f"Performance improved by 150x (45s → 0.3s)",
        title="Summary",
        border_style="green"
    ))
    
    # Show reflection
    console.print("\n[cyan]Agent Reflection:[/cyan]")
    console.print("📝 What worked well:")
    console.print("  • Systematic analysis identified root cause quickly")
    console.print("  • Had fallback plan when dev environment failed")
    console.print("📝 Lessons learned:")
    console.print("  • Should check environment availability before starting")
    console.print("  • Composite indexes very effective for time-range queries")
    console.print("📝 Stored for future use:")
    console.print("  • Query optimization checklist")
    console.print("  • Environment fallback strategy")


async def demonstrate_proactive_behavior():
    """
    Show ADAM's proactive capabilities
    
    This demonstrates the shift from reactive to proactive:
    - Monitoring for opportunities
    - Suggesting optimizations
    - Preventing problems before they occur
    """
    console.print(Panel.fit(
        "[bold cyan]Proactive Agent Demo[/bold cyan]\n\n"
        "ADAM monitors and suggests actions before being asked",
        border_style="cyan"
    ))
    
    console.print("\n[dim]Starting proactive monitoring...[/dim]\n")
    
    # Simulate proactive suggestions over time
    proactive_events = [
        {
            "time": "09:00",
            "trigger": "Schedule check",
            "suggestion": "You have a deployment scheduled today. Shall I run the pre-deployment checklist?"
        },
        {
            "time": "10:30",
            "trigger": "Resource monitor",
            "suggestion": "Database CPU usage trending up (85%). I can analyze slow queries to prevent issues."
        },
        {
            "time": "11:00",
            "trigger": "Pattern detection",
            "suggestion": "I noticed you've searched for React hooks 3 times today. Would you like me to create a comprehensive guide?"
        },
        {
            "time": "14:00",
            "trigger": "Error detection",
            "suggestion": "API response times increased 40% in the last hour. I can investigate the cause."
        },
        {
            "time": "16:00",
            "trigger": "Workflow optimization",
            "suggestion": "You often run these 5 commands together. I can create an automated workflow to save time."
        }
    ]
    
    for event in proactive_events:
        console.print(f"[dim]{event['time']}[/dim] - [yellow]{event['trigger']}[/yellow]")
        await asyncio.sleep(1)
        console.print(f"💡 [cyan]ADAM:[/cyan] {event['suggestion']}")
        console.print("[dim]Press Enter to accept, or type 'later' to defer[/dim]")
        console.print("")
        await asyncio.sleep(2)
    
    console.print("[green]This proactive behavior prevents issues and improves efficiency[/green]")


async def demonstrate_learning_improvement():
    """
    Show how ADAM learns and improves over time
    
    This illustrates why reflection matters:
    - Past failures inform future strategies
    - Successful patterns are reinforced
    - Performance improves with experience
    """
    console.print(Panel.fit(
        "[bold cyan]Learning & Improvement Demo[/bold cyan]\n\n"
        "Watch ADAM get better at similar tasks over time",
        border_style="cyan"
    ))
    
    task = "Deploy application update to production"
    
    console.print(f"\n[yellow]Repeated Task:[/yellow] {task}\n")
    
    attempts = [
        {
            "attempt": 1,
            "date": "Week 1",
            "approach": "Basic deployment",
            "issues": ["Forgot to run tests", "No rollback plan", "Users not notified"],
            "time": "45 mins",
            "success": False,
            "lesson": "Need comprehensive checklist"
        },
        {
            "attempt": 2,
            "date": "Week 2",
            "approach": "With checklist",
            "issues": ["Tests passed but staging wasn't updated", "Notification sent late"],
            "time": "35 mins",
            "success": True,
            "lesson": "Add staging deployment step"
        },
        {
            "attempt": 3,
            "date": "Week 3",
            "approach": "Full pipeline",
            "issues": ["Minor: Metrics dashboard not updated"],
            "time": "25 mins",
            "success": True,
            "lesson": "Add post-deployment verification"
        },
        {
            "attempt": 4,
            "date": "Week 4",
            "approach": "Optimized pipeline",
            "issues": [],
            "time": "15 mins",
            "success": True,
            "lesson": "Process is now reliable and efficient"
        }
    ]
    
    for attempt in attempts:
        console.print(f"[cyan]Attempt {attempt['attempt']} - {attempt['date']}[/cyan]")
        console.print(f"Approach: {attempt['approach']}")
        
        if attempt['issues']:
            console.print(f"[red]Issues encountered:[/red]")
            for issue in attempt['issues']:
                console.print(f"  • {issue}")
        else:
            console.print("[green]No issues![/green]")
        
        console.print(f"Time taken: {attempt['time']}")
        console.print(f"Result: {'✅ Success' if attempt['success'] else '❌ Failed'}")
        console.print(f"[dim]Lesson learned: {attempt['lesson']}[/dim]")
        console.print("")
        await asyncio.sleep(2)
    
    # Show improvement graph
    console.print("[bold]Performance Improvement:[/bold]")
    console.print("Time:    45m ████████████████████")
    console.print("         35m ██████████████")
    console.print("         25m ██████████")
    console.print("         15m ██████")
    console.print("              1    2    3    4  (Attempt)")
    
    console.print("\n[green]ADAM learned from each experience:[/green]")
    console.print("• Created comprehensive deployment checklist")
    console.print("• Automated repetitive steps")
    console.print("• Added verification at each stage")
    console.print("• Reduced deployment time by 67%")


async def run_interactive_demo():
    """
    Interactive demonstration where users can see ADAM in action
    """
    console.print(Panel.fit(
        "[bold cyan]ADAM Agent System - Interactive Demo[/bold cyan]\n\n"
        "Experience the transformation from Q&A to Goal Achievement",
        border_style="cyan"
    ))
    
    # Show menu
    demos = [
        ("Goal Decomposition", demonstrate_goal_decomposition),
        ("Agent Architectures", demonstrate_agent_architectures),
        ("Complex Execution", demonstrate_complex_execution),
        ("Proactive Behavior", demonstrate_proactive_behavior),
        ("Learning & Improvement", demonstrate_learning_improvement)
    ]
    
    while True:
        console.print("\n[yellow]Choose a demonstration:[/yellow]")
        for i, (name, _) in enumerate(demos, 1):
            console.print(f"{i}. {name}")
        console.print("0. Exit")
        
        try:
            choice = int(input("\nEnter choice (0-5): "))
            
            if choice == 0:
                console.print("[green]Thanks for exploring ADAM's agent capabilities![/green]")
                break
            elif 1 <= choice <= len(demos):
                await demos[choice-1][1]()
                console.print("\n[dim]Press Enter to continue...[/dim]")
                input()
            else:
                console.print("[red]Invalid choice[/red]")
        except ValueError:
            console.print("[red]Please enter a number[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Demo interrupted[/yellow]")
            break


async def main():
    """Main entry point"""
    console.print(Panel.fit(
        "[bold cyan]ADAM: From Assistant to Agent[/bold cyan]\n\n"
        "[yellow]Traditional Assistant:[/yellow]\n"
        "User: What's the weather?\n"
        "Assistant: It's 72°F and sunny.\n"
        "[dim]→ Reactive, one-shot, no memory[/dim]\n\n"
        "[green]ADAM as Agent:[/green]\n"
        "User: Keep my development environment optimized\n"
        "ADAM: I'll monitor performance, prevent issues, and\n"
        "      continuously improve your workflow.\n"
        "[dim]→ Proactive, persistent, learning[/dim]",
        border_style="cyan"
    ))
    
    await asyncio.sleep(3)
    
    # Run interactive demo
    await run_interactive_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo terminated by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()