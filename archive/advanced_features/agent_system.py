#!/usr/bin/env python3
"""
ADAM's Proactive Agent System - From Reactive Q&A to Goal-Directed Behavior
===========================================================================

This module transforms ADAM from a passive question-answering system into an
active agent capable of pursuing complex objectives autonomously.

THE FUNDAMENTAL SHIFT:
=====================
Traditional Q&A: User asks → ADAM responds → Done
Agentic System: User sets goal → ADAM plans → executes → monitors → achieves

WHY AGENTS NEED MORE THAN CHAINED LLM CALLS:
===========================================
1. **Memory Between Steps**: Each action informs the next
2. **Error Recovery**: Real-world execution has failures
3. **Resource Management**: Not all paths are worth exploring  
4. **Progress Tracking**: Know when to pivot vs persist
5. **Learning**: Past experiences improve future performance

AGENT ARCHITECTURES IMPLEMENTED:
================================
1. **ReAct (Reasoning + Acting)**: Think-then-act loops
2. **Plan-and-Execute**: Upfront planning with monitored execution
3. **Reflexion**: Self-reflection for continuous improvement

Each architecture has trade-offs in terms of flexibility, reliability,
and computational cost. This implementation lets ADAM choose based on task.
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
import logging
from collections import defaultdict

# LangChain/LangGraph imports
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langchain_core.runnables import RunnablePassthrough

# Rich for beautiful output
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import our tools
from .agent_tools import get_all_tools, get_tool_categories

# Import existing ADAM components
from .memory_network import MemoryNetworkSystem
from .conversation_aware_memory import ConversationAwareMemorySystem

console = Console()
logger = logging.getLogger(__name__)


# ==============================================================================
# Core Data Structures
# ==============================================================================

class TaskStatus(Enum):
    """Status of a task in the execution plan"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Waiting on dependencies
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority levels for task scheduling"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class AgentMode(Enum):
    """Different agent operation modes"""
    REACT = "react"  # Reasoning + Acting loops
    PLAN_EXECUTE = "plan_execute"  # Plan then execute
    REFLEXION = "reflexion"  # Self-reflective improvement
    HYBRID = "hybrid"  # Adaptive mode selection


@dataclass
class Goal:
    """
    Represents a high-level objective the agent is working towards
    
    Goals are the fundamental unit of agent behavior. They transform
    vague user requests into concrete, achievable objectives.
    """
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    success_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    deadline: Optional[datetime] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    parent_goal_id: Optional[str] = None  # For hierarchical goals
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if goal has passed its deadline"""
        return self.deadline and datetime.now() > self.deadline


@dataclass
class Task:
    """
    Represents an atomic action the agent can take
    
    Tasks are the executable units that achieve goals. They map to
    tool calls or other concrete actions.
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal_id: str = ""
    description: str = ""
    action: str = ""  # Tool or action to execute
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Other task IDs
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    result: Optional[Any] = None
    error: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    
    def can_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are satisfied"""
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def mark_started(self):
        """Mark task as started"""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()
        self.attempts += 1
    
    def mark_completed(self, result: Any):
        """Mark task as completed with result"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.result = result
    
    def mark_failed(self, error: str):
        """Mark task as failed with error"""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()


@dataclass
class ExecutionPlan:
    """
    Represents a complete plan to achieve a goal
    
    Plans are the bridge between goals and tasks. They organize
    tasks into executable sequences with proper dependency handling.
    """
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal_id: str = ""
    tasks: List[Task] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    total_estimated_duration: Optional[timedelta] = None
    
    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[Task]:
        """Get tasks that are ready to execute"""
        return [
            task for task in self.tasks
            if task.status == TaskStatus.PENDING
            and task.can_execute(completed_tasks)
        ]
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Find a task by ID"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def is_complete(self) -> bool:
        """Check if all tasks are done (completed or failed)"""
        return all(
            task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
            for task in self.tasks
        )
    
    def success_rate(self) -> float:
        """Calculate the success rate of executed tasks"""
        executed = [t for t in self.tasks if t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]]
        if not executed:
            return 0.0
        successful = sum(1 for t in executed if t.status == TaskStatus.COMPLETED)
        return successful / len(executed)


@dataclass
class Reflection:
    """
    Represents agent's self-reflection on an execution
    
    Reflections enable learning and improvement over time.
    They transform experience into better future performance.
    """
    reflection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal_id: str = ""
    plan_id: str = ""
    outcome: str = ""  # What happened
    analysis: str = ""  # Why it happened
    lessons: List[str] = field(default_factory=list)  # What to do differently
    confidence: float = 0.0  # How confident in the analysis
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """
    Complete state of the agent at any point in time
    
    This is what flows through the LangGraph state machine.
    It contains everything needed to resume from interruption.
    """
    # Current activity
    current_goal: Optional[Goal] = None
    current_plan: Optional[ExecutionPlan] = None
    current_task: Optional[Task] = None
    
    # History
    completed_goals: List[Goal] = field(default_factory=list)
    completed_tasks: Set[str] = field(default_factory=set)
    reflections: List[Reflection] = field(default_factory=list)
    
    # Mode and configuration
    mode: AgentMode = AgentMode.HYBRID
    
    # Messages for LLM context
    messages: List[BaseMessage] = field(default_factory=list)
    
    # Metrics
    total_actions: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    def add_message(self, message: BaseMessage):
        """Add a message to context"""
        self.messages.append(message)
        # Keep context window manageable
        if len(self.messages) > 50:
            self.messages = self.messages[-30:]  # Keep recent context
    
    def get_success_rate(self) -> float:
        """Calculate overall success rate"""
        total = self.successful_actions + self.failed_actions
        return self.successful_actions / total if total > 0 else 0.0


# ==============================================================================
# Agent Base Class and Implementations
# ==============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for different agent architectures
    
    This defines the interface that all agent types must implement,
    allowing us to swap architectures based on task requirements.
    """
    
    def __init__(self, tools: List[BaseTool], memory_system: Optional[MemoryNetworkSystem] = None):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.memory_system = memory_system
        self.state = AgentState()
        
    @abstractmethod
    async def plan(self, goal: Goal) -> ExecutionPlan:
        """Create an execution plan for a goal"""
        pass
    
    @abstractmethod
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a single task"""
        pass
    
    @abstractmethod
    async def reflect(self, goal: Goal, plan: ExecutionPlan, outcome: str) -> Reflection:
        """Reflect on execution results"""
        pass
    
    async def pursue_goal(self, goal: Goal) -> Dict[str, Any]:
        """
        Main entry point: pursue a goal to completion
        
        This orchestrates the full agent lifecycle:
        1. Planning
        2. Execution with monitoring
        3. Reflection and learning
        """
        logger.info(f"Pursuing goal: {goal.description}")
        self.state.current_goal = goal
        
        try:
            # Create execution plan
            plan = await self.plan(goal)
            self.state.current_plan = plan
            
            # Execute plan
            execution_result = await self._execute_plan(plan)
            
            # Reflect on results
            outcome = "success" if execution_result["success"] else "failure"
            reflection = await self.reflect(goal, plan, outcome)
            self.state.reflections.append(reflection)
            
            # Mark goal as completed
            self.state.completed_goals.append(goal)
            self.state.current_goal = None
            
            return {
                "success": execution_result["success"],
                "goal": goal,
                "plan": plan,
                "execution_result": execution_result,
                "reflection": reflection
            }
            
        except Exception as e:
            logger.error(f"Error pursuing goal: {e}")
            return {
                "success": False,
                "goal": goal,
                "error": str(e)
            }
    
    async def _execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Execute a plan with proper task scheduling and error handling
        
        This is where the rubber meets the road - actual execution
        with all its messiness and need for adaptation.
        """
        console.print(f"\n[cyan]Executing plan with {len(plan.tasks)} tasks[/cyan]")
        
        results = []
        failed_tasks = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            while not plan.is_complete():
                # Get ready tasks
                ready_tasks = plan.get_ready_tasks(self.state.completed_tasks)
                
                if not ready_tasks and not self._has_running_tasks(plan):
                    # Deadlock - no tasks can proceed
                    console.print("[red]Plan deadlocked - no tasks can proceed[/red]")
                    break
                
                # Execute ready tasks (could be parallel in production)
                for task in ready_tasks:
                    task_progress = progress.add_task(task.description, total=None)
                    
                    try:
                        task.mark_started()
                        self.state.current_task = task
                        
                        # Execute with retries
                        result = await self._execute_task_with_retries(task)
                        
                        task.mark_completed(result)
                        self.state.completed_tasks.add(task.task_id)
                        self.state.successful_actions += 1
                        results.append(result)
                        
                        progress.update(task_progress, completed=True)
                        console.print(f"[green]✓ Completed: {task.description}[/green]")
                        
                    except Exception as e:
                        task.mark_failed(str(e))
                        self.state.failed_actions += 1
                        failed_tasks.append(task)
                        
                        progress.update(task_progress, completed=True)
                        console.print(f"[red]✗ Failed: {task.description} - {e}[/red]")
                        
                        # Decide whether to continue or abort
                        if task.priority == TaskPriority.CRITICAL:
                            console.print("[red]Critical task failed - aborting plan[/red]")
                            break
                
                # Small delay to prevent spinning
                await asyncio.sleep(0.1)
        
        success = len(failed_tasks) == 0 and plan.is_complete()
        
        return {
            "success": success,
            "completed_tasks": len(results),
            "failed_tasks": len(failed_tasks),
            "results": results,
            "failed_task_details": [
                {"task": t.description, "error": t.error} for t in failed_tasks
            ],
            "success_rate": plan.success_rate()
        }
    
    async def _execute_task_with_retries(self, task: Task) -> Any:
        """Execute a task with retry logic"""
        last_error = None
        
        while task.attempts < task.max_attempts:
            try:
                result = await self.execute_task(task)
                return result
            except Exception as e:
                last_error = e
                if task.attempts < task.max_attempts:
                    console.print(f"[yellow]Retry {task.attempts}/{task.max_attempts} for {task.description}[/yellow]")
                    await asyncio.sleep(2 ** task.attempts)  # Exponential backoff
        
        raise last_error
    
    def _has_running_tasks(self, plan: ExecutionPlan) -> bool:
        """Check if any tasks are currently running"""
        return any(task.status == TaskStatus.IN_PROGRESS for task in plan.tasks)


class ReActAgent(BaseAgent):
    """
    ReAct (Reasoning + Acting) Agent
    
    This architecture interleaves thinking and acting. The agent:
    1. Observes current state
    2. Thinks about what to do
    3. Acts
    4. Observes result
    5. Repeats until goal achieved
    
    Best for: Exploratory tasks, debugging, situations requiring adaptation
    Trade-off: More LLM calls but more flexible
    """
    
    async def plan(self, goal: Goal) -> ExecutionPlan:
        """
        ReAct doesn't do upfront planning - it plans one step at a time
        
        This is both a strength (adaptability) and weakness (can meander)
        """
        # Create a single-task plan that will be extended during execution
        initial_task = Task(
            goal_id=goal.goal_id,
            description="Analyze goal and determine first action",
            action="think",
            parameters={"goal": goal.description}
        )
        
        return ExecutionPlan(
            goal_id=goal.goal_id,
            tasks=[initial_task]
        )
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task in ReAct style - think, act, observe
        
        This is where ReAct's interleaved reasoning happens
        """
        if task.action == "think":
            # Reasoning step - use LLM to decide next action
            thought = await self._think_about_next_action(task)
            
            # Create next task based on reasoning
            if thought["next_action"] != "goal_achieved":
                next_task = Task(
                    goal_id=task.goal_id,
                    description=thought["action_description"],
                    action=thought["next_action"],
                    parameters=thought["action_parameters"]
                )
                
                # Add to plan dynamically
                if self.state.current_plan:
                    self.state.current_plan.tasks.append(next_task)
            
            return thought
            
        else:
            # Action step - execute tool
            if task.action in self.tool_map:
                tool = self.tool_map[task.action]
                result = await tool.arun(**task.parameters)
                
                # Create observation task
                observe_task = Task(
                    goal_id=task.goal_id,
                    description=f"Observe and think about result of {task.action}",
                    action="think",
                    parameters={
                        "previous_action": task.action,
                        "result": result
                    }
                )
                
                # Add to plan
                if self.state.current_plan:
                    self.state.current_plan.tasks.append(observe_task)
                
                return result
            else:
                raise ValueError(f"Unknown action: {task.action}")
    
    async def _think_about_next_action(self, task: Task) -> Dict[str, Any]:
        """
        Use LLM to reason about next action
        
        This is the "Reasoning" in ReAct - explicit thinking steps
        """
        # In production, this would call the LLM with current context
        # For now, return a mock reasoning result
        
        # Build context from state and previous results
        context = self._build_reasoning_context()
        
        # Mock LLM reasoning (would be actual LLM call)
        return {
            "thought": "Based on the goal and current progress, I should search for relevant information",
            "next_action": "search_tool",
            "action_description": "Search for information about the topic",
            "action_parameters": {"query": "example query", "max_results": 5},
            "confidence": 0.8
        }
    
    def _build_reasoning_context(self) -> str:
        """Build context for LLM reasoning"""
        context_parts = []
        
        if self.state.current_goal:
            context_parts.append(f"Goal: {self.state.current_goal.description}")
        
        # Add recent task results
        if self.state.current_plan:
            recent_tasks = [
                t for t in self.state.current_plan.tasks[-5:]
                if t.status == TaskStatus.COMPLETED
            ]
            for task in recent_tasks:
                context_parts.append(f"Completed: {task.description}")
                if task.result:
                    context_parts.append(f"Result: {task.result}")
        
        return "\n".join(context_parts)
    
    async def reflect(self, goal: Goal, plan: ExecutionPlan, outcome: str) -> Reflection:
        """
        Reflect on ReAct execution
        
        ReAct reflection focuses on reasoning quality and action choices
        """
        # Analyze reasoning/action patterns
        reasoning_tasks = [t for t in plan.tasks if t.action == "think"]
        action_tasks = [t for t in plan.tasks if t.action != "think"]
        
        # Mock reflection (would use LLM in production)
        return Reflection(
            goal_id=goal.goal_id,
            plan_id=plan.plan_id,
            outcome=outcome,
            analysis=f"Executed {len(reasoning_tasks)} reasoning steps and {len(action_tasks)} actions",
            lessons=[
                "Consider batching similar actions for efficiency",
                "Add more specific success criteria checks",
                "Improve context management for better reasoning"
            ],
            confidence=0.75
        )


class PlanAndExecuteAgent(BaseAgent):
    """
    Plan-and-Execute Agent
    
    This architecture separates planning from execution:
    1. Create complete plan upfront
    2. Execute plan step by step
    3. Handle failures with replanning
    
    Best for: Well-defined tasks, multi-step procedures, predictable domains
    Trade-off: Less flexible but more efficient
    """
    
    async def plan(self, goal: Goal) -> ExecutionPlan:
        """
        Create a complete execution plan upfront
        
        This is where Plan-and-Execute shines - thoughtful upfront planning
        """
        console.print(f"\n[cyan]Creating execution plan for: {goal.description}[/cyan]")
        
        # Decompose goal into subtasks
        subtasks = await self._decompose_goal(goal)
        
        # Convert subtasks to executable tasks with dependencies
        tasks = []
        task_map = {}  # For dependency resolution
        
        for i, subtask in enumerate(subtasks):
            task = Task(
                goal_id=goal.goal_id,
                description=subtask["description"],
                action=subtask["action"],
                parameters=subtask["parameters"],
                dependencies=subtask.get("dependencies", []),
                priority=TaskPriority(subtask.get("priority", 3)),
                estimated_duration=timedelta(seconds=subtask.get("estimated_seconds", 60))
            )
            tasks.append(task)
            task_map[subtask["id"]] = task.task_id
        
        # Resolve dependency IDs
        for task in tasks:
            task.dependencies = [
                task_map.get(dep, dep) for dep in task.dependencies
            ]
        
        # Calculate total estimated duration
        total_duration = sum(
            (t.estimated_duration for t in tasks if t.estimated_duration),
            timedelta()
        )
        
        plan = ExecutionPlan(
            goal_id=goal.goal_id,
            tasks=tasks,
            total_estimated_duration=total_duration
        )
        
        # Visualize plan
        self._visualize_plan(plan)
        
        return plan
    
    async def _decompose_goal(self, goal: Goal) -> List[Dict[str, Any]]:
        """
        Decompose a high-level goal into executable subtasks
        
        This is the critical planning step - breaking down complexity
        """
        # In production, this would use LLM with planning prompts
        # For demonstration, return a mock decomposition
        
        # Example: "Research and summarize recent advances in quantum computing"
        subtasks = [
            {
                "id": "search_recent",
                "description": "Search for recent quantum computing papers",
                "action": "search_tool",
                "parameters": {
                    "query": "quantum computing advances 2023-2024",
                    "search_type": "academic",
                    "max_results": 10
                },
                "dependencies": [],
                "priority": 2,
                "estimated_seconds": 30
            },
            {
                "id": "analyze_papers",
                "description": "Analyze and extract key findings",
                "action": "analyze_data_tool",
                "parameters": {
                    "analysis_type": "summary"
                },
                "dependencies": ["search_recent"],
                "priority": 2,
                "estimated_seconds": 60
            },
            {
                "id": "search_applications",
                "description": "Search for practical applications",
                "action": "search_tool",
                "parameters": {
                    "query": "quantum computing real-world applications",
                    "search_type": "general",
                    "max_results": 5
                },
                "dependencies": [],
                "priority": 3,
                "estimated_seconds": 30
            },
            {
                "id": "create_summary",
                "description": "Create comprehensive summary document",
                "action": "file_operation_tool",
                "parameters": {
                    "operation": "write",
                    "path": "./adam_workspace/quantum_computing_summary.md"
                },
                "dependencies": ["analyze_papers", "search_applications"],
                "priority": 1,
                "estimated_seconds": 45
            }
        ]
        
        return subtasks
    
    def _visualize_plan(self, plan: ExecutionPlan):
        """Visualize the execution plan as a tree"""
        tree = Tree(f"[bold]Execution Plan ({len(plan.tasks)} tasks)[/bold]")
        
        # Group tasks by dependencies
        no_deps = [t for t in plan.tasks if not t.dependencies]
        
        def add_task_to_tree(task: Task, parent):
            task_node = parent.add(
                f"[cyan]{task.description}[/cyan] "
                f"[dim]({task.action}) [{task.priority.name}][/dim]"
            )
            
            # Find tasks that depend on this one
            dependents = [
                t for t in plan.tasks
                if task.task_id in t.dependencies
            ]
            
            for dep in dependents:
                add_task_to_tree(dep, task_node)
        
        # Add root tasks
        for task in no_deps:
            add_task_to_tree(task, tree)
        
        console.print(tree)
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a planned task
        
        Plan-and-Execute execution is straightforward - just run the task
        """
        if task.action in self.tool_map:
            tool = self.tool_map[task.action]
            
            # Add progress monitoring
            console.print(f"[dim]Executing: {task.action} with {task.parameters}[/dim]")
            
            result = await tool.arun(**task.parameters)
            
            # Store result in memory if valuable
            if self.memory_system and self._is_valuable_result(result):
                self.memory_system.add_memory(
                    memory_id=f"task_{task.task_id}",
                    conversation_id=f"goal_{task.goal_id}",
                    query=task.description,
                    response=json.dumps(result),
                    topics=["agent_execution", task.action],
                    memory_type="task_result"
                )
            
            return result
        else:
            raise ValueError(f"Unknown action: {task.action}")
    
    def _is_valuable_result(self, result: Any) -> bool:
        """Determine if a result is worth storing in memory"""
        # Store non-trivial results
        if isinstance(result, dict):
            return result.get("success", False) and len(str(result)) > 100
        return False
    
    async def reflect(self, goal: Goal, plan: ExecutionPlan, outcome: str) -> Reflection:
        """
        Reflect on plan execution
        
        Plan-and-Execute reflection focuses on plan quality and execution efficiency
        """
        # Analyze plan execution
        completed = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED]
        failed = [t for t in plan.tasks if t.status == TaskStatus.FAILED]
        
        # Calculate metrics
        avg_duration = sum(
            (t.completed_at - t.started_at).total_seconds()
            for t in completed
            if t.completed_at and t.started_at
        ) / len(completed) if completed else 0
        
        lessons = []
        
        if failed:
            lessons.append(f"Failed tasks indicate plan gaps: {[t.description for t in failed]}")
        
        if avg_duration > 60:
            lessons.append("Consider breaking down long-running tasks")
        
        if plan.success_rate() < 0.8:
            lessons.append("Low success rate suggests need for more robust planning")
        
        return Reflection(
            goal_id=goal.goal_id,
            plan_id=plan.plan_id,
            outcome=outcome,
            analysis=f"Plan had {len(completed)} successes and {len(failed)} failures",
            lessons=lessons,
            confidence=0.8,
            metadata={
                "avg_task_duration": avg_duration,
                "success_rate": plan.success_rate()
            }
        )


class ReflexionAgent(BaseAgent):
    """
    Reflexion Agent
    
    This architecture emphasizes learning from experience:
    1. Execute tasks
    2. Reflect deeply on outcomes
    3. Update strategies based on reflection
    4. Retry with improved approach
    
    Best for: Complex tasks requiring iteration, learning scenarios
    Trade-off: Slower but continuously improving
    """
    
    def __init__(self, tools: List[BaseTool], memory_system: Optional[MemoryNetworkSystem] = None):
        super().__init__(tools, memory_system)
        self.strategy_memory = {}  # Learned strategies
        self.failure_patterns = defaultdict(list)  # Track failure patterns
    
    async def plan(self, goal: Goal) -> ExecutionPlan:
        """
        Create plan informed by past reflections
        
        Reflexion planning incorporates learned strategies
        """
        # Check if we have strategies for similar goals
        similar_strategies = self._find_similar_strategies(goal)
        
        if similar_strategies:
            console.print(f"[green]Found {len(similar_strategies)} relevant strategies from past experience[/green]")
            plan = await self._plan_with_strategies(goal, similar_strategies)
        else:
            console.print("[yellow]No prior experience - creating initial plan[/yellow]")
            plan = await self._create_initial_plan(goal)
        
        return plan
    
    def _find_similar_strategies(self, goal: Goal) -> List[Dict[str, Any]]:
        """Find strategies from similar past goals"""
        similar = []
        
        for past_goal_id, strategy in self.strategy_memory.items():
            # Simple similarity check (in production, use embeddings)
            if any(keyword in goal.description.lower() for keyword in strategy["keywords"]):
                similar.append(strategy)
        
        return similar
    
    async def _plan_with_strategies(self, goal: Goal, strategies: List[Dict[str, Any]]) -> ExecutionPlan:
        """Create plan incorporating learned strategies"""
        # Merge strategies and create improved plan
        tasks = []
        
        # Add tasks from successful strategies
        for strategy in strategies:
            for task_template in strategy["successful_patterns"]:
                task = Task(
                    goal_id=goal.goal_id,
                    description=task_template["description"],
                    action=task_template["action"],
                    parameters=task_template["parameters"],
                    priority=TaskPriority.HIGH  # Proven strategies get priority
                )
                tasks.append(task)
        
        # Avoid tasks from failure patterns
        for pattern in self.failure_patterns.get(goal.description, []):
            console.print(f"[yellow]Avoiding previously failed approach: {pattern}[/yellow]")
        
        return ExecutionPlan(goal_id=goal.goal_id, tasks=tasks)
    
    async def _create_initial_plan(self, goal: Goal) -> ExecutionPlan:
        """Create an initial plan without prior experience"""
        # Start with exploration
        tasks = [
            Task(
                goal_id=goal.goal_id,
                description="Explore approaches to goal",
                action="search_tool",
                parameters={"query": goal.description, "search_type": "general"}
            ),
            Task(
                goal_id=goal.goal_id,
                description="Analyze findings and create approach",
                action="analyze_data_tool",
                parameters={"analysis_type": "summary"}
            )
        ]
        
        return ExecutionPlan(goal_id=goal.goal_id, tasks=tasks)
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute with reflection checkpoints
        
        Reflexion execution includes periodic self-assessment
        """
        # Execute the task
        result = await super().execute_task(task)
        
        # Reflect on immediate outcome
        if not result.get("success", True):
            # Record failure pattern
            self.failure_patterns[self.state.current_goal.description].append({
                "task": task.description,
                "action": task.action,
                "error": result.get("error", "Unknown error")
            })
        
        return result
    
    async def reflect(self, goal: Goal, plan: ExecutionPlan, outcome: str) -> Reflection:
        """
        Deep reflection with strategy extraction
        
        This is where Reflexion truly shines - learning from experience
        """
        # Perform deep analysis
        reflection = await self._deep_reflect(goal, plan, outcome)
        
        # Extract and store strategies
        if outcome == "success":
            strategy = self._extract_strategy(goal, plan, reflection)
            self.strategy_memory[goal.goal_id] = strategy
            console.print("[green]Stored successful strategy for future use[/green]")
        
        # Update failure patterns
        if outcome == "failure":
            self._update_failure_patterns(goal, plan)
        
        return reflection
    
    async def _deep_reflect(self, goal: Goal, plan: ExecutionPlan, outcome: str) -> Reflection:
        """Perform deep reflection on execution"""
        # Analyze what worked and what didn't
        successful_tasks = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in plan.tasks if t.status == TaskStatus.FAILED]
        
        # Look for patterns
        successful_actions = [t.action for t in successful_tasks]
        failed_actions = [t.action for t in failed_tasks]
        
        analysis_parts = []
        lessons = []
        
        if successful_tasks:
            most_successful = Counter(successful_actions).most_common(1)[0]
            analysis_parts.append(f"{most_successful[0]} was most reliable ({most_successful[1]} successes)")
            lessons.append(f"Prioritize {most_successful[0]} for similar goals")
        
        if failed_tasks:
            most_failed = Counter(failed_actions).most_common(1)[0]
            analysis_parts.append(f"{most_failed[0]} had issues ({most_failed[1]} failures)")
            lessons.append(f"Improve error handling for {most_failed[0]}")
        
        # Check timing patterns
        if successful_tasks:
            durations = [
                (t.completed_at - t.started_at).total_seconds()
                for t in successful_tasks
                if t.completed_at and t.started_at
            ]
            if durations:
                avg_duration = sum(durations) / len(durations)
                if avg_duration > 30:
                    lessons.append("Consider parallel execution for time-consuming tasks")
        
        return Reflection(
            goal_id=goal.goal_id,
            plan_id=plan.plan_id,
            outcome=outcome,
            analysis=" ".join(analysis_parts),
            lessons=lessons,
            confidence=0.85,
            metadata={
                "successful_actions": successful_actions,
                "failed_actions": failed_actions
            }
        )
    
    def _extract_strategy(self, goal: Goal, plan: ExecutionPlan, reflection: Reflection) -> Dict[str, Any]:
        """Extract reusable strategy from successful execution"""
        successful_tasks = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED]
        
        # Create task templates
        successful_patterns = [
            {
                "description": task.description,
                "action": task.action,
                "parameters": task.parameters
            }
            for task in successful_tasks
        ]
        
        # Extract keywords for matching
        keywords = set()
        for word in goal.description.lower().split():
            if len(word) > 3:  # Skip short words
                keywords.add(word)
        
        return {
            "goal_description": goal.description,
            "keywords": list(keywords),
            "successful_patterns": successful_patterns,
            "lessons": reflection.lessons,
            "success_rate": plan.success_rate(),
            "created_at": datetime.now().isoformat()
        }
    
    def _update_failure_patterns(self, goal: Goal, plan: ExecutionPlan):
        """Update patterns to avoid in future"""
        failed_tasks = [t for t in plan.tasks if t.status == TaskStatus.FAILED]
        
        for task in failed_tasks:
            pattern = {
                "action": task.action,
                "parameters": task.parameters,
                "error": task.error,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store by goal type for pattern matching
            goal_type = self._categorize_goal(goal)
            self.failure_patterns[goal_type].append(pattern)
    
    def _categorize_goal(self, goal: Goal) -> str:
        """Categorize goal for pattern matching"""
        # Simple keyword-based categorization
        description_lower = goal.description.lower()
        
        if any(word in description_lower for word in ["research", "find", "search"]):
            return "research"
        elif any(word in description_lower for word in ["create", "write", "generate"]):
            return "creation"
        elif any(word in description_lower for word in ["analyze", "evaluate", "assess"]):
            return "analysis"
        else:
            return "general"


class HybridAgent(BaseAgent):
    """
    Hybrid Adaptive Agent
    
    This agent dynamically selects the best architecture for each goal:
    - ReAct for exploration
    - Plan-and-Execute for structured tasks
    - Reflexion for learning opportunities
    
    This demonstrates meta-learning: learning which approach works best when
    """
    
    def __init__(self, tools: List[BaseTool], memory_system: Optional[MemoryNetworkSystem] = None):
        super().__init__(tools, memory_system)
        
        # Initialize sub-agents
        self.react_agent = ReActAgent(tools, memory_system)
        self.plan_execute_agent = PlanAndExecuteAgent(tools, memory_system)
        self.reflexion_agent = ReflexionAgent(tools, memory_system)
        
        # Track performance of each approach
        self.approach_performance = {
            AgentMode.REACT: {"successes": 0, "failures": 0},
            AgentMode.PLAN_EXECUTE: {"successes": 0, "failures": 0},
            AgentMode.REFLEXION: {"successes": 0, "failures": 0}
        }
    
    def _select_approach(self, goal: Goal) -> AgentMode:
        """
        Select the best approach for a given goal
        
        This is the key innovation - adaptive strategy selection
        """
        goal_lower = goal.description.lower()
        
        # Heuristics for approach selection
        if any(word in goal_lower for word in ["explore", "investigate", "debug", "understand"]):
            # Exploratory tasks benefit from ReAct's flexibility
            return AgentMode.REACT
            
        elif any(word in goal_lower for word in ["implement", "create", "build", "deploy"]):
            # Structured tasks benefit from planning
            return AgentMode.PLAN_EXECUTE
            
        elif any(word in goal_lower for word in ["improve", "optimize", "learn", "iterate"]):
            # Learning tasks benefit from reflection
            return AgentMode.REFLEXION
            
        else:
            # Default: choose based on past performance
            return self._choose_by_performance()
    
    def _choose_by_performance(self) -> AgentMode:
        """Choose approach based on historical performance"""
        best_mode = AgentMode.PLAN_EXECUTE  # Default
        best_rate = 0.0
        
        for mode, perf in self.approach_performance.items():
            total = perf["successes"] + perf["failures"]
            if total > 0:
                success_rate = perf["successes"] / total
                if success_rate > best_rate:
                    best_rate = success_rate
                    best_mode = mode
        
        return best_mode
    
    async def pursue_goal(self, goal: Goal) -> Dict[str, Any]:
        """Override to add approach selection"""
        # Select approach
        approach = self._select_approach(goal)
        console.print(f"\n[cyan]Selected approach: {approach.value}[/cyan]")
        
        # Delegate to appropriate agent
        if approach == AgentMode.REACT:
            agent = self.react_agent
        elif approach == AgentMode.PLAN_EXECUTE:
            agent = self.plan_execute_agent
        else:
            agent = self.reflexion_agent
        
        # Execute
        result = await agent.pursue_goal(goal)
        
        # Update performance tracking
        if result["success"]:
            self.approach_performance[approach]["successes"] += 1
        else:
            self.approach_performance[approach]["failures"] += 1
        
        # Add approach info to result
        result["approach_used"] = approach.value
        
        return result
    
    async def plan(self, goal: Goal) -> ExecutionPlan:
        """Delegate to selected agent"""
        approach = self._select_approach(goal)
        
        if approach == AgentMode.REACT:
            return await self.react_agent.plan(goal)
        elif approach == AgentMode.PLAN_EXECUTE:
            return await self.plan_execute_agent.plan(goal)
        else:
            return await self.reflexion_agent.plan(goal)
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Delegate to selected agent"""
        # Use the agent that created the plan
        if hasattr(self, '_current_agent'):
            return await self._current_agent.execute_task(task)
        else:
            # Fallback to plan-execute
            return await self.plan_execute_agent.execute_task(task)
    
    async def reflect(self, goal: Goal, plan: ExecutionPlan, outcome: str) -> Reflection:
        """Comprehensive reflection across all approaches"""
        # Get reflections from all agents for comparison
        reflections = []
        
        if hasattr(self, '_current_agent'):
            reflection = await self._current_agent.reflect(goal, plan, outcome)
            reflections.append(reflection)
        
        # Meta-reflection on approach selection
        meta_lessons = []
        
        if outcome == "success":
            meta_lessons.append(f"Approach selection was correct for this goal type")
        else:
            meta_lessons.append(f"Consider different approach for similar goals")
        
        # Combine reflections
        if reflections:
            main_reflection = reflections[0]
            main_reflection.lessons.extend(meta_lessons)
            return main_reflection
        else:
            return Reflection(
                goal_id=goal.goal_id,
                plan_id=plan.plan_id,
                outcome=outcome,
                analysis="Hybrid agent reflection",
                lessons=meta_lessons,
                confidence=0.9
            )


# ==============================================================================
# Agent System Integration with LangGraph
# ==============================================================================

class ADAMAgentSystem:
    """
    Main agent system that integrates with ADAM's existing architecture
    
    This class bridges the gap between the agent implementations above
    and ADAM's existing LangGraph-based conversation system.
    """
    
    def __init__(
        self,
        memory_system: MemoryNetworkSystem,
        conversation_memory: ConversationAwareMemorySystem,
        mode: AgentMode = AgentMode.HYBRID
    ):
        self.memory_system = memory_system
        self.conversation_memory = conversation_memory
        
        # Get all available tools
        self.tools = get_all_tools()
        
        # Initialize appropriate agent
        if mode == AgentMode.HYBRID:
            self.agent = HybridAgent(self.tools, memory_system)
        elif mode == AgentMode.REACT:
            self.agent = ReActAgent(self.tools, memory_system)
        elif mode == AgentMode.PLAN_EXECUTE:
            self.agent = PlanAndExecuteAgent(self.tools, memory_system)
        else:
            self.agent = ReflexionAgent(self.tools, memory_system)
        
        # Active goals
        self.active_goals: List[Goal] = []
        self.goal_history: List[Dict[str, Any]] = []
        
        # Create LangGraph state machine
        self.graph = self._create_state_graph()
        
        # Proactive monitoring
        self.monitoring_active = False
        self.monitoring_task = None
    
    def _create_state_graph(self) -> StateGraph:
        """
        Create LangGraph state machine for agent operation
        
        This integrates with ADAM's existing conversation flow
        """
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("receive_input", self._receive_input)
        workflow.add_node("create_goal", self._create_goal)
        workflow.add_node("plan", self._plan)
        workflow.add_node("execute", self._execute)
        workflow.add_node("reflect", self._reflect)
        workflow.add_node("respond", self._respond)
        
        # Add edges
        workflow.add_edge("receive_input", "create_goal")
        workflow.add_edge("create_goal", "plan")
        workflow.add_edge("plan", "execute")
        workflow.add_edge("execute", "reflect")
        workflow.add_edge("reflect", "respond")
        workflow.add_edge("respond", END)
        
        # Set entry point
        workflow.set_entry_point("receive_input")
        
        return workflow.compile()
    
    async def _receive_input(self, state: AgentState) -> AgentState:
        """Process user input and determine if it's a goal or question"""
        # This would integrate with ADAM's existing input processing
        return state
    
    async def _create_goal(self, state: AgentState) -> AgentState:
        """Convert user input into a structured goal"""
        # Extract goal from messages
        if state.messages:
            last_message = state.messages[-1]
            
            # Create goal from user message
            goal = Goal(
                description=last_message.content,
                success_criteria=self._extract_success_criteria(last_message.content),
                constraints=self._extract_constraints(last_message.content),
                priority=self._determine_priority(last_message.content)
            )
            
            state.current_goal = goal
            self.active_goals.append(goal)
        
        return state
    
    def _extract_success_criteria(self, text: str) -> List[str]:
        """Extract success criteria from goal description"""
        # In production, use NLP or LLM
        criteria = []
        
        if "summarize" in text.lower():
            criteria.append("Create comprehensive summary")
        if "research" in text.lower():
            criteria.append("Find relevant information")
        if "implement" in text.lower():
            criteria.append("Create working implementation")
        
        return criteria if criteria else ["Complete the requested task"]
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Extract constraints from goal description"""
        constraints = []
        
        if "quick" in text.lower() or "fast" in text.lower():
            constraints.append("Time-sensitive - optimize for speed")
        if "accurate" in text.lower() or "precise" in text.lower():
            constraints.append("Accuracy is critical")
        if "simple" in text.lower():
            constraints.append("Keep solution simple")
        
        return constraints
    
    def _determine_priority(self, text: str) -> TaskPriority:
        """Determine priority from goal description"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["urgent", "asap", "critical"]):
            return TaskPriority.CRITICAL
        elif any(word in text_lower for word in ["important", "priority"]):
            return TaskPriority.HIGH
        elif any(word in text_lower for word in ["whenever", "low priority"]):
            return TaskPriority.LOW
        else:
            return TaskPriority.MEDIUM
    
    async def _plan(self, state: AgentState) -> AgentState:
        """Create execution plan"""
        if state.current_goal:
            plan = await self.agent.plan(state.current_goal)
            state.current_plan = plan
        return state
    
    async def _execute(self, state: AgentState) -> AgentState:
        """Execute the plan"""
        if state.current_goal and state.current_plan:
            result = await self.agent.pursue_goal(state.current_goal)
            
            # Store in goal history
            self.goal_history.append({
                "goal": state.current_goal,
                "result": result,
                "timestamp": datetime.now()
            })
            
            # Remove from active goals
            self.active_goals = [
                g for g in self.active_goals
                if g.goal_id != state.current_goal.goal_id
            ]
        
        return state
    
    async def _reflect(self, state: AgentState) -> AgentState:
        """Reflect on execution"""
        # Reflection happens within pursue_goal
        return state
    
    async def _respond(self, state: AgentState) -> AgentState:
        """Generate response to user"""
        if state.current_goal:
            # Create response message
            response = self._generate_response(state)
            state.add_message(AIMessage(content=response))
        
        return state
    
    def _generate_response(self, state: AgentState) -> str:
        """Generate human-friendly response"""
        goal = state.current_goal
        plan = state.current_plan
        
        if not goal or not plan:
            return "I'm ready to help you achieve your goals. What would you like me to work on?"
        
        # Build response
        response_parts = [
            f"I've completed working on: {goal.description}\n"
        ]
        
        if plan.is_complete():
            success_rate = plan.success_rate()
            if success_rate == 1.0:
                response_parts.append("✅ All tasks completed successfully!")
            elif success_rate > 0.5:
                response_parts.append(f"⚠️ Partially completed ({success_rate:.0%} success rate)")
            else:
                response_parts.append(f"❌ Encountered difficulties ({success_rate:.0%} success rate)")
        
        # Add summary of what was done
        completed_tasks = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED]
        if completed_tasks:
            response_parts.append("\nWhat I did:")
            for task in completed_tasks[:5]:  # Show first 5
                response_parts.append(f"- {task.description}")
        
        # Add key results
        if state.reflections:
            latest_reflection = state.reflections[-1]
            if latest_reflection.lessons:
                response_parts.append("\nKey learnings:")
                for lesson in latest_reflection.lessons[:3]:
                    response_parts.append(f"- {lesson}")
        
        return "\n".join(response_parts)
    
    # ===========================================================================
    # Proactive Capabilities
    # ===========================================================================
    
    async def start_proactive_monitoring(self):
        """
        Start proactive monitoring for opportunities to help
        
        This is what transforms ADAM from reactive to proactive
        """
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        console.print("[green]Proactive monitoring started[/green]")
    
    async def stop_proactive_monitoring(self):
        """Stop proactive monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        console.print("[yellow]Proactive monitoring stopped[/yellow]")
    
    async def _monitoring_loop(self):
        """
        Main monitoring loop that looks for opportunities to help
        
        This demonstrates true agency - acting without being asked
        """
        while self.monitoring_active:
            try:
                # Check various conditions
                suggestions = []
                
                # Monitor system resources
                resource_suggestion = await self._check_system_resources()
                if resource_suggestion:
                    suggestions.append(resource_suggestion)
                
                # Monitor for scheduled tasks
                scheduled_suggestion = await self._check_scheduled_tasks()
                if scheduled_suggestion:
                    suggestions.append(scheduled_suggestion)
                
                # Monitor for optimization opportunities
                optimization_suggestion = await self._check_optimization_opportunities()
                if optimization_suggestion:
                    suggestions.append(optimization_suggestion)
                
                # Present suggestions
                if suggestions:
                    await self._present_suggestions(suggestions)
                
                # Sleep before next check
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_system_resources(self) -> Optional[Dict[str, Any]]:
        """Check system resources and suggest optimizations"""
        # Use monitoring tool
        result = await self.agent.tool_map["monitor_resources_tool"].arun()
        
        if result.get("success"):
            system = result.get("system", {})
            
            # Check for issues
            if system.get("cpu_percent", 0) > 80:
                return {
                    "type": "resource_optimization",
                    "issue": "High CPU usage detected",
                    "suggestion": "I can help identify and optimize CPU-intensive processes",
                    "priority": TaskPriority.HIGH
                }
            
            if system.get("disk_percent", 0) > 90:
                return {
                    "type": "resource_optimization", 
                    "issue": "Low disk space",
                    "suggestion": "I can help clean up unnecessary files and optimize storage",
                    "priority": TaskPriority.CRITICAL
                }
        
        return None
    
    async def _check_scheduled_tasks(self) -> Optional[Dict[str, Any]]:
        """Check for scheduled or recurring tasks"""
        # Check active goals for deadlines
        for goal in self.active_goals:
            if goal.deadline:
                time_remaining = goal.deadline - datetime.now()
                
                if timedelta(0) < time_remaining < timedelta(hours=1):
                    return {
                        "type": "deadline_reminder",
                        "issue": f"Goal deadline approaching: {goal.description}",
                        "suggestion": "I should prioritize completing this goal",
                        "priority": TaskPriority.CRITICAL,
                        "goal": goal
                    }
        
        return None
    
    async def _check_optimization_opportunities(self) -> Optional[Dict[str, Any]]:
        """Check for opportunities to optimize workflows"""
        # Analyze recent task patterns
        if len(self.goal_history) >= 5:
            # Look for repeated similar goals
            recent_goals = self.goal_history[-10:]
            
            # Simple pattern detection (in production, use better NLP)
            goal_types = defaultdict(int)
            for entry in recent_goals:
                goal_desc = entry["goal"].description.lower()
                if "search" in goal_desc:
                    goal_types["search"] += 1
                elif "summarize" in goal_desc:
                    goal_types["summarize"] += 1
                elif "analyze" in goal_desc:
                    goal_types["analyze"] += 1
            
            # Suggest automation for frequent tasks
            most_common = max(goal_types.items(), key=lambda x: x[1]) if goal_types else None
            
            if most_common and most_common[1] >= 3:
                return {
                    "type": "workflow_optimization",
                    "issue": f"Frequent {most_common[0]} tasks detected",
                    "suggestion": f"I can create an automated workflow for {most_common[0]} tasks",
                    "priority": TaskPriority.MEDIUM
                }
        
        return None
    
    async def _present_suggestions(self, suggestions: List[Dict[str, Any]]):
        """Present proactive suggestions to user"""
        console.print("\n[cyan]🤖 ADAM has some suggestions:[/cyan]")
        
        for i, suggestion in enumerate(suggestions, 1):
            console.print(f"\n{i}. [{suggestion['priority'].name}] {suggestion['issue']}")
            console.print(f"   💡 {suggestion['suggestion']}")
        
        # In a real implementation, this would create goals
        # or wait for user approval
    
    # ===========================================================================
    # Public Interface
    # ===========================================================================
    
    async def process_message(self, message: str) -> str:
        """
        Main entry point for processing user messages
        
        This integrates with ADAM's existing conversation system
        """
        # Create state with message
        state = AgentState()
        state.add_message(HumanMessage(content=message))
        
        # Run through state graph
        final_state = await self.graph.ainvoke(state)
        
        # Return response
        if final_state.messages:
            return final_state.messages[-1].content
        else:
            return "I'm processing your request..."
    
    def get_active_goals(self) -> List[Goal]:
        """Get list of active goals"""
        return self.active_goals
    
    def get_goal_history(self) -> List[Dict[str, Any]]:
        """Get history of completed goals"""
        return self.goal_history
    
    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            "active_goals": len(self.active_goals),
            "completed_goals": len(self.goal_history),
            "success_rate": self.agent.state.get_success_rate(),
            "total_actions": self.agent.state.total_actions,
            "uptime": (datetime.now() - self.agent.state.start_time).total_seconds()
        }


# ==============================================================================
# Monitoring Dashboard
# ==============================================================================

class AgentMonitoringDashboard:
    """
    Real-time monitoring dashboard for agent activities
    
    This provides visibility into what the agent is doing,
    essential for trust and debugging.
    """
    
    def __init__(self, agent_system: ADAMAgentSystem):
        self.agent_system = agent_system
        self.console = Console()
    
    def display(self):
        """Display the monitoring dashboard"""
        # Clear screen
        self.console.clear()
        
        # Header
        self.console.print(Panel.fit(
            "[bold cyan]ADAM Agent Monitoring Dashboard[/bold cyan]\n" +
            f"[dim]Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            border_style="cyan"
        ))
        
        # Metrics
        metrics = self.agent_system.get_agent_metrics()
        metrics_table = Table(title="System Metrics", box=None)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        
        metrics_table.add_row("Active Goals", str(metrics["active_goals"]))
        metrics_table.add_row("Completed Goals", str(metrics["completed_goals"]))
        metrics_table.add_row("Success Rate", f"{metrics['success_rate']:.1%}")
        metrics_table.add_row("Total Actions", str(metrics["total_actions"]))
        metrics_table.add_row("Uptime", f"{metrics['uptime']:.0f}s")
        
        self.console.print(metrics_table)
        
        # Active Goals
        if self.agent_system.active_goals:
            goals_table = Table(title="Active Goals", box=None)
            goals_table.add_column("Goal", style="yellow")
            goals_table.add_column("Priority", style="magenta")
            goals_table.add_column("Created", style="dim")
            
            for goal in self.agent_system.active_goals[:5]:
                goals_table.add_row(
                    goal.description[:50] + "...",
                    goal.priority.name,
                    goal.created_at.strftime("%H:%M:%S")
                )
            
            self.console.print(goals_table)
        
        # Current Activity
        agent_state = self.agent_system.agent.state
        if agent_state.current_task:
            self.console.print(Panel(
                f"[yellow]Current Task:[/yellow] {agent_state.current_task.description}\n" +
                f"[dim]Action: {agent_state.current_task.action}[/dim]",
                title="Current Activity",
                border_style="yellow"
            ))
        
        # Recent Completions
        if self.agent_system.goal_history:
            recent = self.agent_system.goal_history[-3:]
            
            completions = Table(title="Recent Completions", box=None)
            completions.add_column("Goal", style="green")
            completions.add_column("Result", style="cyan")
            completions.add_column("Time", style="dim")
            
            for entry in recent:
                goal = entry["goal"]
                result = entry["result"]
                completions.add_row(
                    goal.description[:40] + "...",
                    "✅ Success" if result.get("success") else "❌ Failed",
                    entry["timestamp"].strftime("%H:%M:%S")
                )
            
            self.console.print(completions)
    
    async def live_monitor(self):
        """Run live monitoring dashboard"""
        with Live(self.display(), refresh_per_second=1, console=self.console) as live:
            while True:
                await asyncio.sleep(1)
                live.update(self.display())


if __name__ == "__main__":
    # Quick test of agent system
    console.print("[bold cyan]ADAM Agent System Test[/bold cyan]\n")
    
    # This would integrate with actual ADAM systems
    # For now, show the architecture
    console.print("Agent Architectures Available:")
    console.print("1. [cyan]ReAct[/cyan] - Reasoning + Acting loops")
    console.print("2. [green]Plan-and-Execute[/green] - Upfront planning")
    console.print("3. [yellow]Reflexion[/yellow] - Self-reflective learning")
    console.print("4. [magenta]Hybrid[/magenta] - Adaptive selection")
    
    console.print("\nThis transforms ADAM from:")
    console.print("❌ User asks → ADAM responds → Done")
    console.print("\nInto:")
    console.print("✅ User sets goal → ADAM plans → executes → monitors → achieves → learns")