"""
Autonomous Agent Orchestrator using LangGraph
Manages multi-step task execution without waiting for user input
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import asyncio
import json
import logging
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_RETRY = "needs_retry"

@dataclass
class Task:
    """Represents a single task to execute"""
    action: str
    params: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self):
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        return data

class AgentState(TypedDict):
    """State that flows through the graph"""
    # Input
    user_message: str
    workspace_path: Optional[str]
    
    # Planning
    intent: str
    tasks: List[Task]
    current_task_index: int
    
    # Execution
    results: List[Dict[str, Any]]
    intermediate_outputs: List[str]
    
    # Validation
    validation_passed: bool
    validation_feedback: str
    retry_needed: bool
    
    # Output
    final_response: str
    error: Optional[str]
    status: str
    execution_time: float

class ADAMOrchestrator:
    """
    Main orchestrator that manages autonomous task execution
    Uses LangGraph to create a workflow of nodes and edges
    """
    
    def __init__(self):
        """Initialize the orchestrator with LangGraph workflow"""
        self.workflow = None
        self.app = None
        self._setup_workflow()
        logger.info("🚀 ADAM Orchestrator initialized with LangGraph")
    
    def _setup_workflow(self):
        """Create the LangGraph workflow"""
        # Create the state graph
        self.workflow = StateGraph(AgentState)
        
        # Add nodes - each node is a step in the process
        self.workflow.add_node("interpret", self.interpret_node)
        self.workflow.add_node("plan", self.plan_node)
        self.workflow.add_node("execute", self.execute_node)
        self.workflow.add_node("validate", self.validate_node)
        self.workflow.add_node("respond", self.respond_node)
        self.workflow.add_node("error", self.error_node)
        
        # Set entry point
        self.workflow.set_entry_point("interpret")
        
        # Add edges - define the flow
        self.workflow.add_edge("interpret", "plan")
        self.workflow.add_edge("plan", "execute")
        
        # Conditional edge from execute
        self.workflow.add_conditional_edges(
            "execute",
            self._check_execution_status,
            {
                "continue": "execute",  # More tasks to execute
                "validate": "validate",  # All tasks done, validate
                "error": "error"        # Error occurred
            }
        )
        
        # Conditional edge from validate
        self.workflow.add_conditional_edges(
            "validate",
            self._check_validation,
            {
                "retry": "execute",     # Retry execution
                "success": "respond",   # Validation passed
                "error": "error"       # Validation failed
            }
        )
        
        # Terminal edges
        self.workflow.add_edge("respond", END)
        self.workflow.add_edge("error", END)
        
        # Compile the workflow
        self.app = self.workflow.compile()
        logger.info("✅ Workflow compiled successfully")
    
    async def interpret_node(self, state: AgentState) -> AgentState:
        """
        Interpret the user's message and understand the intent
        Uses GPT-4o-mini for fast interpretation
        """
        logger.info(f"🎯 Interpreting: {state['user_message'][:100]}...")
        
        from adam.llm.client import UnifiedLLMClient
        llm = UnifiedLLMClient()
        
        prompt = f"""
        Analyze this user request and identify the main intent:
        "{state['user_message']}"
        
        Determine:
        1. What is the user asking for? (intent)
        2. What information do they need?
        3. What actions are required?
        
        Be specific and actionable.
        """
        
        try:
            # Call async method synchronously
            response = await llm.complete(
                prompt=prompt,
                model="gpt-5-mini"
            )
            
            state['intent'] = response.content
            state['status'] = "interpreting"
            logger.info(f"✅ Intent identified: {state['intent'][:100]}")
            
        except Exception as e:
            logger.error(f"❌ Interpretation failed: {e}")
            state['error'] = str(e)
            state['status'] = "error"
        
        return state
    
    async def plan_node(self, state: AgentState) -> AgentState:
        """
        Plan the tasks needed to fulfill the user's request
        Breaks down the intent into executable steps
        """
        logger.info("📋 Planning tasks...")
        
        from adam.llm.client import UnifiedLLMClient
        llm = UnifiedLLMClient()
        
        prompt = f"""
        User request: "{state['user_message']}"
        Intent: {state['intent']}
        
        Break this down into SIMPLE executable tasks using ONLY these valid actions:
        - list_files: params must have ONLY "path" and optionally "recursive" (true/false)
        - read_file: params must have ONLY "file_path"
        - run_command: params must have ONLY "command"
        
        Keep it simple! For showing folder contents, usually just list_files is enough.
        
        Example:
        [
            {{"action": "list_files", "params": {{"path": "/some/path", "recursive": false}}}}
        ]
        
        Return ONLY a valid JSON array with simple tasks, no other text.
        """
        
        try:
            # Call async method and parse JSON from response
            response = await llm.complete(
                prompt=prompt + "\n\nRemember to return ONLY valid JSON array, nothing else.",
                model="gpt-5-mini"
            )
            
            # Parse the response - be more lenient
            content = response.content.strip()
            
            # Try to extract JSON from the response
            if '```json' in content:
                # Extract JSON from markdown code block
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                # Extract from generic code block
                content = content.split('```')[1].split('```')[0].strip()
            
            # Try to parse as JSON
            try:
                if content.startswith('['):
                    task_list = json.loads(content)
                elif content.startswith('{'):
                    parsed = json.loads(content)
                    task_list = parsed.get('tasks', [])
                else:
                    # If no valid JSON, create a simple task list
                    logger.warning("Could not parse JSON, creating default task")
                    task_list = [
                        {"action": "list_files", "params": {"path": state.get('workspace_path', '.')}}
                    ]
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}, creating default task")
                task_list = [
                    {"action": "list_files", "params": {"path": state.get('workspace_path', '.')}}
                ]
            
            # Convert to Task objects
            state['tasks'] = [
                Task(action=t['action'], params=t['params'])
                for t in task_list
            ]
            state['current_task_index'] = 0
            state['results'] = []
            state['intermediate_outputs'] = []
            
            logger.info(f"📋 Planned {len(state['tasks'])} tasks")
            for i, task in enumerate(state['tasks']):
                logger.info(f"  {i+1}. {task.action} with {task.params}")
            
        except Exception as e:
            logger.error(f"❌ Planning failed: {e}")
            state['error'] = str(e)
            state['status'] = "error"
        
        return state
    
    async def execute_node(self, state: AgentState) -> AgentState:
        """
        Execute the current task in the plan
        This node runs multiple times until all tasks are done
        """
        tasks = state['tasks']
        current_index = state['current_task_index']
        
        if current_index >= len(tasks):
            logger.info("✅ All tasks executed")
            return state
        
        current_task = tasks[current_index]
        logger.info(f"🔧 Executing task {current_index + 1}/{len(tasks)}: {current_task.action}")
        
        # Update task status
        current_task.status = TaskStatus.RUNNING
        
        try:
            # Import the appropriate tool based on action
            result = await self._execute_tool(current_task)
            
            # Store result
            current_task.status = TaskStatus.COMPLETED
            current_task.result = result
            
            state['results'].append({
                'task': current_task.action,
                'params': current_task.params,
                'result': result,
                'status': 'success'
            })
            
            # Add to intermediate outputs for user feedback
            if isinstance(result, dict) and 'message' in result:
                state['intermediate_outputs'].append(result['message'])
            elif isinstance(result, str):
                state['intermediate_outputs'].append(result[:500])  # Truncate long results
            
            logger.info(f"✅ Task completed: {current_task.action}")
            
            # Check if this task generates new tasks (e.g., after listing files, read them)
            new_tasks = self._generate_followup_tasks(current_task, result)
            if new_tasks:
                logger.info(f"📋 Generated {len(new_tasks)} follow-up tasks")
                state['tasks'].extend(new_tasks)
            
        except Exception as e:
            logger.error(f"❌ Task execution failed: {e}")
            current_task.status = TaskStatus.FAILED
            current_task.error = str(e)
            
            state['results'].append({
                'task': current_task.action,
                'params': current_task.params,
                'error': str(e),
                'status': 'failed'
            })
        
        # Move to next task
        state['current_task_index'] = current_index + 1
        
        return state
    
    async def validate_node(self, state: AgentState) -> AgentState:
        """
        Validate that the executed tasks fulfill the user's request
        Uses GPT-4o-mini to check completeness and quality
        """
        logger.info("✔️ Validating results...")
        
        from adam.llm.client import UnifiedLLMClient
        llm = UnifiedLLMClient()
        
        prompt = f"""
        User requested: {state['user_message']}
        Intent: {state['intent']}
        
        We executed these tasks and got these results:
        {json.dumps(state['results'], indent=2)[:3000]}  # Truncate for context
        
        Validate VERY PRAGMATICALLY:
        1. Do we have ANY useful information about what the user asked?
        2. Can we provide a reasonable response with what we have?
        
        BE VERY LENIENT - if we have ANYTHING useful, mark it as passed.
        The user asked a simple question, we don't need perfect information.
        If we listed files or got any data, that's usually enough!
        
        Return JSON:
        {{
            "validation_passed": true/false,
            "confidence": 0.0-1.0,
            "feedback": "explanation",
            "missing_info": ["ONLY if critically important items are missing"]
        }}
        """
        
        try:
            # Call async method and parse JSON from response
            response = await llm.complete(
                prompt=prompt + "\n\nRemember to return ONLY valid JSON.",
                model="gpt-5-mini"
            )
            
            validation = json.loads(response.content)
            state['validation_passed'] = validation['validation_passed']
            state['validation_feedback'] = validation.get('feedback', '')
            
            if not validation['validation_passed'] and validation.get('missing_info'):
                # Generate tasks for missing info
                logger.info(f"⚠️ Validation failed, missing: {validation['missing_info']}")
                state['retry_needed'] = True
            else:
                logger.info(f"✅ Validation passed with confidence: {validation.get('confidence', 1.0)}")
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            state['validation_passed'] = False
            state['error'] = str(e)
        
        return state
    
    async def respond_node(self, state: AgentState) -> AgentState:
        """
        Generate the final response for the user
        Synthesizes all results into a comprehensive answer
        """
        logger.info("💬 Generating final response...")
        
        from adam.llm.client import UnifiedLLMClient
        llm = UnifiedLLMClient()
        
        # Build context from results
        results_summary = []
        for result in state['results']:
            if result['status'] == 'success':
                results_summary.append(f"- {result['task']}: {str(result.get('result', ''))[:200]}")
        
        prompt = f"""
        User asked: {state['user_message']}
        
        Based on our execution results:
        {chr(10).join(results_summary)}
        
        Provide a comprehensive, well-structured response that:
        1. Directly answers their question
        2. Includes all relevant details from our findings
        3. Is organized and easy to understand
        4. Uses markdown formatting where appropriate
        
        Be thorough but concise.
        """
        
        try:
            # Call async method with better model for final response
            response = await llm.complete(
                prompt=prompt,
                model="gpt-5"
            )
            
            state['final_response'] = response.content
            state['status'] = "completed"
            logger.info("✅ Response generated successfully")
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            state['error'] = str(e)
            state['final_response'] = "I encountered an error while generating the response."
            state['status'] = "error"
        
        return state
    
    async def error_node(self, state: AgentState) -> AgentState:
        """Handle errors gracefully"""
        logger.error(f"❌ Error in workflow: {state.get('error', 'Unknown error')}")
        
        state['status'] = "error"
        state['final_response'] = f"I encountered an error: {state.get('error', 'Unknown error')}"
        
        return state
    
    def _check_execution_status(self, state: AgentState) -> str:
        """Determine next step after execution"""
        if state.get('error'):
            return "error"
        
        current_index = state['current_task_index']
        total_tasks = len(state['tasks'])
        
        if current_index < total_tasks:
            return "continue"  # More tasks to execute
        else:
            return "validate"  # All done, validate
    
    def _check_validation(self, state: AgentState) -> str:
        """Determine next step after validation"""
        if state.get('error'):
            return "error"
        
        if state['validation_passed']:
            return "success"
        elif state.get('retry_needed'):
            # Check if we've already retried too many times
            retry_count = state.get('retry_count', 0)
            if retry_count >= 1:  # Max 1 retry to prevent excessive looping
                logger.warning(f"⚠️ Max retries reached ({retry_count}), proceeding with partial results")
                return "success"  # Proceed with what we have
            
            # Increment retry counter
            state['retry_count'] = retry_count + 1
            state['current_task_index'] = 0
            return "retry"
        else:
            # If validation failed but no retry needed, proceed anyway
            return "success"
    
    async def _execute_tool(self, task: Task) -> Any:
        """Execute a specific tool/action"""
        # Import tools dynamically
        from adam.tools import (
            ListFilesTool, ReadFileTool, WriteFileTool,
            SearchFilesTool, DeleteFileTool
        )
        from adam.tools.system_tools import RunCommandTool
        
        # Map actions to tools
        tools = {
            'list_files': ListFilesTool(),
            'read_file': ReadFileTool(),
            'write_file': WriteFileTool(),
            'search_files': SearchFilesTool(),
            'delete_file': DeleteFileTool(),
            'run_command': RunCommandTool(),
        }
        
        tool = tools.get(task.action)
        if not tool:
            # If no specific tool, try to use a general LLM call
            return await self._execute_llm_task(task)
        
        # Execute the tool - check if it's async or sync
        import asyncio
        import inspect
        
        if inspect.iscoroutinefunction(tool.execute):
            # Tool is async, call directly
            result = await tool.execute(**task.params)
        else:
            # Tool is synchronous, run in thread
            result = await asyncio.to_thread(tool.execute, **task.params)
        
        # Convert result to serializable format
        if hasattr(result, 'to_dict'):
            return result.to_dict()
        elif hasattr(result, 'data'):
            return {'data': result.data, 'message': getattr(result, 'message', '')}
        else:
            return str(result)
    
    async def _execute_llm_task(self, task: Task) -> Any:
        """Execute a task using LLM when no specific tool exists"""
        from adam.llm.client import UnifiedLLMClient
        llm = UnifiedLLMClient()
        
        prompt = f"""
        Execute this task:
        Action: {task.action}
        Parameters: {json.dumps(task.params)}
        
        Provide the result.
        """
        
        # Call async method
        response = await llm.complete(
            prompt=prompt,
            model="gpt-5-mini"
        )
        
        return response.content
    
    def _generate_followup_tasks(self, completed_task: Task, result: Any) -> List[Task]:
        """Generate follow-up tasks based on completed task results"""
        followup_tasks = []
        
        # If we listed files, automatically read important ones
        if completed_task.action == 'list_files' and isinstance(result, dict):
            files = result.get('data', '').split('\n') if isinstance(result.get('data'), str) else []
            
            for file_line in files[:10]:  # Limit to first 10 files
                if any(ext in file_line for ext in ['.py', '.js', '.ts', '.md', '.txt']):
                    # Extract file path from the formatted output
                    if '📄' in file_line:
                        file_path = file_line.split('📄')[1].split('(')[0].strip()
                        base_path = completed_task.params.get('path', '.')
                        full_path = f"{base_path}/{file_path}" if base_path != '.' else file_path
                        
                        followup_tasks.append(Task(
                            action='read_file',
                            params={'file_path': full_path}
                        ))
        
        return followup_tasks
    
    async def process_request(self, user_message: str, workspace_path: str = None) -> Dict[str, Any]:
        """
        Main entry point to process a user request autonomously
        """
        logger.info(f"🚀 Processing request: {user_message[:100]}...")
        
        start_time = datetime.now()
        
        # Initialize state
        initial_state = {
            'user_message': user_message,
            'workspace_path': workspace_path or '.',
            'intent': '',
            'tasks': [],
            'current_task_index': 0,
            'results': [],
            'intermediate_outputs': [],
            'validation_passed': False,
            'validation_feedback': '',
            'retry_needed': False,
            'retry_count': 0,  # Track retry attempts
            'final_response': '',
            'error': None,
            'status': 'starting',
            'execution_time': 0.0
        }
        
        try:
            # Run the workflow with recursion limit
            final_state = await self.app.ainvoke(
                initial_state,
                config={"recursion_limit": 25}  # Increased for complex folder analysis
            )
            
            # Calculate execution time
            final_state['execution_time'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Request processed in {final_state['execution_time']:.2f} seconds")
            
            return {
                'status': final_state['status'],
                'response': final_state['final_response'],
                'intermediate_outputs': final_state['intermediate_outputs'],
                'execution_time': final_state['execution_time'],
                'tasks_executed': len(final_state['results']),
                'error': final_state.get('error')
            }
            
        except Exception as e:
            logger.error(f"❌ Orchestrator error: {e}")
            return {
                'status': 'error',
                'response': f"I encountered an error: {str(e)}",
                'error': str(e),
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
    
    async def process_with_streaming(self, user_message: str, workspace_path: str = None):
        """
        Process request with streaming updates
        Yields status updates as the workflow progresses
        """
        logger.info(f"🚀 Processing with streaming: {user_message[:100]}...")
        
        # Initial acknowledgment
        yield {
            'type': 'status',
            'message': "I'm analyzing your request...",
            'stage': 'interpreting'
        }
        
        # Run through the workflow with updates
        initial_state = {
            'user_message': user_message,
            'workspace_path': workspace_path or '.',
            'intent': '',
            'tasks': [],
            'current_task_index': 0,
            'results': [],
            'intermediate_outputs': [],
            'validation_passed': False,
            'validation_feedback': '',
            'retry_needed': False,
            'final_response': '',
            'error': None,
            'status': 'starting',
            'execution_time': 0.0
        }
        
        # Stream updates as we go through nodes
        async for event in self.app.astream(initial_state):
            for node_name, node_state in event.items():
                if node_name == 'interpret':
                    yield {
                        'type': 'status',
                        'message': f"Understanding your request: {node_state.get('intent', '')[:100]}",
                        'stage': 'interpreting'
                    }
                elif node_name == 'plan':
                    task_count = len(node_state.get('tasks', []))
                    yield {
                        'type': 'status',
                        'message': f"Planning {task_count} tasks to complete your request",
                        'stage': 'planning'
                    }
                elif node_name == 'execute':
                    current_task = node_state.get('current_task_index', 0)
                    total_tasks = len(node_state.get('tasks', []))
                    if current_task < total_tasks:
                        task = node_state['tasks'][current_task]
                        yield {
                            'type': 'progress',
                            'message': f"Executing task {current_task + 1}/{total_tasks}: {task.action}",
                            'stage': 'executing',
                            'progress': current_task / total_tasks
                        }
                elif node_name == 'validate':
                    yield {
                        'type': 'status',
                        'message': "Validating results...",
                        'stage': 'validating'
                    }
                elif node_name == 'respond':
                    yield {
                        'type': 'complete',
                        'message': node_state.get('final_response', ''),
                        'stage': 'completed'
                    }
                elif node_name == 'error':
                    yield {
                        'type': 'error',
                        'message': node_state.get('error', 'Unknown error'),
                        'stage': 'error'
                    }

# Singleton instance
_orchestrator = None

def get_orchestrator() -> ADAMOrchestrator:
    """Get or create the orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ADAMOrchestrator()
    return _orchestrator