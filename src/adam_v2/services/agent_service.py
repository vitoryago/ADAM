"""
Agent Service - Manages agent execution
Real background execution with task queuing
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
from enum import Enum
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentTask:
    """Represents a task for an agent to execute"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request: str = ""
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)

class AgentService:
    """
    Service that manages agent execution in the background
    Similar to how Claude Code runs tasks
    """
    
    def __init__(self):
        self.tasks: Dict[str, AgentTask] = {}
        self.runtime = None
        self._initialize_runtime()
        
        # Start background task processor
        self.task_queue = asyncio.Queue()
        self.background_task = None
        
    def _initialize_runtime(self):
        """Initialize the agent runtime"""
        try:
            from agents.agent_runtime import get_agent_runtime
            self.runtime = get_agent_runtime()
            logger.info("Agent runtime initialized")
        except Exception as e:
            logger.error(f"Failed to initialize agent runtime: {e}")
    
    async def start_background_processor(self):
        """Start the background task processor"""
        if not self.background_task:
            self.background_task = asyncio.create_task(self._process_tasks())
            logger.info("Background task processor started")
    
    async def _process_tasks(self):
        """Process tasks from the queue"""
        while True:
            try:
                # Get next task from queue
                task_id = await self.task_queue.get()
                task = self.tasks.get(task_id)
                
                if not task:
                    continue
                
                # Update task status
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                
                logger.info(f"Processing task {task_id}: {task.request}")
                
                # Execute through agent runtime
                if self.runtime:
                    result = await self.runtime.process_request(task.request)
                    
                    # Update task with results
                    task.result = result
                    task.status = TaskStatus.COMPLETED if result.get('status') == 'success' else TaskStatus.FAILED
                    task.error = result.get('error')
                    task.steps = result.get('steps', [])
                else:
                    task.status = TaskStatus.FAILED
                    task.error = "Agent runtime not available"
                
                task.completed_at = datetime.now()
                logger.info(f"Task {task_id} completed with status: {task.status}")
                
            except Exception as e:
                logger.error(f"Error processing task: {e}")
                if task:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.now()
    
    async def submit_task(self, request: str) -> str:
        """
        Submit a task for background execution
        Returns task ID for tracking
        """
        
        # Create new task
        task = AgentTask(request=request)
        self.tasks[task.id] = task
        
        # Add to queue
        await self.task_queue.put(task.id)
        
        logger.info(f"Task {task.id} submitted: {request[:100]}...")
        
        # Ensure processor is running
        await self.start_background_processor()
        
        return task.id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task"""
        
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return {
            'id': task.id,
            'status': task.status.value,
            'request': task.request,
            'result': task.result,
            'error': task.error,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'steps': task.steps
        }
    
    async def execute_immediate(self, request: str) -> Dict[str, Any]:
        """
        Execute a request immediately (not in background)
        Used for simple, quick tasks
        """
        
        if not self.runtime:
            return {
                'status': 'error',
                'error': 'Agent runtime not available'
            }
        
        try:
            return await self.runtime.process_request(request)
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        """List all tasks, optionally filtered by status"""
        
        tasks = []
        for task in self.tasks.values():
            if status is None or task.status == status:
                tasks.append(self.get_task_status(task.id))
        
        return sorted(tasks, key=lambda x: x['created_at'], reverse=True)


# Singleton instance
_agent_service = None

def get_agent_service() -> AgentService:
    """Get or create the agent service"""
    global _agent_service
    
    if _agent_service is None:
        _agent_service = AgentService()
        # Background processor will be started on first task submission
    
    return _agent_service