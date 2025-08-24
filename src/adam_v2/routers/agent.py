"""
Agent Router - Autonomous agent execution endpoints
Provides streaming and batch execution for multi-step tasks
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

class AgentRequest(BaseModel):
    """Request model for agent execution"""
    message: str
    workspace_path: Optional[str] = None
    mode: str = "autonomous"  # "autonomous" or "interactive"
    stream_updates: bool = True
    project_id: Optional[str] = None

class AgentResponse(BaseModel):
    """Response model for agent execution"""
    status: str
    response: str
    intermediate_outputs: list = []
    execution_time: float
    tasks_executed: int
    error: Optional[str] = None

@router.post("/execute", response_model=AgentResponse)
async def execute_autonomous_task(request: AgentRequest):
    """
    Execute a multi-step task autonomously
    The agent will complete all necessary steps without waiting for user input
    """
    logger.info(f"🚀 Agent execution requested: {request.message[:100]}...")
    
    try:
        from agents.orchestrator import get_orchestrator
        orchestrator = get_orchestrator()
        
        # Execute the request
        result = await orchestrator.process_request(
            user_message=request.message,
            workspace_path=request.workspace_path
        )
        
        return AgentResponse(
            status=result['status'],
            response=result['response'],
            intermediate_outputs=result.get('intermediate_outputs', []),
            execution_time=result['execution_time'],
            tasks_executed=result.get('tasks_executed', 0),
            error=result.get('error')
        )
        
    except Exception as e:
        logger.error(f"❌ Agent execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def stream_autonomous_execution(request: AgentRequest):
    """
    Execute task with streaming updates
    Returns Server-Sent Events (SSE) stream with progress updates
    """
    logger.info(f"🚀 Streaming agent execution: {request.message[:100]}...")
    
    async def generate_stream():
        try:
            from agents.orchestrator import get_orchestrator
            orchestrator = get_orchestrator()
            
            # Send initial acknowledgment
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting autonomous execution...', 'timestamp': datetime.now().isoformat()})}\n\n"
            
            # Process with streaming
            async for update in orchestrator.process_with_streaming(
                user_message=request.message,
                workspace_path=request.workspace_path
            ):
                # Add timestamp to each update
                update['timestamp'] = datetime.now().isoformat()
                yield f"data: {json.dumps(update)}\n\n"
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.1)
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'end', 'message': 'Execution complete', 'timestamp': datetime.now().isoformat()})}\n\n"
            
        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'timestamp': datetime.now().isoformat()})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        }
    )

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the status of a running task
    (For future implementation when we add persistent task tracking)
    """
    # TODO: Implement task status tracking
    return {
        "task_id": task_id,
        "status": "not_implemented",
        "message": "Task status tracking will be implemented in the next version"
    }

@router.get("/health")
async def agent_health():
    """Check agent system health"""
    try:
        from agents.orchestrator import get_orchestrator
        orchestrator = get_orchestrator()
        
        return {
            "status": "healthy",
            "orchestrator": "ready" if orchestrator else "not_initialized",
            "langgraph": "enabled",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }