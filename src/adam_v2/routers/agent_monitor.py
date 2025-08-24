"""
Agent Monitor API - Real-time visibility into agent execution
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
import json
import logging
import asyncio
from datetime import datetime

from services.agent_service import get_agent_service

router = APIRouter()
logger = logging.getLogger(__name__)

# Store WebSocket connections for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@router.get("/agent/status")
async def get_agent_status():
    """Get current agent system status"""
    agent_service = get_agent_service()
    
    return {
        "status": "active" if agent_service.runtime else "inactive",
        "runtime_available": agent_service.runtime is not None,
        "background_processor": agent_service.background_task is not None,
        "queue_size": agent_service.task_queue.qsize() if agent_service.task_queue else 0,
        "tasks": agent_service.list_tasks()
    }

@router.get("/agent/tasks")
async def get_agent_tasks():
    """Get all agent tasks"""
    agent_service = get_agent_service()
    return agent_service.list_tasks()

@router.get("/agent/tasks/{task_id}")
async def get_task_details(task_id: str):
    """Get details of a specific task"""
    agent_service = get_agent_service()
    task = agent_service.get_task_status(task_id)
    
    if not task:
        return {"error": "Task not found"}
    
    return task

@router.websocket("/agent/monitor")
async def websocket_monitor(websocket: WebSocket):
    """WebSocket for real-time agent monitoring"""
    await manager.connect(websocket)
    
    try:
        # Send initial status
        agent_service = get_agent_service()
        await websocket.send_json({
            "type": "status",
            "data": {
                "runtime_available": agent_service.runtime is not None,
                "tasks_in_queue": agent_service.task_queue.qsize() if agent_service.task_queue else 0
            }
        })
        
        # Keep connection alive and send updates
        while True:
            await asyncio.sleep(1)
            # Could send periodic updates here
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Global function to send updates to monitor
async def send_monitor_update(update_type: str, data: Dict[str, Any]):
    """Send update to all monitoring clients"""
    await manager.broadcast({
        "type": update_type,
        "timestamp": datetime.now().isoformat(),
        "data": data
    })