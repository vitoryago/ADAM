"""
Onboarding API endpoints for ADAM v2
"""

from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from services.onboarding_service import OnboardingService, OnboardingPhase, MilestoneStatus
from services.memory_service import ProjectMemoryService
from services.llm_service import LLMService
from database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/onboarding", tags=["onboarding"])

# Initialize services  
# We'll initialize these per-project in the endpoints
llm_service = LLMService()
onboarding_services: Dict[str, OnboardingService] = {}

@router.post("/create-path")
async def create_onboarding_path(
    project_path: str = Body(...),
    user_level: str = Body("beginner"),
    focus_area: Optional[str] = Body(None),
    custom_requirements: Optional[str] = Body(None)
) -> Dict[str, Any]:
    """
    Create a personalized onboarding path for a project
    
    Args:
        project_path: Path to the project directory
        user_level: User experience level (beginner, intermediate, advanced)
        focus_area: Optional area to focus on (e.g., "marketing", "data")
        custom_requirements: Optional custom requirements or preferences
    """
    try:
        # Create a default project ID for onboarding
        from pathlib import Path
        project_id = f"onboarding_{Path(project_path).name}"
        
        # Get or create onboarding service for this project
        if project_id not in onboarding_services:
            onboarding_services[project_id] = OnboardingService(
                project_id=project_id,
                project_name=Path(project_path).name,
                llm_service=llm_service
            )
        
        onboarding_service = onboarding_services[project_id]
        
        path = await onboarding_service.create_onboarding_path(
            project_path=project_path,
            user_level=user_level,
            focus_area=focus_area,
            custom_requirements=custom_requirements
        )
        
        return {
            "status": "success",
            "path_id": path.id,
            "project_name": path.project_name,
            "milestones": len(path.milestones),
            "estimated_time": path.estimated_total_time,
            "data": {
                "id": path.id,
                "project_name": path.project_name,
                "user_level": path.user_level,
                "focus_area": path.focus_area,
                "milestones": [
                    {
                        "id": m.id,
                        "title": m.title,
                        "description": m.description,
                        "phase": m.phase.value,
                        "status": m.status.value,
                        "estimated_time": m.estimated_time,
                        "tasks": m.tasks
                    }
                    for m in path.milestones
                ],
                "progress": path.progress,
                "created_at": path.created_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating onboarding path: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/path/{path_id}")
async def get_onboarding_path(path_id: str) -> Dict[str, Any]:
    """
    Get details of an onboarding path
    
    Args:
        path_id: ID of the onboarding path
    """
    try:
        # Find the service that has this path
        onboarding_service = None
        for service in onboarding_services.values():
            if path_id in service.paths:
                onboarding_service = service
                break
        
        if not onboarding_service:
            raise HTTPException(status_code=404, detail=f"Path {path_id} not found")
            
        path = onboarding_service.paths[path_id]
        
        return {
            "id": path.id,
            "project_name": path.project_name,
            "user_level": path.user_level,
            "focus_area": path.focus_area,
            "milestones": [
                {
                    "id": m.id,
                    "title": m.title,
                    "description": m.description,
                    "phase": m.phase.value,
                    "status": m.status.value,
                    "estimated_time": m.estimated_time,
                    "tasks": m.tasks,
                    "resources": m.resources,
                    "completion_criteria": m.completion_criteria
                }
                for m in path.milestones
            ],
            "current_milestone": path.current_milestone,
            "progress": path.progress,
            "estimated_total_time": path.estimated_total_time,
            "created_at": path.created_at.isoformat(),
            "updated_at": path.updated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting onboarding path: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/path/{path_id}/progress")
async def update_progress(
    path_id: str,
    milestone_id: str = Body(...),
    task_id: str = Body(...),
    completed: bool = Body(...)
) -> Dict[str, Any]:
    """
    Update progress on a specific task
    
    Args:
        path_id: ID of the onboarding path
        milestone_id: ID of the milestone
        task_id: ID of the task
        completed: Whether the task is completed
    """
    try:
        # Find the service that has this path
        onboarding_service = None
        for service in onboarding_services.values():
            if path_id in service.paths:
                onboarding_service = service
                break
        
        if not onboarding_service:
            raise ValueError(f"Path {path_id} not found")
            
        path = await onboarding_service.update_milestone_progress(
            path_id=path_id,
            milestone_id=milestone_id,
            task_id=task_id,
            completed=completed
        )
        
        return {
            "status": "success",
            "progress": path.progress,
            "milestone_status": next(
                m.status.value for m in path.milestones if m.id == milestone_id
            )
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/path/{path_id}/recommendation")
async def get_next_recommendation(path_id: str) -> Dict[str, Any]:
    """
    Get AI-powered recommendation for the next step
    
    Args:
        path_id: ID of the onboarding path
    """
    try:
        # Find the service that has this path
        onboarding_service = None
        for service in onboarding_services.values():
            if path_id in service.paths:
                onboarding_service = service
                break
        
        if not onboarding_service:
            raise ValueError(f"Path {path_id} not found")
            
        recommendation = await onboarding_service.get_next_recommendation(path_id)
        return recommendation
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/path/{path_id}/export")
async def export_path(
    path_id: str,
    format: str = "json"
) -> Any:
    """
    Export onboarding path in various formats
    
    Args:
        path_id: ID of the onboarding path
        format: Export format (json, markdown)
    """
    try:
        # Find the service that has this path
        onboarding_service = None
        for service in onboarding_services.values():
            if path_id in service.paths:
                onboarding_service = service
                break
        
        if not onboarding_service:
            raise ValueError(f"Path {path_id} not found")
            
        result = onboarding_service.export_path(path_id, format)
        
        if format == "json":
            import json
            return json.loads(result)
        else:
            return {"format": format, "content": result}
            
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting path: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/paths")
async def list_paths() -> List[Dict[str, Any]]:
    """List all available onboarding paths"""
    try:
        paths = []
        # Collect paths from all services
        for service in onboarding_services.values():
            for path_id, path in service.paths.items():
                paths.append({
                    "id": path.id,
                    "project_name": path.project_name,
                    "user_level": path.user_level,
                    "focus_area": path.focus_area,
                    "progress": path.progress,
                    "milestones_total": len(path.milestones),
                    "milestones_completed": sum(
                        1 for m in path.milestones 
                        if m.status == MilestoneStatus.COMPLETED
                    ),
                    "created_at": path.created_at.isoformat()
                })
            
        return paths
        
    except Exception as e:
        logger.error(f"Error listing paths: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/path/{path_id}/chat")
async def chat_about_path(
    path_id: str,
    message: str = Body(...)
) -> Dict[str, Any]:
    """
    Chat with ADAM about the onboarding path
    
    Args:
        path_id: ID of the onboarding path
        message: User's message
    """
    try:
        # Find the service that has this path
        onboarding_service = None
        for service in onboarding_services.values():
            if path_id in service.paths:
                onboarding_service = service
                break
        
        if not onboarding_service:
            raise HTTPException(status_code=404, detail=f"Path {path_id} not found")
            
        path = onboarding_service.paths[path_id]
        
        # Build context for LLM
        context = f"""
        User is onboarding to project: {path.project_name}
        User level: {path.user_level}
        Focus area: {path.focus_area or 'Full project'}
        Current progress: {path.progress:.1f}%
        
        Current milestone: {next((m.title for m in path.milestones if m.id == path.current_milestone), 'None')}
        
        User message: {message}
        """
        
        response = await llm_service.generate_response(context)
        
        return {
            "message": response,
            "context": {
                "project": path.project_name,
                "progress": path.progress,
                "current_milestone": path.current_milestone
            }
        }
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))