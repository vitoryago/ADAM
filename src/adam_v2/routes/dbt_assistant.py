"""
DBT Assistant API Routes
Natural language interface to DBT projects
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

from ..services.dbt_assistant import get_dbt_assistant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dbt-assistant", tags=["dbt-assistant"])


# Request/Response models
class DetectProjectsRequest(BaseModel):
    workspace_path: str

class LoadProjectRequest(BaseModel):
    project_path: str
    force_reload: bool = False

class QueryRequest(BaseModel):
    query: str
    project_path: Optional[str] = None


# Routes
@router.post("/detect-projects")
async def detect_projects(request: DetectProjectsRequest):
    """
    Detect all DBT projects in workspace
    """
    assistant = get_dbt_assistant()
    projects = assistant.detect_dbt_projects(request.workspace_path)

    return {
        "success": True,
        "workspace_path": request.workspace_path,
        "projects": projects,
        "count": len(projects)
    }


@router.post("/load-project")
async def load_project(request: LoadProjectRequest):
    """
    Load a DBT project
    """
    assistant = get_dbt_assistant()
    result = assistant.load_project(request.project_path, request.force_reload)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error"))

    return result


@router.post("/query")
async def query(request: QueryRequest):
    """
    Natural language query about DBT models

    Examples:
    - "Tell me about base_ll1_loan"
    - "What does base_ll1_loan depend on?"
    - "What models use base_ll1_loan?"
    - "Show me complex models"
    - "Optimize base_ll1_loan for snowflake"
    - "Find models with loan in the name"
    - "How many models do we have?"
    """
    assistant = get_dbt_assistant()
    result = assistant.handle_query(request.query, request.project_path)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error"))

    return result


@router.get("/current-project")
async def get_current_project():
    """
    Get currently loaded project
    """
    assistant = get_dbt_assistant()

    if not assistant.current_project:
        return {
            "success": True,
            "loaded": False,
            "message": "No project currently loaded"
        }

    reader = assistant.get_current_reader()
    stats = reader.get_project_stats()

    return {
        "success": True,
        "loaded": True,
        "project_path": assistant.current_project,
        "project_name": stats["project_name"],
        "statistics": stats
    }
