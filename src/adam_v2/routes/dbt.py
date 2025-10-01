"""
DBT API Routes
Endpoints for DBT project analysis and documentation
"""

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from ..services.dbt_service import get_dbt_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dbt", tags=["dbt"])


# Request/Response Models
class IngestProjectRequest(BaseModel):
    project_path: str = Field(..., description="Path to DBT project")
    project_id: str = Field(..., description="Unique project identifier")


class DocumentModelRequest(BaseModel):
    project_path: str
    project_id: str
    model_name: str
    platform: str = Field(default="redshift", description="Target platform (redshift or snowflake)")


class LineageRequest(BaseModel):
    project_path: str
    project_id: str
    model_name: str


class OptimizationRequest(BaseModel):
    project_path: str
    project_id: str
    model_name: str
    platform: str = Field(default="redshift")


class SearchModelsRequest(BaseModel):
    project_id: str
    query: str
    limit: int = Field(default=5, ge=1, le=20)


class ProjectStatsRequest(BaseModel):
    project_path: str
    project_id: str


class ListModelsRequest(BaseModel):
    project_path: str
    project_id: str
    tag: Optional[str] = None


class FindProjectsRequest(BaseModel):
    workspace_path: str = Field(..., description="Path to workspace to search")


@router.post("/ingest")
async def ingest_project(request: IngestProjectRequest):
    """
    Ingest a DBT project into ADAM's memory

    This endpoint analyzes a DBT project and stores all models, sources,
    tests, and lineage information in ADAM's memory for semantic search
    and intelligent assistance.
    """
    service = get_dbt_service()
    result = service.ingest_project(request.project_path, request.project_id)

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))

    return result


@router.post("/document")
async def document_model(request: DocumentModelRequest):
    """
    Generate comprehensive documentation for a DBT model

    Returns markdown documentation including:
    - Model overview and metadata
    - Data lineage (upstream/downstream)
    - Column descriptions
    - Test coverage
    - Optimization suggestions
    - Usage examples
    """
    service = get_dbt_service()
    result = service.get_model_documentation(
        request.project_path,
        request.project_id,
        request.model_name,
        request.platform
    )

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error"))

    return result


@router.post("/lineage")
async def analyze_lineage(request: LineageRequest):
    """
    Analyze data lineage for a model

    Returns:
    - Direct upstream and downstream dependencies
    - Full lineage tree
    - Impact analysis (criticality, DAG depth)
    - Change risk assessment
    """
    service = get_dbt_service()
    result = service.analyze_lineage(
        request.project_path,
        request.project_id,
        request.model_name
    )

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error"))

    return result


@router.post("/optimize")
async def get_optimizations(request: OptimizationRequest):
    """
    Get SQL optimization suggestions for a model

    Analyzes the model's SQL and provides platform-specific
    optimization recommendations for Redshift or Snowflake.
    """
    service = get_dbt_service()
    result = service.get_optimizations(
        request.project_path,
        request.project_id,
        request.model_name,
        request.platform
    )

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error"))

    return result


@router.post("/search")
async def search_models(request: SearchModelsRequest):
    """
    Search for models using semantic search

    Uses ADAM's memory system to find relevant models based on
    natural language queries.
    """
    service = get_dbt_service()
    result = service.search_models(
        request.project_id,
        request.query,
        request.limit
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))

    return result


@router.post("/stats")
async def get_project_stats(request: ProjectStatsRequest):
    """
    Get statistics for a DBT project

    Returns:
    - Model counts by materialization type
    - Test coverage metrics
    - Source and exposure counts
    """
    service = get_dbt_service()
    result = service.get_project_stats(
        request.project_path,
        request.project_id
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))

    return result


@router.post("/models/list")
async def list_models(request: ListModelsRequest):
    """
    List all models in a project

    Optionally filter by tag.
    """
    service = get_dbt_service()
    result = service.list_models(
        request.project_path,
        request.project_id,
        request.tag
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))

    return result


@router.post("/projects/find")
async def find_projects(request: FindProjectsRequest):
    """
    Find all DBT projects in a workspace

    Searches recursively for dbt_project.yml files and reports
    which projects have compiled manifests available.
    """
    service = get_dbt_service()
    projects = service.find_dbt_projects(request.workspace_path)

    return {
        "success": True,
        "total_projects": len(projects),
        "projects": projects
    }


@router.delete("/cache/{project_id}")
async def clear_cache(project_id: Optional[str] = None):
    """
    Clear cached parsers and memory for a project

    Use this when a DBT project has been updated and needs to be re-parsed.
    """
    service = get_dbt_service()
    service.clear_cache(project_id)

    return {
        "success": True,
        "message": f"Cache cleared for project: {project_id}" if project_id else "All caches cleared"
    }


# Health check for DBT functionality
@router.get("/health")
async def health_check():
    """Check if DBT analyzer is available"""
    try:
        service = get_dbt_service()
        return {
            "status": "healthy",
            "service": "dbt_analyzer",
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }