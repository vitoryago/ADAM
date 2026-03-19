"""
Memory management endpoints for ADAM 4.0
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, Dict, Any
from datetime import datetime

from adam.database import get_db
from adam.api.models import Project
from adam.memory.core import MemoryType
from pydantic import BaseModel, Field

import logging

logger = logging.getLogger(__name__)

# Check if memory system is available
try:
    from adam.memory.project import ProjectAwareMemory
    ADAM_MEMORY_AVAILABLE = True
except ImportError:
    ADAM_MEMORY_AVAILABLE = False
    logger.warning("Memory system not available")


# Request/Response schemas
class MemorySearchRequest(BaseModel):
    """Request for memory search"""
    query: str = Field(..., description="Search query")
    limit: int = Field(5, ge=1, le=20, description="Maximum results to return")
    memory_types: Optional[List[str]] = Field(None, description="Filter by memory types")
    min_relevance: float = Field(0.5, ge=0.0, le=1.0, description="Minimum relevance score")
    conversation_id: Optional[str] = Field(None, description="Filter by conversation")


class MemoryStoreRequest(BaseModel):
    """Request to store a memory"""
    content: str = Field(..., description="Memory content")
    memory_type: str = Field("conversation", description="Type of memory")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    conversation_id: Optional[str] = Field(None, description="Associated conversation ID")
    cost: Optional[float] = Field(None, description="Generation cost")


class MemoryResponse(BaseModel):
    """Memory response"""
    id: str
    content: str
    memory_type: str
    relevance_score: Optional[float] = None
    timestamp: str
    metadata: Dict[str, Any]


class MemoryStatsResponse(BaseModel):
    """Memory statistics response"""
    total_memories: int
    memory_types: Dict[str, int]
    total_cost: float
    avg_access_count: float = 0.0
    oldest_memory: Optional[str] = None
    newest_memory: Optional[str] = None


class MemoryExportResponse(BaseModel):
    """Memory export response"""
    project_id: str
    project_name: str
    export_date: str
    memory_count: int
    memories: List[Dict[str, Any]]


# Router setup
router = APIRouter()


# Dependency to get project and memory service
async def get_memory_service(
    project_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get memory service for a project"""
    # Verify project exists
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )

    if not ADAM_MEMORY_AVAILABLE:
        return None

    from adam.memory.project import ProjectAwareMemory
    return ProjectAwareMemory(project.id, project.name)


@router.post("/projects/{project_id}/memories/search", response_model=List[MemoryResponse])
async def search_memories(
    project_id: str,
    request: MemorySearchRequest,
    memory_service=Depends(get_memory_service)
):
    """
    Search memories within a project.

    Searches are isolated to the specific project's memory collection.
    """
    if not ADAM_MEMORY_AVAILABLE or memory_service is None:
        return []

    # Convert memory types
    memory_types = None
    if request.memory_types:
        memory_types = []
        for type_str in request.memory_types:
            try:
                memory_types.append(MemoryType[type_str.upper()])
            except KeyError:
                # Skip invalid types
                pass

    # Perform search
    results = await memory_service.search_memories(
        query=request.query,
        limit=request.limit,
        memory_types=memory_types,
        min_relevance=request.min_relevance,
        conversation_id=request.conversation_id
    )

    # Convert to response format
    return [
        MemoryResponse(
            id=result.get("id", ""),
            content=result.get("content", ""),
            memory_type=result.get("memory_type", "conversation"),
            relevance_score=result.get("relevance_score"),
            timestamp=result.get("timestamp", datetime.now().isoformat()),
            metadata=result.get("metadata", {})
        )
        for result in results
    ]


@router.post("/projects/{project_id}/memories", response_model=MemoryResponse)
async def store_memory(
    project_id: str,
    request: MemoryStoreRequest,
    memory_service=Depends(get_memory_service)
):
    """
    Store a new memory in the project's collection.

    This is typically called automatically by the LLM service for valuable responses.
    """
    if not ADAM_MEMORY_AVAILABLE or memory_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory system not available"
        )

    # Convert memory type
    try:
        memory_type = MemoryType[request.memory_type.upper()]
    except KeyError:
        memory_type = MemoryType.CONVERSATION

    # Store memory
    memory_id = await memory_service.store_memory(
        content=request.content,
        memory_type=memory_type,
        metadata=request.metadata,
        conversation_id=request.conversation_id,
        cost=request.cost
    )

    return MemoryResponse(
        id=memory_id,
        content=request.content,
        memory_type=memory_type.value,
        timestamp=datetime.now().isoformat(),
        metadata=request.metadata or {}
    )


@router.get("/projects/{project_id}/memories/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(
    project_id: str,
    memory_service=Depends(get_memory_service)
):
    """Get statistics about the project's memory collection"""
    if not ADAM_MEMORY_AVAILABLE or memory_service is None:
        return MemoryStatsResponse(
            total_memories=0,
            memory_types={},
            total_cost=0.0,
            avg_access_count=0.0,
            oldest_memory=None,
            newest_memory=None
        )

    stats = await memory_service.get_memory_stats()
    return MemoryStatsResponse(
        total_memories=stats.get("total_memories", 0),
        memory_types=stats.get("memory_types", {}),
        total_cost=stats.get("total_cost", 0.0),
        avg_access_count=stats.get("avg_access_count", 0.0),
        oldest_memory=stats.get("oldest_memory"),
        newest_memory=stats.get("newest_memory"),
    )


@router.delete("/projects/{project_id}/memories")
async def clear_memories(
    project_id: str,
    confirm: bool = Query(False, description="Confirm deletion"),
    memory_service=Depends(get_memory_service)
):
    """
    Clear all memories for a project.

    Requires confirmation parameter to be true.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirmation required. Set confirm=true to delete all memories."
        )

    if not ADAM_MEMORY_AVAILABLE or memory_service is None:
        return {"message": "Memory system not available", "deleted": 0}

    count = await memory_service.clear_memories()

    return {
        "message": f"Deleted {count} memories from project",
        "deleted": count
    }


@router.get("/projects/{project_id}/memories/export", response_model=MemoryExportResponse)
async def export_memories(
    project_id: str,
    memory_service=Depends(get_memory_service)
):
    """Export all memories for backup or migration"""
    if not ADAM_MEMORY_AVAILABLE or memory_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory system not available"
        )

    export_data = await memory_service.export_memories()
    return MemoryExportResponse(**export_data)


@router.post("/projects/{project_id}/memories/import")
async def import_memories(
    project_id: str,
    export_data: MemoryExportResponse,
    memory_service=Depends(get_memory_service)
):
    """Import memories from an export"""
    if not ADAM_MEMORY_AVAILABLE or memory_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory system not available"
        )

    # Verify the export is for the same project or user wants to override
    if export_data.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Export is from different project ({export_data.project_id}). "
                   "Create new endpoint if cross-project import is needed."
        )

    imported = await memory_service.import_memories(export_data.dict())

    return {
        "message": f"Imported {imported} memories",
        "imported": imported,
        "total_in_export": export_data.memory_count
    }


@router.get("/memory-types")
async def get_memory_types():
    """Get available memory types"""
    if not ADAM_MEMORY_AVAILABLE:
        return {
            "available": False,
            "types": []
        }

    return {
        "available": True,
        "types": [
            {
                "value": mt.value,
                "name": mt.name,
                "description": mt.value.replace("_", " ").title()
            }
            for mt in MemoryType
        ]
    }
