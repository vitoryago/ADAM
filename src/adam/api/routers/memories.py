"""
Memory management endpoints for ADAM 4.0

Provides two sets of endpoints:
  1. Project-scoped: /projects/{project_id}/memories/...  (original REST style)
  2. Convenience:     /memories/search, /memories/stats, /memories/store
     These accept project_id as a query parameter and are easier to use
     from browsers and debugging tools.

All endpoints gracefully handle ChromaDB being unavailable at import time
or at runtime (e.g. missing native libraries, disk errors).
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, Dict, Any
from datetime import datetime

from adam.database import get_db
from adam.api.models import Project
from pydantic import BaseModel, Field

import logging

logger = logging.getLogger(__name__)

# Check if memory system is available at import time
try:
    from adam.memory.core import MemoryType
    from adam.memory.project import ProjectAwareMemory
    ADAM_MEMORY_AVAILABLE = True
except ImportError as _import_err:
    ADAM_MEMORY_AVAILABLE = False
    logger.warning(f"Memory system not available: {_import_err}")

    # Provide a stub MemoryType so endpoint code that references it
    # doesn't break when the memory subsystem is missing.
    from enum import Enum

    class MemoryType(str, Enum):  # type: ignore[no-redef]
        CONVERSATION = "conversation"


# Request/Response schemas
class MemorySearchRequest(BaseModel):
    """Request for memory search (POST body)"""
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
    memory_available: bool = True
    message: Optional[str] = None


class MemoryExportResponse(BaseModel):
    """Memory export response"""
    project_id: str
    project_name: str
    export_date: str
    memory_count: int
    memories: List[Dict[str, Any]]


# Router setup
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_stats(message: str = "Memory system not available") -> MemoryStatsResponse:
    """Return a safe empty stats response."""
    return MemoryStatsResponse(
        total_memories=0,
        memory_types={},
        total_cost=0.0,
        avg_access_count=0.0,
        oldest_memory=None,
        newest_memory=None,
        memory_available=False,
        message=message,
    )


def _try_create_memory_service(project_id: str, project_name: str):
    """
    Attempt to instantiate ProjectAwareMemory.
    Returns the service instance, or None if ChromaDB / embeddings fail at runtime.
    """
    if not ADAM_MEMORY_AVAILABLE:
        return None
    try:
        from adam.memory.project import ProjectAwareMemory as _PAM
        return _PAM(project_id, project_name)
    except Exception as exc:
        logger.warning(f"Failed to create memory service for project {project_id}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Dependency: project-scoped memory service (used by /projects/{project_id}/...)
# ---------------------------------------------------------------------------

async def get_memory_service(
    project_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get memory service for a project (verifies project exists in DB)."""
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )

    return _try_create_memory_service(project.id, project.name)


# ---------------------------------------------------------------------------
# Dependency: lightweight memory service (used by /memories/... convenience endpoints)
# Skips DB project lookup -- uses project_id directly.
# ---------------------------------------------------------------------------

def get_memory_service_light(project_id: str):
    """Get memory service without verifying the project in the database."""
    return _try_create_memory_service(project_id, project_id)


# ===========================================================================
# Convenience endpoints: /memories/search, /memories/stats, /memories/store
# ===========================================================================

@router.get("/memories/search", response_model=List[MemoryResponse])
async def search_memories_get(
    query: str = Query(..., description="Search query"),
    project_id: str = Query(..., description="Project ID to search within"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results to return"),
    min_relevance: float = Query(0.3, ge=0.0, le=1.0, description="Minimum relevance score"),
):
    """
    Search memories by query (GET convenience endpoint).

    Accepts project_id as a query parameter instead of a path parameter.
    Returns an empty list with 200 if the memory system is unavailable.
    """
    memory_service = get_memory_service_light(project_id)
    if memory_service is None:
        return []

    try:
        results = await memory_service.search_memories(
            query=query,
            limit=limit,
            min_relevance=min_relevance,
        )
    except Exception as exc:
        logger.error(f"Memory search failed: {exc}")
        return []

    return [
        MemoryResponse(
            id=r.get("id", ""),
            content=r.get("content", ""),
            memory_type=r.get("memory_type", "conversation"),
            relevance_score=r.get("relevance_score"),
            timestamp=r.get("timestamp", datetime.now().isoformat()),
            metadata=r.get("metadata", {}),
        )
        for r in results
    ]


@router.get("/memories/stats", response_model=MemoryStatsResponse)
async def get_memory_stats_convenience(
    project_id: str = Query(..., description="Project ID"),
):
    """
    Memory analytics (GET convenience endpoint).

    Returns total memories, type distribution, total cost, etc.
    Returns zeros with memory_available=False when ChromaDB is unavailable.
    """
    memory_service = get_memory_service_light(project_id)
    if memory_service is None:
        return _empty_stats()

    try:
        stats = await memory_service.get_memory_stats()
    except Exception as exc:
        logger.error(f"Memory stats failed: {exc}")
        return _empty_stats(f"Error retrieving stats: {exc}")

    return MemoryStatsResponse(
        total_memories=stats.get("total_memories", 0),
        memory_types=stats.get("memory_types", {}),
        total_cost=stats.get("total_cost", 0.0),
        avg_access_count=stats.get("avg_access_count", 0.0),
        oldest_memory=stats.get("oldest_memory"),
        newest_memory=stats.get("newest_memory"),
        memory_available=True,
    )


@router.post("/memories/store", response_model=MemoryResponse)
async def store_memory_convenience(
    request: MemoryStoreRequest,
    project_id: str = Query(..., description="Project ID"),
):
    """
    Manual memory storage (POST convenience endpoint).

    Stores a memory in the given project's collection.
    Returns 503 if the memory system is unavailable.
    """
    memory_service = get_memory_service_light(project_id)
    if memory_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory system not available (ChromaDB may be missing)",
        )

    # Convert memory type safely
    memory_type_enum = MemoryType.CONVERSATION
    if ADAM_MEMORY_AVAILABLE:
        try:
            from adam.memory.core import MemoryType as _MT
            memory_type_enum = _MT[request.memory_type.upper()]
        except (KeyError, ImportError):
            pass

    try:
        memory_id = await memory_service.store_memory(
            content=request.content,
            memory_type=memory_type_enum,
            metadata=request.metadata,
            conversation_id=request.conversation_id,
            cost=request.cost,
        )
    except Exception as exc:
        logger.error(f"Memory store failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to store memory: {exc}",
        )

    return MemoryResponse(
        id=memory_id,
        content=request.content,
        memory_type=request.memory_type,
        timestamp=datetime.now().isoformat(),
        metadata=request.metadata or {},
    )


# ===========================================================================
# Project-scoped endpoints (original REST style)
# ===========================================================================

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
    if memory_service is None:
        return []

    # Convert memory types
    memory_types = None
    if ADAM_MEMORY_AVAILABLE and request.memory_types:
        memory_types = []
        for type_str in request.memory_types:
            try:
                from adam.memory.core import MemoryType as _MT
                memory_types.append(_MT[type_str.upper()])
            except (KeyError, ImportError):
                pass

    try:
        results = await memory_service.search_memories(
            query=request.query,
            limit=request.limit,
            memory_types=memory_types,
            min_relevance=request.min_relevance,
            conversation_id=request.conversation_id
        )
    except Exception as exc:
        logger.error(f"Memory search failed: {exc}")
        return []

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
    if memory_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory system not available"
        )

    # Convert memory type
    memory_type_enum = MemoryType.CONVERSATION
    if ADAM_MEMORY_AVAILABLE:
        try:
            from adam.memory.core import MemoryType as _MT
            memory_type_enum = _MT[request.memory_type.upper()]
        except (KeyError, ImportError):
            pass

    try:
        memory_id = await memory_service.store_memory(
            content=request.content,
            memory_type=memory_type_enum,
            metadata=request.metadata,
            conversation_id=request.conversation_id,
            cost=request.cost
        )
    except Exception as exc:
        logger.error(f"Memory store failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to store memory: {exc}",
        )

    return MemoryResponse(
        id=memory_id,
        content=request.content,
        memory_type=request.memory_type,
        timestamp=datetime.now().isoformat(),
        metadata=request.metadata or {}
    )


@router.get("/projects/{project_id}/memories/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(
    project_id: str,
    memory_service=Depends(get_memory_service)
):
    """Get statistics about the project's memory collection"""
    if memory_service is None:
        return _empty_stats()

    try:
        stats = await memory_service.get_memory_stats()
    except Exception as exc:
        logger.error(f"Memory stats failed: {exc}")
        return _empty_stats(f"Error retrieving stats: {exc}")

    return MemoryStatsResponse(
        total_memories=stats.get("total_memories", 0),
        memory_types=stats.get("memory_types", {}),
        total_cost=stats.get("total_cost", 0.0),
        avg_access_count=stats.get("avg_access_count", 0.0),
        oldest_memory=stats.get("oldest_memory"),
        newest_memory=stats.get("newest_memory"),
        memory_available=True,
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

    if memory_service is None:
        return {"message": "Memory system not available", "deleted": 0}

    try:
        count = await memory_service.clear_memories()
    except Exception as exc:
        logger.error(f"Memory clear failed: {exc}")
        return {"message": f"Error clearing memories: {exc}", "deleted": 0}

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
    if memory_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory system not available"
        )

    try:
        export_data = await memory_service.export_memories()
    except Exception as exc:
        logger.error(f"Memory export failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Export failed: {exc}",
        )

    return MemoryExportResponse(**export_data)


@router.post("/projects/{project_id}/memories/import")
async def import_memories(
    project_id: str,
    export_data: MemoryExportResponse,
    memory_service=Depends(get_memory_service)
):
    """Import memories from an export"""
    if memory_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory system not available"
        )

    if export_data.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Export is from different project ({export_data.project_id}). "
                   "Create new endpoint if cross-project import is needed."
        )

    try:
        imported = await memory_service.import_memories(export_data.dict())
    except Exception as exc:
        logger.error(f"Memory import failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Import failed: {exc}",
        )

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
            "types": [],
            "message": "Memory system not available (ChromaDB may be missing)",
        }

    try:
        from adam.memory.core import MemoryType as _MT
        return {
            "available": True,
            "types": [
                {
                    "value": mt.value,
                    "name": mt.name,
                    "description": mt.value.replace("_", " ").title()
                }
                for mt in _MT
            ]
        }
    except Exception as exc:
        logger.error(f"Failed to list memory types: {exc}")
        return {
            "available": False,
            "types": [],
            "message": f"Error: {exc}",
        }
