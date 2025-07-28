"""
Project management endpoints for ADAM v2.0
"""

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List
import logging

from database import get_db
from models import (
    Project, ProjectCreate, ProjectUpdate, ProjectResponse,
    Conversation
)
from services.memory_service import ProjectMemoryService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new project with isolated memory space"""
    # Create project in database
    project = Project(
        name=project_data.name,
        description=project_data.description,
        settings=project_data.settings
    )
    
    db.add(project)
    await db.commit()
    await db.refresh(project)
    
    # Initialize memory collection for this project
    try:
        memory_service = ProjectMemoryService(project.id, project.name)
        # Collection is initialized automatically in __init__
        logger.info(f"Created project {project.id} with memory collection")
    except Exception as e:
        logger.error(f"Failed to create memory collection: {e}")
        # Don't fail the request, project can work without memory initially
    
    # Calculate conversation count
    conv_count_result = await db.execute(
        select(func.count(Conversation.id))
        .where(Conversation.project_id == project.id)
    )
    conversation_count = conv_count_result.scalar() or 0
    
    response = ProjectResponse.model_validate(project)
    response.conversation_count = conversation_count
    response.memory_count = 0  # TODO: Get from ChromaDB
    
    return response


@router.get("/", response_model=List[ProjectResponse])
async def list_projects(
    include_archived: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """List all projects"""
    query = select(Project)
    
    if not include_archived:
        query = query.where(Project.is_archived == False)
    
    query = query.order_by(Project.created_at.desc())
    
    result = await db.execute(query)
    projects = result.scalars().all()
    
    # Calculate counts for each project
    project_responses = []
    for project in projects:
        # Get conversation count
        conv_count_result = await db.execute(
            select(func.count(Conversation.id))
            .where(Conversation.project_id == project.id)
        )
        conversation_count = conv_count_result.scalar() or 0
        
        # Get memory count
        try:
            memory_service = ProjectMemoryService(project.id, project.name)
            stats = await memory_service.get_memory_stats()
            memory_count = stats.get('total_memories', 0)
        except:
            memory_count = 0
        
        response = ProjectResponse.model_validate(project)
        response.conversation_count = conversation_count
        response.memory_count = memory_count
        project_responses.append(response)
    
    return project_responses


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific project"""
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    # Get memory count
    try:
        memory_service = ProjectMemoryService(project.id, project.name)
        stats = await memory_service.get_memory_stats()
        memory_count = stats.get('total_memories', 0)
    except:
        project.memory_count = 0
    
    # Calculate conversation count
    conv_count_result = await db.execute(
        select(func.count(Conversation.id))
        .where(Conversation.project_id == project.id)
    )
    conversation_count = conv_count_result.scalar() or 0
    
    response = ProjectResponse.model_validate(project)
    response.conversation_count = conversation_count
    response.memory_count = 0  # TODO: Get from ChromaDB
    
    return response


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    update: ProjectUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a project"""
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    # Update fields
    if update.name is not None:
        project.name = update.name
    if update.description is not None:
        project.description = update.description
    if update.settings is not None:
        project.settings = {**project.settings, **update.settings}
    if update.is_archived is not None:
        project.is_archived = update.is_archived
    
    await db.commit()
    await db.refresh(project)
    
    # Calculate conversation count
    conv_count_result = await db.execute(
        select(func.count(Conversation.id))
        .where(Conversation.project_id == project.id)
    )
    conversation_count = conv_count_result.scalar() or 0
    
    response = ProjectResponse.model_validate(project)
    response.conversation_count = conversation_count
    response.memory_count = 0  # TODO: Get from ChromaDB
    
    return response


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a project and its memory collection"""
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    # Delete memory collection
    try:
        from services.memory_service import ProjectMemoryService
        memory_service = ProjectMemoryService(project_id, project.name)
        await memory_service.clear_memories()
        logger.info(f"Deleted memory collection for project {project_id}")
    except Exception as e:
        logger.error(f"Failed to delete memory collection: {e}")
    
    # Delete project (cascades to conversations and messages)
    await db.delete(project)
    await db.commit()


@router.post("/{project_id}/archive", response_model=ProjectResponse)
async def archive_project(
    project_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Archive a project (keeps memory but hides from main list)"""
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    project.is_archived = True
    await db.commit()
    await db.refresh(project)
    
    # Calculate conversation count
    conv_count_result = await db.execute(
        select(func.count(Conversation.id))
        .where(Conversation.project_id == project.id)
    )
    conversation_count = conv_count_result.scalar() or 0
    
    response = ProjectResponse.model_validate(project)
    response.conversation_count = conversation_count
    response.memory_count = 0  # TODO: Get from ChromaDB
    
    return response


@router.get("/{project_id}/stats")
async def get_project_stats(
    project_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get project statistics including memory usage"""
    # Get project
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    # Get conversation count
    conv_count_result = await db.execute(
        select(func.count(Conversation.id))
        .where(Conversation.project_id == project_id)
    )
    conversation_count = conv_count_result.scalar() or 0
    
    # Get memory stats
    try:
        from services.memory_service import ProjectMemoryService
        memory_service = ProjectMemoryService(project_id, project.name)
        memory_stats = await memory_service.get_memory_stats()
    except:
        memory_stats = {"total_memories": 0, "error": "Memory collection not found"}
    
    return {
        "project_id": project_id,
        "name": project.name,
        "conversation_count": conversation_count,
        "memory_stats": memory_stats,
        "created_at": project.created_at,
        "last_active": project.updated_at
    }