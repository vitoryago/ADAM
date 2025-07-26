"""
Conversation management endpoints for ADAM v2.0
"""

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List
import logging

from database import get_db
from models import (
    Conversation, ConversationCreate, ConversationUpdate, ConversationResponse,
    Project, Message
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/projects/{project_id}/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    project_id: str,
    conversation_data: ConversationCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new conversation in a project"""
    # Verify project exists
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    # Create conversation
    conversation = Conversation(
        project_id=project_id,
        title=conversation_data.title
    )
    
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    
    logger.info(f"Created conversation {conversation.id} in project {project_id}")
    
    return conversation


@router.get("/projects/{project_id}/conversations", response_model=List[ConversationResponse])
async def list_project_conversations(
    project_id: str,
    db: AsyncSession = Depends(get_db)
):
    """List all conversations in a project"""
    # Verify project exists
    project_result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    if not project_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    # Get conversations, pinned first, then by last activity
    query = select(Conversation).where(
        Conversation.project_id == project_id
    ).order_by(
        Conversation.is_pinned.desc(),
        Conversation.updated_at.desc()
    )
    
    result = await db.execute(query)
    conversations = result.scalars().all()
    
    return [ConversationResponse.model_validate(conv) for conv in conversations]


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific conversation"""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return conversation


@router.put("/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    update: ConversationUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a conversation (rename, pin/unpin)"""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Update fields
    if update.title is not None:
        conversation.title = update.title
    if update.is_pinned is not None:
        conversation.is_pinned = update.is_pinned
    
    await db.commit()
    await db.refresh(conversation)
    
    return conversation


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a conversation and all its messages"""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Delete conversation (cascades to messages)
    await db.delete(conversation)
    await db.commit()
    
    logger.info(f"Deleted conversation {conversation_id}")


@router.post("/conversations/{conversation_id}/pin", response_model=ConversationResponse)
async def pin_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Pin a conversation"""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    conversation.is_pinned = True
    await db.commit()
    await db.refresh(conversation)
    
    return conversation


@router.post("/conversations/{conversation_id}/unpin", response_model=ConversationResponse)
async def unpin_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Unpin a conversation"""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    conversation.is_pinned = False
    await db.commit()
    await db.refresh(conversation)
    
    return conversation


@router.get("/conversations/{conversation_id}/stats")
async def get_conversation_stats(
    conversation_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get conversation statistics"""
    # Get conversation
    conv_result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = conv_result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Get message count and costs
    msg_stats_result = await db.execute(
        select(
            func.count(Message.id).label("message_count"),
            func.sum(Message.tokens_used).label("total_tokens"),
            func.sum(Message.cost).label("total_cost")
        ).where(Message.conversation_id == conversation_id)
    )
    stats = msg_stats_result.one()
    
    return {
        "conversation_id": conversation_id,
        "title": conversation.title,
        "message_count": stats.message_count or 0,
        "total_tokens": stats.total_tokens or 0,
        "total_cost": float(stats.total_cost or 0),
        "created_at": conversation.created_at,
        "last_active": conversation.updated_at,
        "is_pinned": conversation.is_pinned
    }