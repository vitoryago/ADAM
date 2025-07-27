"""
Message management endpoints for ADAM v2.0
Handles sending messages, LLM integration, and streaming responses
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, AsyncGenerator
import json
import logging
from datetime import datetime

from database import get_db
from models import (
    Message, MessageCreate, MessageResponse,
    Conversation, Project
)
from services.llm_service import LLMService
from services.memory_service import ProjectMemoryService, ADAM_MEMORY_AVAILABLE

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def send_message(
    conversation_id: str,
    message_data: MessageCreate,
    db: AsyncSession = Depends(get_db)
):
    """Send a message and get AI response"""
    # Verify conversation exists and get project
    conv_result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = conv_result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Get project for settings and memory
    project_result = await db.execute(
        select(Project).where(Project.id == conversation.project_id)
    )
    project = project_result.scalar_one_or_none()
    
    # Create user message
    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=message_data.content,
        has_image=message_data.has_image,
        image_url=message_data.image_data if message_data.has_image else None
    )
    
    db.add(user_message)
    await db.commit()
    await db.refresh(user_message)
    
    # Get conversation history
    history_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .limit(10)  # Last 10 messages for context
    )
    history = history_result.scalars().all()
    
    # Initialize services with project ID for memory storage
    llm_service = LLMService(project_settings=project.settings, project_id=project.id)
    
    # Get memory context if enabled
    memory_context = ""
    if message_data.use_memory and ADAM_MEMORY_AVAILABLE:
        try:
            # Try to use advanced memory service
            try:
                from services.advanced_memory_service import AdvancedMemoryService
                memory_service = AdvancedMemoryService(project.id, project.name)
                
                # Use advanced search if available
                memories = await memory_service.advanced_search(
                    query=message_data.content,
                    conversation_id=conversation_id,
                    limit=5,
                    use_bm25=True,
                    use_semantic=True
                )
            except (ImportError, AttributeError):
                # Fallback to basic service
                memory_service = ProjectMemoryService(project.id, project.name)
                memories = await memory_service.search_memories(
                    query=message_data.content,
                    conversation_id=conversation_id,
                    limit=5
                )
            
            if memories:
                memory_context = "\n\n=== Relevant Memories ===\n"
                for mem in memories[:3]:  # Top 3 memories
                    memory_context += f"- {mem.content[:200]}...\n"
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
    
    # Generate AI response
    try:
        response = await llm_service.generate_response(
            message=message_data.content,
            history=history,
            memory_context=memory_context,
            model=message_data.model,
            image_data=message_data.image_data if message_data.has_image else None
        )
        
        # Create assistant message
        assistant_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=response.content,
            model=response.model_used,
            tokens_used=response.tokens_used,
            cost=response.cost
        )
        
        db.add(assistant_message)
        await db.commit()
        await db.refresh(assistant_message)
        
        # Store in memory if worthy
        if message_data.use_memory and response.cost > 0.001:  # Store if cost > $0.001
            try:
                memory_manager = ProjectMemoryManager(project.id)
                memory_manager.store_memory(
                    query=message_data.content,
                    response=response.content,
                    conversation_id=conversation_id,
                    memory_type="conversation",
                    metadata={
                        "model": response.model_used,
                        "cost": response.cost,
                        "tokens": response.tokens_used
                    }
                )
            except Exception as e:
                logger.error(f"Error storing memory: {e}")
        
        # Return both messages
        return [
            MessageResponse.model_validate(user_message),
            MessageResponse.model_validate(assistant_message)
        ]
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate response"
        )


@router.post("/conversations/{conversation_id}/messages/stream")
async def send_message_stream(
    conversation_id: str,
    message_data: MessageCreate,
    db: AsyncSession = Depends(get_db)
):
    """Send a message and stream the AI response"""
    # Verify conversation and get project (same as above)
    conv_result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = conv_result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Get project
    project_result = await db.execute(
        select(Project).where(Project.id == conversation.project_id)
    )
    project = project_result.scalar_one_or_none()
    
    # Create user message
    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=message_data.content,
        has_image=message_data.has_image,
        image_url=message_data.image_data if message_data.has_image else None
    )
    
    db.add(user_message)
    await db.commit()
    await db.refresh(user_message)
    
    # Get history
    history_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .limit(10)
    )
    history = history_result.scalars().all()
    
    # Initialize services with project ID for memory storage
    llm_service = LLMService(project_settings=project.settings, project_id=project.id)
    
    # Get memory context
    memory_context = ""
    if message_data.use_memory and ADAM_MEMORY_AVAILABLE:
        try:
            # Try to use advanced memory service
            try:
                from services.advanced_memory_service import AdvancedMemoryService
                memory_service = AdvancedMemoryService(project.id, project.name)
                
                # Use advanced search if available
                memories = await memory_service.advanced_search(
                    query=message_data.content,
                    conversation_id=conversation_id,
                    limit=5,
                    use_bm25=True,
                    use_semantic=True
                )
            except (ImportError, AttributeError):
                # Fallback to basic service
                memory_service = ProjectMemoryService(project.id, project.name)
                memories = await memory_service.search_memories(
                    query=message_data.content,
                    conversation_id=conversation_id,
                    limit=5
                )
            
            if memories:
                memory_context = "\n\n=== Relevant Memories ===\n"
                for mem in memories[:3]:
                    memory_context += f"- {mem.content[:200]}...\n"
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
    
    async def stream_response():
        """Stream the response as Server-Sent Events"""
        try:
            # Stream user message first
            yield f"data: {json.dumps({'type': 'user_message', 'id': user_message.id, 'content': user_message.content})}\n\n"
            
            # Stream AI response
            full_response = ""
            tokens_used = 0
            cost = 0.0
            model_used = message_data.model or "automatic"
            
            async for chunk in llm_service.stream_response(
                message=message_data.content,
                history=history,
                memory_context=memory_context,
                model=message_data.model,
                image_data=message_data.image_data if message_data.has_image else None
            ):
                full_response += chunk.content
                tokens_used = chunk.tokens_used
                cost = chunk.cost
                model_used = chunk.model_used
                
                # Send chunk
                yield f"data: {json.dumps({'type': 'assistant_chunk', 'content': chunk.content})}\n\n"
            
            # Create assistant message in database
            assistant_message = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_response,
                model=model_used,
                tokens_used=tokens_used,
                cost=cost
            )
            
            db.add(assistant_message)
            await db.commit()
            await db.refresh(assistant_message)
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'complete', 'id': assistant_message.id, 'tokens': tokens_used, 'cost': cost, 'model': model_used})}\n\n"
            
            # Store in memory if worthy
            if message_data.use_memory and cost > 0.001:
                try:
                    memory_manager = ProjectMemoryManager(project.id)
                    memory_manager.store_memory(
                        query=message_data.content,
                        response=full_response,
                        conversation_id=conversation_id,
                        memory_type="conversation",
                        metadata={
                            "model": model_used,
                            "cost": cost,
                            "tokens": tokens_used
                        }
                    )
                except Exception as e:
                    logger.error(f"Error storing memory: {e}")
                    
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # Disable Nginx buffering
        }
    )


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    conversation_id: str,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """Get messages in a conversation"""
    # Verify conversation exists
    conv_result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    if not conv_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Get messages
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    messages = result.scalars().all()
    
    # Reverse to show oldest first
    messages.reverse()
    
    return [MessageResponse.model_validate(msg) for msg in messages]


@router.delete("/messages/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(
    message_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a message (soft delete for audit trail)"""
    result = await db.execute(
        select(Message).where(Message.id == message_id)
    )
    message = result.scalar_one_or_none()
    
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )
    
    # For now, we'll actually delete. In production, consider soft delete
    await db.delete(message)
    await db.commit()


@router.post("/conversations/{conversation_id}/regenerate")
async def regenerate_last_response(
    conversation_id: str,
    model: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Regenerate the last assistant response"""
    # Get last two messages (should be user then assistant)
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(2)
    )
    last_messages = result.scalars().all()
    
    if len(last_messages) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not enough messages to regenerate"
        )
    
    # Verify pattern is assistant then user (reversed due to desc order)
    if last_messages[0].role != "assistant" or last_messages[1].role != "user":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last message must be from assistant to regenerate"
        )
    
    # Delete the last assistant message
    await db.delete(last_messages[0])
    await db.commit()
    
    # Resend the user message
    message_data = MessageCreate(
        content=last_messages[1].content,
        model=model,
        has_image=last_messages[1].has_image,
        use_memory=True
    )
    
    return await send_message(conversation_id, message_data, db)