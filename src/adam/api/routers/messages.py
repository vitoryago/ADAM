"""
Message management endpoints for ADAM 4.0
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

from adam.database import get_db
from adam.api.models import (
    Message, MessageCreate, MessageResponse,
    Conversation, Project
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Check memory availability at module level
try:
    from adam.memory.project import ProjectAwareMemory
    ADAM_MEMORY_AVAILABLE = True
except ImportError:
    ADAM_MEMORY_AVAILABLE = False
    logger.warning("Memory system not available")


def _get_llm_service(project_settings, project_id):
    """Lazy-load LLMService to avoid import-time failures."""
    from adam.services.llm_service import LLMService
    return LLMService(project_settings=project_settings, project_id=project_id)


def _get_memory_service(project_id, project_name):
    """Lazy-load memory service."""
    if not ADAM_MEMORY_AVAILABLE:
        return None
    from adam.memory.project import ProjectAwareMemory
    return ProjectAwareMemory(project_id, project_name)


async def _get_memory_context(message_content, project_id, project_name):
    """Retrieve memory context for a message."""
    memory_context = ""
    if not ADAM_MEMORY_AVAILABLE:
        return memory_context

    try:
        memory_service = _get_memory_service(project_id, project_name)
        if memory_service:
            memories = await memory_service.search_memories(
                query=message_content,
                conversation_id=None,  # Search across all conversations in the project
                limit=20
            )

            if memories:
                memory_context = "\n\n=== Relevant Memories ===\n"
                for mem in memories[:10]:
                    content = mem.get("content", "") if isinstance(mem, dict) else str(mem)
                    memory_context += f"- {content[:200]}...\n"
    except Exception as e:
        logger.error(f"Error retrieving memories: {e}")

    return memory_context


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
        image_url=f"data:image/jpeg;base64,{message_data.image_data}" if message_data.has_image and message_data.image_data else None
    )

    db.add(user_message)
    await db.commit()
    await db.refresh(user_message)

    # Get conversation history
    history_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .limit(30)
    )
    history = history_result.scalars().all()

    # Initialize LLM service with project ID for memory storage
    llm_service = _get_llm_service(project.settings, project.id)

    # Get memory context if enabled
    memory_context = ""
    if message_data.use_memory:
        memory_context = await _get_memory_context(
            message_data.content, project.id, project.name
        )

    # Check if VSCode sent workspace context with file content
    enhanced_content = message_data.content
    workspace_instructions = ""

    # Check for file operation requests AND set concise mode for VSCode
    if message_data.workspace_context:
        # Always use concise style in VSCode unless specified
        if not message_data.response_style:
            message_data.response_style = "concise"

        # Detect file operations via keyword fallback
        if "create" in message_data.content.lower() and "file" in message_data.content.lower():
            workspace_instructions = "\n\n[SYSTEM: User is in VSCode. You can create files. Be concise.]"
        elif any(phrase in message_data.content.lower() for phrase in ['debug', 'fix']):
            workspace_instructions = "\n\n[SYSTEM: User is in VSCode. For code fixes, provide the corrected version. Be concise.]"

    if message_data.workspace_context and message_data.workspace_context.get('activeFile'):
        active_file = message_data.workspace_context['activeFile']
        if isinstance(active_file, dict) and active_file.get('content'):
            # Detect if the user is referring to a file/code contextually
            content_lower = message_data.content.lower()

            file_reference_indicators = ['this', 'that', 'here', 'current', 'above', 'it', 'the file', 'the code']
            action_verbs = ['read', 'analyze', 'explain', 'review', 'check', 'look', 'help', 'understand', 'optimize', 'improve', 'debug', 'fix']

            seems_to_reference_file = (
                (len(content_lower.split()) <= 8 and any(ref in content_lower for ref in file_reference_indicators)) or
                (any(verb in content_lower for verb in action_verbs) and any(ref in content_lower for ref in file_reference_indicators)) or
                content_lower in ['?', 'and this?', 'this one?', 'how about this?', 'what about this one?']
            )

            if seems_to_reference_file:
                file_info = f"\n\n**Current file: {active_file.get('file', 'unknown')}** ({active_file.get('language', 'unknown')} file)\n"
                file_info += f"```{active_file.get('language', '')}\n{active_file.get('content', '')}\n```"
                enhanced_content = f"{message_data.content}\n{file_info}"

    # Add workspace instructions if any
    enhanced_content = enhanced_content + workspace_instructions

    # Use memory context for AI response
    combined_context = memory_context

    # Generate AI response
    try:
        response = await llm_service.generate_response(
            message=enhanced_content,
            history=history,
            memory_context=combined_context,
            model=message_data.model,
            image_data=message_data.image_data if message_data.has_image else None,
            use_search=message_data.use_search,
            search_mode=message_data.search_mode,
            response_style=message_data.response_style
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

        # If response includes citations, append them
        if response.metadata and 'citations' in response.metadata:
            citations = response.metadata['citations']
            if citations:
                citation_text = "\n\n---\n**Sources:**\n"
                for i, citation in enumerate(citations, 1):
                    citation_text += f"{i}. [{citation.get('title', 'Source')}]({citation.get('url', '#')})\n"
                assistant_message.content += citation_text

        db.add(assistant_message)
        await db.commit()
        await db.refresh(assistant_message)

        # Store in memory if worthy
        is_substantial = len(response.content.split()) > 100 or any(
            term in response.content.upper() for term in ["DBT", "PDT", "MODEL", "SQL", "CREATE", "SELECT"]
        )
        cost_threshold = 0.000001
        token_threshold = 200 if is_substantial else 500

        if message_data.use_memory and (response.cost > cost_threshold or response.tokens_used > token_threshold):
            try:
                memory_service = _get_memory_service(project.id, project.name)
                if memory_service:
                    from adam.memory.core import MemoryType
                    memory_id = await memory_service.store_memory(
                        content=f"Q: {message_data.content}\n\nA: {response.content}",
                        memory_type=MemoryType.CONVERSATION,
                        metadata={
                            "model": response.model_used,
                            "cost": response.cost,
                            "tokens": response.tokens_used
                        },
                        conversation_id=conversation_id,
                        cost=response.cost
                    )
                    logger.info(f"Stored memory {memory_id} for conversation {conversation_id}")
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
    # Verify conversation and get project
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
        image_url=f"data:image/jpeg;base64,{message_data.image_data}" if message_data.has_image and message_data.image_data else None
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

    # Initialize services
    llm_service = _get_llm_service(project.settings, project.id)

    # Check if VSCode sent workspace context
    enhanced_content = message_data.content
    workspace_instructions = ""

    if message_data.workspace_context:
        if not message_data.response_style:
            message_data.response_style = "concise"
        if any(phrase in message_data.content.lower() for phrase in ['debug', 'fix', 'create a file', 'edit this', 'modify this']):
            workspace_instructions = "\n\n[SYSTEM: User is in VSCode. For code fixes, provide the corrected version. Be concise.]"

    if message_data.workspace_context and message_data.workspace_context.get('activeFile'):
        active_file = message_data.workspace_context['activeFile']
        if isinstance(active_file, dict) and active_file.get('content'):
            content_lower = message_data.content.lower()
            file_reference_indicators = ['this', 'that', 'here', 'current', 'above', 'it', 'the file', 'the code']
            action_verbs = ['read', 'analyze', 'explain', 'review', 'check', 'look', 'help', 'understand', 'optimize', 'improve', 'debug', 'fix']

            seems_to_reference_file = (
                (len(content_lower.split()) <= 8 and any(ref in content_lower for ref in file_reference_indicators)) or
                (any(verb in content_lower for verb in action_verbs) and any(ref in content_lower for ref in file_reference_indicators)) or
                content_lower in ['?', 'and this?', 'this one?', 'how about this?', 'what about this one?']
            )

            if seems_to_reference_file:
                file_info = f"\n\n**Current file: {active_file.get('file', 'unknown')}** ({active_file.get('language', 'unknown')} file)\n"
                file_info += f"```{active_file.get('language', '')}\n{active_file.get('content', '')}\n```"
                enhanced_content = f"{message_data.content}\n{file_info}"

    enhanced_content = enhanced_content + workspace_instructions

    # Get memory context
    memory_context = ""
    if message_data.use_memory:
        memory_context = await _get_memory_context(
            message_data.content, project.id, project.name
        )

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
                message=enhanced_content,
                history=history,
                memory_context=memory_context,
                model=message_data.model,
                image_data=message_data.image_data if message_data.has_image else None,
                response_style=message_data.response_style,
                use_search=message_data.use_search,
                search_mode=message_data.search_mode
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
            is_substantial = len(full_response.split()) > 100 or any(
                term in full_response.upper() for term in ["DBT", "PDT", "MODEL", "SQL", "CREATE", "SELECT"]
            )
            cost_threshold = 0.000001
            token_threshold = 200 if is_substantial else 500

            if message_data.use_memory and (cost > cost_threshold or tokens_used > token_threshold):
                try:
                    memory_service = _get_memory_service(project.id, project.name)
                    if memory_service:
                        from adam.memory.core import MemoryType
                        await memory_service.store_memory(
                            content=f"Q: {message_data.content}\n\nA: {full_response}",
                            memory_type=MemoryType.CONVERSATION,
                            metadata={
                                "model": model_used,
                                "cost": cost,
                                "tokens": tokens_used
                            },
                            conversation_id=conversation_id,
                            cost=cost
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
            "X-Accel-Buffering": "no"
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
    """Delete a message"""
    result = await db.execute(
        select(Message).where(Message.id == message_id)
    )
    message = result.scalar_one_or_none()

    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )

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
