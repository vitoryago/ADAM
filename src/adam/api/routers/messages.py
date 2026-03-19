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

# Check worthiness evaluator availability
try:
    from adam.memory.core import MemoryWorthinessEvaluator, QueryComplexity, MemoryType
    _worthiness_evaluator = MemoryWorthinessEvaluator()
    WORTHINESS_EVALUATOR_AVAILABLE = True
except ImportError:
    _worthiness_evaluator = None
    WORTHINESS_EVALUATOR_AVAILABLE = False
    logger.warning("Memory worthiness evaluator not available")


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


def _format_memory_context(memories: list) -> str:
    """Format memory search results into a readable context string for the LLM.

    Each memory entry includes a relevance percentage so the model can weigh
    the importance of each recalled piece of context.
    """
    if not memories:
        return ""

    memory_parts = []
    for mem in memories:
        content = mem.get("content", mem.get("document", ""))
        if not isinstance(content, str):
            content = str(content)
        # Truncate long content to keep the prompt manageable
        content = content[:300]
        similarity = mem.get("relevance_score", mem.get("similarity", 0))
        memory_parts.append(f"[Relevance: {similarity:.0%}] {content}")

    return "Relevant context from previous conversations:\n" + "\n---\n".join(memory_parts)


async def _get_memory_context(message_content, project_id, project_name):
    """Retrieve memory context for a message.

    Searches the project-scoped memory collection and formats results with
    relevance scores. Returns an empty string on any failure so that the
    chat pipeline is never blocked by memory issues.
    """
    if not ADAM_MEMORY_AVAILABLE:
        return ""

    try:
        memory_service = _get_memory_service(project_id, project_name)
        if not memory_service:
            return ""

        memories = await memory_service.search_memories(
            query=message_content,
            conversation_id=None,  # Search across all conversations in the project
            limit=5,
        )

        if memories:
            context = _format_memory_context(memories)
            logger.info(
                "Memory recall returned %d results for query '%.50s...'",
                len(memories),
                message_content,
            )
            return context
    except Exception as e:
        logger.warning(f"Memory recall failed: {e}")

    return ""


async def _store_memory_if_worthy(
    query: str,
    response_content: str,
    response_cost: float,
    response_tokens: int,
    model_used: str,
    project_id: str,
    project_name: str,
    conversation_id: str,
):
    """Evaluate whether a response is worth storing in memory and persist it.

    Uses the MemoryWorthinessEvaluator when available, otherwise falls back
    to a simple heuristic based on cost and token count.  All failures are
    caught and logged -- memory storage must never crash the chat.
    """
    should_store = False
    store_reason = "heuristic"

    if WORTHINESS_EVALUATOR_AVAILABLE and _worthiness_evaluator:
        try:
            complexity = _worthiness_evaluator.assess_query_complexity(query)
            should_store, store_reason = _worthiness_evaluator.should_store_memory(
                query, response_content, response_cost, complexity
            )
        except Exception as e:
            logger.warning(f"Worthiness evaluation failed, falling back to heuristic: {e}")
            should_store = None  # signal to use fallback

    # Fallback heuristic when evaluator is unavailable or errored
    if should_store is None or (not WORTHINESS_EVALUATOR_AVAILABLE):
        is_substantial = len(response_content.split()) > 100 or any(
            term in response_content.upper()
            for term in ["DBT", "PDT", "MODEL", "SQL", "CREATE", "SELECT"]
        )
        cost_threshold = 0.000001
        token_threshold = 200 if is_substantial else 500
        should_store = response_cost > cost_threshold or response_tokens > token_threshold
        store_reason = "heuristic-fallback"

    if not should_store:
        logger.debug(
            "Memory storage skipped: %s (query='%.50s...')", store_reason, query
        )
        return

    try:
        memory_service = _get_memory_service(project_id, project_name)
        if not memory_service:
            return

        mem_type = MemoryType.CONVERSATION if WORTHINESS_EVALUATOR_AVAILABLE else None
        if mem_type is None:
            from adam.memory.core import MemoryType as MT
            mem_type = MT.CONVERSATION

        memory_id = await memory_service.store_memory(
            content=f"Q: {query}\n\nA: {response_content}",
            memory_type=mem_type,
            metadata={
                "model": model_used,
                "cost": response_cost,
                "tokens": response_tokens,
                "store_reason": store_reason,
            },
            conversation_id=conversation_id,
            cost=response_cost,
        )
        logger.info(
            "Stored memory %s for conversation %s (reason: %s)",
            memory_id,
            conversation_id,
            store_reason,
        )
    except Exception as e:
        logger.error(f"Error storing memory: {e}")


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

    # Get full conversation history (sliding window in LLMService handles truncation)
    history_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
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

        # Store in memory if worthy (uses MemoryWorthinessEvaluator when available)
        if message_data.use_memory:
            await _store_memory_if_worthy(
                query=message_data.content,
                response_content=response.content,
                response_cost=response.cost,
                response_tokens=response.tokens_used,
                model_used=response.model_used,
                project_id=project.id,
                project_name=project.name,
                conversation_id=conversation_id,
            )

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

    # Get full conversation history (sliding window in LLMService handles truncation)
    history_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
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

            # Store in memory if worthy (uses MemoryWorthinessEvaluator when available)
            if message_data.use_memory:
                await _store_memory_if_worthy(
                    query=message_data.content,
                    response_content=full_response,
                    response_cost=cost,
                    response_tokens=tokens_used,
                    model_used=model_used,
                    project_id=project.id,
                    project_name=project.name,
                    conversation_id=conversation_id,
                )

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
