"""
Deep Discussion API endpoints for ADAM 4.0

Handles creating, configuring, and running Deep Discussion sessions
with SSE streaming support.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from pydantic import BaseModel
import json
import logging
from datetime import datetime

from adam.database import get_db
from adam.api.models import (
    DeepDiscussionSessionDB,
    DeepDiscussionSessionCreate,
    DeepDiscussionSessionUpdate,
    DeepDiscussionSessionResponse,
    Conversation, Message, Project,
)
from adam.deep_discussion.config import get_smart_defaults

router = APIRouter()
logger = logging.getLogger(__name__)


class FromConversationRequest(BaseModel):
    """Request body for creating a session from a conversation."""
    question: str


@router.post("/sessions", response_model=DeepDiscussionSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    session_data: DeepDiscussionSessionCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new Deep Discussion session with smart defaults."""
    # Verify project exists
    result = await db.execute(
        select(Project).where(Project.id == session_data.project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    # Apply smart defaults for model assignments
    smart_defaults = get_smart_defaults()

    session = DeepDiscussionSessionDB(
        project_id=session_data.project_id,
        conversation_id=session_data.conversation_id,
        question=session_data.question,
        pattern=session_data.pattern,
        model_assignments=smart_defaults,
    )

    db.add(session)
    await db.commit()
    await db.refresh(session)

    logger.info("Created deep discussion session %s for project %s", session.id, session.project_id)
    return DeepDiscussionSessionResponse.model_validate(session)


@router.put("/sessions/{session_id}/config", response_model=DeepDiscussionSessionResponse)
async def update_session_config(
    session_id: str,
    update: DeepDiscussionSessionUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update model assignments, pattern, or budget for a session."""
    result = await db.execute(
        select(DeepDiscussionSessionDB).where(DeepDiscussionSessionDB.id == session_id)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # Only update non-None fields
    if update.model_assignments is not None:
        session.model_assignments = update.model_assignments
    if update.pattern is not None:
        session.pattern = update.pattern
    if update.budget is not None:
        session.budget = update.budget

    await db.commit()
    await db.refresh(session)

    logger.info("Updated config for deep discussion session %s", session_id)
    return DeepDiscussionSessionResponse.model_validate(session)


@router.post("/sessions/{session_id}/start")
async def start_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Begin the discussion. Returns an SSE stream of progress events."""
    result = await db.execute(
        select(DeepDiscussionSessionDB).where(DeepDiscussionSessionDB.id == session_id)
    )
    session_record = result.scalar_one_or_none()

    if not session_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # Build a DeepDiscussionSession from the DB record
    from adam.deep_discussion.config import SessionConfig
    from adam.deep_discussion.session import DeepDiscussionSession

    config = SessionConfig(
        question=session_record.question,
        pattern=session_record.pattern,
        model_assignments=session_record.model_assignments or {},
        budget=session_record.budget,
        conversation_id=session_record.conversation_id,
    )
    dd_session = DeepDiscussionSession(config)
    dd_session.id = session_record.id

    async def stream_events():
        """Stream session events as SSE."""
        try:
            # Lazy-load an LLM client for the pattern pipeline
            from adam.services.llm_service import LLMService
            llm_service = LLMService()
            llm_client = llm_service.llm_client

            async for event in dd_session.run_stream(llm_client):
                yield f"data: {json.dumps(event)}\n\n"

            # Update the DB record on completion
            session_record.status = dd_session.status
            session_record.result = dd_session.result
            session_record.total_cost = dd_session.total_cost
            session_record.scratchpad_data = dd_session.scratchpad_data
            session_record.completed_at = dd_session.completed_at
            await db.commit()

        except Exception as exc:
            logger.error("Error in deep discussion stream for session %s: %s", session_id, exc)
            yield f"data: {json.dumps({'type': 'session_error', 'session_id': session_id, 'error': str(exc)})}\n\n"

            # Mark as failed in DB
            session_record.status = "failed"
            session_record.completed_at = datetime.utcnow()
            try:
                await db.commit()
            except Exception:
                logger.exception("Failed to update session status to failed")

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions/{session_id}", response_model=DeepDiscussionSessionResponse)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get session state and results."""
    result = await db.execute(
        select(DeepDiscussionSessionDB).where(DeepDiscussionSessionDB.id == session_id)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    return DeepDiscussionSessionResponse.model_validate(session)


@router.post("/sessions/{session_id}/replay", response_model=DeepDiscussionSessionResponse, status_code=status.HTTP_201_CREATED)
async def replay_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Create a new session cloned from an existing one."""
    result = await db.execute(
        select(DeepDiscussionSessionDB).where(DeepDiscussionSessionDB.id == session_id)
    )
    original = result.scalar_one_or_none()

    if not original:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # Clone with fresh id and status
    cloned = DeepDiscussionSessionDB(
        project_id=original.project_id,
        conversation_id=original.conversation_id,
        question=original.question,
        pattern=original.pattern,
        model_assignments=original.model_assignments,
        budget=original.budget,
    )

    db.add(cloned)
    await db.commit()
    await db.refresh(cloned)

    logger.info("Replayed session %s as %s", session_id, cloned.id)
    return DeepDiscussionSessionResponse.model_validate(cloned)


@router.post(
    "/sessions/from-conversation/{conversation_id}",
    response_model=DeepDiscussionSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session_from_conversation(
    conversation_id: str,
    body: FromConversationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a session from a chat conversation's recent messages."""
    # Verify conversation exists
    conv_result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = conv_result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    # Fetch last 10 messages
    msg_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(10)
    )
    messages = list(msg_result.scalars().all())
    messages.reverse()  # oldest first

    # Format as context
    context_lines = ["CONVERSATION CONTEXT:"]
    for msg in messages:
        role_label = "user" if msg.role == "user" else "assistant"
        context_lines.append(f"[{role_label}]: {msg.content}")
    context_lines.append("---")
    context_lines.append(f"QUESTION: {body.question}")
    full_question = "\n".join(context_lines)

    # Apply smart defaults
    smart_defaults = get_smart_defaults()

    session = DeepDiscussionSessionDB(
        project_id=conversation.project_id,
        conversation_id=conversation_id,
        question=full_question,
        pattern="peer_review",
        model_assignments=smart_defaults,
    )

    db.add(session)
    await db.commit()
    await db.refresh(session)

    logger.info(
        "Created deep discussion session %s from conversation %s",
        session.id,
        conversation_id,
    )
    return DeepDiscussionSessionResponse.model_validate(session)


@router.get("/sessions", response_model=List[DeepDiscussionSessionResponse])
async def list_sessions(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """List all Deep Discussion sessions for a project."""
    result = await db.execute(
        select(DeepDiscussionSessionDB)
        .where(DeepDiscussionSessionDB.project_id == project_id)
        .order_by(DeepDiscussionSessionDB.created_at.desc())
    )
    sessions = result.scalars().all()

    return [DeepDiscussionSessionResponse.model_validate(s) for s in sessions]
