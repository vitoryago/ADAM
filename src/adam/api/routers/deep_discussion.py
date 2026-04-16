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
import asyncio
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
from adam.llm.config import LLMConfig

router = APIRouter()
logger = logging.getLogger(__name__)
ACTIVE_SESSION_TASKS: dict[str, asyncio.Task] = {}
UNSUPPORTED_DEEP_DISCUSSION_MODELS = {
    "grok-4.20-multi-agent-0309": (
        "Model 'grok-4.20-multi-agent-0309' is not supported in Deep Discussion "
        "role slots because xAI only allows it through a dedicated multi-agent surface."
    ),
}


class FromConversationRequest(BaseModel):
    """Request body for creating a session from a conversation."""
    question: str


def _validate_model_assignments(model_assignments: dict[str, str]) -> None:
    """Reject unknown or unavailable cloud model assignments."""
    llm_config = LLMConfig()
    known_models = llm_config.models

    for role, model_id in model_assignments.items():
        if model_id in UNSUPPORTED_DEEP_DISCUSSION_MODELS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{UNSUPPORTED_DEEP_DISCUSSION_MODELS[model_id]} Role: '{role}'.",
            )

        model_config = llm_config.get_model_config(model_id)
        if model_config is None:
            if model_id.startswith(("grok-", "claude-", "gpt-", "gemini-", "o")):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unknown model '{model_id}' for role '{role}'.",
                )
            continue

        if model_id not in llm_config.get_available_models():
            provider_name = model_config.provider.value
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Model '{model_id}' for role '{role}' is unavailable because "
                    f"the {provider_name} provider is not configured."
                ),
            )


@router.get("/available-models")
async def get_available_models():
    """Return the cloud models that are actually usable with current credentials."""
    llm_config = LLMConfig()
    available = [
        model_id
        for model_id in llm_config.get_available_models()
        if model_id not in UNSUPPORTED_DEEP_DISCUSSION_MODELS
    ]
    return {"available_models": available}


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
    smart_defaults = get_smart_defaults(prefer_local=session_data.prefer_local)
    _validate_model_assignments(smart_defaults)

    budget = 2.0
    if session_data.prefer_local:
        local_count = sum(
            1 for m in smart_defaults.values()
            if not any(m.startswith(p) for p in ("grok-", "claude-", "gpt-", "gemini-"))
        )
        if local_count >= 3:
            budget = 0.50

    session = DeepDiscussionSessionDB(
        project_id=session_data.project_id,
        conversation_id=session_data.conversation_id,
        question=session_data.question,
        pattern=session_data.pattern,
        model_assignments=smart_defaults,
        prefer_local=session_data.prefer_local,
        budget=budget,
    )

    db.add(session)
    await db.commit()
    await db.refresh(session)

    logger.info("Created deep discussion session %s for project %s", session.id, session.project_id)
    return DeepDiscussionSessionResponse.model_validate(session)


async def _update_session_config(
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
        _validate_model_assignments(update.model_assignments)
        session.model_assignments = update.model_assignments
    if update.pattern is not None:
        session.pattern = update.pattern
    if update.budget is not None:
        session.budget = update.budget

    await db.commit()
    await db.refresh(session)

    logger.info("Updated config for deep discussion session %s", session_id)
    return DeepDiscussionSessionResponse.model_validate(session)


@router.put("/sessions/{session_id}/config", response_model=DeepDiscussionSessionResponse)
async def update_session_config(
    session_id: str,
    update: DeepDiscussionSessionUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update model assignments, pattern, or budget for a session."""
    return await _update_session_config(session_id, update, db)


@router.patch("/sessions/{session_id}/config", response_model=DeepDiscussionSessionResponse)
async def patch_session_config(
    session_id: str,
    update: DeepDiscussionSessionUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Backwards-compatible PATCH alias for session config updates."""
    return await _update_session_config(session_id, update, db)


async def _start_session_stream(
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

    _validate_model_assignments(session_record.model_assignments or {})

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
        task = asyncio.current_task()
        if task is not None:
            ACTIVE_SESSION_TASKS[session_id] = task

        try:
            session_record.status = "running"
            await db.commit()

            # Lazy-load an LLM client for the pattern pipeline
            from adam.services.llm_service import LLMService
            llm_service = LLMService()
            llm_client = llm_service.llm_client

            async for event in dd_session.run_stream(llm_client):
                event_type = event.get("type")

                if event_type == "session_complete":
                    session_record.status = dd_session.status
                    session_record.result = dd_session.result
                    session_record.total_cost = dd_session.total_cost
                    session_record.scratchpad_data = dd_session.scratchpad_data
                    session_record.completed_at = dd_session.completed_at
                    await db.commit()
                elif event_type == "session_error":
                    session_record.status = "failed"
                    session_record.completed_at = datetime.now()
                    await db.commit()

                yield f"data: {json.dumps(event)}\n\n"

        except asyncio.CancelledError:
            logger.info("Deep discussion stream for session %s cancelled", session_id)
            session_record.status = "cancelled"
            session_record.completed_at = datetime.now()
            try:
                await db.commit()
            except Exception:
                logger.exception("Failed to update session status to cancelled")
            raise
        except Exception as exc:
            logger.error("Error in deep discussion stream for session %s: %s", session_id, exc)
            yield f"data: {json.dumps({'type': 'session_error', 'session_id': session_id, 'error': str(exc)})}\n\n"

            # Mark as failed in DB
            session_record.status = "failed"
            session_record.completed_at = datetime.now()
            try:
                await db.commit()
            except Exception:
                logger.exception("Failed to update session status to failed")
        finally:
            ACTIVE_SESSION_TASKS.pop(session_id, None)

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/sessions/{session_id}/start")
async def start_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Begin the discussion with a POST request."""
    return await _start_session_stream(session_id, db)


@router.get("/sessions/{session_id}/start")
async def start_session_sse(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Begin the discussion with EventSource-compatible GET semantics."""
    return await _start_session_stream(session_id, db)


@router.post("/sessions/{session_id}/stop", response_model=DeepDiscussionSessionResponse)
async def stop_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Cancel an active deep discussion session."""
    result = await db.execute(
        select(DeepDiscussionSessionDB).where(DeepDiscussionSessionDB.id == session_id)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    if session.status not in {"completed", "failed", "cancelled"}:
        session.status = "cancelled"
        session.completed_at = datetime.now()
        await db.commit()
        await db.refresh(session)

    active_task = ACTIVE_SESSION_TASKS.get(session_id)
    if active_task and not active_task.done():
        active_task.cancel()

    logger.info("Stopped deep discussion session %s", session_id)
    return DeepDiscussionSessionResponse.model_validate(session)


@router.get("/sessions/{session_id}/run")
async def run_session_legacy(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Legacy GET alias retained for existing frontend EventSource clients."""
    return await _start_session_stream(session_id, db)


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


@router.post(
    "/from-conversation",
    response_model=DeepDiscussionSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session_from_conversation_legacy(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Legacy body-based alias for older frontend clients."""
    conversation_id = body.get("conversation_id")
    question = body.get("question")

    if not conversation_id or not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="conversation_id and question are required",
        )

    return await create_session_from_conversation(
        conversation_id=conversation_id,
        body=FromConversationRequest(question=question),
        db=db,
    )


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
