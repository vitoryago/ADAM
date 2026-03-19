"""
Voice endpoints for ADAM 4.0
Handles audio transcription and synthesis
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List
import logging
import json
import asyncio
import base64

from adam.database import get_db
from adam.api.models import MessageCreate, Conversation, Project, Message
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_voice_service():
    """Lazy-load VoiceService."""
    from adam.services.voice_service import VoiceService
    return VoiceService()


def _get_voice_handler():
    """Lazy-load VoiceConversationHandler."""
    from adam.services.voice_conversation_handler import VoiceConversationHandler
    return VoiceConversationHandler()


def _get_voice_websocket_config():
    """Lazy-load WebSocket voice config."""
    from adam.services.voice_websocket import ElevenLabsWebSocket, WebSocketVoiceConfig
    return ElevenLabsWebSocket, WebSocketVoiceConfig


def _get_llm_service(project_settings, project_id):
    """Lazy-load LLMService."""
    from adam.services.llm_service import LLMService
    return LLMService(project_settings=project_settings, project_id=project_id)


class TranscriptionRequest(BaseModel):
    """Request model for audio transcription"""
    audio_data: str  # Base64 encoded audio
    format: str = "webm"
    language: Optional[str] = None

class TranscriptionResponse(BaseModel):
    """Response model for transcription"""
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None

class TTSRequest(BaseModel):
    """Request model for text-to-speech"""
    text: str
    voice_id: Optional[str] = None
    stream: bool = False

class VoiceInfo(BaseModel):
    """Voice information"""
    id: str
    name: str
    preview_url: Optional[str] = None
    labels: dict = {}

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    """Transcribe audio to text using Speech-to-Text"""
    try:
        voice_service = _get_voice_service()
        result = await voice_service.transcribe_audio(
            audio_data=request.audio_data,
            format=request.format,
            language=request.language
        )

        return TranscriptionResponse(
            text=result.text,
            language=result.language,
            confidence=result.confidence
        )
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/transcribe/file", response_model=TranscriptionResponse)
async def transcribe_audio_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    """Transcribe audio file to text"""
    try:
        voice_service = _get_voice_service()
        audio_data = await file.read()
        format = file.filename.split('.')[-1] if '.' in file.filename else 'webm'

        result = await voice_service.transcribe_audio(
            audio_data=audio_data,
            format=format,
            language=language
        )

        return TranscriptionResponse(
            text=result.text,
            language=result.language,
            confidence=result.confidence
        )
    except Exception as e:
        logger.error(f"File transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """Synthesize speech from text using Text-to-Speech"""
    try:
        voice_service = _get_voice_service()
        if request.stream:
            async def audio_stream():
                async for chunk in await voice_service.synthesize_speech(
                    text=request.text,
                    voice_id=request.voice_id,
                    stream=True
                ):
                    yield chunk

            return StreamingResponse(
                audio_stream(),
                media_type="audio/mpeg",
                headers={
                    "Cache-Control": "no-cache",
                    "Transfer-Encoding": "chunked"
                }
            )
        else:
            audio_response = await voice_service.synthesize_speech(
                text=request.text,
                voice_id=request.voice_id,
                stream=False
            )

            return Response(
                content=audio_response.audio_data,
                media_type=f"audio/{audio_response.format}",
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{audio_response.format}"
                }
            )
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/voices", response_model=list[VoiceInfo])
async def get_available_voices():
    """Get list of available voices for TTS"""
    try:
        voice_service = _get_voice_service()
        voices = await voice_service.get_available_voices()
        return [VoiceInfo(**voice) for voice in voices]
    except Exception as e:
        logger.error(f"Error fetching voices: {e}")
        return []

@router.post("/voice-chat")
async def voice_chat_endpoint(
    audio: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
    use_memory: bool = Form(True),
    model: Optional[str] = Form(None),
    voice_id: Optional[str] = Form(None),
    use_search: bool = Form(False),
    search_mode: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Complete voice chat endpoint:
    1. Transcribe audio to text
    2. Process through ADAM chat
    3. Synthesize response to speech
    """
    try:
        voice_service = _get_voice_service()
        voice_handler = _get_voice_handler()

        # Step 1: Transcribe audio
        audio_data = await audio.read()
        format = audio.filename.split('.')[-1] if '.' in audio.filename else 'webm'

        transcription = await voice_service.transcribe_audio(
            audio_data=audio_data,
            format=format
        )

        user_text = transcription.text
        logger.info(f"Transcribed text: {user_text}")

        # Step 2: Process through ADAM chat
        if not conversation_id:
            raise HTTPException(
                status_code=400,
                detail="conversation_id is required for voice chat"
            )

        # Verify conversation exists and get project
        conv_result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = conv_result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        project_result = await db.execute(
            select(Project).where(Project.id == conversation.project_id)
        )
        project = project_result.scalar_one_or_none()

        # Create user message
        user_message = Message(
            conversation_id=conversation_id,
            role="user",
            content=user_text
        )

        db.add(user_message)
        await db.commit()
        await db.refresh(user_message)

        # Get conversation history
        history_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .limit(10)
        )
        history = history_result.scalars().all()

        # Initialize LLM service
        llm_service = _get_llm_service(project.settings, project.id)

        # Get memory context if enabled
        memory_context = ""
        if use_memory:
            try:
                from adam.memory.project import ProjectAwareMemory
                memory_service = ProjectAwareMemory(project.id, project.name)
                memories = await memory_service.search_memories(
                    query=user_text,
                    conversation_id=None,
                    limit=15
                )

                if memories:
                    memory_context = "\n\n=== Relevant Memories ===\n"
                    for mem in memories:
                        content = mem.get("content", "") if isinstance(mem, dict) else str(mem)
                        memory_context += f"- {content[:200]}...\n"
            except Exception as e:
                logger.error(f"Error retrieving memories: {e}")

        # Define async callback for LLM
        async def llm_callback(messages, system_prompt):
            return await llm_service.generate_response(
                message=messages[-1]['content'] if messages else user_text,
                history=history,
                memory_context=memory_context,
                model=model,
                use_search=use_search,
                search_mode=search_mode,
                system_prompt=system_prompt
            )

        # Process through voice conversation handler
        voice_response = await voice_handler.process_voice_input(
            transcribed_text=user_text,
            conversation_id=conversation_id,
            llm_callback=llm_callback
        )

        # Get the full response for database storage
        full_response = await llm_service.generate_response(
            message=user_text,
            history=history,
            memory_context=memory_context,
            model=model,
            use_search=use_search,
            search_mode=search_mode
        )

        # Create assistant message with full content
        assistant_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=full_response.content,
            model=full_response.model_used,
            tokens_used=full_response.tokens_used,
            cost=full_response.cost
        )

        db.add(assistant_message)
        await db.commit()

        # Store in memory if worthy
        if use_memory and full_response.cost > 0.000001:
            try:
                from adam.memory.project import ProjectAwareMemory
                from adam.memory.core import MemoryType
                memory_service = ProjectAwareMemory(project.id, project.name)
                await memory_service.store_memory(
                    content=f"Q: {user_text}\n\nA: {full_response.content}",
                    memory_type=MemoryType.CONVERSATION,
                    metadata={
                        "model": full_response.model_used,
                        "cost": full_response.cost,
                        "tokens": full_response.tokens_used,
                        "voice": True,
                        "spoken_response": voice_response['spoken_text']
                    },
                    conversation_id=conversation_id,
                    cost=full_response.cost
                )
            except Exception as e:
                logger.error(f"Error storing memory: {e}")

        # Step 3: Synthesize only the spoken response
        audio_response = await voice_service.synthesize_speech(
            text=voice_response['spoken_text'],
            voice_id=voice_id,
            stream=False
        )

        import urllib.parse

        return Response(
            content=audio_response.audio_data,
            media_type=f"audio/{audio_response.format}",
            headers={
                "X-Transcription": urllib.parse.quote(user_text, safe=''),
                "X-Response-Text": urllib.parse.quote(voice_response['spoken_text'], safe=''),
                "X-Full-Response": urllib.parse.quote(full_response.content[:500], safe=''),
                "X-Model-Used": full_response.model_used,
                "X-Tokens": str(full_response.tokens_used),
                "X-Has-Code": str(bool(voice_response.get('code_blocks'))),
                "X-Wait-For-Response": str(voice_response.get('wait_for_response', False))
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/voice-stream")
async def voice_stream_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time voice streaming"""
    await websocket.accept()

    voice_service = _get_voice_service()
    ElevenLabsWebSocket, WebSocketVoiceConfig = _get_voice_websocket_config()
    ws_config = WebSocketVoiceConfig()
    elevenlabs_ws = ElevenLabsWebSocket(ws_config)

    try:
        await elevenlabs_ws.connect()

        async def receive_from_client():
            try:
                while True:
                    data = await websocket.receive_json()
                    if data.get("type") == "text":
                        text = data.get("text", "")
                        flush = data.get("flush", False)
                        await elevenlabs_ws.send_text(text, flush)
                    elif data.get("type") == "audio":
                        audio_data = data.get("data", "")
                        fmt = data.get("format", "webm")
                        if audio_data:
                            try:
                                transcription = await voice_service.transcribe_audio(
                                    audio_data=audio_data, format=fmt
                                )
                                await websocket.send_json({
                                    "type": "transcription",
                                    "text": transcription.text,
                                    "language": transcription.language
                                })
                                await elevenlabs_ws.send_text(transcription.text, flush=True)
                            except Exception as e:
                                logger.error(f"Error transcribing audio: {e}")
                                await websocket.send_json({
                                    "type": "error", "message": "Transcription failed"
                                })
            except WebSocketDisconnect:
                pass

        async def send_to_client():
            try:
                async for audio_data in elevenlabs_ws.receive_audio():
                    await websocket.send_json({
                        "type": "audio",
                        "data": base64.b64encode(audio_data["audio"]).decode(),
                        "is_final": audio_data["is_final"],
                        "alignment": audio_data.get("alignment"),
                        "normalized_alignment": audio_data.get("normalized_alignment")
                    })
            except Exception as e:
                logger.error(f"Error sending to client: {e}")

        receive_task = asyncio.create_task(receive_from_client())
        send_task = asyncio.create_task(send_to_client())

        done, pending = await asyncio.wait(
            [receive_task, send_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in pending:
            task.cancel()

    except WebSocketDisconnect:
        logger.info("Client disconnected from voice stream")
    except Exception as e:
        logger.error(f"Voice stream error: {e}")
    finally:
        await elevenlabs_ws.close()


@router.post("/synthesize/with-timing")
async def synthesize_with_timing(request: TTSRequest):
    """Synthesize speech with character-level timing information"""
    try:
        voice_service = _get_voice_service()
        result = await voice_service.synthesize_speech(
            text=request.text,
            voice_id=request.voice_id,
            stream=False,
            with_timing=True
        )
        return result
    except Exception as e:
        logger.error(f"Synthesis with timing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
