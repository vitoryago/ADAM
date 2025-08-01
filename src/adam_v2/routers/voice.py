"""
Voice endpoints for ADAM v2.0
Handles audio transcription and synthesis
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
import logging
import json
import asyncio
import base64

from database import get_db
from services.voice_service import VoiceService, VoiceConfig, VoiceProvider, TTSModel
from services.voice_websocket import ElevenLabsWebSocket, WebSocketVoiceConfig
from services.llm_service import LLMService
from services.voice_conversation_handler import VoiceConversationHandler
from models import MessageCreate, Conversation, Project, Message
from sqlalchemy import select
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize voice service
voice_service = VoiceService()
voice_handler = VoiceConversationHandler()

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
    """
    Transcribe audio to text using Speech-to-Text
    """
    try:
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
    """
    Transcribe audio file to text
    """
    try:
        # Read file content
        audio_data = await file.read()
        
        # Get format from filename
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
    """
    Synthesize speech from text using Text-to-Speech
    """
    try:
        if request.stream:
            # Return streaming response
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
            # Return complete audio
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
    """
    Get list of available voices for TTS
    """
    try:
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
            # If no conversation_id provided, we need to either create one or error
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
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )
        
        # Get project for settings
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
        llm_service = LLMService(project_settings=project.settings, project_id=project.id)
        
        # Get memory context if enabled
        memory_context = ""
        if use_memory:
            try:
                from services.memory_service import ProjectMemoryService, ADAM_MEMORY_AVAILABLE
                if ADAM_MEMORY_AVAILABLE:
                    memory_service = ProjectMemoryService(project.id, project.name)
                    memories = await memory_service.search_memories(
                        query=user_text,
                        conversation_id=None,
                        limit=5
                    )
                    
                    if memories:
                        memory_context = "\n\n=== Relevant Memories ===\n"
                        for mem in memories[:3]:
                            memory_context += f"- {mem.content[:200]}...\n"
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
        if use_memory and full_response.cost > 0.0001:
            try:
                from services.memory_service import ProjectMemoryService
                memory_service = ProjectMemoryService(project.id, project.name)
                await memory_service.store_memory(
                    content=f"Q: {user_text}\n\nA: {full_response.content}",
                    memory_type="voice_conversation",
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
        
        # Include both spoken and visual content in headers (safely encoded)
        import urllib.parse
        
        return Response(
            content=audio_response.audio_data,
            media_type=f"audio/{audio_response.format}",
            headers={
                "X-Transcription": urllib.parse.quote(user_text, safe=''),
                "X-Response-Text": urllib.parse.quote(voice_response['spoken_text'], safe=''),
                "X-Full-Response": urllib.parse.quote(full_response.content[:500], safe=''),  # Limit size
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
    """
    WebSocket endpoint for real-time voice streaming
    Handles bidirectional audio streaming
    """
    await websocket.accept()
    
    # Initialize ElevenLabs WebSocket
    ws_config = WebSocketVoiceConfig()
    elevenlabs_ws = ElevenLabsWebSocket(ws_config)
    
    try:
        # Connect to ElevenLabs
        await elevenlabs_ws.connect()
        
        # Create tasks for handling incoming and outgoing data
        receive_task = asyncio.create_task(receive_from_client(websocket, elevenlabs_ws))
        send_task = asyncio.create_task(send_to_client(websocket, elevenlabs_ws))
        
        # Wait for either task to complete
        done, pending = await asyncio.wait(
            [receive_task, send_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            
    except WebSocketDisconnect:
        logger.info("Client disconnected from voice stream")
    except Exception as e:
        logger.error(f"Voice stream error: {e}")
    finally:
        await elevenlabs_ws.close()
        

async def receive_from_client(websocket: WebSocket, elevenlabs_ws: ElevenLabsWebSocket):
    """Receive text/audio from client and forward to ElevenLabs"""
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "text":
                # Send text to ElevenLabs for synthesis
                text = data.get("text", "")
                flush = data.get("flush", False)
                await elevenlabs_ws.send_text(text, flush)
                
            elif data.get("type") == "audio":
                # Handle audio transcription
                audio_data = data.get("data", "")
                format = data.get("format", "webm")
                
                if audio_data:
                    try:
                        # Transcribe the audio chunk
                        transcription = await voice_service.transcribe_audio(
                            audio_data=audio_data,
                            format=format
                        )
                        
                        # Send transcription back to client
                        await websocket.send_json({
                            "type": "transcription",
                            "text": transcription.text,
                            "language": transcription.language
                        })
                        
                        # Forward transcribed text to ElevenLabs
                        await elevenlabs_ws.send_text(transcription.text, flush=True)
                        
                    except Exception as e:
                        logger.error(f"Error transcribing audio: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Transcription failed"
                        })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Error receiving from client: {e}")
        

async def send_to_client(websocket: WebSocket, elevenlabs_ws: ElevenLabsWebSocket):
    """Receive audio from ElevenLabs and forward to client"""
    try:
        async for audio_data in elevenlabs_ws.receive_audio():
            # Send audio chunk to client
            await websocket.send_json({
                "type": "audio",
                "data": base64.b64encode(audio_data["audio"]).decode(),
                "is_final": audio_data["is_final"],
                "alignment": audio_data.get("alignment"),
                "normalized_alignment": audio_data.get("normalized_alignment")
            })
            
    except Exception as e:
        logger.error(f"Error sending to client: {e}")
        

@router.post("/synthesize/with-timing")
async def synthesize_with_timing(request: TTSRequest):
    """
    Synthesize speech with character-level timing information
    """
    try:
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