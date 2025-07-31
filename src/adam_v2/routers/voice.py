"""
Voice endpoints for ADAM v2.0
Handles audio transcription and synthesis
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import logging

from database import get_db
from services.voice_service import VoiceService, VoiceConfig, VoiceProvider, TTSModel
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize voice service
voice_service = VoiceService()

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
    voice_id: Optional[str] = Form(None)
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
        
        # Step 2: Process through ADAM chat (simplified for now)
        # In production, this would call the existing message endpoint
        user_text = transcription.text
        
        # TODO: Call message processing service here
        # For now, return a placeholder response
        assistant_response = f"I heard you say: '{user_text}'. This is where ADAM would process and respond."
        
        # Step 3: Synthesize response
        audio_response = await voice_service.synthesize_speech(
            text=assistant_response,
            voice_id=voice_id,
            stream=False
        )
        
        return Response(
            content=audio_response.audio_data,
            media_type=f"audio/{audio_response.format}",
            headers={
                "X-Transcription": user_text,
                "X-Response-Text": assistant_response
            }
        )
        
    except Exception as e:
        logger.error(f"Voice chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))