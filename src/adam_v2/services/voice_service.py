"""
Voice Service for ADAM v2.0
Handles Speech-to-Text (STT) and Text-to-Speech (TTS) operations
"""

import os
import asyncio
import logging
from typing import Optional, Union, AsyncGenerator
from dataclasses import dataclass
import base64
import httpx
from enum import Enum

logger = logging.getLogger(__name__)

class VoiceProvider(Enum):
    """Available voice service providers"""
    OPENAI_WHISPER = "openai_whisper"
    ELEVENLABS = "elevenlabs"
    XTTS = "xtts"  # Open source option
    AZURE = "azure"
    
class TTSModel(Enum):
    """Available TTS models based on our research"""
    ELEVENLABS_MULTILINGUAL = "eleven_multilingual_v2"
    ELEVENLABS_TURBO = "eleven_turbo_v2"
    XTTS_V2 = "xtts_v2"
    AZURE_NEURAL = "azure_neural"

@dataclass
class VoiceConfig:
    """Configuration for voice services"""
    stt_provider: VoiceProvider = VoiceProvider.OPENAI_WHISPER
    tts_provider: VoiceProvider = VoiceProvider.ELEVENLABS
    tts_model: TTSModel = TTSModel.ELEVENLABS_MULTILINGUAL
    voice_id: Optional[str] = None  # For ElevenLabs voice selection
    language: str = "en"
    speaking_rate: float = 1.0
    pitch: float = 0.0

@dataclass
class TranscriptionResult:
    """Result from Speech-to-Text"""
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None

@dataclass
class AudioResponse:
    """Result from Text-to-Speech"""
    audio_data: bytes  # Raw audio data
    format: str = "mp3"  # Audio format
    sample_rate: int = 44100
    
class VoiceService:
    """Unified voice service for ADAM"""
    
    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self._setup_clients()
        
    def _setup_clients(self):
        """Initialize API clients based on configuration"""
        self.clients = {}
        
        # OpenAI client for Whisper
        if self.config.stt_provider == VoiceProvider.OPENAI_WHISPER:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                try:
                    from openai import AsyncOpenAI
                    self.clients["openai"] = AsyncOpenAI(api_key=openai_key)
                    logger.info("OpenAI Whisper client initialized")
                except ImportError:
                    logger.warning("OpenAI SDK not installed for Whisper")
                    
        # ElevenLabs client
        if self.config.tts_provider == VoiceProvider.ELEVENLABS:
            elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
            if elevenlabs_key:
                self.clients["elevenlabs_key"] = elevenlabs_key
                logger.info("ElevenLabs API key configured")
                
    async def transcribe_audio(
        self, 
        audio_data: Union[bytes, str],
        format: str = "webm",
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Convert speech to text
        
        Args:
            audio_data: Audio bytes or base64 string
            format: Audio format (wav, mp3, webm, etc.)
            language: Optional language hint
            
        Returns:
            TranscriptionResult with transcribed text
        """
        # Convert base64 to bytes if needed
        if isinstance(audio_data, str):
            audio_data = base64.b64decode(audio_data)
            
        if self.config.stt_provider == VoiceProvider.OPENAI_WHISPER:
            return await self._transcribe_whisper(audio_data, format, language)
        else:
            raise NotImplementedError(f"STT provider {self.config.stt_provider} not implemented")
            
    async def _transcribe_whisper(
        self, 
        audio_data: bytes,
        format: str,
        language: Optional[str]
    ) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper"""
        client = self.clients.get("openai")
        if not client:
            raise ValueError("OpenAI client not initialized")
            
        try:
            # Create a file-like object for the API
            import io
            audio_file = io.BytesIO(audio_data)
            audio_file.name = f"audio.{format}"
            
            # Use Whisper API
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language or self.config.language
            )
            
            return TranscriptionResult(
                text=response.text,
                language=language or self.config.language
            )
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            raise
            
    async def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        stream: bool = False
    ) -> Union[AudioResponse, AsyncGenerator[bytes, None]]:
        """
        Convert text to speech
        
        Args:
            text: Text to synthesize
            voice_id: Optional voice ID override
            stream: Whether to stream audio chunks
            
        Returns:
            AudioResponse or async generator of audio chunks
        """
        if self.config.tts_provider == VoiceProvider.ELEVENLABS:
            return await self._synthesize_elevenlabs(text, voice_id, stream)
        else:
            raise NotImplementedError(f"TTS provider {self.config.tts_provider} not implemented")
            
    async def _synthesize_elevenlabs(
        self,
        text: str,
        voice_id: Optional[str],
        stream: bool
    ) -> Union[AudioResponse, AsyncGenerator[bytes, None]]:
        """Synthesize using ElevenLabs"""
        api_key = self.clients.get("elevenlabs_key")
        if not api_key:
            raise ValueError("ElevenLabs API key not configured")
            
        # Use provided voice_id or default
        voice_id = voice_id or self.config.voice_id or "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
        
        # ElevenLabs API endpoint
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "text": text,
            "model_id": self.config.tts_model.value,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        if stream:
            # Return async generator for streaming
            return self._stream_elevenlabs(url, headers, data)
        else:
            # Return complete audio
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers)
                response.raise_for_status()
                
                return AudioResponse(
                    audio_data=response.content,
                    format="mp3",
                    sample_rate=44100
                )
                
    async def _stream_elevenlabs(self, url: str, headers: dict, data: dict):
        """Stream audio from ElevenLabs"""
        # Add stream parameter
        data["stream"] = True
        url += "/stream"
        
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=data, headers=headers) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
                    
    async def get_available_voices(self) -> list:
        """Get list of available voices for TTS"""
        if self.config.tts_provider == VoiceProvider.ELEVENLABS:
            api_key = self.clients.get("elevenlabs_key")
            if not api_key:
                return []
                
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.elevenlabs.io/v1/voices",
                    headers={"xi-api-key": api_key}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return [
                        {
                            "id": voice["voice_id"],
                            "name": voice["name"],
                            "preview_url": voice.get("preview_url"),
                            "labels": voice.get("labels", {})
                        }
                        for voice in data.get("voices", [])
                    ]
        return []