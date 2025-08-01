"""
Voice Service for ADAM v2.0
Handles Speech-to-Text (STT) and Text-to-Speech (TTS) operations
"""

import os
import asyncio
import logging
from typing import Optional, Union, AsyncGenerator, List, Dict
from dataclasses import dataclass
import base64
import httpx
from enum import Enum
import io

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
    ELEVENLABS_TURBO_V2_5 = "eleven_turbo_v2_5"
    ELEVENLABS_V3 = "eleven_v3"  # For dialogue
    XTTS_V2 = "xtts_v2"
    AZURE_NEURAL = "azure_neural"

@dataclass
class VoiceConfig:
    """Configuration for voice services"""
    stt_provider: VoiceProvider = VoiceProvider.OPENAI_WHISPER
    tts_provider: VoiceProvider = VoiceProvider.ELEVENLABS
    tts_model: TTSModel = TTSModel.ELEVENLABS_V3  # Better for natural dialogue
    voice_id: Optional[str] = None  # For ElevenLabs voice selection
    language: str = "en"
    speaking_rate: float = 0.85  # Slower speech rate
    pitch: float = 0.0
    stability: float = 0.75  # More stable for clearer speech
    similarity_boost: float = 0.75  # Slightly less boost for naturalness
    style: float = 0.0
    use_speaker_boost: bool = True

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
    
@dataclass
class TimingInfo:
    """Character timing information"""
    characters: List[str]
    character_start_times_seconds: List[float]
    character_end_times_seconds: List[float]
    
class VoiceService:
    """Unified voice service for ADAM"""
    
    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        # Set default voice ID if not provided
        if not self.config.voice_id:
            self.config.voice_id = "ZthjuvLPty3kTMaNKVKb"  # Your provided voice ID
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
            elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "sk_314914abf445bf785fc32d48d544e5271b0c1511cfb74adc")
            if elevenlabs_key:
                try:
                    from elevenlabs.client import ElevenLabs
                    self.clients["elevenlabs"] = ElevenLabs(api_key=elevenlabs_key)
                    self.clients["elevenlabs_key"] = elevenlabs_key
                    logger.info("ElevenLabs client initialized")
                except ImportError:
                    logger.warning("ElevenLabs SDK not installed. Install with: pip install elevenlabs")
                    # Fallback to HTTP client
                    self.clients["elevenlabs_key"] = elevenlabs_key
                
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
        logger.info(f"Transcribing audio: format={format}, data_type={type(audio_data).__name__}")
        
        # Convert base64 to bytes if needed
        if isinstance(audio_data, str):
            try:
                audio_data = base64.b64decode(audio_data)
                logger.info(f"Decoded base64 audio, size={len(audio_data)} bytes")
            except Exception as e:
                logger.error(f"Failed to decode base64 audio: {e}")
                raise ValueError(f"Invalid base64 audio data: {e}")
        
        # Check minimum size - WebM files need more data
        min_size = 5000 if format == "webm" else 1000
        if len(audio_data) < min_size:
            logger.warning(f"Audio data too small: {len(audio_data)} bytes (minimum: {min_size})")
            raise ValueError(f"Audio data too small for transcription. Received {len(audio_data)} bytes, need at least {min_size}")
            
        # Log audio info for debugging
        logger.info(f"Audio data size: {len(audio_data)} bytes, format: {format}")
            
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
            import tempfile
            import os
            
            # Log file details for debugging
            logger.info(f"Processing {format} file, first 20 bytes (hex): {audio_data[:20].hex()}")
            logger.info(f"First 4 bytes (raw): {audio_data[:4]}")
            
            # Check for common file signatures
            if audio_data[:4] == b'RIFF':
                logger.info("Detected WAV file signature")
            elif audio_data[:4] == b'\x1a\x45\xdf\xa3':
                logger.info("Detected WebM/Matroska file signature")
            elif audio_data[:3] == b'ID3':
                logger.info("Detected MP3 file signature")
            else:
                logger.warning(f"Unknown file signature: {audio_data[:4].hex()}")
            
            # For WebM from browser, we need special handling
            if format == "webm" and not (audio_data[:4] == b'\x1a\x45\xdf\xa3'):
                # Browser WebM might have different structure
                logger.info("Browser-generated WebM detected, using special handling")
                
                # Try to fix the WebM header or use as-is
                # Sometimes browser WebM has a slightly different structure
                format = "webm"  # Keep format as webm
                
            # Always use temp file approach for reliability
            temp_path = None
            try:
                # Create temp file with proper extension
                with tempfile.NamedTemporaryFile(mode='wb', suffix=f'.{format}', delete=False) as tmp_file:
                    tmp_file.write(audio_data)
                    temp_path = tmp_file.name
                    logger.info(f"Created temp file: {temp_path}, size: {len(audio_data)} bytes")
                
                # First attempt: Try with the original format
                try:
                    with open(temp_path, 'rb') as audio_file_handle:
                        response = await client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file_handle,
                            language=language or self.config.language,
                            response_format="text"
                        )
                        logger.info("Transcription successful with original format")
                except Exception as first_error:
                    logger.warning(f"First attempt failed: {first_error}")
                    
                    # Second attempt: Force WAV conversion for any format
                    if os.path.exists("/usr/local/bin/ffmpeg"):
                        logger.info(f"Converting {format} to WAV using ffmpeg")
                        wav_path = temp_path.replace(f'.{format}', '.wav')
                        
                        import subprocess
                        # Use more compatible ffmpeg settings
                        result = subprocess.run(
                            [
                                '/usr/local/bin/ffmpeg',
                                '-i', temp_path,
                                '-acodec', 'pcm_s16le',  # 16-bit PCM
                                '-ar', '16000',           # 16kHz sample rate
                                '-ac', '1',               # Mono
                                '-f', 'wav',
                                wav_path
                            ],
                            capture_output=True,
                            text=True
                        )
                        
                        if result.returncode == 0:
                            logger.info(f"Converted to WAV successfully")
                            with open(wav_path, 'rb') as wav_file:
                                response = await client.audio.transcriptions.create(
                                    model="whisper-1",
                                    file=wav_file,
                                    language=language or self.config.language,
                                    response_format="text"
                                )
                                logger.info("Transcription successful with converted WAV")
                            os.unlink(wav_path)
                        else:
                            logger.error(f"FFmpeg conversion failed: {result.stderr}")
                            raise Exception(f"Audio conversion failed: {result.stderr}")
                    else:
                        raise first_error
                    
            except Exception as e:
                logger.error(f"Whisper API error: {e}")
                
                # If WebM fails, try converting to WAV using ffmpeg
                if format == "webm" and os.path.exists("/usr/local/bin/ffmpeg"):
                    logger.info("Attempting to convert WebM to WAV using ffmpeg")
                    wav_path = None
                    try:
                        # Convert WebM to WAV
                        wav_path = temp_path.replace('.webm', '.wav')
                        import subprocess
                        result = subprocess.run(
                            ['/usr/local/bin/ffmpeg', '-i', temp_path, '-ar', '16000', '-ac', '1', '-f', 'wav', wav_path],
                            capture_output=True,
                            text=True
                        )
                        
                        if result.returncode == 0:
                            logger.info(f"Converted to WAV successfully: {wav_path}")
                            # Try again with WAV
                            with open(wav_path, 'rb') as wav_file:
                                response = await client.audio.transcriptions.create(
                                    model="whisper-1",
                                    file=wav_file,
                                    language=language or self.config.language,
                                    response_format="text"
                                )
                                logger.info("Transcription successful with converted WAV")
                        else:
                            logger.error(f"FFmpeg conversion failed: {result.stderr}")
                            raise Exception(f"Audio conversion failed: {result.stderr}")
                            
                    finally:
                        if wav_path and os.path.exists(wav_path):
                            os.unlink(wav_path)
                else:
                    raise
                    
            finally:
                # Clean up temp file
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
            
            return TranscriptionResult(
                text=response.text if isinstance(response, object) and hasattr(response, 'text') else response,
                language=language or self.config.language
            )
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            raise
            
    async def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        stream: bool = False,
        with_timing: bool = False,
        use_openai: bool = False
    ) -> Union[AudioResponse, AsyncGenerator[bytes, None], Dict]:
        """
        Convert text to speech
        
        Args:
            text: Text to synthesize
            voice_id: Optional voice ID override
            stream: Whether to stream audio chunks
            with_timing: Whether to include character timing information
            use_openai: Force use of OpenAI TTS instead of configured provider
            
        Returns:
            AudioResponse, async generator of audio chunks, or dict with timing
        """
        # Allow override to use OpenAI TTS
        if use_openai and self.clients.get("openai"):
            return await self._synthesize_openai(text, voice_id, stream)
        
        if self.config.tts_provider == VoiceProvider.ELEVENLABS:
            return await self._synthesize_elevenlabs(text, voice_id, stream, with_timing)
        else:
            raise NotImplementedError(f"TTS provider {self.config.tts_provider} not implemented")
            
    async def _synthesize_elevenlabs(
        self,
        text: str,
        voice_id: Optional[str],
        stream: bool,
        with_timing: bool = False
    ) -> Union[AudioResponse, AsyncGenerator[bytes, None], Dict]:
        """Synthesize using ElevenLabs"""
        # Use provided voice_id or default
        voice_id = voice_id or self.config.voice_id
        
        # Try to use SDK first
        elevenlabs_client = self.clients.get("elevenlabs")
        if elevenlabs_client:
            try:
                if stream:
                    # Use SDK streaming
                    from elevenlabs import stream as elevenlabs_stream
                    audio_stream = elevenlabs_client.text_to_speech.stream(
                        text=text,
                        voice_id=voice_id,
                        model_id=self.config.tts_model.value,
                        voice_settings={
                            "stability": self.config.stability,
                            "similarity_boost": self.config.similarity_boost,
                            "style": self.config.style,
                            "use_speaker_boost": self.config.use_speaker_boost
                        }
                    )
                    
                    async def stream_generator():
                        for chunk in audio_stream:
                            if isinstance(chunk, bytes):
                                yield chunk
                    
                    return stream_generator()
                else:
                    # Use SDK for regular synthesis
                    audio = elevenlabs_client.text_to_speech.convert(
                        text=text,
                        voice_id=voice_id,
                        model_id=self.config.tts_model.value,
                        voice_settings={
                            "stability": self.config.stability,
                            "similarity_boost": self.config.similarity_boost,
                            "style": self.config.style,
                            "use_speaker_boost": self.config.use_speaker_boost
                        }
                    )
                    
                    # Convert generator to bytes
                    audio_bytes = b''.join(chunk for chunk in audio)
                    
                    return AudioResponse(
                        audio_data=audio_bytes,
                        format="mp3",
                        sample_rate=44100
                    )
            except Exception as e:
                logger.warning(f"ElevenLabs SDK failed, falling back to HTTP: {e}")
        
        # Fallback to HTTP API
        api_key = self.clients.get("elevenlabs_key")
        if not api_key:
            raise ValueError("ElevenLabs API key not configured")
            
        # Choose endpoint based on options
        if with_timing:
            endpoint = f"/with-timestamps"
        elif stream:
            endpoint = "/stream"
        else:
            endpoint = ""
            
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}{endpoint}"
        
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "text": text,
            "model_id": self.config.tts_model.value,
            "voice_settings": {
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost,
                "style": self.config.style,
                "use_speaker_boost": self.config.use_speaker_boost
            }
        }
        
        if stream:
            # Return async generator for streaming
            return self._stream_elevenlabs_http(url, headers, data)
        elif with_timing:
            # Return response with timing data
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers)
                response.raise_for_status()
                return response.json()
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
                
    async def _stream_elevenlabs_http(self, url: str, headers: dict, data: dict):
        """Stream audio from ElevenLabs via HTTP"""
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=data, headers=headers) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
                    
    async def _synthesize_openai(
        self,
        text: str,
        voice_id: Optional[str],
        stream: bool
    ) -> Union[AudioResponse, AsyncGenerator[bytes, None]]:
        """Synthesize using OpenAI TTS"""
        client = self.clients.get("openai")
        if not client:
            raise ValueError("OpenAI client not initialized")
            
        try:
            # OpenAI voice options: alloy, echo, fable, onyx, nova, shimmer
            voice = voice_id or "nova"  # Default to nova voice
            
            if stream:
                # Stream audio
                async def stream_generator():
                    response = await client.audio.speech.create(
                        model="tts-1",  # or "tts-1-hd" for higher quality
                        voice=voice,
                        input=text,
                        response_format="mp3",
                        speed=1.0
                    )
                    
                    # OpenAI returns the full audio, not chunks
                    # So we'll chunk it ourselves for streaming
                    audio_bytes = response.content
                    chunk_size = 4096
                    
                    for i in range(0, len(audio_bytes), chunk_size):
                        yield audio_bytes[i:i + chunk_size]
                        await asyncio.sleep(0.01)  # Small delay between chunks
                
                return stream_generator()
            else:
                # Generate complete audio
                response = await client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=text,
                    response_format="mp3",
                    speed=1.0
                )
                
                return AudioResponse(
                    audio_data=response.content,
                    format="mp3",
                    sample_rate=24000  # OpenAI TTS sample rate
                )
                
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}")
            # Fallback to ElevenLabs if OpenAI fails
            logger.info("Falling back to ElevenLabs TTS")
            return await self._synthesize_elevenlabs(text, self.config.voice_id, stream, False)
                    
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