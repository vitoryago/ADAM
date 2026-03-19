"""
WebSocket Voice Service for ADAM
Handles real-time voice streaming with ElevenLabs WebSocket API
"""

import os
import json
import asyncio
import logging
import websockets
from typing import Optional, AsyncGenerator, Dict, Any
from dataclasses import dataclass
import base64

logger = logging.getLogger(__name__)

@dataclass
class WebSocketVoiceConfig:
    """Configuration for WebSocket voice streaming"""
    voice_id: str = "ZthjuvLPty3kTMaNKVKb"  # Default voice
    model_id: str = "eleven_multilingual_v2"
    stability: float = 0.5
    similarity_boost: float = 0.8
    style: float = 0.0
    use_speaker_boost: bool = True
    optimize_streaming_latency: int = 0
    output_format: str = "mp3_44100_128"

class ElevenLabsWebSocket:
    """Handle WebSocket connection to ElevenLabs for real-time TTS"""

    def __init__(self, config: Optional[WebSocketVoiceConfig] = None):
        self.config = config or WebSocketVoiceConfig()
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.websocket = None
        self.is_connected = False

    async def connect(self):
        """Establish WebSocket connection"""
        ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.config.voice_id}/stream-input"

        params = {
            "model_id": self.config.model_id,
            "output_format": self.config.output_format,
            "optimize_streaming_latency": str(self.config.optimize_streaming_latency)
        }

        param_str = "&".join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{ws_url}?{param_str}"

        try:
            self.websocket = await websockets.connect(
                full_url,
                extra_headers={"xi-api-key": self.api_key}
            )
            self.is_connected = True
            logger.info("Connected to ElevenLabs WebSocket")

            await self._send_initial_config()

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise

    async def _send_initial_config(self):
        """Send initial voice configuration"""
        config_message = {
            "text": " ",
            "voice_settings": {
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost,
                "style": self.config.style,
                "use_speaker_boost": self.config.use_speaker_boost
            },
            "xi_api_key": self.api_key
        }

        await self.websocket.send(json.dumps(config_message))

    async def send_text(self, text: str, flush: bool = False):
        """Send text chunk for synthesis"""
        if not self.is_connected:
            raise RuntimeError("WebSocket not connected")

        message = {
            "text": text,
            "try_trigger_generation": flush
        }

        await self.websocket.send(json.dumps(message))

    async def receive_audio(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Receive audio chunks from WebSocket"""
        if not self.is_connected:
            raise RuntimeError("WebSocket not connected")

        while True:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)

                if "audio" in data:
                    audio_bytes = base64.b64decode(data["audio"])

                    yield {
                        "audio": audio_bytes,
                        "is_final": data.get("isFinal", False),
                        "alignment": data.get("alignment"),
                        "normalized_alignment": data.get("normalizedAlignment")
                    }

                if data.get("isFinal", False):
                    break

            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error receiving audio: {e}")
                break

    async def close(self):
        """Close WebSocket connection"""
        if self.websocket and self.is_connected:
            await self.send_text("")
            await self.websocket.close()
            self.is_connected = False
            logger.info("WebSocket connection closed")

    async def stream_text_to_speech(self, text_generator: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
        """
        Stream text chunks and yield audio chunks in real-time

        Args:
            text_generator: Async generator yielding text chunks

        Yields:
            Audio bytes
        """
        try:
            await self.connect()

            send_task = asyncio.create_task(self._send_text_chunks(text_generator))

            async for audio_data in self.receive_audio():
                if audio_data["audio"]:
                    yield audio_data["audio"]

            await send_task

        finally:
            await self.close()

    async def _send_text_chunks(self, text_generator: AsyncGenerator[str, None]):
        """Send text chunks from generator"""
        async for chunk in text_generator:
            await self.send_text(chunk, flush=True)

        await self.send_text("", flush=True)


class MultiContextWebSocket:
    """Handle multi-context WebSocket for managing multiple voice streams"""

    def __init__(self, voice_id: str = "ZthjuvLPty3kTMaNKVKb"):
        self.voice_id = voice_id
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.websocket = None
        self.is_connected = False
        self.contexts = {}

    async def connect(self):
        """Establish multi-context WebSocket connection"""
        ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/multi-stream-input"

        try:
            self.websocket = await websockets.connect(
                ws_url,
                extra_headers={"xi-api-key": self.api_key}
            )
            self.is_connected = True
            logger.info("Connected to ElevenLabs Multi-Context WebSocket")

        except Exception as e:
            logger.error(f"Multi-context WebSocket connection failed: {e}")
            raise

    async def create_context(self, context_id: str, voice_settings: Optional[Dict] = None):
        """Create a new context for independent audio generation"""
        if not self.is_connected:
            raise RuntimeError("WebSocket not connected")

        settings = voice_settings or {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True
        }

        message = {
            "text": " ",
            "voice_settings": settings,
            "context_id": context_id
        }

        await self.websocket.send(json.dumps(message))
        self.contexts[context_id] = True

    async def send_to_context(self, context_id: str, text: str, flush: bool = False):
        """Send text to a specific context"""
        if not self.is_connected:
            raise RuntimeError("WebSocket not connected")

        if context_id not in self.contexts:
            raise ValueError(f"Context {context_id} not found")

        message = {
            "text": text,
            "context_id": context_id,
            "flush": flush
        }

        await self.websocket.send(json.dumps(message))

    async def close_context(self, context_id: str):
        """Close a specific context"""
        if context_id in self.contexts:
            message = {
                "context_id": context_id,
                "close_context": True
            }
            await self.websocket.send(json.dumps(message))
            del self.contexts[context_id]

    async def close(self):
        """Close all contexts and WebSocket connection"""
        if self.websocket and self.is_connected:
            for context_id in list(self.contexts.keys()):
                await self.close_context(context_id)

            message = {"close_socket": True}
            await self.websocket.send(json.dumps(message))

            await self.websocket.close()
            self.is_connected = False
            logger.info("Multi-context WebSocket connection closed")
