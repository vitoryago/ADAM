"""
Real-time Voice Streaming Endpoint for ADAM 4.0

Provides ChatGPT-like voice conversation with:
- Streaming LLM responses
- Sentence-by-sentence TTS synthesis
- Low latency audio playback
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import json
import asyncio
import base64
import logging
import re
import os
from typing import Optional, AsyncGenerator

from adam.database import get_db
from adam.api.models import Message, Conversation, Project
from sqlalchemy import select

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_voice_service():
    """Lazy-load VoiceService."""
    from adam.services.voice_service import VoiceService
    return VoiceService()


def _get_voice_handler():
    """Lazy-load VoiceConversationHandler."""
    from adam.services.voice_conversation_handler import VoiceConversationHandler, VOICE_SYSTEM_PROMPT
    return VoiceConversationHandler(), VOICE_SYSTEM_PROMPT


def _get_llm_service(project_settings, project_id):
    """Lazy-load LLMService."""
    from adam.services.llm_service import LLMService
    return LLMService(project_settings=project_settings, project_id=project_id)


class StreamingVoiceSession:
    """Manages a streaming voice conversation session"""

    def __init__(self, websocket: WebSocket, conversation_id: str, db: AsyncSession):
        self.websocket = websocket
        self.conversation_id = conversation_id
        self.db = db
        self.llm_service = None
        self.is_processing = False
        self.current_audio_queue = asyncio.Queue()
        self.sentence_buffer = ""
        self.audio_buffer = []
        self.spoken_sentences = set()
        self.tts_queue = asyncio.Queue()
        self.tts_worker_task = None
        self.actual_model_used = None
        self.voice_service = _get_voice_service()

    async def initialize(self):
        """Initialize the session with conversation data"""
        conv_result = await self.db.execute(
            select(Conversation).where(Conversation.id == self.conversation_id)
        )
        conversation = conv_result.scalar_one_or_none()

        if not conversation:
            raise ValueError("Conversation not found")

        project_result = await self.db.execute(
            select(Project).where(Project.id == conversation.project_id)
        )
        project = project_result.scalar_one_or_none()

        self.llm_service = _get_llm_service(project.settings, project.id)

    async def process_audio(self, audio_data: str, is_final: bool = True, format: str = "webm", mime_type: str = "audio/webm") -> None:
        """Process incoming audio data"""
        if self.is_processing:
            await self.send_message({
                "type": "error",
                "message": "Already processing, please wait"
            })
            return

        self.is_processing = True

        try:
            logger.info(f"Transcribing audio, base64 data size: {len(audio_data)} chars")

            transcription = await self.voice_service.transcribe_audio(
                audio_data=audio_data,
                format=format,
                language="en"
            )

            await self.send_message({
                "type": "transcription",
                "text": transcription.text
            })

            await self.stream_llm_response(transcription.text)

        except ValueError as ve:
            logger.warning(f"Audio validation error: {ve}")
            await self.send_message({
                "type": "error",
                "message": f"Audio validation failed: {str(ve)}. Please speak for at least 2 seconds."
            })
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            await self.send_message({
                "type": "error",
                "message": str(e)
            })
        finally:
            self.is_processing = False

    async def process_audio_chunk(self, audio_data: str, is_final: bool) -> None:
        """Process streaming audio chunks"""
        try:
            self.audio_buffer.append(audio_data)

            if len(self.audio_buffer) >= 1:
                combined_audio = "".join(self.audio_buffer)

                transcription = await self.voice_service.transcribe_audio(
                    audio_data=combined_audio,
                    format="webm"
                )

                await self.send_message({
                    "type": "transcription",
                    "text": transcription.text,
                    "final": False
                })

                if is_final:
                    self.audio_buffer = []
                    await self.send_message({
                        "type": "transcription",
                        "text": transcription.text,
                        "final": True
                    })
                    await self.stream_llm_response(transcription.text)

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")

    async def stream_llm_response(self, user_text: str) -> None:
        """Stream LLM response with sentence-by-sentence TTS"""
        try:
            _, VOICE_SYSTEM_PROMPT = _get_voice_handler()

            history_result = await self.db.execute(
                select(Message)
                .where(Message.conversation_id == self.conversation_id)
                .order_by(Message.created_at.desc())
                .limit(10)
            )
            history = list(reversed(history_result.scalars().all()))

            sentence_count = 0
            full_response = ""
            self.sentence_buffer = ""
            self.spoken_sentences.clear()

            self.tts_worker_task = asyncio.create_task(self._tts_worker())

            voice_messages = [{"role": "system", "content": VOICE_SYSTEM_PROMPT}]
            for msg in history:
                voice_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            logger.info(f"Starting LLM stream for user text: {user_text[:50]}...")

            stream_generator = self.llm_service.stream_response(
                message=user_text,
                history=history,
                system_prompt=VOICE_SYSTEM_PROMPT
            )

            async for chunk in stream_generator:
                if chunk.content:
                    self.sentence_buffer += chunk.content
                    full_response += chunk.content

                if hasattr(chunk, 'model_used') and chunk.model_used:
                    self.actual_model_used = chunk.model_used

                    sentences = self.extract_complete_sentences()

                    for sentence in sentences:
                        sentence_hash = hash(sentence.strip())
                        if sentence_hash in self.spoken_sentences:
                            continue

                        sentence_count += 1

                        if self.should_speak_sentence(sentence):
                            self.spoken_sentences.add(sentence_hash)
                            await self.tts_queue.put((sentence, sentence_count))

                    await self.send_message({
                        "type": "text_chunk",
                        "content": chunk.content
                    })

            if self.sentence_buffer.strip():
                remaining_text = self.sentence_buffer.strip()
                remaining_hash = hash(remaining_text)
                if remaining_hash not in self.spoken_sentences and self.should_speak_sentence(remaining_text):
                    sentence_count += 1
                    self.spoken_sentences.add(remaining_hash)
                    await self.tts_queue.put((remaining_text, sentence_count))

            await self.tts_queue.put((None, None))

            if self.tts_worker_task:
                await self.tts_worker_task

            await self.save_messages(user_text, full_response)

            use_openai_tts = os.getenv("USE_OPENAI_TTS", "false").lower() == "true"
            await self.send_message({
                "type": "completion",
                "full_text": full_response,
                "user_text": user_text,
                "model": self.actual_model_used or getattr(self.llm_service, 'default_model', None) or "grok-3-mini-high",
                "tts_provider": "ElevenLabs" if not use_openai_tts else "OpenAI"
            })

        except Exception as e:
            logger.error(f"Error streaming LLM response: {e}")
            await self.send_message({
                "type": "error",
                "message": f"LLM error: {str(e)}"
            })

    def extract_complete_sentences(self) -> list[str]:
        """Extract complete sentences from buffer"""
        sentences = []
        pattern = r'([.!?]+)\s+(?=[A-Z])'
        match = re.search(pattern, self.sentence_buffer)

        if match:
            sentence = self.sentence_buffer[:match.end()].strip()
            if len(sentence) > 10:
                sentences.append(sentence)
                self.sentence_buffer = self.sentence_buffer[match.end():]
        elif len(self.sentence_buffer) > 300:
            sentence = self.sentence_buffer.strip()
            sentences.append(sentence)
            self.sentence_buffer = ""

        return sentences

    def should_speak_sentence(self, sentence: str) -> bool:
        """Determine if a sentence should be spoken aloud"""
        if any(pattern in sentence for pattern in ['```', 'function', 'const', 'import', '{', '}']):
            return False
        if 'http://' in sentence or 'https://' in sentence:
            return False
        if len(sentence.strip()) < 10:
            return False
        return True

    async def _tts_worker(self) -> None:
        """Worker to process TTS queue in order"""
        while True:
            try:
                text, sequence = await self.tts_queue.get()
                if text is None:
                    break
                await self.synthesize_and_send(text, sequence)
            except Exception as e:
                logger.error(f"TTS worker error: {e}")

    async def synthesize_and_send(self, text: str, sequence: int) -> None:
        """Synthesize speech and send audio chunks"""
        try:
            use_openai_tts = os.getenv("USE_OPENAI_TTS", "false").lower() == "true"

            audio_stream = await self.voice_service.synthesize_speech(
                text=text,
                stream=True,
                use_openai=use_openai_tts
            )

            chunk_number = 0
            total_bytes = 0
            async for audio_chunk in audio_stream:
                chunk_number += 1
                total_bytes += len(audio_chunk)

                await self.send_message({
                    "type": "audio_chunk",
                    "sequence": sequence,
                    "chunk": chunk_number,
                    "data": base64.b64encode(audio_chunk).decode(),
                    "text": text if chunk_number == 1 else None
                })

            logger.info(f"Sent {chunk_number} audio chunks, total {total_bytes} bytes for sentence {sequence}")

        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}", exc_info=True)

    async def save_messages(self, user_text: str, assistant_text: str) -> None:
        """Save messages to database"""
        try:
            user_msg = Message(
                conversation_id=self.conversation_id,
                role="user",
                content=user_text
            )
            self.db.add(user_msg)

            assistant_msg = Message(
                conversation_id=self.conversation_id,
                role="assistant",
                content=assistant_text,
                model=self.actual_model_used or getattr(self.llm_service, 'default_model', None) or "grok-3-mini-high"
            )
            self.db.add(assistant_msg)

            await self.db.commit()
            logger.info("Messages saved successfully")

        except Exception as e:
            logger.error(f"Error saving messages: {e}")
            await self.db.rollback()

    async def send_message(self, data: dict) -> None:
        """Send JSON message to client"""
        await self.websocket.send_json(data)


@router.websocket("/ws/voice-stream/{conversation_id}")
async def voice_stream_endpoint(
    websocket: WebSocket,
    conversation_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    WebSocket endpoint for real-time voice streaming

    Protocol:
    Client -> Server: {"type": "audio", "data": "base64_encoded_audio"}
    Server -> Client: {"type": "transcription", "text": "..."}
    Server -> Client: {"type": "text_chunk", "content": "..."}
    Server -> Client: {"type": "audio_chunk", "sequence": 1, "chunk": 1, "data": "...", "text": "..."}
    Server -> Client: {"type": "completion", "full_text": "..."}
    """
    await websocket.accept()
    session = StreamingVoiceSession(websocket, conversation_id, db)

    try:
        await session.initialize()

        while True:
            data = await websocket.receive_json()

            if data.get("type") == "audio":
                audio_data = data.get("data", "")
                is_final = data.get("final", True)
                format = data.get("format", "webm")
                mime_type = data.get("mimeType", "audio/webm")
                if audio_data:
                    asyncio.create_task(session.process_audio(
                        audio_data, is_final, format, mime_type
                    ))

            elif data.get("type") == "audio_chunk":
                audio_data = data.get("data", "")
                is_final = data.get("final", False)
                if audio_data:
                    asyncio.create_task(session.process_audio_chunk(audio_data, is_final))

            elif data.get("type") == "ping":
                await session.send_message({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from voice stream: {conversation_id}")
    except Exception as e:
        logger.error(f"Voice stream error: {e}")
        try:
            await session.send_message({
                "type": "error",
                "message": f"Stream error: {str(e)}"
            })
        except Exception:
            pass
    finally:
        await db.close()
