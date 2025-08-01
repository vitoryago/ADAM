"""
Real-time Voice Streaming Endpoint

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

from database import get_db
from services.voice_service import VoiceService
from services.llm_service import LLMService
from services.voice_conversation_handler import VoiceConversationHandler, VOICE_SYSTEM_PROMPT
from models import Message, Conversation, Project
from sqlalchemy import select

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
voice_service = VoiceService()
voice_handler = VoiceConversationHandler()


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
        self.audio_buffer = []  # Buffer for streaming audio chunks
        
    async def initialize(self):
        """Initialize the session with conversation data"""
        # Get conversation and project
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
        
        self.llm_service = LLMService(
            project_settings=project.settings, 
            project_id=project.id
        )
        
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
            # Step 1: Transcribe audio
            logger.info(f"Transcribing audio, base64 data size: {len(audio_data)} chars")
            logger.info(f"Format: {format}, MIME type: {mime_type}")
            logger.info(f"First 50 chars of base64: {audio_data[:50]}...")
                
            transcription = await voice_service.transcribe_audio(
                audio_data=audio_data,
                format=format
            )
            
            await self.send_message({
                "type": "transcription",
                "text": transcription.text
            })
            
            # Step 2: Stream LLM response
            await self.stream_llm_response(transcription.text)
            
        except ValueError as ve:
            # Handle known validation errors
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
            # Add to buffer
            self.audio_buffer.append(audio_data)
            
            # Transcribe current buffer
            if len(self.audio_buffer) >= 1:  # Process every chunk for real-time
                # Combine chunks
                combined_audio = "".join(self.audio_buffer)
                
                # Transcribe
                transcription = await voice_service.transcribe_audio(
                    audio_data=combined_audio,
                    format="webm"
                )
                
                # Send streaming transcription
                await self.send_message({
                    "type": "transcription",
                    "text": transcription.text,
                    "final": False
                })
                
                # If final, clear buffer and process full response
                if is_final:
                    self.audio_buffer = []
                    await self.send_message({
                        "type": "transcription",
                        "text": transcription.text,
                        "final": True
                    })
                    # Process LLM response
                    await self.stream_llm_response(transcription.text)
                    
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    async def stream_llm_response(self, user_text: str) -> None:
        """Stream LLM response with sentence-by-sentence TTS"""
        try:
            # Get conversation history
            history_result = await self.db.execute(
                select(Message)
                .where(Message.conversation_id == self.conversation_id)
                .order_by(Message.created_at.desc())
                .limit(10)
            )
            history = list(reversed(history_result.scalars().all()))
            
            # Start streaming LLM response
            sentence_count = 0
            full_response = ""
            
            # Add a simple system prompt for voice
            voice_messages = [{"role": "system", "content": VOICE_SYSTEM_PROMPT}]
            for msg in history:
                voice_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            logger.info(f"Starting LLM stream for user text: {user_text[:50]}...")
            
            async for chunk in self.llm_service.stream_response(
                message=user_text,
                history=history,
                system_prompt=VOICE_SYSTEM_PROMPT
            ):
                if chunk.content:
                    self.sentence_buffer += chunk.content
                    full_response += chunk.content
                    
                    # Check for complete sentences
                    sentences = self.extract_complete_sentences()
                    
                    for sentence in sentences:
                        sentence_count += 1
                        logger.info(f"Extracted sentence {sentence_count}: {sentence[:50]}...")
                        
                        # Skip code blocks and technical content for voice
                        if self.should_speak_sentence(sentence):
                            logger.info(f"Speaking sentence {sentence_count}")
                            # Start TTS for this sentence immediately
                            asyncio.create_task(
                                self.synthesize_and_send(sentence, sentence_count)
                            )
                            # Add small pause between sentences for natural flow
                            await asyncio.sleep(0.1)  # Reduced for smoother flow
                        else:
                            logger.info(f"Skipping sentence {sentence_count} (not suitable for speech)")
                    
                    # Send text chunk to frontend
                    await self.send_message({
                        "type": "text_chunk",
                        "content": chunk.content
                    })
            
            # Process any remaining text
            if self.sentence_buffer.strip():
                if self.should_speak_sentence(self.sentence_buffer):
                    await self.synthesize_and_send(self.sentence_buffer, sentence_count + 1)
            
            # Save to database
            await self.save_messages(user_text, full_response)
            
            # Signal completion
            await self.send_message({
                "type": "completion",
                "full_text": full_response
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
        
        # Look for sentence endings but be more careful about abbreviations
        # Only split if followed by space and capital letter or end of text
        pattern = r'([.!?]+)(\s+)(?=[A-Z]|$)'
        
        matches = list(re.finditer(pattern, self.sentence_buffer))
        
        if matches:
            last_end = 0
            for match in matches:
                # Get the sentence including the punctuation
                sentence = self.sentence_buffer[last_end:match.end()].strip()
                if len(sentence) > 10:  # Only add meaningful sentences
                    sentences.append(sentence)
                last_end = match.end()
            
            # Keep remaining text in buffer
            self.sentence_buffer = self.sentence_buffer[last_end:]
        
        # If buffer is getting too long without sentence end, treat as complete
        elif len(self.sentence_buffer) > 150:
            sentences.append(self.sentence_buffer.strip())
            self.sentence_buffer = ""
            
        return sentences
    
    def should_speak_sentence(self, sentence: str) -> bool:
        """Determine if a sentence should be spoken aloud"""
        # Skip code-like content
        if any(pattern in sentence for pattern in ['```', 'function', 'const', 'import', '{', '}']):
            return False
            
        # Skip URLs
        if 'http://' in sentence or 'https://' in sentence:
            return False
            
        # Skip very short sentences
        if len(sentence.strip()) < 10:
            return False
            
        return True
    
    async def synthesize_and_send(self, text: str, sequence: int) -> None:
        """Synthesize speech and send audio chunks"""
        try:
            # Check if we should use OpenAI TTS
            use_openai_tts = os.getenv("USE_OPENAI_TTS", "false").lower() == "true"
            logger.info(f"Synthesizing speech for: {text[:50]}... (use_openai={use_openai_tts})")
            
            # Stream TTS audio
            audio_stream = await voice_service.synthesize_speech(
                text=text,
                stream=True,
                use_openai=use_openai_tts
            )
            
            chunk_number = 0
            total_bytes = 0
            async for audio_chunk in audio_stream:
                chunk_number += 1
                total_bytes += len(audio_chunk)
                
                # Send audio chunk
                await self.send_message({
                    "type": "audio_chunk",
                    "sequence": sequence,
                    "chunk": chunk_number,
                    "data": base64.b64encode(audio_chunk).decode(),
                    "text": text if chunk_number == 1 else None
                })
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.02)  # Balanced for smooth streaming
            
            logger.info(f"Sent {chunk_number} audio chunks, total {total_bytes} bytes for sentence {sequence}")
                
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}", exc_info=True)
    
    async def save_messages(self, user_text: str, assistant_text: str) -> None:
        """Save messages to database"""
        try:
            # Save user message
            user_msg = Message(
                conversation_id=self.conversation_id,
                role="user",
                content=user_text
            )
            self.db.add(user_msg)
            
            # Save assistant message
            assistant_msg = Message(
                conversation_id=self.conversation_id,
                role="assistant",
                content=assistant_text,
                model=self.llm_service.default_model
            )
            self.db.add(assistant_msg)
            
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Error saving messages: {e}")
    
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
    Client → Server:
    {
        "type": "audio",
        "data": "base64_encoded_audio"
    }
    
    Server → Client:
    {
        "type": "transcription",
        "text": "what user said"
    }
    {
        "type": "text_chunk",
        "content": "partial response"
    }
    {
        "type": "audio_chunk",
        "sequence": 1,
        "chunk": 1,
        "data": "base64_encoded_audio",
        "text": "sentence being spoken"
    }
    {
        "type": "completion",
        "full_text": "complete response"
    }
    """
    await websocket.accept()
    session = StreamingVoiceSession(websocket, conversation_id, db)
    
    try:
        await session.initialize()
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            if data.get("type") == "audio":
                audio_data = data.get("data", "")
                is_final = data.get("final", True)
                format = data.get("format", "webm")
                mime_type = data.get("mimeType", "audio/webm")
                if audio_data:
                    # Process in background with format info
                    asyncio.create_task(session.process_audio(
                        audio_data, is_final, format, mime_type
                    ))
                    
            elif data.get("type") == "audio_chunk":
                # Handle streaming audio chunks
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
        except:
            pass
    finally:
        await db.close()