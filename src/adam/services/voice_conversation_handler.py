"""
Voice Conversation Handler

Manages voice conversation flow with intelligent timing, context awareness,
and natural turn-taking.
"""

import asyncio
import time
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum

from adam.services.voice_response_formatter import VoiceResponseFormatter, VoiceResponse, VOICE_SYSTEM_PROMPT, VOICE_FORMAT_PROMPT

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Voice conversation states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    WAITING_FOR_RESPONSE = "waiting_for_response"


@dataclass
class VoiceConversationContext:
    """Context for ongoing voice conversation"""
    conversation_id: str
    start_time: datetime
    last_interaction: datetime
    turn_count: int = 0
    is_voice_mode: bool = True
    current_topic: Optional[str] = None
    pending_actions: List[Dict] = field(default_factory=list)
    voice_history: List[Dict] = field(default_factory=list)
    state: ConversationState = ConversationState.IDLE

    def add_turn(self, user_text: str, assistant_response: VoiceResponse):
        """Add a conversation turn to history"""
        self.turn_count += 1
        self.last_interaction = datetime.now()
        self.voice_history.append({
            'turn': self.turn_count,
            'user': user_text,
            'assistant_spoken': assistant_response.spoken_text,
            'has_code': bool(assistant_response.code_blocks),
            'timestamp': self.last_interaction
        })


class VoiceConversationHandler:
    """Handles voice conversation flow and timing"""

    def __init__(self):
        self.formatter = VoiceResponseFormatter()
        self.contexts: Dict[str, VoiceConversationContext] = {}
        self.voice_detection_delay = 1.0
        self.max_response_length = 150

    def get_or_create_context(self, conversation_id: str) -> VoiceConversationContext:
        """Get or create a voice conversation context"""
        if conversation_id not in self.contexts:
            self.contexts[conversation_id] = VoiceConversationContext(
                conversation_id=conversation_id,
                start_time=datetime.now(),
                last_interaction=datetime.now()
            )
        return self.contexts[conversation_id]

    async def process_voice_input(
        self,
        transcribed_text: str,
        conversation_id: str,
        llm_callback: Callable,
        voice_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Process voice input with intelligent response handling

        Args:
            transcribed_text: The transcribed user speech
            conversation_id: Conversation ID
            llm_callback: Async function to get LLM response
            voice_callback: Optional callback for voice synthesis progress

        Returns:
            Dict with spoken response, visual content, and metadata
        """
        context = self.get_or_create_context(conversation_id)
        context.state = ConversationState.PROCESSING

        try:
            voice_enhanced_messages = self._prepare_voice_messages(
                transcribed_text,
                context
            )

            llm_response = await llm_callback(
                messages=voice_enhanced_messages,
                system_prompt=VOICE_SYSTEM_PROMPT
            )

            response_content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

            voice_response = self.formatter.format_response(
                response_content,
                context={'conversation_context': context}
            )

            context.add_turn(transcribed_text, voice_response)

            if voice_response.wait_for_response:
                context.state = ConversationState.WAITING_FOR_RESPONSE
            else:
                context.state = ConversationState.IDLE

            response = {
                'spoken_text': voice_response.spoken_text,
                'visual_content': voice_response.visual_content,
                'code_blocks': voice_response.code_blocks,
                'action_prompt': voice_response.action_prompt,
                'wait_for_response': voice_response.wait_for_response,
                'conversation_state': context.state.value,
                'turn_count': context.turn_count
            }

            if voice_callback:
                await voice_callback({
                    'type': 'response_ready',
                    'response': response
                })

            return response

        except Exception as e:
            logger.error(f"Error processing voice input: {e}")
            context.state = ConversationState.IDLE
            raise

    def _prepare_voice_messages(
        self,
        user_text: str,
        context: VoiceConversationContext
    ) -> List[Dict]:
        """Prepare messages with voice-specific context"""
        messages = []

        if context.voice_history:
            recent_history = context.voice_history[-3:]
            for turn in recent_history:
                messages.append({
                    'role': 'user',
                    'content': turn['user']
                })
                messages.append({
                    'role': 'assistant',
                    'content': turn['assistant_spoken']
                })

        messages.append({
            'role': 'user',
            'content': f"{user_text}\n\n{VOICE_FORMAT_PROMPT}"
        })

        return messages

    async def handle_voice_detection(
        self,
        audio_stream: asyncio.Queue,
        conversation_id: str,
        on_speech_end: Callable
    ):
        """
        Handle voice activity detection with proper timing

        Args:
            audio_stream: Queue of audio chunks
            conversation_id: Conversation ID
            on_speech_end: Callback when speech ends
        """
        context = self.get_or_create_context(conversation_id)
        silence_start = None

        while True:
            try:
                chunk = await asyncio.wait_for(
                    audio_stream.get(),
                    timeout=0.1
                )

                if chunk is None:
                    break

                is_speech = self._detect_speech_activity(chunk)

                if is_speech:
                    silence_start = None
                    context.state = ConversationState.LISTENING
                else:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= self.voice_detection_delay:
                        await on_speech_end()
                        silence_start = None

            except asyncio.TimeoutError:
                if context.state == ConversationState.LISTENING:
                    if silence_start and time.time() - silence_start >= self.voice_detection_delay:
                        await on_speech_end()
                        silence_start = None

    def _detect_speech_activity(self, audio_chunk: bytes) -> bool:
        """
        Simple voice activity detection

        In production, use WebRTC VAD or similar
        """
        return len(audio_chunk) > 100

    def should_interrupt(self, conversation_id: str) -> bool:
        """Check if the assistant should be interrupted"""
        context = self.contexts.get(conversation_id)
        if not context:
            return False

        return context.state in [
            ConversationState.SPEAKING,
            ConversationState.WAITING_FOR_RESPONSE
        ]

    def get_conversation_summary(self, conversation_id: str) -> Optional[Dict]:
        """Get a summary of the voice conversation"""
        context = self.contexts.get(conversation_id)
        if not context:
            return None

        duration = (datetime.now() - context.start_time).total_seconds()

        return {
            'conversation_id': conversation_id,
            'duration_seconds': duration,
            'turn_count': context.turn_count,
            'current_state': context.state.value,
            'has_pending_actions': bool(context.pending_actions),
            'last_interaction': context.last_interaction.isoformat()
        }

    def end_conversation(self, conversation_id: str) -> Optional[Dict]:
        """End a voice conversation and return summary"""
        summary = self.get_conversation_summary(conversation_id)
        if conversation_id in self.contexts:
            del self.contexts[conversation_id]
        return summary


# Utility functions for voice interactions
def create_voice_intro() -> str:
    """Create a natural voice introduction"""
    intros = [
        "Hi! I'm Adam, your AI coding partner. How can I help you today?",
        "Hello! I'm Adam. What are we working on today?",
        "Hey there! Adam here. What coding challenge can I help you with?",
        "Hi! I'm ready to help. What's on your mind?",
    ]
    import random
    return random.choice(intros)


def create_voice_acknowledgment(action: str) -> str:
    """Create natural acknowledgments for actions"""
    acknowledgments = {
        'thinking': [
            "Let me think about that...",
            "Good question, let me consider this...",
            "Hmm, let me look into that...",
            "Interesting, give me a moment...",
        ],
        'searching': [
            "Let me search for that...",
            "I'll look that up for you...",
            "Let me find that information...",
            "Searching now...",
        ],
        'coding': [
            "Let me write that code for you...",
            "I'll create that solution...",
            "Let me implement that...",
            "Working on the code now...",
        ],
        'error': [
            "I see there's an issue. Let me help...",
            "Looks like we have an error to fix...",
            "I notice a problem. Let me solve it...",
            "There's an error here. I'll fix it...",
        ]
    }

    import random
    return random.choice(acknowledgments.get(action, ["Let me help with that..."]))
