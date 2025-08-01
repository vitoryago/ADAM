# ADAM Voice Conversation Guide

## Overview

ADAM's Voice Conversation system provides intelligent, natural voice interactions with context-aware response formatting. Unlike traditional voice assistants that read everything aloud, ADAM knows what to speak versus what to display visually.

## Key Features

### 1. **Intelligent Response Filtering**
- **Spoken responses** are concise and natural (2-3 sentences max)
- **Code snippets** are displayed visually, not read aloud
- **File paths** are simplified to just the filename when spoken
- **URLs** are described, not read verbatim
- **Long explanations** are broken into conversational chunks

### 2. **Natural Conversation Flow**
- **Auto-stop recording** after 1 second of silence
- **Turn-taking awareness** - ADAM knows when to wait for your response
- **Context preservation** across multiple turns
- **Interruption handling** - you can interrupt ADAM while speaking

### 3. **Voice-Aware Prompting**
The system uses specialized prompts for voice mode:
```python
VOICE_SYSTEM_PROMPT = """You are ADAM, an AI voice assistant and coding partner. 
You're having a natural voice conversation.

VOICE INTERACTION RULES:
1. Keep spoken responses concise and natural - aim for 2-3 sentences max
2. NEVER read code verbatim - instead describe what the code does
3. When presenting code, say something like "I've prepared a code snippet for you"
4. Use natural pauses with commas and periods
5. Ask one question at a time and wait for responses
"""
```

## Architecture

### Components

1. **Voice Response Formatter** (`voice_response_formatter.py`)
   - Processes LLM responses for voice output
   - Extracts code blocks and visual content
   - Formats text for natural speech

2. **Voice Conversation Handler** (`voice_conversation_handler.py`)
   - Manages conversation state and flow
   - Handles turn-taking and timing
   - Maintains conversation context

3. **Voice Endpoints** (`routers/voice.py`)
   - `/voice-chat` - Complete voice interaction endpoint
   - `/transcribe` - Speech-to-text
   - `/synthesize` - Text-to-speech
   - `/ws/voice-stream` - Real-time streaming

4. **Frontend Components**
   - `VoiceConversation` - Main voice UI component
   - `AudioPlayer` - Handles audio playback with controls
   - `VoiceInput` - Microphone recording interface

### Voice Processing Pipeline

```
User Speech → Microphone → WebM Audio → STT (Whisper) → 
→ Voice Handler → LLM with Voice Prompts → Response Formatter →
→ Spoken Text + Visual Content → TTS (ElevenLabs) → Audio Output
```

## Response Examples

### Example 1: Code Debugging
**User**: "I'm getting a TypeError in my React component when I try to map over users"

**ADAM speaks**: "I see the issue. The error happens because users might be undefined when the component first renders. I've prepared a fix for you."

**ADAM displays**:
```javascript
// Add a default value or check
const users = props.users || [];

// Or use optional chaining
{users?.map(user => (
  <UserCard key={user.id} user={user} />
))}
```

### Example 2: Explaining Concepts
**User**: "How does async await work in JavaScript?"

**ADAM speaks**: "Async await makes asynchronous code look synchronous. When you mark a function as async, it always returns a promise. The await keyword pauses execution until that promise resolves. Should I show you an example?"

**ADAM displays** (if user says yes):
```javascript
// Traditional promises
fetchUser()
  .then(user => fetchPosts(user.id))
  .then(posts => console.log(posts))
  .catch(error => console.error(error));

// With async/await
async function getUserPosts() {
  try {
    const user = await fetchUser();
    const posts = await fetchPosts(user.id);
    console.log(posts);
  } catch (error) {
    console.error(error);
  }
}
```

## Configuration

### Voice Settings
```python
# ElevenLabs voice configuration
DEFAULT_VOICE_ID = "ZthjuvLPty3kTMaNKVKb"  # Your custom voice
VOICE_SETTINGS = {
    "stability": 0.75,
    "similarity_boost": 0.85,
    "style": 0.5,
    "use_speaker_boost": True
}
```

### Timing Configuration
```python
# Voice activity detection
SILENCE_THRESHOLD = 1.0  # seconds
MAX_RECORDING_TIME = 60  # seconds
AUDIO_CHUNK_SIZE = 1024

# Response timing
RESPONSE_DELAY = 0.5  # seconds before responding
INTERRUPTION_THRESHOLD = 0.3  # seconds
```

## Usage Tips

### For Natural Conversations
1. **Speak naturally** - no need for keywords or specific phrases
2. **Pause when done** - ADAM will detect when you've finished
3. **Ask follow-ups** - ADAM remembers the conversation context
4. **Interrupt if needed** - Just start speaking while ADAM is talking

### For Technical Discussions
1. **Describe the problem** first, then ask for code
2. **Use phrases like**:
   - "Show me how to..."
   - "Can you write code for..."
   - "What's wrong with this approach..."
3. **Reference previous code** - "Can you modify that function to..."

### Voice Commands That Work Well
- ✅ "Help me debug this error"
- ✅ "Explain how React hooks work"
- ✅ "Search for Python async best practices"
- ✅ "What's the difference between let and const?"
- ✅ "Can you optimize this database query?"

### What to Avoid
- ❌ Reading code line by line
- ❌ Spelling out variable names
- ❌ Long, unstructured monologues
- ❌ Multiple questions without pausing

## Advanced Features

### Multi-Modal Responses
ADAM can provide different content for voice and display:
```python
voice_response = VoiceResponse(
    spoken_text="I found the issue and prepared a fix",
    visual_content=full_explanation_with_code,
    code_blocks=[...],
    action_prompt="Should I explain how this fix works?"
)
```

### Context-Aware Formatting
The formatter adapts based on content type:
- **Errors**: Spoken summary + visual stack trace
- **API docs**: Spoken overview + visual details
- **Code reviews**: Spoken findings + visual annotations

### Voice Personality
ADAM adapts its voice style based on context:
- **Debugging**: Focused and analytical
- **Learning**: Patient and explanatory  
- **Brainstorming**: Creative and encouraging
- **Code review**: Constructive and detailed

## Troubleshooting

### Common Issues

1. **Microphone not working**
   - Check browser permissions
   - Ensure HTTPS connection
   - Test with browser's audio settings

2. **Voice not synthesizing**
   - Verify ElevenLabs API key
   - Check voice ID configuration
   - Monitor API rate limits

3. **Poor transcription**
   - Reduce background noise
   - Speak clearly and at normal pace
   - Check microphone quality

### Debug Mode
Enable debug logging for troubleshooting:
```python
# In voice_conversation_handler.py
logger.setLevel(logging.DEBUG)

# Frontend console
localStorage.setItem('debug:voice', 'true')
```

## Future Enhancements

### Planned Features
1. **Wake word detection** ("Hey ADAM")
2. **Multi-language support**
3. **Voice cloning** for personalized responses
4. **Emotion detection** for empathetic responses
5. **Background listening** mode

### Integration Ideas
1. **IDE voice commands**
2. **Voice-driven code refactoring**
3. **Pair programming mode**
4. **Voice annotations for code review**
5. **Meeting transcription and summary**

## Best Practices

### For Developers
1. Test with various accents and speaking speeds
2. Handle edge cases (silence, noise, interruptions)
3. Provide visual feedback for voice states
4. Cache common voice responses
5. Monitor voice API costs

### For Users
1. Find a quiet environment
2. Use a good quality microphone
3. Speak at a normal pace
4. Give ADAM time to process
5. Use the visual display for complex content

## API Reference

### Voice Chat Endpoint
```http
POST /api/voice/voice-chat
Content-Type: multipart/form-data

Parameters:
- audio: Audio file (WebM/MP3/WAV)
- conversation_id: String
- model: String (optional)
- use_search: Boolean (optional)
- voice_id: String (optional)

Response Headers:
- X-Response-Text: Spoken response
- X-Full-Response: Complete response
- X-Has-Code: Boolean
- X-Wait-For-Response: Boolean
```

### WebSocket Streaming
```javascript
const ws = new WebSocket('ws://localhost:8000/api/voice/ws/voice-stream');

// Send audio chunks
ws.send(JSON.stringify({
  type: 'audio',
  data: base64AudioChunk,
  format: 'webm'
}));

// Receive responses
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'transcription') {
    console.log('User said:', data.text);
  } else if (data.type === 'audio') {
    playAudio(data.data);
  }
};
```

## Conclusion

ADAM's Voice Conversation system represents a new paradigm in AI interaction - one that understands the difference between what should be spoken and what should be shown. By combining natural speech processing with intelligent response formatting, ADAM provides a truly conversational coding partner experience.

For the latest updates and examples, visit the [Voice Integration Guide](./VOICE_INTEGRATION.md).