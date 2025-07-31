# ADAM Voice Integration Guide

## Overview

This guide explains how to integrate voice capabilities into ADAM, enabling users to speak to ADAM and receive spoken responses.

## Architecture

```
User Speech → Microphone → STT (Whisper) → Text → ADAM (Grok/OpenAI) → Response Text → TTS (ElevenLabs) → Speaker
```

## Setup Instructions

### 1. Environment Variables

Add the following to your `.env` file:

```bash
# For Speech-to-Text (Whisper)
OPENAI_API_KEY=your_openai_key_here

# For Text-to-Speech (ElevenLabs)
ELEVENLABS_API_KEY=your_elevenlabs_key_here

# Optional: Default voice ID
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM  # Rachel voice
```

### 2. Install Dependencies

```bash
# Backend
pip install openai httpx

# Frontend (already included in package.json)
```

### 3. Register Voice Routes

Update `src/adam_v2/main.py`:

```python
from routers import voice

# Add to router registration
app.include_router(voice.router, prefix="/api/voice", tags=["voice"])
```

### 4. Update Message Input

In `message-input.tsx`, add the voice input button:

```tsx
import { VoiceInput } from "./voice-input";

// In the component
<VoiceInput 
  onTranscription={(text) => {
    // Set the input value
    setValue(text);
  }}
  disabled={disabled}
/>
```

### 5. Add Voice Playback to Messages

In `message-bubble.tsx`, add voice player for assistant messages:

```tsx
import { VoicePlayer } from "./voice-player";

// For assistant messages
{message.role === 'assistant' && (
  <VoicePlayer 
    text={message.content}
    autoPlay={false}
  />
)}
```

## Voice Models Comparison

Based on our research, here are the recommended models:

### Speech-to-Text (STT)
1. **OpenAI Whisper** (Recommended)
   - High accuracy
   - Multi-language support
   - Good with accents
   - Cost: ~$0.006/minute

2. **Alternative: Local Whisper**
   - Use `whisper` Python package
   - No API costs
   - Requires local GPU

### Text-to-Speech (TTS)
1. **ElevenLabs** (Recommended for quality)
   - Most realistic voices
   - Voice cloning capability
   - Cost: ~$0.18/1000 chars (Pro)
   
2. **XTTS-v2** (Recommended for open-source)
   - Free to run
   - Good quality
   - Requires local GPU

3. **Azure Neural TTS** (Enterprise)
   - Good for scale
   - Many languages
   - Cost: ~$16/1M chars

## Usage Flow

1. **Voice Input**:
   - User clicks microphone button
   - Records audio (WebM format)
   - Sends to `/api/voice/transcribe`
   - Receives text transcription
   - Text appears in message input

2. **Processing**:
   - Text sent to ADAM chat endpoint
   - Grok/OpenAI processes the query
   - Returns text response

3. **Voice Output**:
   - Response text sent to `/api/voice/synthesize`
   - ElevenLabs generates audio
   - Audio played to user

## Advanced Features

### Voice Selection
```typescript
// Get available voices
const voices = await fetch('/api/voice/voices');

// Use specific voice
<VoicePlayer 
  text={message.content}
  voiceId="21m00Tcm4TlvDq8ikWAM"  // Rachel
/>
```

### Streaming TTS
For long responses, use streaming:

```typescript
const response = await fetch('/api/voice/synthesize', {
  method: 'POST',
  body: JSON.stringify({
    text: longText,
    stream: true
  })
});

// Handle streaming audio
```

### Voice Settings
Configure in `VoiceConfig`:
- `speaking_rate`: Speed (0.5-2.0)
- `pitch`: Voice pitch (-2.0 to 2.0)
- `language`: Language code

## Cost Optimization

1. **Cache TTS responses** for repeated content
2. **Use shorter system prompts** for voice interactions
3. **Implement client-side VAD** (Voice Activity Detection)
4. **Consider local models** for high-volume use

## Security Considerations

1. **Validate audio file size** (max 10MB recommended)
2. **Rate limit voice endpoints**
3. **Sanitize transcribed text** before processing
4. **Use secure WebSocket** for real-time voice

## Next Steps

1. Implement voice activity detection
2. Add wake word detection ("Hey ADAM")
3. Support multiple TTS providers
4. Add voice conversation mode
5. Implement local model fallbacks