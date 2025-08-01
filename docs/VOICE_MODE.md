# ADAM Voice Mode Configuration

## Overview

ADAM supports real-time voice conversations with:
- Speech-to-Text using OpenAI Whisper
- Text-to-Speech using ElevenLabs or OpenAI
- Voice Activity Detection with automatic silence detection
- Streaming transcription and responses

## Setup

### 1. Install FFmpeg (Required for audio processing)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### 2. Configure Voice Providers

Add to your `.env` file:

```bash
# Required for Speech-to-Text
OPENAI_API_KEY=your-openai-api-key

# For ElevenLabs TTS (optional)
ELEVENLABS_API_KEY=your-elevenlabs-api-key

# To use OpenAI TTS instead of ElevenLabs
USE_OPENAI_TTS=true
```

### 3. Voice Provider Options

#### ElevenLabs (Default)
- High-quality, natural-sounding voices
- Multiple voice options
- Supports emotion and style control
- Requires separate API key

#### OpenAI TTS
- Integrated with OpenAI API
- Good quality voices
- No additional API key needed
- Set `USE_OPENAI_TTS=true` to enable

## Usage

1. Click the microphone button in the chat interface
2. Speak naturally - ADAM will detect when you stop talking (1 second of silence)
3. ADAM will transcribe your speech and respond with voice
4. The conversation is automatically saved to the chat history

## Troubleshooting

### "Invalid file format" error
- Ensure FFmpeg is installed: `ffmpeg -version`
- Try using OpenAI TTS: Set `USE_OPENAI_TTS=true`
- Check browser compatibility (Chrome/Edge recommended)

### No audio playback
- Check browser permissions for microphone access
- Ensure speakers/headphones are connected
- Try refreshing the page

### Voice cuts off or stutters
- Check your internet connection
- Try reducing concurrent requests
- Consider using OpenAI TTS for better stability

## Browser Compatibility

Best support:
- Chrome/Chromium (recommended)
- Microsoft Edge
- Safari (macOS)

Limited support:
- Firefox (may have WebM codec issues)

## Advanced Configuration

### Custom Voice Settings

For ElevenLabs in `voice_service.py`:
```python
VoiceConfig(
    stability=0.75,          # Voice consistency (0-1)
    similarity_boost=0.75,   # Voice clarity (0-1)
    style=0.0,              # Style intensity (0-1)
    use_speaker_boost=True   # Enhanced clarity
)
```

### OpenAI Voice Options
Available voices: alloy, echo, fable, onyx, nova, shimmer

To change the default OpenAI voice, modify `voice_service.py`:
```python
voice = voice_id or "nova"  # Change "nova" to your preferred voice
```