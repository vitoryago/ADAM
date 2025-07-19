# ADAM Command-Line Interfaces

This directory contains the command-line interface tools for interacting with ADAM.

## Available CLIs

### adam_chat.py
The main chat interface for conversing with ADAM.
- Features intelligent model routing
- Memory system integration
- Conversation tracking
- Cost monitoring

**Usage:**
```bash
python cli/adam_chat.py
```

### adam_complete.py
Full-featured interface with complete transparency showing:
- Model selection reasoning
- Memory search results
- Cost tracking
- SQL analysis capabilities
- All internal operations

**Usage:**
```bash
python cli/adam_complete.py
# Run in quiet mode
python cli/adam_complete.py --quiet
# Run system test
python cli/adam_complete.py --test
```

## Requirements
- Python 3.8+
- All dependencies from requirements.txt
- API keys set in .env file (XAI_API_KEY and/or OPENAI_API_KEY)

## Tips
- Use adam_chat.py for regular conversations
- Use adam_complete.py when you need to see what ADAM is doing internally
- Both interfaces support multi-turn conversations and remember context