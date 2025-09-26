# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ADAM (Analytics Data Assistant with Memory) is a sophisticated AI assistant system with persistent memory, intelligent model routing, and multimodal capabilities. The system has evolved over ~1.5 years from a simple chatbot to a comprehensive AI coworker platform.

## Core Architecture

The project consists of four main components that interact:

1. **Core ADAM System** (`/src/adam/`): Python-based memory system with ChromaDB vector storage, LLM integration (Grok/OpenAI), and intelligent query routing
2. **ADAM v2 Backend** (`/src/adam_v2/`): FastAPI server providing RESTful APIs, project isolation, and enhanced services
3. **Frontend Applications**: Multiple interfaces including React app (`/frontend/AdamChat/`), Streamlit web UI (`/web/`), and CLI tools (`/cli/`)
4. **VSCode Extension** (`/vscode-extension/adam-code/`): TypeScript extension for IDE integration (currently in development)

### Key Architectural Patterns

- **Memory System**: Dual-storage using SQLite for metadata and ChromaDB for vector embeddings with BM25 + semantic search
- **Model Routing**: Automatic selection between grok-4-reasoning (complex), grok-4 (standard), and grok-3-mini-high (simple) based on query complexity
- **Project Isolation**: Each project has its own memory space and conversation history
- **Cost Optimization**: Smart routing reduces API costs by 63-89% through query analysis and caching

## Essential Commands

### Backend Development

```bash
# Setup and run main backend (from root)
pip install -e .
pip install -r requirements.txt
python -m adam_v2.main  # FastAPI server on http://localhost:8000

# Alternative ways to run backend
cd src/adam_v2 && python main.py
uvicorn adam_v2.main:app --reload --host 0.0.0.0 --port 8000

# Run CLI interfaces
python cli/adam_chat.py        # Simple chat
python cli/adam_complete.py    # Full interface with transparency
streamlit run web/adam_web.py  # Web UI (MUST use streamlit run)
```

### Frontend Development (React)

```bash
cd frontend/AdamChat
npm install
npm run dev         # Development server on http://localhost:5173
npm run build       # Production build
npm run preview     # Preview production build
npm run dev-server  # Run Express backend (if needed)
```

### VSCode Extension Development

```bash
cd vscode-extension/adam-code
npm install
npm run compile     # Compile TypeScript
npm run watch       # Watch mode for development

# Testing the extension
# 1. Open VSCode
# 2. Press F5 to launch Extension Development Host
# 3. Use Cmd+Shift+A to open ADAM chat
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=adam --cov-report=html

# Run specific test categories
pytest -m "not integration"  # Skip integration tests
pytest tests/unit/           # Unit tests only

# ADAM v2 specific tests
cd src/adam_v2
python run_tests.py
```

## Environment Configuration

Create `.env` file in project root with:

```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here        # For GPT models and Whisper
XAI_API_KEY=your_xai_api_key_here              # For Grok models
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here    # For text-to-speech

# Optional: Model Preferences
USE_GROK=true                  # Enable Grok models
PREFER_GROK=true              # Prefer Grok over OpenAI
USE_OPENAI_TTS=false          # Use OpenAI TTS instead of ElevenLabs

# Memory Configuration
MEMORY_PATH=./data/adam_memory
PROJECT_PATH=./data/adam_projects
CONVERSATION_PATH=./conversations
```

## Database Setup

The system automatically creates necessary databases on first run:

- **SQLite** (`adam_v2.db`): Projects, conversations, messages metadata
- **ChromaDB** (`/data/adam_memory/`): Vector storage for semantic search

No manual migration needed - databases are created automatically with proper schemas.

## API Endpoints

Main backend runs on `http://localhost:8000` with:
- `/api/docs` - Interactive API documentation
- `/api/projects` - Project management
- `/api/conversations` - Conversation handling
- `/api/messages` - Message processing with memory
- `/api/voice` - Voice processing endpoints

## Critical Implementation Details

### Memory System Flow
1. User message → Query analyzer determines complexity
2. Model router selects appropriate LLM
3. Memory search retrieves relevant context
4. LLM generates response with context
5. Important exchanges saved to memory
6. Memory connections updated in graph

### VSCode Extension Integration
- Connects to backend via HTTP (WebSocket planned but not implemented)
- Project ID hardcoded: `3a859e97-16fd-46c6-b018-1ede9fade704`
- Backend must be running on `localhost:8000`

### Voice Processing
- STT: OpenAI Whisper API
- TTS: ElevenLabs with voice ID `ZthjuvLPty3kTMaNKVKb`
- Audio format: MP3 at 44.1kHz

## Common Development Tasks

### Adding New LLM Provider
1. Update `/src/adam/llm/config.py` with provider config
2. Implement client in `/src/adam/llm/client.py`
3. Add routing logic to query analyzer
4. Update cost calculations in pricing manager

### Modifying Memory Behavior
- Core memory: `/src/adam/memory.py`
- Project isolation: `/src/adam/project_aware_memory.py`
- Search algorithms: `/src/adam/advanced_rag.py`
- Memory lifecycle: `/src/adam/memory_lifecycle.py`

### Debugging Tips
- Enable debug logging: `export LOG_LEVEL=DEBUG`
- Check backend logs: `backend.log`
- Memory stats: `python check_memory.py`
- Cost analysis: `python examples/cost_analysis_demo.py`

## Project State Notes

- VSCode extension is in active development - core features work but needs testing
- WebSocket support planned but not implemented - using HTTP polling
- Frontend uses React with Vite, not Next.js
- Multiple abandoned UI attempts exist in archive folders
- Test data in `adam_v2/data/` is 154MB and can be deleted

## Key Files for Understanding Architecture

- `/src/adam/__init__.py` - Core system exports and version
- `/src/adam/llm/client.py` - Unified LLM client implementation
- `/src/adam_v2/main.py` - FastAPI application setup
- `/vscode-extension/adam-code/src/extension.ts` - VSCode extension entry
- `/frontend/AdamChat/src/App.tsx` - React app main component