# ADAM v2.0 - Project-Based AI Assistant

A modern, project-based AI assistant with isolated memory spaces, built with FastAPI, HTMX, and advanced RAG capabilities.

## Features

- 🗂️ **Project Organization** - Keep conversations organized by project
- 🧠 **Isolated Memory** - Each project has its own ChromaDB memory space
- 💬 **Multiple Conversations** - Manage multiple conversation threads per project
- 🔄 **Real-time Streaming** - SSE-based streaming responses
- 🎯 **Smart Model Routing** - Automatic model selection based on query complexity
- 🔍 **Advanced RAG** - BM25 + Semantic search fusion for better memory recall
- 💰 **Cost Tracking** - Track costs per message, conversation, and project
- 🖼️ **Image Support** - Upload images for vision model analysis

## Quick Start

### Prerequisites

- Python 3.8+
- API keys for LLM providers (OpenAI, xAI)

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the application:
```bash
python main.py
```

4. Open your browser to http://localhost:8000

## Usage

### Creating a Project

1. Click "New Project" on the home page
2. Enter project name and optional description
3. Select default model (or use automatic routing)

### Starting a Conversation

1. Click on a project to open it
2. Click "New Conversation"
3. Start chatting with ADAM!

### Memory System

- **Automatic Storage**: Valuable responses are automatically stored
- **Memory Search**: Use the "Browse Memories" button to search project memories
- **Memory Toggle**: Enable/disable memory usage per message

### Keyboard Shortcuts

- `Cmd/Ctrl + K`: Quick search (coming soon)
- `Cmd/Ctrl + N`: New project
- `Enter`: Send message
- `Shift + Enter`: New line in message

## API Documentation

Once running, visit http://localhost:8000/docs for interactive API documentation.

## Testing

Run the test suite:
```bash
python run_tests.py
```

Or run specific tests:
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests  
pytest tests/integration/ -v

# Memory tests
pytest tests/unit/test_memory_service.py -v
pytest tests/unit/test_advanced_memory.py -v
```

## Architecture

- **FastAPI** - Modern async web framework
- **HTMX** - Dynamic UI without complex JavaScript
- **SQLAlchemy** - Async ORM for database
- **ChromaDB** - Vector database for memories
- **Tailwind CSS** - Utility-first styling

## Example Usage

### Test Messaging
```bash
python examples/test_messaging.py
```

### Test Memory
```bash
python examples/test_memory.py
```

## Configuration

### Environment Variables

- `DATABASE_URL`: SQLite database path (default: `sqlite+aiosqlite:///./data/adam_v2.db`)
- `OPENAI_API_KEY`: OpenAI API key
- `XAI_API_KEY`: xAI (Grok) API key
- `DEFAULT_MODEL`: Default LLM model
- `ADAM_V2_MEMORY_PATH`: ChromaDB storage path

### Project Settings

Each project can have custom settings:
- `model`: Default model for the project
- `temperature`: LLM temperature (0-1)
- `max_tokens`: Maximum response tokens

## Development

### Project Structure
```
adam_v2/
├── main.py              # FastAPI application
├── models.py            # SQLAlchemy models
├── database.py          # Database configuration
├── routers/            # API endpoints
│   ├── projects.py
│   ├── conversations.py
│   ├── messages.py
│   ├── memories.py
│   └── ui.py
├── services/           # Business logic
│   ├── llm_service.py
│   ├── memory_service.py
│   └── advanced_memory_service.py
├── templates/          # HTML templates
│   ├── base.html
│   ├── index.html
│   └── conversation.html
├── static/            # Static files
└── tests/             # Test suite
```

## Troubleshooting

### Memory System Not Available
- Ensure ChromaDB is installed: `pip install chromadb`
- Check that ADAM v1 modules are accessible

### No LLM Responses
- Verify API keys in `.env` file
- Check console for error messages

### Database Issues
- Delete `data/adam_v2.db` to reset
- Run `python main.py` to recreate

## License

This project is part of the ADAM ecosystem.