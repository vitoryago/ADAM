# ADAM - Analytics Data Assistant with Memory

<div align="center">
  <h1>🧠 ADAM</h1>
  <p><strong>Your Personal AI Assistant with Perfect Memory</strong></p>
  <p>
    <a href="#features">Features</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#documentation">Documentation</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#contributing">Contributing</a>
  </p>
</div>

## Overview

ADAM (Analytics Data Assistant with Memory) is an intelligent AI assistant that remembers everything. Unlike traditional chatbots that forget your conversation after each session, ADAM builds a persistent memory network that grows stronger with each interaction.

### Key Capabilities

- **🧠 Perfect Memory**: Remembers all conversations and builds connections between related topics
- **💬 Multi-Model Support**: Intelligently routes queries to the most appropriate AI model (Ollama, OpenAI, Anthropic, X.AI)
- **💰 Cost Optimization**: Reduces API costs by 63-89% through smart routing and caching
- **🎙️ Voice Interface**: Natural voice conversations with speech recognition and synthesis
- **👁️ Vision Processing**: Analyzes images and screenshots for visual understanding
- **🔗 Memory Network**: Creates a neural-network-like graph of interconnected memories
- **⚡ Real-time Processing**: LangGraph state machine for intelligent decision-making

## Features

### Advanced Memory System
- **Conversation-Aware Memory**: Links conversations to memory storage seamlessly
- **Memory Decay**: Old, unused memories naturally fade like human memory
- **Reference Resolution**: Tracks relationships between memories with weighted connections
- **Semantic Search**: Find relevant memories using natural language queries

### Intelligent Query Routing
- **Grok-3-mini**: Simple queries and basic questions ($0.4/M tokens)
- **O3**: Complex analysis and reasoning ($5/M tokens)
- **Claude Opus 4**: Advanced coding and technical tasks ($45/M tokens)

### Cost Management
- Dynamic model selection based on query complexity
- Real-time API pricing updates
- Detailed cost tracking and reporting
- Smart caching to avoid redundant API calls

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ADAM.git
cd ADAM

# Run the setup script
./setup_adam.sh

# Or manually install
pip install -e .
```

### Basic Usage

```python
# Simple chat interface
python src/adam_v1_basic.py

# Advanced interface with full memory capabilities
python src/adam_v2_memory.py
```

### Programmatic Usage

```python
from adam import IntegratedConversationSystem

# Initialize ADAM
adam = IntegratedConversationSystem()

# Start a conversation
response = adam.process_message("Tell me about quantum computing")
print(response)

# ADAM remembers context
response = adam.process_message("How does it relate to cryptography?")
print(response)
```

## Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get up and running in minutes
- **[Architecture Overview](docs/architecture/)** - Deep dive into system design
- **[API Reference](docs/api/)** - Complete API documentation
- **[Security Guide](docs/SECURITY.md)** - Security best practices
- **[Daily Development Logs](docs/daily_logs/)** - Follow the evolution of ADAM

## Architecture

ADAM uses a sophisticated multi-layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│              (Voice / Text / Programmatic API)              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 LangGraph State Machine                      │
│         (Query Analysis & Intelligent Routing)               │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Conversation System + Memory Network            │
│      (Context Management & Memory Graph Operations)          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    Multi-Model Backend                       │
│        (Ollama / OpenAI / Anthropic / X.AI)                │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

- **`conversation_system.py`** - Manages all conversations and sessions
- **`memory_network.py`** - Graph-based memory storage and retrieval
- **`langgraph_conversation.py`** - State machine for intelligent processing
- **`pricing_manager.py`** - Real-time cost optimization
- **`integrated_conversation_system.py`** - Unified interface bringing everything together

## Project Structure

```
ADAM/
├── src/adam/           # Core library code
│   ├── core/          # Core abstractions and interfaces
│   ├── memory/        # Memory management modules
│   ├── tools/         # Utility tools and helpers
│   ├── vision/        # Image processing capabilities
│   └── voice/         # Speech recognition and synthesis
├── adam_memory_advanced/  # Persistent storage
├── examples/          # Example usage scripts
├── tests/            # Comprehensive test suite
├── docs/             # Documentation
└── notebooks/        # Jupyter notebooks for experimentation
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=adam tests/
```

### Key Technologies

- **LangChain & LangGraph** - LLM orchestration and state management
- **ChromaDB** - Vector storage for semantic search
- **NetworkX** - Memory graph operations
- **OpenAI Whisper** - Speech recognition
- **Sentence Transformers** - Text embeddings

## Performance

Based on real-world usage:
- **Cost Reduction**: 63-89% compared to using high-end models exclusively
- **Response Time**: <2s for simple queries, <5s for complex analysis
- **Memory Retrieval**: Sub-second semantic search across thousands of memories
- **Accuracy**: 95%+ relevance in memory retrieval

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with ❤️ using LangChain and LangGraph
- Inspired by the human memory system
- Special thanks to the open-source AI community

---

<div align="center">
  <p><strong>ADAM - Where Every Conversation Matters</strong></p>
  <p>🧠 Remember Everything • 💡 Understand Context • 🚀 Evolve Continuously</p>
</div>