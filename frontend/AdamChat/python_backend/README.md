# ADAM Python Backend Integration

This directory contains the sophisticated Python backend for ADAM (Advanced Data Analytics Model) that integrates with the Node.js web application to provide advanced AI capabilities.

## 🚀 Quick Start

1. **Install Python Dependencies**:
   ```bash
   cd python_backend
   python setup.py --full  # Creates venv and installs all dependencies
   ```

2. **Configure Environment**:
   ```bash
   cp .env.template .env
   # Edit .env with your API keys (XAI_API_KEY, OPENAI_API_KEY, etc.)
   ```

3. **Test the Backend**:
   ```bash
   python setup.py --test  # Verify installation
   python main.py  # Test the service
   ```

4. **Start Web Application**:
   ```bash
   cd ..
   npm run dev  # The Node.js app will automatically connect to Python backend
   ```

## 🤖 New Coworker Features

ADAM now includes powerful coworker capabilities:

- **Project Isolation**: Each project has its own memory space and context
- **Screen Vision**: ADAM can see your screen for better assistance
- **Multi-model Intelligence**: Automatic routing between Grok-4, Grok-3-Mini, and O4-Mini
- **Cost Optimization**: Real-time budget monitoring and model selection

## 🏗️ Architecture Overview

The integration works through a bridge pattern where:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Web Browser   │    │   Node.js App    │    │  Python ADAM Core   │
│                 │◄──►│                  │◄──►│                     │
│ - React UI      │    │ - Express API    │    │ - Advanced RAG      │
│ - WebSockets    │    │ - WebSocket      │    │ - Memory Networks   │
│ - Project Mgmt  │    │ - Database       │    │ - LLM Integration   │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

### Key Integration Points

1. **Real-time Chat**: WebSocket messages route through Python ADAM for intelligent responses
2. **Project Memory**: Each project maintains context in both PostgreSQL and ADAM's memory network
3. **Cost Monitoring**: Real-time tracking of LLM usage and costs
4. **Multi-model Routing**: Automatic selection between Grok-3, O1, and Claude based on query complexity

## 📁 File Structure

```
python_backend/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── setup.py                           # Setup and installation script
├── main.py                            # Main entry point
├── test_integration.py               # Integration testing
├── .env.template                      # Environment configuration template
│
├── src/adam/                          # Core ADAM package
│   ├── __init__.py                    # Package initialization
│   ├── adam_service.py               # Main service bridge
│   │
│   ├── 🧠 Core Memory System:
│   ├── memory.py                      # Base memory system with ChromaDB
│   ├── memory_config.py               # Memory configuration
│   ├── memory_network.py              # Graph-based memory relationships
│   ├── memory_lifecycle.py            # Memory aging and cleanup
│   ├── advanced_rag.py               # BM25 + semantic search fusion
│   ├── memory_search_enhanced.py      # Enhanced memory retrieval
│   ├── temporal_memory_scoring.py     # Time-based memory scoring
│   │
│   ├── 🤖 LLM Integration:
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py                  # Unified LLM client (Grok, OpenAI)
│   │   ├── config.py                  # Model configurations and routing
│   │   └── query_analyzer.py          # Query complexity analysis
│   │
│   ├── 💬 Conversation Management:
│   ├── conversation_system.py         # Session management
│   ├── conversation_aware_memory.py   # Context-aware memory
│   ├── langgraph_conversation.py      # LangGraph state machine
│   ├── integrated_conversation_system.py # System integration
│   │
│   ├── 👥 Coworker Features:
│   ├── project_manager.py             # Project-based isolation
│   ├── project_aware_memory.py        # Project memory integration
│   ├── screen_capture.py              # Screen vision capabilities
│   │
│   ├── 💰 Cost & Performance:
│   ├── cost_monitor.py                # API cost tracking
│   ├── pricing_manager.py             # Model pricing management
│   ├── activity_tracker.py           # Usage statistics
│   ├── memory_compressor.py          # Memory compression
│   │
│   ├── ⚙️ Core Infrastructure:
│   ├── config.py                     # System configuration
│   └── errors.py                     # Error handling
└
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# API Keys - Get these from respective providers
OPENAI_API_KEY=sk-...                    # For O1 models
ANTHROPIC_API_KEY=sk-ant-...             # For Claude models  
XAI_API_KEY=xai-...                      # For Grok models

# Model Selection
DEFAULT_SIMPLE_MODEL=grok-3-mini-reasoning-high
DEFAULT_COMPLEX_MODEL=o1-mini-high
DEFAULT_CODING_MODEL=claude-opus-4

# Cost Management
DAILY_COST_LIMIT=1.00                    # USD per day
MONTHLY_COST_LIMIT=30.00                 # USD per month

# Memory Configuration
MEMORY_STORAGE_PATH=./adam_memory_advanced
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
MEMORY_CONFIDENCE_THRESHOLD=0.7

# Optional Features
ENABLE_VOICE=true                        # Text-to-speech
LOG_LEVEL=INFO
```

### Model Costs (as of 2024)

| Model | Input (per 1K tokens) | Output (per 1K tokens) | Use Case |
|-------|----------------------|------------------------|----------|
| Grok-3 Mini | $0.15 | $0.60 | Simple queries, explanations |
| O1 Mini | $15.00 | $60.00 | Complex reasoning, analysis |
| Claude Opus 4 | $15.00 | $75.00 | Code generation, debugging |

## 🧠 Advanced Features

### 1. Three-Stage Retrieval (RAG)

The system uses a sophisticated retrieval approach:

- **BM25**: Keyword-based search for exact term matching
- **Vector Search**: Semantic similarity using embeddings  
- **Graph Traversal**: Following memory network connections
- **Temporal Scoring**: Natural time-based relevance

### 2. Memory Networks

Memories are connected in a graph structure:

- **Memory Nodes**: Individual memories with metadata
- **Weighted Connections**: Stronger relationships between related memories
- **Conversation Threads**: Track evolving understanding
- **Memory Decay**: Unused memories naturally fade

### 3. Intelligent Model Routing

Queries are automatically routed to optimal models:

```python
# Simple query → Grok-3 Mini (cost-effective)
"What is machine learning?"

# Complex analysis → O1 Mini (reasoning)
"Analyze the performance bottlenecks in this distributed system"

# Code debugging → Claude Opus 4 (code specialist)
"Fix this Python error: TypeError in line 45"
```

### 4. Cost Optimization

- **Real-time monitoring**: Track costs as they occur
- **Budget limits**: Hard stops at daily/monthly limits
- **Model fallbacks**: Cheaper alternatives when budget is tight
- **Query analysis**: Route to most cost-effective model

## 🔌 Integration API

### Python Service → Node.js Communication

The Python service communicates via stdin/stdout JSON messages:

**Request Format**:
```json
{
  "requestId": "req_123",
  "type": "QUERY",
  "data": {
    "query": "How do I optimize this SQL query?",
    "conversationId": "conv_456",
    "projectId": "proj_789",
    "userId": "user_101",
    "context": {
      "previousMessages": [...],
      "projectMemory": "Previous SQL work...",
      "userPreferences": {}
    }
  }
}
```

**Response Format**:
```json
{
  "requestId": "req_123",
  "response": {
    "response": "To optimize your SQL query...",
    "cost": 0.02,
    "modelUsed": "claude-opus-4",
    "processingTime": 1.2,
    "memoryConfidence": 0.85,
    "sources": [...],
    "conversationState": {
      "complexity": "complex",
      "memoryFound": true,
      "shouldStore": true
    }
  }
}
```

## 🚦 Getting Started Checklist

- [ ] Install Python 3.8+ and pip
- [ ] Run `python setup.py --full` to create environment
- [ ] Copy `.env.template` to `.env` and add API keys
- [ ] Test with `python setup.py --test`
- [ ] Start Python service with `python adam_service.py`
- [ ] Start Node.js app with `npm run dev`
- [ ] Create a new project and test chat functionality

## 🔍 Troubleshooting

### Common Issues

1. **"Module not found" errors**:
   ```bash
   pip install -r requirements.txt
   ```

2. **API key errors**:
   - Ensure `.env` file exists with valid API keys
   - Check key permissions and quotas

3. **ChromaDB initialization issues**:
   ```bash
   pip install --upgrade chromadb
   ```

4. **Python service not responding**:
   - Check `adam_service.log` for errors
   - Ensure Python path is correct in Node.js integration

### Debug Mode

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
python adam_service.py
```

### Performance Tuning

For production use:
- Use `sentence-transformers/all-mpnet-base-v2` for better embeddings
- Enable memory compression for long-running sessions
- Set appropriate cost limits
- Monitor memory usage with activity tracker

## 📊 Monitoring

The system provides comprehensive monitoring:

1. **Cost Dashboard**: Real-time LLM usage costs
2. **Memory Analytics**: Memory network visualization
3. **Conversation Metrics**: Session and query statistics
4. **Performance Tracking**: Response times and model efficiency

Access monitoring data via the `/api/adam/cost-summary` endpoint.

## 🤝 Contributing

The Python backend is modular and extensible. Key extension points:

- **Custom LLM Integration**: Add new models in `integrated_conversation_system.py`
- **Memory Strategies**: Extend memory lifecycle in `memory_lifecycle.py`  
- **Retrieval Methods**: Add new RAG techniques in `advanced_rag.py`
- **Cost Providers**: Support new pricing APIs in `pricing_manager.py`

## 📜 License

This ADAM backend integrates with your existing web application and maintains the same licensing as the parent project.