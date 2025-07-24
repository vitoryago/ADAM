# ADAM Quick Start Guide

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/adam.git
   cd adam
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

## Running ADAM

### Basic Chat Mode (v1)
```bash
python src/adam_v1_basic.py
```

### Advanced Mode with Memory (v2)
```bash
python src/adam_v2_memory.py
```

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run specific test
pytest tests/test_conversation_system.py -v
```

## System Components

### 1. Conversation System
Tracks all conversations, sessions, and exchanges:
```python
from src.adam import ConversationSystem

conv_system = ConversationSystem()
session_id = conv_system.start_session(title="My Project")
```

### 2. Memory Network
Creates a graph of interconnected memories:
```python
from src.adam import MemoryNetworkSystem

memory_network = MemoryNetworkSystem(base_memory, conv_system)
memory_id = memory_network.add_memory_with_references(
    query="How to optimize SQL?",
    response="Use indexes...",
    memory_type="explanation",
    topics=["SQL", "optimization"]
)
```

### 3. Conversation-Aware Memory
Bridges conversations with memory storage:
```python
from src.adam import ConversationAwareMemorySystem

cam_system = ConversationAwareMemorySystem(base_memory)
exchange_id, memory_id = cam_system.process_interaction(
    query="My question",
    response="ADAM's answer",
    topics=["topic1", "topic2"],
    generation_cost=0.01,
    model_used="gpt-4"
)
```

## Common Use Cases

### 1. Continue Previous Conversation
```python
# Continue discussing a topic from yesterday
recap, memory_ids = cam_system.continue_conversation("dbt optimization")
```

### 2. Search Past Knowledge
```python
# Search both conversations and memories
results = cam_system.search_conversations_and_memories(
    "timeout errors",
    lookback_days=30
)
```

### 3. Visualize Memory Network
```python
# Create a visual map of memories about SQL
fig = memory_network.visualize_memory_network(topic="SQL")
fig.savefig("sql_memories.png")
```

### 4. Get Analytics
```python
# Get usage analytics for the last month
analytics = cam_system.get_analytics(days=30)
print(f"Total conversations: {analytics['conversations']['total_sessions']}")
print(f"Memories created: {analytics['conversations']['total_memories']}")
```

## Directory Structure
```
adam/
├── src/
│   ├── adam/                    # Core modules
│   │   ├── __init__.py
│   │   ├── conversation_system.py
│   │   ├── memory_network.py
│   │   └── conversation_aware_memory.py
│   ├── adam_v1_basic.py        # Basic chat interface
│   └── adam_v2_memory.py        # Advanced interface with memory
├── tests/                       # Test files
├── examples/                    # Usage examples
├── docs/                        # Documentation
└── adam_memory_advanced/        # Data storage
    ├── conversations/           # Conversation sessions
    └── memory_network/          # Memory graph data
```

## Troubleshooting

### Issue: ModuleNotFoundError
```bash
# Make sure you're in the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Issue: Ollama not running
```bash
# Start Ollama service
ollama serve

# Pull required model
ollama pull mistral
```

### Issue: Memory decay removing important memories
```python
# Adjust decay threshold
memory_network.DECAY_THRESHOLD_DAYS = 60  # Start decay after 60 days
memory_network.DECAY_RATE = 0.05  # Slower decay rate
```

## Next Steps

1. Check out the `examples/` directory for more usage patterns
2. Read the daily logs in `docs/daily_logs/` to understand the development journey
3. Explore the test files to see how components work together
4. Start building your own integrations!

For more detailed documentation, see the main README.md file.