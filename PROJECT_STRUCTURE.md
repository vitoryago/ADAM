# ADAM Project Structure

This document describes the organized structure of the ADAM (Analytics Data Assistant with Memory) project.

## Directory Structure

```
ADAM/
├── src/adam/              # Core library code
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── core/              # Core abstractions and interfaces
│   ├── llm/               # LLM client implementations
│   ├── memory/            # Memory management modules
│   ├── tools/             # Utility tools (SQL, etc.)
│   ├── vision/            # Image processing capabilities
│   └── voice/             # Speech recognition and synthesis
│
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── artifacts/         # Test output files (PNGs, etc.)
│   ├── manual/            # Manual test scripts
│   └── test_documentation/# Test results documentation
│
├── examples/              # Example usage scripts
│   ├── agent_demo.py
│   ├── conversation_usage.py
│   └── ...
│
├── docs/                  # Documentation
│   ├── architecture/      # Architecture documentation
│   ├── costs/             # Cost analysis
│   ├── daily_logs/        # Development logs
│   ├── tutorials/         # User tutorials
│   └── ...
│
├── images/                # Project images and visualizations
│   ├── rag_comparison_heatmap.png
│   └── rag_method_overlap.png
│
├── notebooks/             # Jupyter notebooks
│   └── 01_getting_started.ipynb
│
├── scripts/               # Utility scripts
│
├── config/                # Configuration files
├── configs/               # Alternative config directory
│
├── knowledge/             # Knowledge base data
│   ├── business_context/
│   ├── languages/
│   ├── patterns/
│   └── schemas/
│
├── data/                  # Data directories
│   ├── conversations/
│   ├── knowledge/
│   └── memories/
│
├── archive/               # Archived/old files
│   ├── old_versions/      # Previous versions of scripts
│   ├── demo_conversations/# Old demo conversations
│   ├── documentation/     # Old documentation
│   ├── setup_scripts/     # Old setup scripts
│   └── temp_files/        # Temporary files
│
├── adam_complete.py       # Main entry point
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── README.md             # Project README
├── QUICKSTART.md         # Quick start guide
└── .gitignore            # Git ignore rules
```

## Key Components

### Core Library (`src/adam/`)
- **config.py**: Central configuration management
- **conversation_system.py**: Manages conversations and sessions
- **memory_network.py**: Graph-based memory storage
- **integrated_conversation_system.py**: Unified interface

### Memory Storage
- **adam_memory/**: Basic memory storage
- **adam_memory_advanced/**: Advanced memory with analytics
- **adam_complete_memory/**: Complete memory system data

### Tests (`tests/`)
- Comprehensive test coverage including unit and integration tests
- Test artifacts are stored in `tests/artifacts/`
- Manual test scripts in `tests/manual/`

### Documentation (`docs/`)
- Architecture guides
- Development roadmap
- Daily development logs
- Security guidelines

## Ignored Files

The following are excluded from version control:
- Virtual environments (`venv/`, `env/`)
- Python cache files (`__pycache__/`, `*.pyc`)
- Database files (`*.db`, `*.sqlite3`)
- Environment files (`.env`)
- Memory data directories
- IDE configurations (`.vscode/`, `.idea/`)

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables (see `.env.example`)
3. Run the main application: `python adam_complete.py`
4. See `docs/QUICKSTART.md` for detailed setup instructions