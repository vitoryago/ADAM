# ADAM Complete File Structure

## Overview
This document contains the complete file structure of the ADAM (Advanced Data Analytics Model) project, including all files, their purposes, and whether they are essential or optional.

Generated on: 2025-07-29

Total files (excluding venv, __pycache__, .git): ~294 files

## Legend
- 🔴 **ESSENTIAL** - Core functionality, cannot run without
- 🟡 **RECOMMENDED** - Enhanced features, strongly suggested
- 🟢 **OPTIONAL** - Nice to have, examples, docs
- ⚪ **UNNECESSARY** - Can be safely removed

---

## Root Directory Structure

```
/Users/vitoryago/ADAM/
├── 🔴 .env                                 # API keys (REQUIRED)
├── 🔴 .env.example                         # Template for API keys
├── 🔴 setup.py                             # Package installation
├── 🔴 requirements.txt                     # Core dependencies
├── 🟡 requirements_web.txt                 # Web UI dependencies
├── 🟡 requirements_coworker.txt           # Screen capture dependencies
├── 🟢 README.md                            # Project documentation
├── 🟢 LICENSE                              # License file
├── 🟢 .gitignore                           # Git ignore rules
├── ⚪ .claude/                             # Claude editor settings
│   └── ⚪ settings.local.json
├── ⚪ adam-frontend/                       # DELETED - React frontend
└── ⚪ how_to_learn_more.md                # Learning resources
```

## Core ADAM Source (`/src/adam/`)

```
/Users/vitoryago/ADAM/src/adam/
├── 🔴 __init__.py                          # Package initialization, exports
├── 🔴 memory.py                            # Core memory system with ChromaDB
├── 🔴 memory_config.py                     # Memory configuration
├── 🔴 llm/                                 # LLM integration
│   ├── 🔴 __init__.py
│   ├── 🔴 client.py                        # Unified LLM client (Grok, OpenAI)
│   ├── 🔴 config.py                        # Model configurations
│   └── 🔴 query_analyzer.py               # Query complexity analysis
├── 🔴 conversation_system.py               # Session management
├── 🔴 errors.py                            # Error handling
├── 🔴 config.py                            # System configuration
├── 🟡 project_manager.py                   # Project-based isolation (NEW)
├── 🟡 project_aware_memory.py              # Project memory integration (NEW)
├── 🟡 screen_capture.py                    # Screen vision capabilities (NEW)
├── 🟡 advanced_rag.py                      # BM25 + semantic search
├── 🟡 memory_network.py                    # Graph-based memory relationships
├── 🟡 conversation_aware_memory.py         # Context-aware memory
├── 🟡 langgraph_conversation.py            # LangGraph state machine
├── 🟡 integrated_conversation_system.py    # System integration
├── 🟡 memory_lifecycle.py                  # Memory aging/cleanup
├── 🟡 memory_search_enhanced.py            # Enhanced retrieval
├── 🟡 temporal_memory_scoring.py           # Time-based scoring
├── 🟢 cost_monitor.py                      # API cost tracking
├── 🟢 pricing_manager.py                   # Model pricing
├── 🟢 activity_tracker.py                  # Usage statistics
├── 🟢 memory_compressor.py                 # Memory compression
└── 🟢 tools/                               # Utility tools
    ├── 🟢 __init__.py
    └── 🟢 sql_tools.py                     # SQL utilities
```

## ADAM V2 (`/src/adam_v2/`)

```
/Users/vitoryago/ADAM/src/adam_v2/
├── 🟡 services/                            # Enhanced services
│   ├── 🟡 memory_service.py               # Project-based memory
│   ├── 🟡 llm_service.py                  # Streaming LLM service
│   └── 🟡 advanced_memory_service.py      # BM25 + evaluation
├── 🟡 models.py                            # Database models
├── ⚪ main.py                              # FastAPI app (not needed)
├── ⚪ routers/                             # API endpoints (not needed)
│   ├── ⚪ projects.py
│   ├── ⚪ conversations.py
│   ├── ⚪ messages.py
│   ├── ⚪ memories.py
│   └── ⚪ ui.py
├── ⚪ templates/                           # DELETED - HTMX templates
├── ⚪ static/                              # Static files
├── ⚪ data/                                # Test data (154MB - SKIP)
│   └── ⚪ adam_v2_memory/                  # ChromaDB test data
├── ⚪ tests/                               # Test files
├── ⚪ examples/                            # V2 examples
├── ⚪ backup_ui_files/                     # Old UI attempts
├── 🟢 README.md                            # V2 documentation
├── 🟢 PROGRESS.md                          # Development progress
├── ⚪ requirements.txt                     # V2 dependencies
├── ⚪ requirements-minimal.txt             # Minimal deps
├── ⚪ .env                                 # V2 env file
├── ⚪ .env.example                         # V2 env template
├── ⚪ *.py                                 # Various test/utility scripts
└── ⚪ *.log                                # Log files
```

## Web Interface (`/web/`)

```
/Users/vitoryago/ADAM/web/
├── 🟢 adam_web.py                          # Streamlit interface
├── 🟢 README.md                            # Web interface docs
└── ⚪ streamlit.log                        # Streamlit logs
```

## Data Storage (`/data/`)

```
/Users/vitoryago/ADAM/data/
├── 🟡 adam_memory*/                        # ChromaDB collections
├── 🟡 adam_projects/                       # Project metadata
│   └── 🟡 projects.json                   # Project registry
├── 🟡 screen_captures/                     # Captured screenshots
├── 🟢 access_log.json                      # Memory access stats
├── 🟢 cost_savings.json                    # Cost tracking
└── 🟢 activity_log.json                    # Usage statistics
```

## Conversations (`/conversations/`)

```
/Users/vitoryago/ADAM/conversations/
├── 🟡 session_*/                           # Individual sessions
│   ├── 🟡 session_info.json               # Session metadata
│   └── 🟡 exchanges.json                  # Conversation history
└── 🟢 active_session.txt                   # Current session pointer
```

## Examples (`/examples/`)

```
/Users/vitoryago/ADAM/examples/
├── 🟢 advanced_memory_demo.py              # Memory features demo
├── 🟢 coworker_demo.py                     # Coworker features (NEW)
├── 🟢 conversation_network_demo.py         # Network visualization
├── 🟢 cost_analysis_demo.py               # Cost tracking demo
├── 🟢 memory_lifecycle_demo.py            # Lifecycle management
├── 🟢 memory_network_demo.py              # Graph relationships
├── 🟢 test_messaging.py                   # V2 messaging test
├── 🟢 test_memory.py                      # V2 memory test
└── 🟢 test_models.py                      # Model testing
```

## Documentation (`/docs/`)

```
/Users/vitoryago/ADAM/docs/
├── 🟢 ADAM_COWORKER_INTEGRATION.md        # Coworker guide (NEW)
├── 🟢 ADAM_EVOLUTION.md                   # Development history
├── 🟢 cost_analysis_report_*.txt          # Cost reports
├── 🟢 LANGGRAPH_INTEGRATION.md           # LangGraph guide
├── 🟢 memory_analysis_*.txt              # Memory reports
├── 🟢 migration_plan.md                   # V1→V2 migration
├── 🟢 PRICING_UPDATE_*.md                # Pricing updates
└── 🟢 SESSION_REPORT_*.txt               # Session reports
```

## Configuration (`/config/`)

```
/Users/vitoryago/ADAM/config/
├── 🔴 .env.template                        # Environment template
└── 🟢 *.yaml                               # Config files (if any)
```

## Package Info (`/src/adam_assistant.egg-info/`)

```
/Users/vitoryago/ADAM/src/adam_assistant.egg-info/
├── ⚪ dependency_links.txt                 # Package dependencies
├── ⚪ PKG-INFO                            # Package metadata
├── ⚪ SOURCES.txt                         # Source file list
├── ⚪ top_level.txt                       # Top-level modules
└── ⚪ requires.txt                        # Requirements
```

## Virtual Environment (`/venv/`) - EXCLUDED

```
/Users/vitoryago/ADAM/venv/
└── ⚪ [Thousands of library files - NOT INCLUDED IN TRANSFER]
```

---

## Summary by Category

### 🔴 ESSENTIAL Files (Must Have)
- Core memory system: `memory.py`, `memory_config.py`
- LLM integration: `llm/client.py`, `llm/config.py`
- Conversation: `conversation_system.py`
- Configuration: `.env`, `requirements.txt`, `setup.py`
- Error handling: `errors.py`
- Package init: `__init__.py` files

### 🟡 RECOMMENDED Files (Enhanced Features)
- Project isolation: `project_manager.py`, `project_aware_memory.py`
- Screen capture: `screen_capture.py`
- Advanced search: `advanced_rag.py`, `memory_network.py`
- V2 services: `adam_v2/services/*.py`
- Existing data: `data/`, `conversations/`

### 🟢 OPTIONAL Files (Nice to Have)
- Examples: `examples/*.py`
- Documentation: `docs/*.md`, `README.md`
- Web interface: `web/adam_web.py`
- Cost tracking: `cost_monitor.py`, `pricing_manager.py`

### ⚪ UNNECESSARY Files (Can Remove)
- V2 web components: `adam_v2/routers/`, `adam_v2/templates/`
- Test data: `adam_v2/data/` (154MB!)
- Build artifacts: `*.egg-info/`, `__pycache__/`
- Logs: `*.log`
- Test files: `adam_v2/tests/`, `test_*.py`

---

## Compression Recommendations

### Minimal Core (~2MB compressed)
```bash
tar -czf adam_minimal.tar.gz \
  src/adam/*.py \
  src/adam/llm/ \
  requirements.txt \
  setup.py \
  .env.example
```

### Recommended Set (~5MB compressed)
```bash
tar -czf adam_recommended.tar.gz \
  src/adam/ \
  src/adam_v2/services/ \
  src/adam_v2/models.py \
  data/ \
  conversations/ \
  requirements*.txt \
  setup.py \
  .env.example \
  examples/
```

### Complete Archive (~20MB compressed, excluding test data)
```bash
tar -czf adam_complete.tar.gz \
  --exclude="*/venv/*" \
  --exclude="*/__pycache__/*" \
  --exclude="*.pyc" \
  --exclude="adam_v2/data/*" \
  --exclude="*.log" \
  --exclude=".git/*" \
  .
```

This structure represents ~1.5 years of ADAM development, from a simple chatbot to a sophisticated AI coworker with project-based memory and screen vision capabilities.