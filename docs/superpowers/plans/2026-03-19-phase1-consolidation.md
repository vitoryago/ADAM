# Phase 1: Consolidation — "One ADAM" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `src/adam/` (v1) and `src/adam_v2/` (v2) into a single unified `src/adam/` package. Delete all dead code, mocks, and duplicates. End state: one backend, one memory system, one LLM client — everything compiles and tests pass.

**Architecture:** The unified package lives at `src/adam/` with sub-packages: `api/`, `core/`, `memory/`, `llm/`, `knowledge/`, `services/`. A single FastAPI entrypoint at `src/adam/api/main.py` serves the backend. SQLAlchemy + SQLite for persistence, ChromaDB + NetworkX for memory.

**Tech Stack:** Python 3.9+, FastAPI, SQLAlchemy, ChromaDB, NetworkX, LangChain/LangGraph, Pydantic v2, pytest

**Spec:** `docs/superpowers/specs/2026-03-18-adam-roadmap-design.md`

---

## Dependency Graph

```
Task 1 (Scaffold & Purge)
   ├──→ Task 2 (Foundation) ──→ Task 3 (LLM) ──┐
   │                                              ├──→ Task 6 (Services) ──→ Task 7 (API) ──→ Task 8 (Tests)
   ├──→ Task 4 (Memory) ─────────────────────────┤
   └──→ Task 5 (Knowledge) ──────────────────────┘
```

**Parallel opportunities:**
- Tasks 2 + 5 can run in parallel (independent subtrees)
- Tasks 3 + 4 can run in parallel after Task 2 (both depend on config/errors but not each other)
- Task 6 depends on Tasks 3 + 4 + 5 (services import from LLM, memory, AND knowledge)

---

### Task 1: Scaffold & Purge

**Goal:** Create the new directory structure and delete all dead code. After this task, the new tree exists (with empty `__init__.py` files) and all dead files are gone.

**Files:**
- Create: `src/adam/api/__init__.py`
- Create: `src/adam/api/routers/__init__.py`
- Create: `src/adam/core/__init__.py`
- Create: `src/adam/knowledge/__init__.py`
- Create: `src/adam/services/__init__.py`
- Delete: `src/adam/system.py` (lines 70-90 only — the `ADAMMemoryAdvanced` stub. Keep `ADAMSystem` for Task 2)
- Delete: `src/adam/integrated_conversation_system.py`
- Delete: `src/adam/conversation_system.py`
- Delete: `src/adam/legacy_config.py`
- Delete: `src/adam/screen_capture.py`
- Delete: `src/adam/tools/` (entire directory)
- Delete: `src/adam/cli/` (entire directory)
- Delete: `src/adam_v2/routes/` (entire directory)
- Delete: `src/adam_v2/routers/tools.py`
- Delete: `src/adam_v2/routers/onboarding.py`
- Delete: `src/adam_v2/routers/file_watcher.py`
- Delete: `src/adam_v2/services/onboarding_service.py`
- Delete: `src/adam_v2/services/onboarding_integration_service.py`
- Delete: `src/adam_v2/services/file_watcher.py`
- Delete: `src/adam_v2/services/dbt_chat_integration.py`
- Delete: `src/adam_v2/memory_manager.py`
- Delete: `src/adam_v2/run_adam_v2.py`
- Delete: `src/adam_v2/check_latest_message.py`
- Delete: `src/adam_v2/start.sh`
- Delete: `src/adam_v2/start_server.sh`
- Delete: `src/adam_v2/backend.log`
- Delete: `src/adam_v2/server.log`
- Delete: `src/adam_v2/test_markdown.html`
- Delete: `src/adam_v2/README.md`
- Delete: `frontend/AdamChat/python_backend/` (entire directory)
- Delete: `frontend/AdamChat/client/src/pages/test-hover.tsx`
- Delete: `frontend/AdamChat/client/src/pages/test-markdown.tsx`
- Delete: `frontend/AdamChat/client/src/pages/test-message.tsx`
- Delete: `frontend/AdamChat/client/src/components/chat/sidebar-test.tsx`
- Delete: `frontend/AdamChat/client/src/components/chat/streaming-voice-conversation-old.tsx`
- Delete: `web/` (entire directory, if exists)
- Delete: `cli/` (top-level directory)
- Delete: `.env.unified`
- Delete: `start-adam-web.sh`
- Delete: `test_dbt_api.py`
- Delete: `test_dbt_assistant.py`

- [ ] **Step 1: Create new directory structure**

Create the following directories with empty `__init__.py` files:

```
src/adam/api/
src/adam/api/routers/
src/adam/core/
src/adam/knowledge/
src/adam/services/
```

Each `__init__.py` should contain only:
```python
"""ADAM - [subpackage description]"""
```

- [ ] **Step 2: Delete all dead files from src/adam/**

Delete these files (they are mocked, duplicated, or dead per the spec):
- `src/adam/integrated_conversation_system.py`
- `src/adam/conversation_system.py`
- `src/adam/legacy_config.py`
- `src/adam/screen_capture.py`
- `src/adam/tools/` (entire directory including `__init__.py`, `sql_tools.py`, `web_search.py`, `code_executor.py`, `snowflake_executor.py`, `file_generator.py`, `ai_sql_generator.py`, `model_web_search.py`)
- `src/adam/cli/` (entire directory)

Also remove the `ADAMMemoryAdvanced` mock class from `src/adam/system.py` (lines 70-90) but keep the `ADAMSystem` class (lines 13-66).

- [ ] **Step 3: Delete dead files from src/adam_v2/**

Delete:
- `src/adam_v2/routes/` (entire directory — superseded by `routers/`)
- `src/adam_v2/routers/tools.py`
- `src/adam_v2/routers/onboarding.py`
- `src/adam_v2/routers/file_watcher.py`
- `src/adam_v2/services/onboarding_service.py`
- `src/adam_v2/services/onboarding_integration_service.py`
- `src/adam_v2/services/file_watcher.py`
- `src/adam_v2/services/dbt_chat_integration.py`
- `src/adam_v2/memory_manager.py`
- `src/adam_v2/run_adam_v2.py`
- `src/adam_v2/run_tests.py`
- `src/adam_v2/check_latest_message.py`
- `src/adam_v2/start.sh`, `start_server.sh`
- `src/adam_v2/backend.log`, `server.log`, `test_markdown.html`
- `src/adam_v2/README.md`
- `src/adam_v2/logs/` (entire directory)
- `src/adam_v2/adam_v2.db`
- `src/adam_v2/.coverage`, `.pytest_cache/` (runtime artifacts)

- [ ] **Step 4: Delete dead files from frontend and root**

Delete:
- `frontend/AdamChat/python_backend/` (entire directory)
- `frontend/AdamChat/client/src/pages/test-hover.tsx`
- `frontend/AdamChat/client/src/pages/test-markdown.tsx`
- `frontend/AdamChat/client/src/pages/test-message.tsx`
- `frontend/AdamChat/client/src/components/chat/sidebar-test.tsx`
- `frontend/AdamChat/client/src/components/chat/streaming-voice-conversation-old.tsx`
- `.env.unified`
- `start-adam-web.sh`
- `test_dbt_api.py`
- `test_dbt_assistant.py`

If `web/` or `cli/` exist as top-level directories, delete them too.

- [ ] **Step 5: Update .gitignore (do this BEFORE any git add -A)**

Add these lines to `.gitignore` if not already present:
```
*.log
*.db
*.coverage
.pytest_cache/
data/
adam_memory_advanced/
__pycache__/
```

- [ ] **Step 6: Create .env.example**

Create `.env.example` at project root with template variables (no real keys):
```
# LLM Provider API Keys
XAI_API_KEY=your-xai-key-here
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Database
DATABASE_URL=sqlite:///./data/adam.db

# Memory
CHROMADB_PERSIST_DIR=./data/memory

# Server
HOST=0.0.0.0
PORT=8000

# Default Model
DEFAULT_MODEL=gpt-4o-mini
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: scaffold new structure and purge dead code

Create unified directory structure under src/adam/ with api/, core/,
knowledge/, services/ sub-packages. Delete all mocked, duplicated,
and dead code identified in Phase 1 spec."
```

---

### Task 2: Foundation Modules

**Goal:** Consolidate config, errors, utils, and database into the new structure. These are foundational modules that everything else imports.

**Files:**
- Move: `src/adam/config/unified.py` → `src/adam/config.py`
- Keep: `src/adam/errors.py` (already in correct location)
- Keep: `src/adam/utils/` (already in correct location)
- Merge: `src/adam/database/engine.py` + `src/adam_v2/database.py` → `src/adam/database.py`
- Move: `src/adam/system.py` (ADAMSystem class) → `src/adam/core/app.py`
- Move: `src/adam/langgraph_conversation.py` → `src/adam/core/pipeline.py`
- Move: `src/adam/project_manager.py` → `src/adam/services/project_service.py`
- Create: `tests/test_foundation.py`

- [ ] **Step 1: Consolidate config**

Read `src/adam/config/unified.py` and `src/adam/config/__init__.py`. Create a single `src/adam/config.py` that:
- Contains the `get_config()` function and config classes from `unified.py`
- Exports everything that `__init__.py` was re-exporting
- Uses `python-dotenv` to load `.env`

Then delete `src/adam/config/` directory.

- [ ] **Step 2: Consolidate database**

Read `src/adam/database/engine.py`, `src/adam/database/models.py`, and `src/adam_v2/database.py`. Create a single `src/adam/database.py` that:
- Contains the SQLAlchemy engine setup from `engine.py`
- Contains the `Base` declarative base
- Contains `get_engine()` and `get_session()` functions
- Uses `DATABASE_URL` from config, defaulting to `sqlite:///./data/adam.db`

Do NOT include the SQLAlchemy ORM models here — those go in `src/adam/api/models.py` (Task 7).

Delete `src/adam/database/` directory and `src/adam/database/migrations.py`.

- [ ] **Step 3: Move ADAMSystem to core/app.py**

Move the `ADAMSystem` class from `src/adam/system.py` to `src/adam/core/app.py`. Update its imports to use the new `src/adam/config` and `src/adam/llm/async_client` paths. Delete `src/adam/system.py`.

- [ ] **Step 3b: Move langgraph_conversation.py to core/pipeline.py**

Move `src/adam/langgraph_conversation.py` to `src/adam/core/pipeline.py`. Update its imports:
- `from adam.errors import ...` (if needed)
- `from adam.memory import ...` (for memory references)
- Remove any imports to deleted modules (`conversation_system`, `integrated_conversation_system`)

The LangGraph pipeline has mock implementations for `check_memory_node` and `generate_response_node` — leave these as-is for now. They'll be wired to real implementations in Phase 2. Mark them with `# TODO(phase2): connect to real LLM/memory`.

- [ ] **Step 3c: Move project_manager.py to services/project_service.py**

Move `src/adam/project_manager.py` to `src/adam/services/project_service.py`. Update imports to use `adam.` prefix paths. Delete the original.

- [ ] **Step 4: Update src/adam/__init__.py**

Rewrite `src/adam/__init__.py` to export only what exists in the new structure:

```python
"""ADAM - Analytics Data Assistant with Memory"""

__version__ = "4.0.0"
```

Remove all old imports that reference deleted modules. We'll add exports back as modules are consolidated.

- [ ] **Step 5: Write foundation smoke test**

Create `tests/test_foundation.py`:

```python
"""Smoke tests for foundation modules."""

def test_config_loads():
    from adam.config import get_config
    config = get_config()
    assert config is not None

def test_database_engine():
    from adam.database import get_engine
    engine = get_engine()
    assert engine is not None

def test_errors_importable():
    from adam.errors import ADAMError, MemoryError, StorageError
    assert issubclass(MemoryError, ADAMError)

def test_adam_system_importable():
    from adam.core.app import ADAMSystem
    assert ADAMSystem is not None
```

- [ ] **Step 6: Run tests**

Run: `cd /Users/vitoryago/ADAM && python -m pytest tests/test_foundation.py -v`
Expected: All 4 tests PASS. If imports fail, fix the import paths.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: consolidate foundation modules (config, database, errors)

Merge config/ directory into single config.py. Merge database/
directory into single database.py. Move ADAMSystem to core/app.py.
Clean up __init__.py exports. Add foundation smoke tests."
```

---

### Task 3: LLM Layer

**Goal:** Move the LLM modules into the unified structure with updated imports. The LLM layer is already well-organized in `src/adam/llm/` — mostly needs import fixes and router consolidation.

**Files:**
- Keep: `src/adam/llm/client.py` (update imports)
- Keep: `src/adam/llm/async_client.py` (update imports)
- Merge: `src/adam/llm_router.py` + `src/adam_v2/services/fast_routing_service.py` + `src/adam_v2/services/intelligent_routing_service.py` → `src/adam/llm/router.py`
- Keep: `src/adam/llm/query_analyzer.py` (if exists, update imports)
- Keep: `src/adam/llm/config.py` (if exists, update imports)
- Create: `src/adam/llm/__init__.py`
- Create: `tests/test_llm.py`

- [ ] **Step 0: Create llm/__init__.py**

Create `src/adam/llm/__init__.py`:
```python
"""ADAM LLM Layer - Multi-provider LLM client and intelligent routing."""

from .client import UnifiedLLMClient

__all__ = ['UnifiedLLMClient']
```

- [ ] **Step 1: Update llm/client.py imports**

Read `src/adam/llm/client.py`. Update any imports that reference old paths (e.g., `from adam.config.unified import ...` → `from adam.config import ...`). Do not change functionality.

- [ ] **Step 2: Update llm/async_client.py imports**

Same process for `async_client.py`.

- [ ] **Step 3: Consolidate router**

Read these three files:
- `src/adam/llm_router.py` (the `LLMRouter` and `SpecializedRouter` classes)
- `src/adam_v2/services/fast_routing_service.py`
- `src/adam_v2/services/intelligent_routing_service.py`

Create `src/adam/llm/router.py` that:
- Contains the `LLMRouter` class from `llm_router.py` (the core routing logic)
- Incorporates the fast rule-based pre-filter concept from `fast_routing_service.py`
- Incorporates the model selection logic from `intelligent_routing_service.py`
- Exports a single `route_query(query, context) -> RoutingDecision` interface
- Uses `from adam.config import get_config` for API keys
- Uses `from adam.llm.client import UnifiedLLMClient` if needed for the Haiku router call

Delete `src/adam/llm_router.py` (old location) after creating the consolidated version.
Delete `src/adam_v2/services/fast_routing_service.py` and `src/adam_v2/services/intelligent_routing_service.py`.

- [ ] **Step 4: Write LLM layer test**

Create `tests/test_llm.py`:

```python
"""Tests for LLM layer imports and basic structure."""

def test_unified_client_importable():
    from adam.llm.client import UnifiedLLMClient
    assert UnifiedLLMClient is not None

def test_router_importable():
    from adam.llm.router import LLMRouter, RoutingDecision
    assert LLMRouter is not None
    assert RoutingDecision is not None

def test_router_fallback():
    """Test that rule-based fallback works without API keys."""
    from adam.llm.router import LLMRouter
    router = LLMRouter.__new__(LLMRouter)  # Skip __init__ (needs API keys)
    decision = router._fallback_routing("Hello, how are you?")
    assert decision is not None
    assert hasattr(decision, 'model_tier')
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_llm.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: consolidate LLM layer with unified router

Merge llm_router.py, fast_routing_service.py, and
intelligent_routing_service.py into single llm/router.py.
Update all import paths in llm/ modules."
```

---

### Task 4: Memory System

**Goal:** Consolidate 4 memory implementations (~6500 lines) into one unified system under `src/adam/memory/`. This is the largest and most important consolidation task.

**Files:**
- Keep: `src/adam/memory/core.py` — canonical ChromaDB storage (update imports)
- Keep: `src/adam/memory/network.py` — NetworkX graph (update imports)
- Keep: `src/adam/memory/lifecycle.py` — decay/reinforcement (update imports)
- Keep: `src/adam/memory/compressor.py` — memory compression (update imports)
- Keep: `src/adam/memory/config.py` — embedding config (update imports)
- Keep: `src/adam/memory/project.py` — project-scoped memory (update imports)
- Move: `src/adam/advanced_rag.py` → `src/adam/memory/rag.py` (update imports)
- Merge: `src/adam/memory/conversation.py` into `src/adam/memory/core.py` (conversation-aware features)
- Merge: `src/adam/memory/scoring.py` into `src/adam/memory/core.py` (scoring logic)
- Merge: `src/adam/memory/search.py` into `src/adam/memory/core.py` (search logic)
- Merge: `src/adam_v2/services/memory_service.py` concepts into `src/adam/memory/project.py`
- Merge: `src/adam_v2/services/advanced_memory_service.py` BM25 logic into `src/adam/memory/rag.py`
- Create: `tests/test_memory.py`

- [ ] **Step 1: Update core.py imports**

Read `src/adam/memory/core.py`. Update imports:
- `from .config import MemoryConfig` (already correct)
- `from .lifecycle import MemoryLifecycleManager` (already correct)
- `from ..errors import ...` → `from adam.errors import ...`
- Any references to `adam.conversation_system` → remove (that module is deleted)

- [ ] **Step 2: Merge conversation.py, scoring.py, search.py into core.py**

Read `src/adam/memory/conversation.py`, `scoring.py`, and `search.py`. For each:
- Identify classes/functions that are used by other modules
- If they're only used by `core.py`, merge them directly into `core.py`
- If they're used elsewhere, keep them as imports from `core.py`
- Delete the original files after merging

The key class from `conversation.py` is `ConversationAwareMemorySystem` — merge its logic into the main `ADAMMemoryAdvanced` class in `core.py`.

- [ ] **Step 3: Move advanced_rag.py to memory/rag.py**

Move `src/adam/advanced_rag.py` to `src/adam/memory/rag.py`. Update its imports:
- `from .core import ADAMMemoryAdvanced`
- `from .network import MemoryNetworkSystem`
- `from adam.errors import ...`

Read `src/adam_v2/services/advanced_memory_service.py` and extract any BM25 evaluation logic that isn't already in `rag.py`. Merge it in.

> **Note:** The spec says `advanced_memory_service.py` merges into `core.py`, but the BM25 retrieval logic is a better fit for `rag.py` (which is the advanced retrieval module). This is a deliberate deviation from the spec.

Delete `src/adam/advanced_rag.py` and `src/adam_v2/services/advanced_memory_service.py`.

- [ ] **Step 4: Update network.py imports**

Read `src/adam/memory/network.py`. Update:
- `from ..errors import ...` → `from adam.errors import ...`
- Remove any references to deleted modules

- [ ] **Step 5: Update project.py with v2 memory service concepts**

Read `src/adam/memory/project.py` and `src/adam_v2/services/memory_service.py`. The v2 service has project-scoped ChromaDB collection logic — merge that concept into `project.py` so project-aware memory scoping works through a single interface.

Delete `src/adam_v2/services/memory_service.py`.

- [ ] **Step 6: Update memory/__init__.py**

Write `src/adam/memory/__init__.py` that exports the public API:

```python
"""ADAM Memory System - Persistent memory with graph relationships and decay."""

from .core import ADAMMemoryAdvanced, MemoryType, MemoryWorthinessEvaluator
from .network import MemoryNetworkSystem, MemoryNode, ConversationThread
from .lifecycle import MemoryLifecycleManager
from .config import MemoryConfig
from .project import ProjectAwareMemory

__all__ = [
    'ADAMMemoryAdvanced',
    'MemoryType',
    'MemoryWorthinessEvaluator',
    'MemoryNetworkSystem',
    'MemoryNode',
    'ConversationThread',
    'MemoryLifecycleManager',
    'MemoryConfig',
    'ProjectAwareMemory',
]
```

- [ ] **Step 7: Write memory smoke test**

Create `tests/test_memory.py`:

```python
"""Smoke tests for consolidated memory system."""

def test_memory_imports():
    from adam.memory import ADAMMemoryAdvanced, MemoryNetworkSystem, MemoryConfig
    assert ADAMMemoryAdvanced is not None
    assert MemoryNetworkSystem is not None

def test_memory_config():
    from adam.memory.config import MemoryConfig
    config = MemoryConfig()
    assert config.embedding_model_name is not None

def test_memory_types():
    from adam.memory.core import MemoryType
    assert MemoryType.ERROR_SOLUTION.value == "error_solution"
    assert MemoryType.CODE_PATTERN.value == "code_pattern"

def test_worthiness_evaluator():
    from adam.memory.core import MemoryWorthinessEvaluator, QueryComplexity
    evaluator = MemoryWorthinessEvaluator()
    complexity = evaluator.assess_query_complexity("What is Python?")
    assert complexity in [QueryComplexity.TRIVIAL, QueryComplexity.SIMPLE]
```

- [ ] **Step 8: Run tests**

Run: `python -m pytest tests/test_memory.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: consolidate memory system into unified package

Merge 4 memory implementations into single src/adam/memory/ package.
Fold conversation.py, scoring.py, search.py into core.py. Move
advanced_rag.py to memory/rag.py with BM25 logic from v2. Update
project.py with v2 scoping concepts. Add memory smoke tests."
```

---

### Task 5: Knowledge Layer

**Goal:** Move dbt and SQL knowledge services into `src/adam/knowledge/`. This is independent of Tasks 3-4 and can run in parallel.

**Files:**
- Move: `src/adam_v2/services/dbt_knowledge_service.py` → `src/adam/knowledge/dbt_knowledge.py`
- Move: `src/adam_v2/services/sql_knowledge_service.py` → `src/adam/knowledge/sql_knowledge.py`
- Move: `src/adam_v2/services/lineage_service.py` → `src/adam/knowledge/lineage_service.py`
- Move: `src/adam_v2/dbt_analyzer/` → `src/adam/knowledge/dbt_analyzer/`
- Merge: `src/adam_v2/services/dbt_service.py` into `src/adam/knowledge/dbt_knowledge.py`
- Merge: `src/adam_v2/services/dbt_assistant.py` into `src/adam/knowledge/dbt_knowledge.py`
- Merge: `src/adam_v2/services/dbt_integration_service.py` into `src/adam/knowledge/dbt_knowledge.py`
- Merge: `src/adam_v2/services/dbt_column_service.py` into `src/adam/knowledge/dbt_analyzer/column_intelligence.py`
- Create: `tests/test_knowledge.py`

- [ ] **Step 1: Move dbt_knowledge_service.py**

Copy `src/adam_v2/services/dbt_knowledge_service.py` to `src/adam/knowledge/dbt_knowledge.py`. Update imports to use `adam.` prefix paths. Read `dbt_service.py`, `dbt_assistant.py`, and `dbt_integration_service.py` — merge any unique functionality (that isn't already in `dbt_knowledge_service.py`) into the new file.

Delete the original v2 files.

- [ ] **Step 2: Move sql_knowledge_service.py**

Copy `src/adam_v2/services/sql_knowledge_service.py` to `src/adam/knowledge/sql_knowledge.py`. Update imports.

Delete original.

- [ ] **Step 3: Move lineage_service.py**

Copy `src/adam_v2/services/lineage_service.py` to `src/adam/knowledge/lineage_service.py`. Update imports.

Delete original.

- [ ] **Step 4: Move dbt_analyzer directory**

Copy `src/adam_v2/dbt_analyzer/` to `src/adam/knowledge/dbt_analyzer/`. Update all internal imports to use `adam.knowledge.dbt_analyzer.` prefix. Read `dbt_column_service.py` and merge relevant logic into `column_intelligence.py`.

Delete `src/adam_v2/dbt_analyzer/` and `src/adam_v2/services/dbt_column_service.py`.

- [ ] **Step 5: Update knowledge/__init__.py**

```python
"""ADAM Knowledge Layer - Domain-specific knowledge for dbt, SQL, and more."""

from .dbt_knowledge import DBTKnowledgeService
from .sql_knowledge import SQLKnowledgeService

__all__ = ['DBTKnowledgeService', 'SQLKnowledgeService']
```

- [ ] **Step 6: Write knowledge smoke test**

Create `tests/test_knowledge.py`:

```python
"""Smoke tests for knowledge layer."""

def test_dbt_knowledge_importable():
    from adam.knowledge.dbt_knowledge import DBTKnowledgeService
    assert DBTKnowledgeService is not None

def test_sql_knowledge_importable():
    from adam.knowledge.sql_knowledge import SQLKnowledgeService
    assert SQLKnowledgeService is not None

def test_dbt_context_detection():
    from adam.knowledge.dbt_knowledge import DBTKnowledgeService
    service = DBTKnowledgeService()
    assert service.detect_dbt_context("create a dbt incremental model") == True
    assert service.detect_dbt_context("what is the weather today") == False

def test_dbt_analyzer_importable():
    from adam.knowledge.dbt_analyzer.parser import DbtParser
    assert DbtParser is not None
```

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_knowledge.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: consolidate knowledge layer (dbt, SQL, lineage)

Move dbt_knowledge_service, sql_knowledge_service, lineage_service
to src/adam/knowledge/. Move dbt_analyzer/ to knowledge/dbt_analyzer/.
Merge dbt_service, dbt_assistant, dbt_integration_service into
dbt_knowledge.py. Add knowledge smoke tests."
```

---

### Task 6: Services Layer

**Goal:** Consolidate all business logic services into `src/adam/services/`.

**Files:**
- Merge: `src/adam_v2/services/llm_service.py` → `src/adam/services/llm_service.py` (update to use `adam.llm.client`)
- Move: `src/adam_v2/services/voice_service.py` → `src/adam/services/voice_service.py`
- Move: `src/adam_v2/services/voice_websocket.py` → `src/adam/services/voice_websocket.py`
- Move: `src/adam_v2/services/voice_conversation_handler.py` → `src/adam/services/voice_conversation_handler.py`
- Move: `src/adam_v2/services/voice_response_formatter.py` → `src/adam/services/voice_response_formatter.py`
- Move: `src/adam_v2/services/response_style_service.py` → `src/adam/services/response_style_service.py`
- Move: `src/adam_v2/services/markdown_service.py` → `src/adam/services/markdown_service.py`
- Move: `src/adam/cost_monitor.py` → `src/adam/services/cost_monitor.py`
- Move: `src/adam/pricing_manager.py` → `src/adam/services/pricing_manager.py`
- Move: `src/adam/activity_tracker.py` → `src/adam/services/activity_tracker.py`
- Create: `tests/test_services.py`

- [ ] **Step 1: Consolidate llm_service.py**

Read `src/adam_v2/services/llm_service.py`. This is the largest service (~800 lines). Create `src/adam/services/llm_service.py` that:
- Keeps the `LLMService`, `LLMResponse`, `StreamChunk` classes
- Updates all imports to use `adam.llm.client.UnifiedLLMClient` instead of the duplicated path
- Updates `from services.dbt_knowledge_service import ...` → `from adam.knowledge.dbt_knowledge import ...`
- Updates `from services.sql_knowledge_service import ...` → `from adam.knowledge.sql_knowledge import ...`
- Updates `from services.fast_routing_service import ...` → `from adam.llm.router import ...`
- Updates `from services.response_style_service import ...` → `from adam.services.response_style_service import ...`
- Updates memory service imports to use `adam.memory`
- Removes the `sys.path.insert(0, ...)` hack at the top

- [ ] **Step 2: Move voice services**

Copy these 4 files from `src/adam_v2/services/` to `src/adam/services/`:
- `voice_service.py`
- `voice_websocket.py`
- `voice_conversation_handler.py`
- `voice_response_formatter.py`

Update all internal imports to use `adam.` prefix paths.

- [ ] **Step 3: Move remaining services**

Move from `src/adam_v2/services/`:
- `response_style_service.py` → `src/adam/services/`
- `markdown_service.py` → `src/adam/services/`

Move from `src/adam/`:
- `cost_monitor.py` → `src/adam/services/cost_monitor.py`
- `pricing_manager.py` → `src/adam/services/pricing_manager.py`
- `activity_tracker.py` → `src/adam/services/activity_tracker.py`

Update all imports.

- [ ] **Step 4: Update services/__init__.py**

```python
"""ADAM Services - Business logic layer."""

from .llm_service import LLMService, LLMResponse, StreamChunk

__all__ = ['LLMService', 'LLMResponse', 'StreamChunk']
```

- [ ] **Step 5: Write services smoke test**

Create `tests/test_services.py`:

```python
"""Smoke tests for services layer."""

def test_llm_service_importable():
    from adam.services.llm_service import LLMService, LLMResponse, StreamChunk
    assert LLMService is not None

def test_voice_service_importable():
    from adam.services.voice_service import VoiceService
    assert VoiceService is not None

def test_response_style_importable():
    from adam.services.response_style_service import ResponseStyleService
    assert ResponseStyleService is not None

def test_cost_monitor_importable():
    from adam.services.cost_monitor import CostMonitor
    assert CostMonitor is not None
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_services.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: consolidate services layer

Move llm_service, voice services, response_style, markdown,
cost_monitor, pricing_manager, activity_tracker into unified
src/adam/services/. Update all imports. Add service smoke tests."
```

---

### Task 7: API Layer & Main Entrypoint

**Goal:** Create the single FastAPI entrypoint and move all routers into the unified structure.

**Files:**
- Create: `src/adam/api/main.py` (based on `src/adam_v2/main.py`)
- Move: `src/adam_v2/models.py` → `src/adam/api/models.py`
- Move: `src/adam_v2/routers/conversations.py` → `src/adam/api/routers/conversations.py`
- Move: `src/adam_v2/routers/projects.py` → `src/adam/api/routers/projects.py`
- Move: `src/adam_v2/routers/memories.py` → `src/adam/api/routers/memories.py`
- Move: `src/adam_v2/routers/messages.py` → `src/adam/api/routers/messages.py`
- Move: `src/adam_v2/routers/voice.py` → `src/adam/api/routers/voice.py`
- Move: `src/adam_v2/routers/voice_streaming.py` → `src/adam/api/routers/voice_streaming.py`
- Move: `src/adam_v2/routers/lineage.py` → `src/adam/api/routers/lineage.py`
- Move: `src/adam_v2/routers/styles.py` → `src/adam/api/routers/styles.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Move models.py**

Copy `src/adam_v2/models.py` to `src/adam/api/models.py`. Update imports:
- `from database import Base` → `from adam.database import Base`

- [ ] **Step 2: Move all routers**

Copy each router file from `src/adam_v2/routers/` to `src/adam/api/routers/`:
- `conversations.py`, `projects.py`, `memories.py`, `messages.py`
- `voice.py`, `voice_streaming.py`, `lineage.py`, `styles.py`

For each file, update imports:
- `from models import ...` → `from adam.api.models import ...`
- `from database import ...` → `from adam.database import ...`
- `from services.xxx import ...` → `from adam.services.xxx import ...`
- `from services.memory_service import ...` → `from adam.memory import ...`

- [ ] **Step 3: Create main.py entrypoint**

Read `src/adam_v2/main.py`. Create `src/adam/api/main.py` that:
- Creates the FastAPI app instance
- Includes all routers from `adam.api.routers`
- Sets up CORS middleware
- Initializes the database on startup
- Has a health check endpoint at `/health`

```python
"""ADAM API - Single FastAPI entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adam.database import get_engine

app = FastAPI(
    title="ADAM - Analytics Data Assistant with Memory",
    version="4.0.0",
    description="AI assistant with persistent memory and intelligent routing"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from adam.api.routers import (
    conversations, projects, memories, messages,
    voice, voice_streaming, lineage, styles
)

app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(conversations.router, prefix="/api/conversations", tags=["conversations"])
app.include_router(messages.router, prefix="/api/messages", tags=["messages"])
app.include_router(memories.router, prefix="/api/memories", tags=["memories"])
app.include_router(voice.router, prefix="/api/voice", tags=["voice"])
app.include_router(voice_streaming.router, prefix="/api/voice/stream", tags=["voice-streaming"])
app.include_router(lineage.router, prefix="/api/lineage", tags=["lineage"])
app.include_router(styles.router, prefix="/api/styles", tags=["styles"])

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "4.0.0"}

@app.on_event("startup")
async def startup():
    # Initialize database tables
    from adam.api.models import Base
    from adam.database import get_engine
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
```

- [ ] **Step 4: Update routers/__init__.py**

```python
"""ADAM API Routers."""
```

- [ ] **Step 5: Write API smoke test**

Create `tests/test_api.py`:

```python
"""Smoke tests for API layer."""
from fastapi.testclient import TestClient

def test_app_creates():
    from adam.api.main import app
    assert app is not None
    assert app.title == "ADAM - Analytics Data Assistant with Memory"

def test_health_check():
    from adam.api.main import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_routers_registered():
    from adam.api.main import app
    routes = [r.path for r in app.routes]
    assert "/health" in routes
    assert any("/api/projects" in r for r in routes)
    assert any("/api/conversations" in r for r in routes)
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: consolidate API layer with unified FastAPI entrypoint

Move all routers and models from adam_v2/ to adam/api/. Create
single main.py entrypoint. Update all import paths. Add API
smoke tests including health check."
```

---

### Task 8: Final Cleanup, Test Migration & Verification

**Goal:** Delete the now-empty `src/adam_v2/` directory, migrate existing tests, update setup.py, and verify everything works end-to-end.

**Files:**
- Delete: `src/adam_v2/` (entire directory — all useful code has been moved)
- Migrate: `src/adam_v2/tests/` → `tests/` (already done conceptually, now update imports)
- Update: `setup.py`
- Update: `requirements-consolidated.txt`
- Move: `src/adam_v2/pytest.ini` → `pytest.ini` (root)
- Create: `tests/test_integration.py`

- [ ] **Step 0: Preserve examples before bulk delete**

Move `src/adam_v2/examples/` contents to `examples/`:
```bash
cp src/adam_v2/examples/*.py examples/
```

- [ ] **Step 1: Delete src/adam_v2/**

Verify that all useful code has been moved by checking the Task 1-7 commits. Then delete the entire `src/adam_v2/` directory (includes `data/`, `logs/`, `adam_v2.db`, `.env`, runtime artifacts).

```bash
rm -rf src/adam_v2/
```

- [ ] **Step 2: Migrate existing tests**

Copy test files from the git history (or from the deleted directory if not yet committed) to `tests/`. Update all imports in the test files:
- `from models import ...` → `from adam.api.models import ...`
- `from database import ...` → `from adam.database import ...`
- `from services.xxx import ...` → `from adam.services.xxx import ...`

Move `src/adam_v2/pytest.ini` to root `pytest.ini` if it doesn't exist there yet.

- [ ] **Step 3: Update setup.py**

Update `setup.py`:
- Version: `4.0.0`
- `packages=find_packages(where="src")` (should still work since `src/adam/` is the only package)
- Update `install_requires` to match `requirements-consolidated.txt`
- Update entry_points to remove old CLI scripts, add:
  ```python
  entry_points={
      "console_scripts": [
          "adam-server=adam.api.main:app",
      ],
  },
  ```

- [ ] **Step 4: Consolidate requirements**

Read `requirements-consolidated.txt`, `src/adam_v2/requirements.txt`, and `requirements_dbt.txt`. Create a single `requirements.txt` at the project root with all unique dependencies. Delete the old files.

- [ ] **Step 5: Write integration test**

Create `tests/test_integration.py`:

```python
"""Integration tests — verify the full stack works together."""
from fastapi.testclient import TestClient

def test_full_server_startup():
    """Server starts without errors."""
    from adam.api.main import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200

def test_create_project():
    """Can create a project through the API."""
    from adam.api.main import app
    client = TestClient(app)
    response = client.post("/api/projects/", json={
        "name": "Test Project",
        "description": "Integration test project"
    })
    assert response.status_code in [200, 201]
    data = response.json()
    assert data["name"] == "Test Project"

def test_memory_system_initializes():
    """Memory system can be imported and config loaded."""
    from adam.memory import MemoryConfig
    config = MemoryConfig()
    assert config is not None

def test_knowledge_services_load():
    """Knowledge services initialize without errors."""
    from adam.knowledge import DBTKnowledgeService
    service = DBTKnowledgeService()
    assert service is not None

def test_no_adam_v2_imports():
    """Verify no code imports from the deleted adam_v2 package."""
    import os
    import re

    violations = []
    for root, dirs, files in os.walk("src/adam"):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".py"):
                filepath = os.path.join(root, f)
                with open(filepath) as fh:
                    content = fh.read()
                if "adam_v2" in content or "from adam_v2" in content:
                    violations.append(filepath)

    assert violations == [], f"Files still importing from adam_v2: {violations}"
```

- [ ] **Step 6: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 7: Verify server starts**

Run: `cd /Users/vitoryago/ADAM && python -m uvicorn adam.api.main:app --host 0.0.0.0 --port 8000`
Expected: Server starts without errors. Hit `http://localhost:8000/health` — returns `{"status": "ok"}`.

Stop the server after verifying.

- [ ] **Step 8: Final commit**

```bash
git add -A
git commit -m "refactor: complete Phase 1 consolidation

Delete src/adam_v2/ — all code migrated to unified src/adam/ package.
Migrate and update all tests. Consolidate requirements. Update
setup.py to v4.0.0. Full test suite passes, server starts clean.

Phase 1 deliverable: one backend, one memory system, one LLM client.
Everything that exists actually works."
```

---

## Agent Team Execution Strategy

The following tasks can be parallelized using agent worktrees:

| Wave | Tasks | Rationale |
|------|-------|-----------|
| 1 | Task 1 | Foundation — everything depends on this |
| 2 | Task 2 + Task 5 | Independent: foundation modules vs knowledge layer |
| 3 | Task 3 + Task 4 | Independent: LLM vs memory (both depend on Task 2) |
| 4 | Task 6 | Services depend on LLM + memory + knowledge (Tasks 3, 4, 5) |
| 5 | Task 7 | API depends on services |
| 6 | Task 8 | Final cleanup depends on everything |

Total: 8 tasks across 6 waves. Parallel agents reduce this from 8 sequential tasks to 6 rounds.
