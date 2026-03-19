# ADAM Roadmap — Foundation First

**Date:** 2026-03-18
**Status:** Approved
**Approach:** Foundation First — consolidate, then build forward

## Context

ADAM (Analytics Data Assistant with Memory) is a personal AI assistant with persistent memory, intelligent LLM routing, and dbt specialization. The codebase has strong architectural ideas but suffers from split implementations (`src/adam/` v1 and `src/adam_v2/`), mocked features, duplicated systems, and an incomplete frontend. The result: ambitious code that doesn't work end-to-end.

### Audience
- **Now:** Personal productivity tool for the developer
- **Future:** Open-source project for the community

### Core Vision
An AI assistant you can truly talk to — one that remembers everything, reasons deeply using multiple AI providers, and sees the context of your work.

### Strategic Principle
Each phase delivers a complete, working product. Nothing is built on mocks or placeholders. This breaks the cycle of ambitious-but-unfinished.

---

## Phase 1: Consolidation — "One ADAM"

**Goal:** Merge the split codebase into a single clean architecture. Delete everything mocked, duplicated, or dead.

**Estimated duration:** 2-3 weeks

### Complete File Disposition

Every file in `src/adam/` and `src/adam_v2/` has a disposition: **keep**, **delete**, or **merge into X**.

#### `src/adam/` — Core Library (v1)

| File/Module | Disposition | Rationale |
|---|---|---|
| `__init__.py` | **Rewrite** | Update exports to match new unified structure |
| `system.py` — `ADAMSystem` class | **Merge** into `src/adam/core/app.py` | Thin async wrapper around LLM client, useful as top-level entrypoint |
| `system.py` — `ADAMMemoryAdvanced` stub | **Delete** | Mock class that returns empty results |
| `integrated_conversation_system.py` | **Delete** | Duplicates LangGraph + v2 logic, has hardcoded TODO LLM clients |
| `conversation_system.py` | **Delete** | File-based JSON session system — replaced by v2's SQLAlchemy conversations |
| `langgraph_conversation.py` | **Keep** → `src/adam/core/pipeline.py` | State machine pattern is sound, needs real LLM calls (Phase 2) |
| `llm_router.py` | **Keep** → `src/adam/llm/router.py` | LLM-based query classification works |
| `llm/client.py` | **Keep** → `src/adam/llm/client.py` | Unified multi-provider LLM client |
| `llm/async_client.py` | **Keep** → `src/adam/llm/async_client.py` | Async variant of LLM client |
| `llm/config.py` (if exists) | **Keep** → `src/adam/llm/config.py` | Model configurations |
| `llm/query_analyzer.py` (if exists) | **Keep** → `src/adam/llm/query_analyzer.py` | Query complexity analysis |
| `legacy_config.py` | **Delete** | Dead weight |
| `config/unified.py` | **Keep** → `src/adam/config.py` | Unified configuration |
| `config/__init__.py` | **Merge** into `src/adam/config.py` | Consolidate config into single module |
| `memory/core.py` | **Keep** → `src/adam/memory/core.py` | ChromaDB vector storage, worthiness evaluation — the canonical memory store |
| `memory/network.py` | **Keep** → `src/adam/memory/network.py` | NetworkX graph, decay, reinforcement, threads — the relationship layer |
| `memory/conversation.py` | **Merge** into `src/adam/memory/core.py` | Conversation-aware memory features fold into the main memory service |
| `memory/lifecycle.py` | **Keep** → `src/adam/memory/lifecycle.py` | Memory decay/health management |
| `memory/compressor.py` | **Keep** → `src/adam/memory/compressor.py` | Memory compression for old memories |
| `memory/scoring.py` | **Merge** into `src/adam/memory/core.py` | Scoring logic lives with retrieval |
| `memory/search.py` | **Merge** into `src/adam/memory/core.py` | Search logic lives with retrieval |
| `memory/project.py` | **Keep** → `src/adam/memory/project.py` | Project-aware memory scoping |
| `memory/config.py` | **Keep** → `src/adam/memory/config.py` | Embedding model configuration |
| `advanced_rag.py` | **Keep** → `src/adam/memory/rag.py` | BM25 + vector + graph retrieval — valuable hybrid approach |
| `activity_tracker.py` | **Keep** → `src/adam/services/activity_tracker.py` | Tracks usage for memory decay |
| `cost_monitor.py` | **Keep** → `src/adam/services/cost_monitor.py` | Cost tracking |
| `pricing_manager.py` | **Keep** → `src/adam/services/pricing_manager.py` | Real-time pricing data |
| `project_manager.py` | **Merge** into `src/adam/services/project_service.py` | Project management logic, combined with v2 project router logic |
| `screen_capture.py` | **Delete** | Not connected, replaced by structured editor state in Phase 4 |
| `errors.py` | **Keep** → `src/adam/errors.py` | Error hierarchy and retry logic |
| `tools/` directory | **Delete** | `sql_tools.py`, `web_search.py`, `code_executor.py`, `snowflake_executor.py`, `file_generator.py`, `ai_sql_generator.py`, `model_web_search.py` — these are standalone utilities not integrated into the pipeline. Re-add as needed in Phase 2+ |
| `cli/` directory | **Delete** | CLI entrypoints (`chat`, `complete`, `server`) — replaced by FastAPI. Can rebuild a CLI client in Phase 2 if wanted |
| `database/` directory | **Merge** into `src/adam/database.py` | Keep `engine.py` and `models.py`, consolidate with v2 database module. Delete `migrations.py` — fresh start per data migration plan, re-implement migration support if needed later |
| `utils/` directory | **Keep** → `src/adam/utils/` | async_utils and helpers |

#### `src/adam_v2/` — FastAPI Backend

| File/Module | Disposition | Rationale |
|---|---|---|
| `models.py` | **Keep** → `src/adam/api/models.py` | SQLAlchemy + Pydantic schemas — canonical data layer |
| `database.py` | **Keep** → `src/adam/database.py` | Merged with v1 database module |
| `routers/conversations.py` | **Keep** → `src/adam/api/routers/conversations.py` | Conversation CRUD endpoints |
| `routers/projects.py` | **Keep** → `src/adam/api/routers/projects.py` | Project endpoints |
| `routers/memories.py` | **Keep** → `src/adam/api/routers/memories.py` | Memory search/browse endpoints |
| `routers/voice.py` | **Keep** → `src/adam/api/routers/voice.py` | Voice endpoints (wired up in Phase 2D) |
| `routers/voice_streaming.py` | **Keep** → `src/adam/api/routers/voice_streaming.py` | WebSocket streaming |
| `routers/lineage.py` | **Keep** → `src/adam/api/routers/lineage.py` | dbt lineage endpoints |
| `routers/styles.py` | **Keep** → `src/adam/api/routers/styles.py` | Response style configuration |
| `routers/messages.py` | **Keep** → `src/adam/api/routers/messages.py` | Message CRUD endpoints |
| `routers/tools.py` | **Delete** | Thin wrapper, not connected |
| `routers/onboarding.py` | **Delete** | Premature — can re-add when preparing for open-source |
| `routers/file_watcher.py` | **Delete** | Not connected, revisit in Phase 4 |
| `services/llm_service.py` | **Merge** into `src/adam/services/llm_service.py` | Keep the streaming logic and DBT/SQL enhancement, but route through v1's `UnifiedLLMClient` instead of duplicating it |
| `services/memory_service.py` | **Merge** into `src/adam/memory/core.py` | Project-scoped ChromaDB wrapper folds into consolidated memory |
| `services/advanced_memory_service.py` | **Merge** into `src/adam/memory/core.py` | BM25 evaluation logic folds into consolidated memory |
| `memory_manager.py` | **Delete** | Thin wrapper superseded by consolidated memory service |
| `services/dbt_knowledge_service.py` | **Keep** → `src/adam/knowledge/dbt_knowledge.py` | dbt pattern retrieval and templates |
| `services/sql_knowledge_service.py` | **Keep** → `src/adam/knowledge/sql_knowledge.py` | SQL best practices |
| `services/voice_service.py` | **Keep** → `src/adam/services/voice_service.py` | Voice processing |
| `services/voice_websocket.py` | **Keep** → `src/adam/services/voice_websocket.py` | WebSocket handler |
| `services/voice_conversation_handler.py` | **Keep** → `src/adam/services/voice_conversation_handler.py` | Voice conversation logic |
| `services/voice_response_formatter.py` | **Keep** → `src/adam/services/voice_response_formatter.py` | TTS formatting |
| `services/response_style_service.py` | **Keep** → `src/adam/services/response_style_service.py` | Personality/style system |
| `services/fast_routing_service.py` | **Merge** into `src/adam/llm/router.py` | Combine with v1 LLM router |
| `services/intelligent_routing_service.py` | **Merge** into `src/adam/llm/router.py` | Combine with v1 LLM router |
| `services/lineage_service.py` | **Keep** → `src/adam/knowledge/lineage_service.py` | dbt lineage |
| `services/onboarding_service.py` | **Delete** | Premature |
| `services/onboarding_integration_service.py` | **Delete** | Premature |
| `services/file_watcher.py` | **Delete** | Not connected, revisit Phase 4 |
| `services/markdown_service.py` | **Keep** → `src/adam/services/markdown_service.py` | Markdown rendering |
| `services/dbt_service.py` | **Merge** into `src/adam/knowledge/dbt_knowledge.py` | dbt service logic folds into consolidated knowledge layer |
| `services/dbt_assistant.py` | **Merge** into `src/adam/knowledge/dbt_knowledge.py` | dbt assistant logic folds into knowledge layer |
| `services/dbt_integration_service.py` | **Merge** into `src/adam/knowledge/dbt_knowledge.py` | dbt integration logic folds into knowledge layer |
| `services/dbt_chat_integration.py` | **Delete** | Chat-specific dbt integration — replaced by the LLM service + knowledge layer pattern |
| `services/dbt_column_service.py` | **Merge** into `src/adam/knowledge/dbt_analyzer/column_intelligence.py` | Column-level intelligence lives in the analyzer |
| `routes/dbt.py` | **Delete** | Superseded by `routers/` endpoints — `routes/` is a duplicate routing directory |
| `routes/dbt_assistant.py` | **Delete** | Superseded by `routers/` endpoints |
| `routes/dbt_columns.py` | **Delete** | Superseded by `routers/` endpoints |
| `main.py` | **Keep** → basis for `src/adam/api/main.py` | FastAPI app entrypoint — adapt to new import paths |
| `dbt_analyzer/` directory | **Keep** → `src/adam/knowledge/dbt_analyzer/` | Full dbt toolchain — parser, lineage, optimizer, documentation, pattern learning, YAML updater |
| `tests/` directory | **Keep** → `tests/` | Migrate to top-level, update imports |
| `examples/` directory | **Keep** → `examples/` | Move to top-level |
| `pytest.ini` | **Keep** → top-level `pytest.ini` | Test configuration |
| `README.md` | **Delete** | v2-specific readme, replaced by updated root README |
| `requirements.txt`, `requirements-minimal.txt` | **Merge** into root `requirements.txt` | Consolidate all dependencies |
| `backend.log`, `server.log`, `test_markdown.html` | **Delete** | Runtime artifacts, add `*.log` to `.gitignore` |
| `data/` directory | **Delete** | v2-specific runtime data, fresh start per data migration plan |
| All `start*.sh`, `run_*.py`, `check_*.py` | **Delete** | Replace with single `main.py` entrypoint |

#### Frontend

| Item | Disposition | Rationale |
|---|---|---|
| `frontend/AdamChat/python_backend/` | **Delete** | Redundant — FastAPI backend serves this role |
| `frontend/AdamChat/client/` | **Keep** | React app, clean up unused test pages |
| `frontend/AdamChat/server/` | **Keep** | Vite dev server config |
| `frontend/AdamChat/client/src/pages/test-*.tsx` | **Delete** | Test pages no longer needed |
| `frontend/AdamChat/client/src/components/chat/sidebar-test.tsx` | **Delete** | Test component |
| `frontend/AdamChat/client/src/components/chat/streaming-voice-conversation-old.tsx` | **Delete** | Old/dead component |

#### Other

| Item | Disposition | Rationale |
|---|---|---|
| `.env` (root) | **Keep** | Already in `.gitignore` and not tracked by git. Create `.env.example` template for documentation |
| `.env.unified` | **Delete** | Consolidate into single `.env` with `.env.example` template |
| `web/` directory | **Delete** | Streamlit frontend, replaced by React |
| `cli/` (top-level) | **Delete** | Old CLI scripts, replaced by FastAPI |
| `examples/demo_dbt_project/` | **Keep** → `examples/` | Useful for testing dbt features |

### New Unified Structure

```
ADAM/
├── src/adam/                    # Single Python package
│   ├── api/                     # FastAPI layer
│   │   ├── main.py              # Single entrypoint (FastAPI app)
│   │   ├── models.py            # Pydantic request/response schemas
│   │   └── routers/             # Route handlers
│   ├── core/                    # Pipeline and state
│   │   ├── app.py               # ADAMSystem entrypoint class
│   │   └── pipeline.py          # LangGraph state machine
│   ├── memory/                  # Consolidated memory system
│   │   ├── core.py              # ChromaDB storage + retrieval + scoring
│   │   ├── network.py           # NetworkX graph relationships
│   │   ├── lifecycle.py         # Decay, reinforcement, cleanup
│   │   ├── compressor.py        # Old memory compression
│   │   ├── rag.py               # Advanced RAG (BM25 + vector + graph)
│   │   ├── project.py           # Project-scoped memory
│   │   └── config.py            # Embedding configuration
│   ├── llm/                     # LLM integration
│   │   ├── client.py            # Unified multi-provider client
│   │   ├── async_client.py      # Async variant
│   │   ├── router.py            # Intelligent model routing (merged)
│   │   ├── query_analyzer.py    # Complexity analysis
│   │   └── config.py            # Model configs and pricing
│   ├── knowledge/               # Domain knowledge
│   │   ├── dbt_knowledge.py     # dbt patterns and templates
│   │   ├── sql_knowledge.py     # SQL best practices
│   │   ├── lineage_service.py   # dbt lineage
│   │   └── dbt_analyzer/        # Full dbt toolchain
│   ├── services/                # Business logic
│   │   ├── llm_service.py       # Orchestrates LLM calls with memory/knowledge
│   │   ├── voice_service.py     # Voice processing
│   │   ├── voice_websocket.py   # WebSocket handler
│   │   ├── cost_monitor.py      # Cost tracking
│   │   ├── pricing_manager.py   # Real-time pricing
│   │   ├── activity_tracker.py  # Usage tracking
│   │   ├── response_style_service.py  # Personality/style
│   │   └── markdown_service.py  # Markdown rendering
│   ├── database.py              # SQLAlchemy engine + models
│   ├── config.py                # Unified configuration
│   ├── errors.py                # Error hierarchy
│   └── utils/                   # Helpers
├── frontend/                    # React app (cleaned up)
├── vscode-extension/            # Untouched until Phase 4
├── examples/                    # Demo projects
├── tests/                       # Migrated from adam_v2/tests
├── data/                        # Runtime data
├── .env.example                 # Template (committed)
├── .env                         # Actual config (gitignored)
└── setup.py                     # Updated package config
```

### Phase 1 Deliverable — Clarified

Phase 1 delivers a **clean, compiling, test-passing codebase** with a single architecture. It does NOT yet deliver a fully working conversational experience — that's Phase 2. Specifically:

- The FastAPI server starts and serves API endpoints
- The database schema works (projects, conversations, messages)
- The memory system initializes and can store/retrieve (ChromaDB + NetworkX)
- The LLM client can make real API calls to configured providers
- All tests pass after import path migration
- No mock implementations remain — code either works or is clearly marked as Phase 2+ work with `raise NotImplementedError`

### Data Migration

For a personal tool, existing data is acceptable to lose during consolidation:
- SQLite databases (`adam_v2.db`, `adam_unified.db`) — schema will change, start fresh
- ChromaDB collections — may be incompatible after restructuring, start fresh
- Conversation history JSON files — deleted with v1 conversation system

If any specific conversations or memories are valuable, export them before Phase 1 begins using `export_knowledge_base()` in `memory/core.py`.

### Testing Strategy

- Migrate existing tests from `src/adam_v2/tests/` to `tests/`, updating imports
- Phase 1 exit criteria: all migrated tests pass with new import paths
- Add smoke tests: server starts, creates a project, creates a conversation, memory initializes
- Each subsequent phase adds tests for its features (integration tests, not mocks)

---

## Phase 2: Conversational Core — "Actually Talk to ADAM"

**Goal:** Make ADAM a genuinely good conversational assistant — real streaming, working memory retrieval, natural voice, and a polished chat UI.

**Estimated duration:** 4-6 weeks

### 2A — Real LLM Integration & Streaming
- Replace all `NotImplementedError` stubs with actual working integrations via `UnifiedLLMClient`
- Implement true SSE (Server-Sent Events) streaming from FastAPI to frontend — currently streaming is disabled and returns full responses as one chunk
- Fix intelligent routing: add a fast rule-based pre-filter, only call the LLM router for ambiguous queries (avoids paying for Haiku classification on obvious simple queries)

### 2B — Memory That Works
- Wire the consolidated memory system end-to-end: query → search → confidence scoring → context injection → selective storage
- Surface memory in conversations — user sees when ADAM leverages past context
- Implement the reinforcement loop: memories that produce good answers get strengthened, unused ones decay
- Add a `/memory` command or UI panel to browse/search stored memories

### 2C — Conversation Flow
- Fix multi-turn context with proper sliding window: summarize old messages, keep recent ones verbatim (replace the current fragile truncation)
- Session continuity across restarts — reopen ADAM, pick up where you left off
- Project-scoped conversations — conversations belong to projects, memory is project-aware

### 2D — Voice Interface
- End-to-end voice: browser mic → Whisper transcription → LLM → TTS response
- Wire up the existing WebSocket infrastructure
- Push-to-talk model, not always-listening

### 2E — Frontend Polish
- Clean up the React chat UI — proper streaming display with markdown rendering
- Conversation sidebar with search, project switching
- Voice button, memory indicator, model/cost display
- Frontend connects to the single FastAPI backend (no python_backend)

### Testing
- Integration tests for each sub-phase: LLM call → streaming → memory storage → retrieval
- Voice end-to-end test (manual, with checklist)
- Frontend tested against real backend (not mocked API)

### Deliverable
Open ADAM, type or speak, get streamed responses that remember past conversations, with a clean UI. The core promise of "an assistant that remembers everything" actually works.

---

## Phase 3: Multi-Agent Reasoning — "The Big Brain"

**Goal:** When ADAM faces a hard problem, specialized agents from different providers collaborate to produce a thorough answer.

**Estimated duration:** 4-6 weeks

### 3A — Agent Abstraction Layer
- `Agent` protocol: provider, role (reasoner/coder/researcher/critic), system prompt, send/receive messages
- Agents are lightweight wrappers around LLM calls — structured prompts routed to different models, not microservices
- Built-in agent roles (model assignments are configurable, not hardcoded):
  - **Reasoner** — breaks down problems, plans approach (best with reasoning-capable models)
  - **Coder** — writes and reviews code (best with code-strong models)
  - **Researcher** — finds current information (best with search-enabled models)
  - **Critic** — reviews other agents' work, finds holes (any model)

### 3B — Orchestration Engine
- Coordinator decides if multi-agent is needed (simple questions stay single-model)
- Two orchestration patterns:
  - **Sequential pipeline:** Reasoner → Coder → Critic → final answer
  - **Debate/consensus:** Two agents solve independently, a third reconciles
- Built on LangGraph — a more complex state machine graph with agent nodes
- Transparent to user: shows "Thinking with 3 agents..." and streams progress

### 3C — Cross-Provider Communication
- Structured message format for sharing context between providers (not raw API passthrough)
- Shared scratchpad for intermediate results — orchestrator controls what each agent sees (prevents token waste)
- Cost guardrails: per-query budget, fallback to single-model if exceeded

### 3D — Memory Integration
- Multi-agent sessions produce rich artifacts (reasoning chains, code, critiques) — store the best parts automatically
- Similar problems later can skip the full pipeline and reuse proven solutions
- This is where the memory system's ROI becomes clear

### Testing
- Unit tests for agent protocol and message format
- Integration test: orchestrator routes a complex query through sequential pipeline
- Cost guardrail test: verify budget limits trigger fallback

### Deliverable
Ask ADAM a hard question. Multiple specialists collaborate — a reasoner plans, a coder implements, a critic finds gaps — and you get a thorough, multi-perspective answer.

---

## Phase 4: IDE & Screen Context — "ADAM Sees What You See"

**Goal:** Rebuild the VS Code extension and add screen context awareness.

**Estimated duration:** 3-4 weeks

### 4A — VS Code Extension Rebuild
- Rebuild `adam-code` as a general-purpose ADAM client that's great at dbt (current version is dbt-only, v0.3.x)
- Core features:
  - Chat panel inside VS Code connected to ADAM backend via API
  - Active file context — ADAM knows what file you're editing, language, surrounding code
  - Error context — terminal errors visible to ADAM
  - dbt-specific features carry over: model analysis, lineage, schema suggestions
- Extension is a **thin client** — all intelligence in the backend

### 4B — Screen/Editor Context
- Send structured editor state instead of literal screen capture
- VS Code extension sends: file path, content, cursor position, open terminals, diagnostics, git status
- Backend includes context in prompts when relevant (not always — only when query relates to current work)

### 4C — Proactive Assistance (Stretch Goal)
- Pattern detection: repeated save-and-error cycles trigger an offer to help
- dbt-specific: auto-check `.sql` model files against best practices on save
- Opt-in and subtle — notification, not interruption

### Testing
- Extension integration test: verify context is sent to backend correctly
- Backend test: verify editor context appears in LLM prompts when relevant

### Deliverable
Coding in VS Code, hit a dbt error, and ADAM already has the context — your model file, the error, your conventions — and gives a targeted fix without copy-pasting.

---

## Phase Sequencing

```
Phase 1 ────→ Phase 2 ────→ Phase 3 ────→ Phase 4
 (2-3 wks)     (4-6 wks)     (4-6 wks)     (3-4 wks)

 Consolidate   Converse      Reason        IDE
 one backend   streaming     multi-agent   VS Code
 delete dead   memory e2e    orchestrate   context
 clean arch    voice/UI      cost guard    proactive
 tests pass    tests pass    tests pass    tests pass
```

**Total estimated: 13-19 weeks** (3-5 months) at steady evenings/weekends pace.

Each phase is independently valuable:
- After Phase 1: Clean, working codebase — a foundation you can build on without fighting
- After Phase 2: A genuinely useful AI assistant with memory — the product works
- After Phase 3: A unique multi-agent reasoning system — the differentiator
- After Phase 4: IDE integration — ADAM meets you where you work

---

## Principles

1. **Ship complete, not ambitious.** Every phase delivers a working product.
2. **One of everything.** One backend, one memory system, one LLM client. No duplication.
3. **Delete before adding.** Remove dead code before writing new code.
4. **Test at boundaries.** Integration tests between real systems, not mocks.
5. **Cost-aware by default.** Every LLM call has a cost estimate and budget check.
6. **Memory earns its keep.** Memory is only valuable if retrieval is fast and relevant. Measure hit rate.
7. **Models are configurable, not hardcoded.** Agent roles reference capabilities, not specific model names.
