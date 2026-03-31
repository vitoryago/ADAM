# Deep Discussion Mode — Project Notebook

> Living document capturing all decisions, architecture, and status for the Deep Discussion feature.

---

## Vision

A dedicated interface where the user configures which AI model powers each agent role, watches them work in real-time, and gets a thorough, multi-perspective answer. Deep Discussion is designed for **thoroughness over speed** — users deliberately trigger it when a problem deserves multiple rounds of feedback.

---

## Status

| Milestone | Status | Date | Commits |
|-----------|--------|------|---------|
| Design spec (backend + React) | Done | 2026-03-25 | `e3727d2` |
| Implementation plan (backend + React) | Done | 2026-03-25 | `150a9fc` |
| Phase 5: Backend + React implementation | Done | 2026-03-25 | `d92573d`..`5ba4423` (11 commits) |
| Design spec (VS Code extension) | Done | 2026-03-25 | `1a2ebe8` |
| Implementation plan (VS Code extension) | Done | 2026-03-25 | `ee83623` |
| VS Code extension implementation | Done | 2026-03-25 | `5514fc0`..`e3783c7` (7 commits) |
| Local model integration spec | Done | 2026-03-31 | `306dbb3` |
| Local model integration plan | Done | 2026-03-31 | `9f4b793` |
| Local model integration implementation | Done | 2026-03-31 | `f1f5a17`..`98571e7` (8 commits) |
| Manual end-to-end testing | **Not done** | — | — |
| README update | Not done | — | — |

---

## Key Decisions (from brainstorming)

| Question | Decision | Rationale |
|----------|----------|-----------|
| Where should it live in UI? | Both: sidebar page + "Go Deep" from chat | Flexibility — fresh sessions or escalate mid-conversation |
| Model config approach | Smart defaults + manual override | Frictionless but full control |
| Providers for v1 | Grok, Claude, OpenAI, Gemini | Big 4 cloud providers only |
| Custom/local models (Ollama, vLLM) | Generic LocalModelProvider (OpenAI-compatible) | Backend-agnostic, future-proof for LoRA fine-tuning |
| Local model UX | "Prefer Local" toggle in Advanced settings | Clean, non-intrusive — cloud-only by default |
| User intervention | Watch-only with replay for v1 | Simpler; mid-discussion intervention deferred |
| New orchestration pattern | Peer Review (with rebuttal loop) | Combines best of Round Table + Peer Review |
| Number of new patterns | 1 for v1 (Peer Review) | Ship 3 total: Sequential, Debate, Peer Review |
| Default model assignments | Reasoner=Grok MA, Coder=Opus, Critic=GPT-5.4, Synth=Sonnet | Provider diversity for different perspectives |
| Synthesizer model | claude-sonnet-4-6 (not GPT) | User preference — Sonnet/Gemini better at synthesis |
| LiteLLM | **NO** — security concerns | Keep direct SDK integrations per provider |
| VS Code entry point | Separate activity bar icon ($(hubot)) | Clean separation from existing chat |
| VS Code "Go Deep" from chat | Deferred | Separate icon only for v1 |
| VS Code config UX | Minimal + expandable Advanced | Smart defaults handle it, advanced for tweaking |
| VS Code live view | Simplified (compact agent rows) | Narrow sidebar, focus on progress + final answer |

---

## Architecture Overview

```
User clicks "Deep Discussion" (React sidebar or VS Code activity bar)
    ↓
Session Configuration (pick pattern, optionally override models/budget)
    ↓
POST /api/deep-discussion/sessions → creates session with smart defaults
PUT  /api/deep-discussion/sessions/{id}/config → (if overrides)
POST /api/deep-discussion/sessions/{id}/start → SSE stream begins
    ↓
DeepDiscussionSession delegates to selected pattern:
    ├── Sequential: Reasoner → Coder → Critic → Synthesizer
    ├── Debate: Debater A → Debater B → Reconciler
    └── Peer Review: Producer → Parallel Reviews → Rebuttal → Parallel React → Synthesize
    ↓
SSE events stream to frontend (step_start, agent_start, agent_chunk, agent_done, session_complete)
    ↓
Results persisted to deep_discussion_sessions table
```

---

## Peer Review Pattern (New)

The signature pattern for Deep Discussion. 5 steps with a rebuttal loop:

```
Step 1: PRODUCE
  Reasoner analyzes the problem → PLAN entry

Step 2: REVIEW (parallel via asyncio.gather)
  Coder reviews → CODE entry
  Critic reviews → CRITIQUE entry

Step 3: REBUTTAL
  Reasoner reads both reviews, responds → REBUTTAL entry

Step 4: REACT (parallel via asyncio.gather)
  Coder reacts to rebuttal → REACTION entry
  Critic reacts to rebuttal → REACTION entry

Step 5: SYNTHESIZE
  Synthesizer merges everything → SYNTHESIS entry
```

**Custom per-step visibility** — bypasses Scratchpad.build_context_for_agent() and builds prompts manually. Each reviewer in Step 4 sees only their own review (not the other reviewer's).

**Budget enforcement** — CostGuard checks between each step. If exceeded, skips to emergency synthesis on partial results.

---

## Model Roster (as of 2026-03-24)

| Provider | Model ID | Display Label | Best For |
|----------|----------|---------------|----------|
| X.AI | `grok-4.20-multi-agent-0309` | Grok Multi-Agent | Multi-agent (default Reasoner) |
| X.AI | `grok-4.20-0309-reasoning` | Grok Reasoning | Complex analysis |
| X.AI | `grok-4.20-0309-non-reasoning` | Grok Standard | Standard queries |
| Anthropic | `claude-opus-4-6` | Claude Opus | Code generation (default Coder) |
| Anthropic | `claude-sonnet-4-6` | Claude Sonnet | Synthesis (default Synthesizer) |
| Anthropic | `claude-haiku-4-5` | Claude Haiku | Fast/cheap, routing |
| OpenAI | `gpt-5.4-2026-03-05` | GPT-5.4 | General analysis (default Critic) |
| OpenAI | `gpt-5.4-mini-2026-03-17` | GPT-5.4 Mini | Fast/cheap |
| Google | `gemini-3.1-pro-preview` | Gemini Pro | Analysis, multi-agent |
| Google | `gemini-3-flash-preview` | Gemini Flash | Fast/cheap |

---

## Backend Components

### New Module: `src/adam/deep_discussion/`

| File | Purpose |
|------|---------|
| `config.py` | SessionConfig dataclass, get_smart_defaults(), AVAILABLE_MODELS |
| `session.py` | DeepDiscussionSession — lifecycle, run/run_stream, role-to-agent mapping per pattern |

### New Pattern: `src/adam/agents/patterns/peer_review.py`

| Function | Purpose |
|----------|---------|
| `run_peer_review_pipeline()` | Non-streaming 5-step pipeline with parallel execution |
| `stream_peer_review_pipeline()` | Streaming variant with SSE events |

### New API Router: `src/adam/api/routers/deep_discussion.py`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/deep-discussion/sessions` | POST | Create session with smart defaults |
| `/api/deep-discussion/sessions/{id}/config` | PUT | Update model assignments, pattern, budget |
| `/api/deep-discussion/sessions/{id}/start` | POST | Begin discussion (SSE stream) |
| `/api/deep-discussion/sessions/{id}` | GET | Get session state and results |
| `/api/deep-discussion/sessions/{id}/replay` | POST | Clone session for re-run |
| `/api/deep-discussion/sessions/from-conversation/{conv_id}` | POST | Create from chat (last 10 messages as context) |
| `/api/deep-discussion/sessions` | GET | List sessions by project |

### Modified Files

| File | Change |
|------|--------|
| `src/adam/llm/config.py` | Added GEMINI provider, 10 new model configs |
| `src/adam/llm/async_client.py` | Added GEMINI provider, gemini- prefix routing |
| `src/adam/llm/router.py` | Updated ModelTier to new model names |
| `src/adam/agents/scratchpad.py` | Added REBUTTAL, REACTION entry types |
| `src/adam/agents/orchestrator.py` | Added PEER_REVIEW to OrchestrationPattern |
| `src/adam/api/models.py` | Added DeepDiscussionSessionDB + Pydantic schemas |
| `src/adam/api/main.py` | Registered deep_discussion router |
| `.env.example` | Added GEMINI_API_KEY |

### Database

New table: `deep_discussion_sessions` — auto-created via metadata.create_all() in lifespan hook.

| Column | Type | Purpose |
|--------|------|---------|
| id | UUID PK | Session identifier |
| project_id | FK → projects | Project scope |
| conversation_id | FK → conversations (nullable) | If escalated from chat |
| question | TEXT | User's question |
| pattern | VARCHAR(20) | sequential/debate/peer_review |
| model_assignments | JSON | {reasoner: model, coder: model, ...} |
| budget | FLOAT | Budget limit (default $2.00) |
| total_cost | FLOAT | Actual spend |
| status | VARCHAR(20) | configuring/running/completed/failed |
| result | TEXT (nullable) | Final synthesized answer |
| scratchpad_data | JSON (nullable) | Full agent contributions |
| created_at | TIMESTAMP | |
| completed_at | TIMESTAMP (nullable) | |

---

## React Frontend Components

| File | Purpose |
|------|---------|
| `pages/deep-discussion.tsx` | Page with sidebar + main content area |
| `components/deep-discussion/config-screen.tsx` | Question + pattern + agent grid + budget + start |
| `components/deep-discussion/live-view.tsx` | SSE streaming, progress bar, agent cards |
| `components/deep-discussion/agent-card.tsx` | Expandable card per agent |
| `components/deep-discussion/progress-bar.tsx` | Step progress visualization |
| `components/deep-discussion/model-selector.tsx` | Dropdown grouped by provider |
| `components/deep-discussion/pattern-selector.tsx` | 3 clickable cards |
| `components/deep-discussion/go-deep-modal.tsx` | Modal for "Go Deep" from chat |
| `lib/deep-discussion-api.ts` | API client for all 7 endpoints |

Routes: `/project/:projectId/deep-discussion` and `/project/:projectId/deep-discussion/:sessionId?`

---

## VS Code Extension (v0.5.0)

| File | Purpose |
|------|---------|
| `src/providers/deepDiscussionProvider.ts` | WebviewViewProvider for the panel |
| `media/deep-discussion.js` | Webview UI (config/live/complete modes) |
| `media/deep-discussion.css` | Styles with VS Code CSS variables |
| `src/client/adamClient.ts` | 6 new API methods added |
| `src/extension.ts` | Registers provider + command |
| `package.json` | Activity bar, view, command, keybinding, config |

**Entry point:** Separate activity bar icon ($(hubot)), keybinding `Cmd+Alt+D`

**Config screen:** Minimal — question + pattern + start button. "Advanced" expands for model overrides + budget slider.

**Live view:** Compact agent rows (collapsed by default), progress bar, cancel link. Cost tracking.

**Complete view:** Final answer prominent, expandable agent contributions, "Run Again" button.

**Error handling:** AbortController for cancellation, SSE drop recovery (fetch session state), "Starting..." disabled state.

---

## Test Coverage

**391 backend tests passing** (1 pre-existing failure: missing `ruamel` module for dbt analyzer)

| Test File | Tests | What it covers |
|-----------|-------|---------------|
| `test_gemini_provider.py` | 17 | Provider routing, model roster |
| `test_scratchpad_entry_types.py` | 14 | REBUTTAL/REACTION types, PEER_REVIEW enum |
| `test_peer_review_pattern.py` | 17 | Full pipeline, budget stops, parallel execution, custom visibility |
| `test_deep_discussion_session.py` | 41 | SessionConfig, smart defaults, role mapping, lifecycle |
| `test_deep_discussion_db.py` | 8 | SQLAlchemy model, Pydantic schemas |
| `test_deep_discussion_api.py` | 18 | All 7 API endpoints |
| `test_deep_discussion_integration.py` | 12 | End-to-end lifecycle, replay, pattern consistency |

**Frontend:** TypeScript compilation verified (React + VS Code extension). No component tests yet.

---

## How to Run

### Backend
```bash
cd /Users/vitoryago/ADAM
pip install -e .
python -m adam.api.main
# → http://localhost:8000 (health: /health, docs: /api/docs)
```

### React Frontend
```bash
cd frontend/AdamChat
npm install
npm run dev
# → http://localhost:5173
# Navigate to project → "Deep Discussion" in sidebar
```

### VS Code Extension
```bash
cd vscode-extension/adam-code
npm install && npm run compile
# In VS Code: Cmd+Shift+P → "Developer: Install Extension from Location..." → select adam-code folder
# Look for 🤖 icon in activity bar, or Cmd+Alt+D
```

### Required .env
```
XAI_API_KEY=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

Smart defaults span all 4 providers. If you only have some keys, override model assignments in Advanced settings.

---

## What's Next (Future Enhancements)

| Feature | Priority | Notes |
|---------|----------|-------|
| Manual end-to-end testing | **High** | Not yet tested with real LLM calls |
| LoRA fine-tuning pipeline (Critic first) | **High** | Synthetic data → LoRA → quantize → serve via Ollama. TurboQuant-style 4-bit for M4 Pro 48GB |
| Mid-discussion intervention | Medium | Pause between steps, inject context, redirect agents |
| ~~Custom/local model providers~~ | **Done** | LocalModelProvider with auto-discovery, "Prefer Local" toggle, zero-cost tracking |
| VS Code "Go Deep" from chat | Medium | Button in chat input to escalate conversation |
| New orchestration patterns | Low | Round Table, Mosaic, etc. |
| Memory integration | Low | Auto-store best multi-agent artifacts, reuse for similar problems |
| README update | Low | Current README references old v1 structure |

---

## Design & Plan Documents

| Document | Path |
|----------|------|
| Backend + React spec | `docs/superpowers/specs/2026-03-25-deep-discussion-design.md` |
| Backend + React plan | `docs/superpowers/plans/2026-03-25-deep-discussion.md` |
| VS Code spec | `docs/superpowers/specs/2026-03-25-vscode-deep-discussion-design.md` |
| VS Code plan | `docs/superpowers/plans/2026-03-25-vscode-deep-discussion.md` |

---

## Technical Notes

- **Gemini integration** uses Google's OpenAI-compatible endpoint (`generativelanguage.googleapis.com/v1beta/openai/`) with `AsyncOpenAI` client — same pattern could enable future custom providers
- **Peer Review bypasses `build_context_for_agent()`** — builds prompts manually per step (same approach as Debate pattern)
- **Role-to-agent mapping per pattern:** Sequential/Peer Review are pass-through; Debate remaps `reasoner→debater_a, coder→debater_b, critic→reconciler`
- **SSE event vocabulary:** `session_start, step_start, agent_start, agent_chunk, agent_done, step_complete, session_complete, session_error`
- **Budget enforcement:** CostGuard checks between steps. If exceeded, emergency synthesis on partial results (status remains "completed", not "failed")
- **Streaming cost tracking:** After streaming completes, estimate tokens via word count heuristic (words × 1.3), calculate cost via PricingManager
