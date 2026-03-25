# Deep Discussion Mode — Design Spec

**Date:** 2026-03-25
**Status:** Draft
**Depends on:** Phase 3 multi-agent system (complete), Phase 4 IDE integration (complete)

## Context

ADAM's multi-agent system (Phase 3) works but is opaque and automatic. The Orchestrator auto-selects patterns and all agents use the same provider tier. Users have no visibility into individual agent contributions and no control over which models power each role.

Deep Discussion Mode turns this into a **user-controlled, transparent orchestration UI** designed for thoroughness over speed. Users deliberately trigger it when a problem deserves multiple perspectives and rounds of feedback.

### Audience
- ADAM's developer/user who wants deep, multi-perspective analysis
- Not for quick questions — regular chat handles those

### Core Vision
A dedicated interface where the user configures which AI model powers each agent role, watches them work in real-time, and gets a thorough, multi-perspective answer.

---

## Scope

### In Scope (v1)
- Two entry points: sidebar page (fresh session) + "Go Deep" button in chat (escalate with context)
- Session configuration: per-agent model assignment with smart defaults + manual override
- 4 LLM providers: X.AI (Grok), Anthropic (Claude), OpenAI (GPT), Google (Gemini)
- 3 orchestration patterns: Sequential, Debate, Peer Review (new)
- Peer Review pattern with rebuttal loop (Producer → Parallel Reviews → Rebuttal → Reviewer React → Synthesize)
- Real-time live discussion view with agent cards, progress bar, cost tracking
- Session persistence and replay (re-run with tweaked settings)
- React web app UI (primary frontend)
- Backend API designed to serve both React and VS Code extension
- Model roster update across entire codebase

### Out of Scope (v1)
- Custom/local model providers (Ollama, vLLM, LM Studio)
- Mid-discussion intervention (pause, redirect, inject context)
- VS Code extension UI (backend API supports it, UI deferred)
- New patterns beyond Peer Review
- Proactive suggestions (e.g., ADAM suggesting "Go Deep" automatically)

---

## Architecture

### Approach: Extend the Orchestrator

Deep Discussion builds on the existing multi-agent infrastructure (Agent, Scratchpad, CostGuard, patterns). A new `DeepDiscussionSession` layer wraps the Orchestrator with session configuration, higher budgets, and the new Peer Review pattern.

### New Module: `src/adam/deep_discussion/`

```
src/adam/deep_discussion/
├── __init__.py
├── session.py          # DeepDiscussionSession — lifecycle management
├── config.py           # SessionConfig — per-agent model assignments, pattern, budget
└── patterns/
    └── peer_review.py  # New Peer Review pattern
```

### Backend Components

**`SessionConfig`** (dataclass):
- `question: str` — the user's topic/question
- `pattern: str` — "sequential", "debate", or "peer_review"
- `model_assignments: Dict[str, str]` — maps role → model ID (e.g., `{"reasoner": "grok-4.20-multi-agent-0309"}`)
- `budget: float` — default $2.00, range $0.50–$5.00
- `conversation_id: Optional[str]` — if escalated from chat

**`DeepDiscussionSession`** (class):
- Holds SessionConfig, manages lifecycle (configuring → running → completed → failed)
- Creates Agent instances with per-agent model overrides from config
- Delegates execution to the selected pattern (Sequential, Debate, or Peer Review)
- Streams SSE events for real-time UI updates
- Persists session and results to database

**Smart Defaults:** When creating a session, auto-suggest models based on available providers:

| Agent Role | Default Model | Rationale |
|-----------|--------------|-----------|
| Reasoner | `grok-4.20-multi-agent-0309` | Purpose-built for multi-agent collaboration |
| Coder | `claude-opus-4-6` | Best at code generation and review |
| Critic | `gpt-5.4-2026-03-05` | Strong general model, different provider for diverse perspective |
| Synthesizer | `claude-sonnet-4-6` | Excellent at coherent, well-structured writing |

Users can override any assignment before starting.

---

## Peer Review Pattern

The new orchestration pattern for Deep Discussion. A structured multi-round review process with rebuttal.

### Flow

```
Step 1: PRODUCE
  Reasoner analyzes the problem, creates structured plan/analysis

Step 2: REVIEW (parallel)
  Coder reviews Reasoner's output (implementation perspective)
  Critic reviews Reasoner's output (quality/gaps perspective)
  → Both run in parallel, both see Reasoner's full output

Step 3: REBUTTAL
  Reasoner reads both reviews, responds to their points
  → Addresses concerns, pushes back where appropriate, revises plan

Step 4: REACT (parallel)
  Coder reacts to Reasoner's rebuttal (accept, double down, new issues)
  Critic reacts to Reasoner's rebuttal (accept, double down, new issues)
  → Both run in parallel

Step 5: SYNTHESIZE
  Synthesizer merges all contributions into final coherent answer
```

### New EntryTypes

The Peer Review pattern requires two new `EntryType` values in `scratchpad.py`:
- `REBUTTAL = "rebuttal"` — Producer's response to reviewer feedback (Step 3)
- `REACTION = "reaction"` — Reviewer's follow-up to the rebuttal (Step 4)

### Agent Visibility Rules (Scratchpad)
- Step 1: Reasoner sees only the user's question
- Step 2: Reviewers see the question + Reasoner's output
- Step 3: Reasoner sees question + own output + both reviews
- Step 4: Reviewers see question + Reasoner's output + own review + Reasoner's rebuttal
- Step 5: Synthesizer sees everything

### Scratchpad Visibility Implementation

The existing `Scratchpad.build_context_for_agent()` uses a **role-based** visibility model (e.g., Reasoner always sees nothing, Critic always sees plan+code+research). This is incompatible with Peer Review, where the same role (Reasoner) needs different visibility at different steps.

**Solution:** The Peer Review pattern **bypasses `build_context_for_agent()`** and constructs prompts manually per step using `scratchpad.entries` directly. Each step function in `peer_review.py` builds a custom context string by filtering entries by `EntryType`:

```
Step 1 (Produce):  context = query only
Step 2 (Review):   context = query + entries where type == PLAN
Step 3 (Rebuttal): context = query + entries where type in (PLAN, CRITIQUE, CODE)
Step 4 (React):    context = query + entries where type in (PLAN, CRITIQUE, CODE, REBUTTAL)
                    + filter to show each reviewer only their own CRITIQUE entry
Step 5 (Synthesize): context = all entries
```

This is the same approach the Debate pattern uses — it builds custom prompts in `debate.py` rather than relying on `build_context_for_agent()`. The existing Sequential pattern is the only one that uses the role-based method.

### Role-to-Agent Mapping Per Pattern

The `model_assignments` dict uses 4 canonical keys: `reasoner`, `coder`, `critic`, `synthesizer`. Each pattern maps these to its agent slots:

| Canonical Key | Sequential | Debate | Peer Review |
|--------------|-----------|--------|-------------|
| `reasoner` | Reasoner | Debater A | Producer (Reasoner) |
| `coder` | Coder | Debater B | Reviewer 1 (Coder) |
| `critic` | Critic | Reconciler | Reviewer 2 (Critic) |
| `synthesizer` | Synthesizer | (not used) | Synthesizer |

For Debate: `reasoner` and `coder` become the two opposing perspectives, `critic` becomes the reconciler who bridges them. `synthesizer` is unused because the Reconciler produces the final output.

---

## Gemini Provider Integration

Extend `AsyncLLMClient` to support Google Gemini via the OpenAI-compatible endpoint.

### Implementation
- Add `GEMINI` to `AsyncLLMProvider` enum
- Initialize using `AsyncOpenAI` client pointed at `https://generativelanguage.googleapis.com/v1beta/openai/` with `GEMINI_API_KEY`
- Add `gemini-` prefix branch to `_get_provider_for_model()`: `elif model.startswith('gemini-'): return AsyncLLMProvider.GEMINI`
- Reuse existing `_complete_openai` / `_stream_openai` methods (same API format)
- Add `ModelProvider.GEMINI` to `llm/config.py`

### Configuration
- New env var: `GEMINI_API_KEY`
- Add to `.env.example`

---

## Model Roster Update

Update all model references across the codebase to current versions.

### Current → Updated

**Grok (X.AI):**
- `grok-4-fast-reasoning` → `grok-4.20-0309-reasoning`
- `grok-4-fast-non-reasoning` → `grok-4.20-0309-non-reasoning`
- (new) `grok-4.20-multi-agent-0309`

**Claude (Anthropic):**
- `claude-3-5-haiku-20241022` → `claude-haiku-4-5`
- (add) `claude-opus-4-6`, `claude-sonnet-4-6`

**OpenAI:**
- (add) `gpt-5.4-2026-03-05`, `gpt-5.4-mini-2026-03-17`

**Gemini (Google):**
- (new) `gemini-3.1-pro-preview`, `gemini-3-flash-preview`

### Files Affected
- `src/adam/llm/router.py` — ModelTier enum, default models
- `src/adam/llm/async_client.py` — provider detection, Gemini client
- `src/adam/llm/config.py` — model configurations and pricing
- `src/adam/agents/orchestrator.py` — default budget
- `src/adam/deep_discussion/config.py` — smart defaults
- `vscode-extension/adam-code/` — default model settings
- `.env.example` — add GEMINI_API_KEY
- Tests — update model name assertions

---

## API Endpoints

New router: `/api/deep-discussion/`

### Endpoints

**`POST /api/deep-discussion/sessions`**
Create a new Deep Discussion session. Returns session with smart default model assignments.

Request body:
```json
{
  "project_id": "uuid",
  "question": "Review my dbt incremental model...",
  "pattern": "peer_review",
  "conversation_id": null
}
```

Response:
```json
{
  "id": "uuid",
  "status": "configuring",
  "question": "...",
  "pattern": "peer_review",
  "model_assignments": {
    "reasoner": "grok-4.20-multi-agent-0309",
    "coder": "claude-opus-4-6",
    "critic": "gpt-5.4-2026-03-05",
    "synthesizer": "claude-sonnet-4-6"
  },
  "budget": 2.00
}
```

**`PUT /api/deep-discussion/sessions/{session_id}/config`**
Update model assignments, pattern, or budget before starting.

**`POST /api/deep-discussion/sessions/{session_id}/start`**
Begin the discussion. Returns SSE stream of agent events.

SSE event types:
- `session_start` — pattern, agents
- `step_start` — step name (produce, review, rebuttal, react, synthesize)
- `agent_start` — agent role, model, step
- `agent_chunk` — streaming token from agent
- `agent_complete` — agent role, content, cost, tokens
- `step_complete` — step name, step cost
- `session_complete` — final result, total cost, total time
- `agent_error` — agent role, error message (single agent failed)
- `session_error` — error message, partial results if any

**`GET /api/deep-discussion/sessions/{session_id}`**
Get session state, results, and full scratchpad data.

**`POST /api/deep-discussion/sessions/{session_id}/replay`**
Create a new session pre-filled with this session's config (for tweaking and re-running).

**`POST /api/deep-discussion/sessions/from-conversation/{conversation_id}`**
Create a session from an existing chat conversation. Carries conversation context as additional input to the agents.

Conversation context extraction: takes the **last 10 messages** from the conversation and formats them as a context block prepended to the user's question. This context block is stored in `SessionConfig.conversation_context` and injected into the Scratchpad's query field as:
```
CONVERSATION CONTEXT:
[user]: ...
[assistant]: ...
---
QUESTION: <user's deep discussion question>
```

**`GET /api/deep-discussion/sessions?project_id={id}`**
List all sessions for a project.

---

## Data Model

### New Table: `deep_discussion_sessions`

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID (PK) | Session identifier |
| `project_id` | UUID (FK → projects) | Project scope |
| `conversation_id` | UUID (FK → conversations, nullable) | If escalated from chat |
| `question` | TEXT | User's question/topic |
| `pattern` | VARCHAR | "sequential" / "debate" / "peer_review" |
| `model_assignments` | JSON | Role → model ID mapping |
| `budget` | FLOAT | Budget limit |
| `total_cost` | FLOAT | Actual spend |
| `status` | VARCHAR | "configuring" / "running" / "completed" / "failed" |
| `result` | TEXT (nullable) | Final synthesized answer |
| `scratchpad_data` | JSON (nullable) | Full agent contributions for display/replay |
| `created_at` | TIMESTAMP | Creation time |
| `completed_at` | TIMESTAMP (nullable) | Completion time |

No changes to existing tables.

### Table Creation
The new table uses SQLAlchemy declarative base, same as existing models in `src/adam/api/models.py`. The `DeepDiscussionSession` SQLAlchemy model is added to `models.py` alongside `Project`, `Conversation`, and `Message`. The table is auto-created via `metadata.create_all()` in the existing lifespan hook — no Alembic migration needed (ADAM uses auto-create for development).

### `OrchestrationPattern` Enum Update
Add `PEER_REVIEW = "peer_review"` to the existing `OrchestrationPattern` enum in `src/adam/agents/orchestrator.py`. The Orchestrator's `select_pattern()` method is not changed — Deep Discussion always receives the pattern from user config, not from auto-detection.

---

## Failure Modes

### Per-Agent Failure
If a single agent fails (e.g., API timeout, rate limit), the session **continues with partial results**. The failed agent's step is marked with an `agent_error` SSE event, and subsequent steps that depend on that agent's output proceed without it. The Synthesizer notes the gap.

### Provider Unavailability
At session creation time, smart defaults only suggest models from **configured providers** (those with valid API keys in `.env`). If a user manually assigns an unconfigured provider, the `/start` endpoint validates all assignments before beginning and returns a 400 error listing which providers are unavailable.

### Mid-Session Budget Exhaustion
CostGuard checks budget between each step of any pattern. If `scratchpad.is_over_budget()` is true after a step completes:
1. Skip remaining steps
2. Run Synthesizer on whatever results exist so far (using the cheapest available model)
3. Emit `session_complete` with a `budget_exceeded: true` flag
4. Mark session status as `completed` (not `failed` — partial results are still valuable)

### Streaming Cost Tracking
`Agent.think_stream()` currently yields raw chunks without tracking cost/tokens. For Deep Discussion:
- After streaming completes, the pattern collects the full response text and estimates tokens using a tokenizer (tiktoken for OpenAI/Gemini, approximate for others)
- Cost is calculated from the estimated tokens using `PricingManager`
- The `agent_complete` SSE event is emitted after streaming with the calculated cost/tokens
- This is an approximation — exact usage data is not available from all providers during streaming

---

## Frontend

### Technology
- React 18 + TypeScript (matches existing AdamChat)
- Radix UI components + Tailwind CSS (existing design system)
- TanStack React Query for server state
- Existing white + green theme

### New Pages / Components

**`/deep-discussion` page** — main entry from sidebar
- Session list (past sessions)
- "New Session" button → config screen

**`DeepDiscussionConfig` component** — session configuration
- Question input (textarea)
- Pattern selector (3 cards: Sequential, Debate, Peer Review)
- Agent configuration grid (4 cards with model dropdowns)
- Budget slider ($0.50–$5.00, default $2.00)
- "Start Deep Discussion" button

**`DeepDiscussionLive` component** — live discussion view
- Header: question, pattern, running status, cost counter
- Progress bar: steps of selected pattern with current step highlighted
- Agent cards: expandable, show role/model/cost/status, stream content
- Completed state: final answer at top, agent cards below for transparency
- "Replay with changes" button

**"Go Deep" integration in chat:**
- Button in chat input bar (next to send)
- Opens `DeepDiscussionConfig` as a modal/drawer
- Pre-fills conversation context
- On completion, posts result back into chat as a message

### Model Selector Component
Shared dropdown for picking models, grouped by provider. Display labels are shortened; full model IDs (with date suffixes) are stored in config and sent to providers.

| Provider | Display Label | Full Model ID |
|----------|-------------|---------------|
| X.AI | Grok Multi-Agent | `grok-4.20-multi-agent-0309` |
| X.AI | Grok Reasoning | `grok-4.20-0309-reasoning` |
| X.AI | Grok Standard | `grok-4.20-0309-non-reasoning` |
| Anthropic | Claude Opus | `claude-opus-4-6` |
| Anthropic | Claude Sonnet | `claude-sonnet-4-6` |
| Anthropic | Claude Haiku | `claude-haiku-4-5` |
| OpenAI | GPT-5.4 | `gpt-5.4-2026-03-05` |
| OpenAI | GPT-5.4 Mini | `gpt-5.4-mini-2026-03-17` |
| Google | Gemini Pro | `gemini-3.1-pro-preview` |
| Google | Gemini Flash | `gemini-3-flash-preview` |

---

## Testing Strategy

- **Unit tests:** SessionConfig validation, smart defaults, Peer Review scratchpad visibility rules
- **Pattern tests:** Peer Review full flow with mocked LLM (verify 5-step execution, parallel reviews, rebuttal sees reviews)
- **API tests:** All 6 endpoints, SSE streaming, session state transitions
- **Integration tests:** End-to-end session from creation to completion with real LLM calls
- **Frontend:** Component tests for config screen and live view (React Testing Library)

---

## Design Principles

1. **Thoroughness over speed** — Deep Discussion is for problems that deserve time. No latency shortcuts.
2. **Transparency** — Every agent's contribution is visible. The user sees how the answer was built.
3. **Provider diversity** — Default assignments spread across providers for diverse perspectives.
4. **Extend, don't replace** — Builds on existing Agent/Scratchpad/CostGuard. Regular chat multi-agent is unaffected.
5. **Ship complete** — Follows ADAM's "Foundation First" principle. Everything in v1 works end-to-end.
