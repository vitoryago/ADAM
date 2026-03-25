# Deep Discussion Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a user-controlled multi-agent orchestration UI where users configure per-agent model assignments, watch agents reason in real-time, and get thorough multi-perspective answers.

**Architecture:** Extends the existing Phase 3 multi-agent system (Agent, Scratchpad, CostGuard) with a new `deep_discussion/` module for session management and the Peer Review pattern. Adds Gemini as a 4th LLM provider. New React page with configuration and live discussion views. Backend API with SSE streaming.

**Tech Stack:** FastAPI (backend), React 18 + TypeScript + Radix UI + Tailwind (frontend), SQLAlchemy (data model), SSE (streaming), AsyncOpenAI (Gemini integration)

**Spec:** `docs/superpowers/specs/2026-03-25-deep-discussion-design.md`

---

### Task 1: Update Model Roster and Add Gemini Provider

**Files:**
- Modify: `src/adam/llm/config.py:14-18` (ModelProvider enum)
- Modify: `src/adam/llm/config.py:44-307` (LLMConfig models dict and defaults)
- Modify: `src/adam/llm/async_client.py:68-73` (AsyncLLMProvider enum)
- Modify: `src/adam/llm/async_client.py:95-147` (_initialize_async_clients)
- Modify: `src/adam/llm/async_client.py:447-457` (_get_provider_for_model)
- Modify: `src/adam/llm/router.py:28-36` (ModelTier enum)
- Modify: `src/adam/llm/router.py:144-148` (MODEL_MAPPING)
- Modify: `.env.example`
- Test: `tests/test_gemini_provider.py`

- [ ] **Step 1: Write test for Gemini provider detection**

```python
# tests/test_gemini_provider.py
"""Tests for Gemini provider integration and model roster update."""
import pytest
from adam.llm.async_client import AsyncLLMProvider, AsyncLLMClient
from adam.llm.config import ModelProvider, LLMConfig


class TestGeminiProviderDetection:
    def test_gemini_model_routes_to_gemini_provider(self):
        client = AsyncLLMClient()
        provider = client._get_provider_for_model("gemini-3.1-pro-preview")
        assert provider == AsyncLLMProvider.GEMINI

    def test_gemini_flash_routes_to_gemini_provider(self):
        client = AsyncLLMClient()
        provider = client._get_provider_for_model("gemini-3-flash-preview")
        assert provider == AsyncLLMProvider.GEMINI

    def test_grok_still_routes_correctly(self):
        client = AsyncLLMClient()
        assert client._get_provider_for_model("grok-4.20-multi-agent-0309") == AsyncLLMProvider.XAI

    def test_claude_still_routes_correctly(self):
        client = AsyncLLMClient()
        assert client._get_provider_for_model("claude-opus-4-6") == AsyncLLMProvider.ANTHROPIC

    def test_gpt_still_routes_correctly(self):
        client = AsyncLLMClient()
        assert client._get_provider_for_model("gpt-5.4-2026-03-05") == AsyncLLMProvider.OPENAI


class TestModelRoster:
    def test_gemini_provider_in_enum(self):
        assert hasattr(ModelProvider, "GEMINI")

    def test_config_has_gemini_api_key_slot(self):
        config = LLMConfig()
        assert ModelProvider.GEMINI in config.api_keys

    def test_config_has_new_grok_models(self):
        config = LLMConfig()
        assert "grok-4.20-multi-agent-0309" in config.models
        assert "grok-4.20-0309-reasoning" in config.models
        assert "grok-4.20-0309-non-reasoning" in config.models

    def test_config_has_new_claude_models(self):
        config = LLMConfig()
        assert "claude-opus-4-6" in config.models
        assert "claude-sonnet-4-6" in config.models
        assert "claude-haiku-4-5" in config.models

    def test_config_has_new_gpt_models(self):
        config = LLMConfig()
        assert "gpt-5.4-2026-03-05" in config.models
        assert "gpt-5.4-mini-2026-03-17" in config.models

    def test_config_has_gemini_models(self):
        config = LLMConfig()
        assert "gemini-3.1-pro-preview" in config.models
        assert "gemini-3-flash-preview" in config.models
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gemini_provider.py -v`
Expected: FAIL — `AsyncLLMProvider` has no `GEMINI`, `ModelProvider` has no `GEMINI`, models don't exist

- [ ] **Step 3: Add GEMINI to ModelProvider enum**

In `src/adam/llm/config.py:14-18`, add `GEMINI = "gemini"` to `ModelProvider`. Add `GEMINI_API_KEY` to `api_keys` dict in `LLMConfig.__init__`. Add all new model configs to the `models` dict:
- `grok-4.20-multi-agent-0309`, `grok-4.20-0309-reasoning`, `grok-4.20-0309-non-reasoning`
- `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5`
- `gpt-5.4-2026-03-05`, `gpt-5.4-mini-2026-03-17`
- `gemini-3.1-pro-preview`, `gemini-3-flash-preview`

Update `default_models` dict to use new model names. Keep old models as backward-compat aliases.

- [ ] **Step 4: Add GEMINI to AsyncLLMProvider and _get_provider_for_model**

In `src/adam/llm/async_client.py`:
- Add `GEMINI = "gemini"` to `AsyncLLMProvider` enum (line 68-73)
- In `_initialize_async_clients` (line 108-147), add Gemini initialization using `AsyncOpenAI` pointed at `https://generativelanguage.googleapis.com/v1beta/openai/` with `GEMINI_API_KEY`
- In `_get_provider_for_model` (line 447-457), add: `elif model.startswith('gemini-'): return AsyncLLMProvider.GEMINI`
- Gemini completion/streaming reuses the OpenAI methods since the API is compatible — in `_complete_single` and `_complete_streaming`, route `GEMINI` to the same methods as `OPENAI`

- [ ] **Step 5: Update ModelTier and MODEL_MAPPING in router**

In `src/adam/llm/router.py:28-36`, update `ModelTier` enum values to new model names:
- `REASONING = "grok-4.20-0309-reasoning"`
- `STANDARD = "grok-4.20-0309-non-reasoning"`
- `FAST = "grok-4.20-0309-non-reasoning"`
- `ROUTER = "claude-haiku-4-5"`

Update `MODEL_MAPPING` (line 144-148) to match.

- [ ] **Step 6: Update .env.example**

Add `GEMINI_API_KEY=your_gemini_api_key_here` to `.env.example`.

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_gemini_provider.py -v`
Expected: All PASS

- [ ] **Step 8: Run full test suite for regressions**

Run: `pytest tests/ -v --timeout=30`
Expected: All existing tests pass (model name references in tests may need updating)

- [ ] **Step 9: Commit**

```bash
git add src/adam/llm/ tests/test_gemini_provider.py .env.example
git commit -m "feat: add Gemini provider and update model roster to current versions"
```

---

### Task 2: Add New EntryTypes and PEER_REVIEW Pattern Enum

**Files:**
- Modify: `src/adam/agents/scratchpad.py:17-24` (EntryType enum)
- Modify: `src/adam/agents/orchestrator.py:22-25` (OrchestrationPattern enum)
- Test: `tests/test_scratchpad_entry_types.py`

- [ ] **Step 1: Write test for new entry types**

```python
# tests/test_scratchpad_entry_types.py
"""Tests for new Peer Review entry types and pattern enum."""
import pytest
from adam.agents.scratchpad import EntryType, Scratchpad, ScratchpadEntry, AgentRole
from adam.agents.orchestrator import OrchestrationPattern


class TestNewEntryTypes:
    def test_rebuttal_entry_type_exists(self):
        assert EntryType.REBUTTAL.value == "rebuttal"

    def test_reaction_entry_type_exists(self):
        assert EntryType.REACTION.value == "reaction"

    def test_scratchpad_filters_by_rebuttal_type(self):
        pad = Scratchpad(query="test")
        entry = ScratchpadEntry(
            agent_role=AgentRole.REASONER,
            agent_name="Reasoner",
            content="rebuttal content",
            entry_type=EntryType.REBUTTAL,
        )
        pad.add_entry(entry)
        results = pad.get_entries_by_type(EntryType.REBUTTAL)
        assert len(results) == 1
        assert results[0].content == "rebuttal content"

    def test_scratchpad_filters_by_reaction_type(self):
        pad = Scratchpad(query="test")
        entry = ScratchpadEntry(
            agent_role=AgentRole.CRITIC,
            agent_name="Critic",
            content="reaction content",
            entry_type=EntryType.REACTION,
        )
        pad.add_entry(entry)
        results = pad.get_entries_by_type(EntryType.REACTION)
        assert len(results) == 1


class TestPeerReviewPatternEnum:
    def test_peer_review_pattern_exists(self):
        assert OrchestrationPattern.PEER_REVIEW.value == "peer_review"

    def test_existing_patterns_unchanged(self):
        assert OrchestrationPattern.SEQUENTIAL.value == "sequential"
        assert OrchestrationPattern.DEBATE.value == "debate"
        assert OrchestrationPattern.SINGLE.value == "single"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scratchpad_entry_types.py -v`
Expected: FAIL — no REBUTTAL/REACTION in EntryType, no PEER_REVIEW in OrchestrationPattern

- [ ] **Step 3: Add REBUTTAL and REACTION to EntryType**

In `src/adam/agents/scratchpad.py:17-24`, add to `EntryType` enum:
```python
REBUTTAL = "rebuttal"    # Producer's response to reviewer feedback
REACTION = "reaction"    # Reviewer's follow-up to the rebuttal
```

- [ ] **Step 4: Add PEER_REVIEW to OrchestrationPattern**

In `src/adam/agents/orchestrator.py:22-25`, add:
```python
PEER_REVIEW = "peer_review"  # Produce -> Review -> Rebut -> React -> Synthesize
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_scratchpad_entry_types.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/adam/agents/scratchpad.py src/adam/agents/orchestrator.py tests/test_scratchpad_entry_types.py
git commit -m "feat: add REBUTTAL/REACTION entry types and PEER_REVIEW pattern"
```

---

### Task 3: Implement Peer Review Pattern

**Files:**
- Create: `src/adam/agents/patterns/peer_review.py`
- Test: `tests/test_peer_review_pattern.py`

- [ ] **Step 1: Write test for Peer Review pattern flow**

```python
# tests/test_peer_review_pattern.py
"""Tests for Peer Review pattern: Produce -> Review -> Rebut -> React -> Synthesize."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass

from adam.agents.scratchpad import Scratchpad, EntryType, AgentRole
from adam.agents.patterns.peer_review import run_peer_review_pipeline, stream_peer_review_pipeline


@dataclass
class FakeLLMResponse:
    content: str
    model: str = "test-model"
    cost: float = 0.01
    total_tokens: int = 100


def make_mock_client(responses: list[str]):
    """Create a mock LLM client that returns responses in order."""
    call_count = 0

    async def mock_complete(prompt, model=None, system_prompt=None,
                            temperature=0.7, max_tokens=None, stream=False, **kwargs):
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        if stream:
            async def gen():
                yield responses[idx]
            return gen()
        return FakeLLMResponse(content=responses[idx])

    client = AsyncMock()
    client.complete = mock_complete
    return client


class TestPeerReviewPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_produces_5_steps(self):
        """Peer Review should produce entries for all 5 steps."""
        client = make_mock_client([
            "Reasoner analysis",     # Step 1: Produce
            "Coder review",          # Step 2a: Review (parallel)
            "Critic review",         # Step 2b: Review (parallel)
            "Reasoner rebuttal",     # Step 3: Rebuttal
            "Coder reaction",        # Step 4a: React (parallel)
            "Critic reaction",       # Step 4b: React (parallel)
            "Final synthesis",       # Step 5: Synthesize
        ])
        pad = Scratchpad(query="Review my code", budget=5.0)
        result = await run_peer_review_pipeline(pad, client)

        # Should have 7 entries (1 produce + 2 review + 1 rebuttal + 2 react + 1 synthesis)
        assert len(result.entries) == 7

        types = [e.entry_type for e in result.entries]
        assert types[0] == EntryType.PLAN          # Produce
        assert types[1] in (EntryType.CODE, EntryType.CRITIQUE)  # Review
        assert types[2] in (EntryType.CODE, EntryType.CRITIQUE)  # Review
        assert types[3] == EntryType.REBUTTAL       # Rebuttal
        assert types[4] == EntryType.REACTION       # React
        assert types[5] == EntryType.REACTION       # React
        assert types[6] == EntryType.SYNTHESIS      # Synthesize

    @pytest.mark.asyncio
    async def test_budget_stops_pipeline(self):
        """Pipeline should stop and synthesize partial results if budget exceeded."""
        client = make_mock_client(["analysis", "review1", "review2", "synthesis"])
        pad = Scratchpad(query="test", budget=0.02)  # Tight budget
        result = await run_peer_review_pipeline(pad, client)

        # Should have fewer than 7 entries due to budget
        assert len(result.entries) < 7
        # Last entry should be synthesis (fallback)
        assert result.entries[-1].entry_type == EntryType.SYNTHESIS

    @pytest.mark.asyncio
    async def test_reviews_run_in_parallel(self):
        """Coder and Critic reviews should run concurrently."""
        timestamps = []

        async def mock_complete(prompt, **kwargs):
            import time
            timestamps.append(time.time())
            await asyncio.sleep(0.05)  # Small delay
            return FakeLLMResponse(content="review")

        client = AsyncMock()
        client.complete = mock_complete
        pad = Scratchpad(query="test", budget=5.0)
        await run_peer_review_pipeline(pad, client)

        # Reviews (calls 2 and 3) should start close together (parallel)
        if len(timestamps) >= 3:
            gap = abs(timestamps[2] - timestamps[1])
            assert gap < 0.1, f"Reviews started {gap}s apart, should be parallel"


class TestPeerReviewStreaming:
    @pytest.mark.asyncio
    async def test_stream_emits_step_events(self):
        """Streaming should emit step_start and agent_start events."""
        client = make_mock_client(["a", "b", "c", "d", "e", "f", "g"])
        pad = Scratchpad(query="test", budget=5.0)

        events = []
        async for event in stream_peer_review_pipeline(pad, client):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "step_start" in event_types
        assert "agent_start" in event_types
        assert "agent_done" in event_types
        assert "done" in event_types
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_peer_review_pattern.py -v`
Expected: FAIL — `peer_review` module does not exist

- [ ] **Step 3: Implement run_peer_review_pipeline**

Create `src/adam/agents/patterns/peer_review.py`. Follow the exact same patterns as `sequential.py` and `debate.py`:
- Import `Agent, AgentConfig` from `..base`
- Import `Scratchpad, ScratchpadEntry, AgentRole, EntryType` from `..scratchpad`
- Import `create_reasoner, create_coder, create_critic, create_synthesizer` from `..roles`
- Define 5 steps matching the spec:
  - Step 1 (PRODUCE): Reasoner thinks, adds PLAN entry
  - Step 2 (REVIEW): Coder + Critic review in parallel via `asyncio.gather`, using custom context that includes the Reasoner's output. Coder adds CODE entry, Critic adds CRITIQUE entry
  - Step 3 (REBUTTAL): Reasoner gets custom context with own PLAN + both reviews, adds REBUTTAL entry
  - Step 4 (REACT): Coder + Critic react in parallel. Each sees their own review + the rebuttal. Both add REACTION entries
  - Step 5 (SYNTHESIZE): Synthesizer sees all entries, adds SYNTHESIS entry
- Budget check between each step. If exceeded, skip to synthesizer on partial results
- Custom context per step (bypass `build_context_for_agent`, build prompts manually like `debate.py` does)
- Accept `model_assignments: Optional[Dict[str, str]]` with keys `reasoner`, `coder`, `critic`, `synthesizer`

- [ ] **Step 4: Implement stream_peer_review_pipeline**

Same file. Follow `stream_sequential_pipeline` pattern — yield events:
- `step_start` with step name
- `agent_start` with agent name, role, model
- `agent_chunk` with streaming tokens
- `agent_done` with cost, tokens
- `step_complete` with step cost
- `done` with total cost, scratchpad

For parallel steps (2 and 4), stream agents sequentially in the streaming variant (simplifies SSE ordering). Non-streaming `run_peer_review_pipeline` still uses true `asyncio.gather` parallelism.

**Streaming cost tracking:** After each agent finishes streaming, estimate tokens using tiktoken (for OpenAI/Gemini models) or approximate word-count heuristic (for others). Calculate cost via `PricingManager`. Emit these in the `agent_done` event.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_peer_review_pattern.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/adam/agents/patterns/peer_review.py tests/test_peer_review_pattern.py
git commit -m "feat: implement Peer Review pattern with rebuttal loop"
```

---

### Task 4: Deep Discussion Session and Config

**Files:**
- Create: `src/adam/deep_discussion/__init__.py`
- Create: `src/adam/deep_discussion/config.py`
- Create: `src/adam/deep_discussion/session.py`
- Test: `tests/test_deep_discussion_session.py`

- [ ] **Step 1: Write test for SessionConfig and smart defaults**

```python
# tests/test_deep_discussion_session.py
"""Tests for Deep Discussion session management."""
import pytest
from adam.deep_discussion.config import SessionConfig, get_smart_defaults
from adam.deep_discussion.session import DeepDiscussionSession


class TestSessionConfig:
    def test_default_budget(self):
        config = SessionConfig(question="test", pattern="peer_review")
        assert config.budget == 2.0

    def test_default_model_assignments(self):
        config = SessionConfig(question="test", pattern="peer_review")
        defaults = get_smart_defaults()
        assert "reasoner" in defaults
        assert "coder" in defaults
        assert "critic" in defaults
        assert "synthesizer" in defaults

    def test_pattern_validation(self):
        config = SessionConfig(question="test", pattern="sequential")
        assert config.pattern == "sequential"

        config2 = SessionConfig(question="test", pattern="debate")
        assert config2.pattern == "debate"

        config3 = SessionConfig(question="test", pattern="peer_review")
        assert config3.pattern == "peer_review"

    def test_conversation_context_optional(self):
        config = SessionConfig(question="test", pattern="peer_review")
        assert config.conversation_id is None
        assert config.conversation_context is None


class TestDeepDiscussionSession:
    def test_initial_status_is_configuring(self):
        config = SessionConfig(question="test", pattern="peer_review")
        session = DeepDiscussionSession(config=config)
        assert session.status == "configuring"

    def test_update_model_assignments(self):
        config = SessionConfig(question="test", pattern="peer_review")
        session = DeepDiscussionSession(config=config)
        session.update_config(model_assignments={"reasoner": "claude-opus-4-6"})
        assert session.config.model_assignments["reasoner"] == "claude-opus-4-6"

    def test_update_budget(self):
        config = SessionConfig(question="test", pattern="peer_review")
        session = DeepDiscussionSession(config=config)
        session.update_config(budget=3.5)
        assert session.config.budget == 3.5

    def test_role_to_agent_mapping_sequential(self):
        config = SessionConfig(question="test", pattern="sequential")
        session = DeepDiscussionSession(config=config)
        mapping = session.get_pattern_model_assignments()
        # Sequential uses role names directly
        assert "reasoner" in str(mapping).lower() or len(mapping) > 0

    def test_role_to_agent_mapping_debate(self):
        config = SessionConfig(
            question="test",
            pattern="debate",
            model_assignments={
                "reasoner": "model-a",
                "coder": "model-b",
                "critic": "model-c",
                "synthesizer": "model-d",
            },
        )
        session = DeepDiscussionSession(config=config)
        mapping = session.get_pattern_model_assignments()
        assert mapping.get("debater_a") == "model-a"
        assert mapping.get("debater_b") == "model-b"
        assert mapping.get("reconciler") == "model-c"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_deep_discussion_session.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Create deep_discussion package with __init__.py**

Create `src/adam/deep_discussion/__init__.py`:
```python
"""Deep Discussion Mode — user-controlled multi-agent orchestration."""
```

- [ ] **Step 4: Implement SessionConfig with smart defaults**

Create `src/adam/deep_discussion/config.py`:
- `SessionConfig` dataclass with: `question`, `pattern`, `model_assignments`, `budget` (default 2.0), `conversation_id` (optional), `conversation_context` (optional)
- `get_smart_defaults()` function returns default model assignments:
  - `reasoner: grok-4.20-multi-agent-0309`
  - `coder: claude-opus-4-6`
  - `critic: gpt-5.4-2026-03-05`
  - `synthesizer: claude-sonnet-4-6`
- `SessionConfig.__post_init__` fills `model_assignments` from `get_smart_defaults()` if not provided
- `AVAILABLE_MODELS` dict mapping display labels to full model IDs, grouped by provider

- [ ] **Step 5: Implement DeepDiscussionSession**

Create `src/adam/deep_discussion/session.py`:
- `DeepDiscussionSession` class with: `id` (uuid), `config: SessionConfig`, `status` (configuring/running/completed/failed), `result`, `scratchpad_data`, `total_cost`, timestamps
- `update_config()` method to change model assignments, pattern, budget
- `get_pattern_model_assignments()` method that maps the 4 canonical role keys to pattern-specific agent slots:
  - Sequential: `{reasoner: x, coder: y, critic: z, synthesizer: w}` (pass-through, uses `AgentRole` keys)
  - Debate: `{debater_a: reasoner_model, debater_b: coder_model, reconciler: critic_model}`
  - Peer Review: `{reasoner: x, coder: y, critic: z, synthesizer: w}` (pass-through)
- `run()` async method that delegates to the correct pattern runner, tracks cost, updates status
- `run_stream()` async generator that wraps pattern stream with session-level events

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_deep_discussion_session.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/adam/deep_discussion/ tests/test_deep_discussion_session.py
git commit -m "feat: add Deep Discussion session management and config"
```

---

### Task 5: Database Model for Deep Discussion Sessions

**Files:**
- Modify: `src/adam/api/models.py` (add DeepDiscussionSession SQLAlchemy model + Pydantic schemas)
- Test: `tests/test_deep_discussion_db.py`

- [ ] **Step 1: Write test for DB model**

```python
# tests/test_deep_discussion_db.py
"""Tests for Deep Discussion database model."""
import pytest
import asyncio
from adam.api.models import DeepDiscussionSessionDB, DeepDiscussionSessionCreate, DeepDiscussionSessionResponse


class TestDeepDiscussionModel:
    def test_model_creates_with_defaults(self):
        session = DeepDiscussionSessionDB(
            project_id="proj-1",
            question="test question",
            pattern="peer_review",
            model_assignments={"reasoner": "grok-4.20-multi-agent-0309"},
            budget=2.0,
        )
        assert session.status == "configuring"
        assert session.total_cost == 0.0
        assert session.result is None

    def test_pydantic_create_schema(self):
        create = DeepDiscussionSessionCreate(
            project_id="proj-1",
            question="Review my code",
            pattern="peer_review",
        )
        assert create.project_id == "proj-1"
        assert create.conversation_id is None

    def test_pydantic_response_schema(self):
        resp = DeepDiscussionSessionResponse(
            id="sess-1",
            project_id="proj-1",
            question="test",
            pattern="peer_review",
            model_assignments={"reasoner": "grok"},
            budget=2.0,
            total_cost=0.5,
            status="completed",
            created_at="2026-03-25T00:00:00",
        )
        assert resp.id == "sess-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_deep_discussion_db.py -v`
Expected: FAIL — models don't exist

- [ ] **Step 3: Add SQLAlchemy model and Pydantic schemas to models.py**

Add to `src/adam/api/models.py`:
- `DeepDiscussionSessionDB(Base)` SQLAlchemy model with all columns from spec
- `DeepDiscussionSessionCreate(BaseModel)` — request schema for creating session
- `DeepDiscussionSessionUpdate(BaseModel)` — request schema for updating config
- `DeepDiscussionSessionResponse(BaseModel)` — response schema

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_deep_discussion_db.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/adam/api/models.py tests/test_deep_discussion_db.py
git commit -m "feat: add DeepDiscussionSession database model and schemas"
```

---

### Task 6: Deep Discussion API Router

**Files:**
- Create: `src/adam/api/routers/deep_discussion.py`
- Modify: `src/adam/api/main.py:66-79` (include new router)
- Test: `tests/test_deep_discussion_api.py`

- [ ] **Step 1: Write test for API endpoints**

```python
# tests/test_deep_discussion_api.py
"""Tests for Deep Discussion API endpoints."""
import pytest
from httpx import AsyncClient, ASGITransport
from adam.api.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
class TestDeepDiscussionAPI:
    async def test_create_session_returns_smart_defaults(self, client):
        resp = await client.post("/api/deep-discussion/sessions", json={
            "project_id": "test-proj",
            "question": "Review my code",
            "pattern": "peer_review",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "configuring"
        assert "reasoner" in data["model_assignments"]
        assert "coder" in data["model_assignments"]
        assert data["budget"] == 2.0

    async def test_update_session_config(self, client):
        # Create first
        create_resp = await client.post("/api/deep-discussion/sessions", json={
            "project_id": "test-proj",
            "question": "test",
            "pattern": "peer_review",
        })
        session_id = create_resp.json()["id"]

        # Update
        resp = await client.put(f"/api/deep-discussion/sessions/{session_id}/config", json={
            "model_assignments": {"reasoner": "claude-opus-4-6"},
            "budget": 3.0,
        })
        assert resp.status_code == 200
        assert resp.json()["model_assignments"]["reasoner"] == "claude-opus-4-6"
        assert resp.json()["budget"] == 3.0

    async def test_get_session(self, client):
        create_resp = await client.post("/api/deep-discussion/sessions", json={
            "project_id": "test-proj",
            "question": "test",
            "pattern": "sequential",
        })
        session_id = create_resp.json()["id"]

        resp = await client.get(f"/api/deep-discussion/sessions/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == session_id

    async def test_list_sessions(self, client):
        resp = await client.get("/api/deep-discussion/sessions", params={"project_id": "test-proj"})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_create_from_conversation(self, client):
        """from-conversation endpoint should carry context."""
        # Note: requires a conversation with messages to exist.
        # For unit test, verify the endpoint exists and returns 404 for missing conversation.
        resp = await client.post(
            "/api/deep-discussion/sessions/from-conversation/nonexistent-conv",
            json={"question": "Go deep on this"},
        )
        assert resp.status_code in (200, 404)  # 404 if conv doesn't exist

    async def test_replay_creates_new_session(self, client):
        create_resp = await client.post("/api/deep-discussion/sessions", json={
            "project_id": "test-proj",
            "question": "test",
            "pattern": "peer_review",
        })
        session_id = create_resp.json()["id"]

        resp = await client.post(f"/api/deep-discussion/sessions/{session_id}/replay")
        assert resp.status_code == 200
        assert resp.json()["id"] != session_id
        assert resp.json()["question"] == "test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_deep_discussion_api.py -v`
Expected: FAIL — router doesn't exist, 404s

- [ ] **Step 3: Implement API router**

Create `src/adam/api/routers/deep_discussion.py` with all endpoints from the spec:
- `POST /sessions` — create session, apply smart defaults, persist to DB
- `PUT /sessions/{id}/config` — update model assignments, pattern, budget
- `POST /sessions/{id}/start` — run the discussion, return SSE stream using `StreamingResponse`. Create `DeepDiscussionSession`, call `run_stream()`, yield SSE events
- `GET /sessions/{id}` — fetch session from DB
- `POST /sessions/{id}/replay` — create new session cloned from existing
- `POST /sessions/from-conversation/{conv_id}` — fetch last 10 messages from conversation, format as context block: `CONVERSATION CONTEXT:\n[user]: ...\n[assistant]: ...\n---\nQUESTION: <question>`. Store in `conversation_context` field
- `GET /sessions` — list by project_id

Follow existing router patterns from `messages.py` and `conversations.py`. The import in `main.py` (line 66-69) uses explicit module imports — extend that line to include `deep_discussion`.

- [ ] **Step 4: Register router in main.py**

In `src/adam/api/main.py`, import and include the new router:
```python
from adam.api.routers import deep_discussion
app.include_router(deep_discussion.router, prefix="/api/deep-discussion", tags=["deep-discussion"])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_deep_discussion_api.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/adam/api/routers/deep_discussion.py src/adam/api/main.py tests/test_deep_discussion_api.py
git commit -m "feat: add Deep Discussion API endpoints with SSE streaming"
```

---

### Task 7: React Frontend — Deep Discussion Page and Config Screen

**Files:**
- Create: `frontend/AdamChat/client/src/pages/deep-discussion.tsx`
- Create: `frontend/AdamChat/client/src/components/deep-discussion/config-screen.tsx`
- Create: `frontend/AdamChat/client/src/components/deep-discussion/model-selector.tsx`
- Create: `frontend/AdamChat/client/src/components/deep-discussion/pattern-selector.tsx`
- Create: `frontend/AdamChat/client/src/lib/deep-discussion-api.ts`
- Modify: `frontend/AdamChat/client/src/App.tsx:17-27` (add route)

- [ ] **Step 1: Create API client for Deep Discussion**

Create `frontend/AdamChat/client/src/lib/deep-discussion-api.ts`:
- `createSession(projectId, question, pattern, conversationId?)` — POST to `/api/deep-discussion/sessions`
- `updateSessionConfig(sessionId, config)` — PUT to `/api/deep-discussion/sessions/{id}/config`
- `startSession(sessionId)` — POST to `/api/deep-discussion/sessions/{id}/start`, returns EventSource for SSE
- `getSession(sessionId)` — GET
- `listSessions(projectId)` — GET
- `replaySession(sessionId)` — POST
- `createFromConversation(conversationId, question)` — POST

- [ ] **Step 2: Create ModelSelector component**

Create `frontend/AdamChat/client/src/components/deep-discussion/model-selector.tsx`:
- Dropdown grouped by provider (X.AI, Anthropic, OpenAI, Google)
- Shows display label, stores full model ID
- Uses Radix UI Select component (matches existing design system)

- [ ] **Step 3: Create PatternSelector component**

Create `frontend/AdamChat/client/src/components/deep-discussion/pattern-selector.tsx`:
- 3 cards: Sequential, Debate, Peer Review
- Each shows name, brief description, step count
- Selected state with green border (matching theme)

- [ ] **Step 4: Create ConfigScreen component**

Create `frontend/AdamChat/client/src/components/deep-discussion/config-screen.tsx`:
- Question textarea at top
- PatternSelector
- 4 agent cards in a 2x2 grid, each with ModelSelector dropdown
- Budget slider ($0.50–$5.00)
- "Start Deep Discussion" button
- Uses Tailwind + Radix UI, white/green theme

- [ ] **Step 5: Create Deep Discussion page**

Create `frontend/AdamChat/client/src/pages/deep-discussion.tsx`:
- Sidebar with navigation (Chat, Deep Discussion) and session history
- Main content area shows ConfigScreen for new sessions
- Uses TanStack React Query for fetching session list

- [ ] **Step 6: Add route to App.tsx**

In `frontend/AdamChat/client/src/App.tsx`, add:
```tsx
import DeepDiscussion from "@/pages/deep-discussion";
// In Router:
<Route path="/project/:projectId/deep-discussion" component={DeepDiscussion} />
<Route path="/project/:projectId/deep-discussion/:sessionId?" component={DeepDiscussion} />
```

- [ ] **Step 7: Verify frontend builds**

Run: `cd frontend/AdamChat && npm run build`
Expected: Build succeeds with no TypeScript errors

- [ ] **Step 8: Commit**

```bash
git add frontend/AdamChat/client/src/
git commit -m "feat: add Deep Discussion config screen and routing in React app"
```

---

### Task 8: React Frontend — Live Discussion View

**Files:**
- Create: `frontend/AdamChat/client/src/components/deep-discussion/live-view.tsx`
- Create: `frontend/AdamChat/client/src/components/deep-discussion/agent-card.tsx`
- Create: `frontend/AdamChat/client/src/components/deep-discussion/progress-bar.tsx`
- Modify: `frontend/AdamChat/client/src/pages/deep-discussion.tsx` (integrate live view)

- [ ] **Step 1: Create ProgressBar component**

Create `frontend/AdamChat/client/src/components/deep-discussion/progress-bar.tsx`:
- Shows steps for the selected pattern (e.g., Produce → Review → Rebuttal → React → Synthesize)
- Current step highlighted in green, completed in green, pending in gray
- Step labels below the bar

- [ ] **Step 2: Create AgentCard component**

Create `frontend/AdamChat/client/src/components/deep-discussion/agent-card.tsx`:
- Shows: agent role, model name, provider icon, cost badge, status (thinking/done)
- Content area: expandable, streams text as it arrives
- Role-specific color dots: Reasoner=blue, Coder=green, Critic=orange, Synthesizer=purple
- Role badges: PRODUCER, REVIEWER, REBUTTAL, FINAL
- Pending state: dashed border, grayed out

- [ ] **Step 3: Create LiveView component**

Create `frontend/AdamChat/client/src/components/deep-discussion/live-view.tsx`:
- Header: question, pattern, running status, live cost counter
- ProgressBar
- AgentCards rendered as events arrive from SSE stream
- Uses EventSource to connect to `/api/deep-discussion/sessions/{id}/start`
- Parses SSE events: `session_start`, `step_start`, `agent_start`, `agent_chunk`, `agent_done`, `step_complete`, `session_complete`, `agent_error`, `session_error`
- Completed state: final synthesized answer at top, all agent cards below
- "Replay with Changes" button

- [ ] **Step 4: Integrate into deep-discussion page**

Modify `frontend/AdamChat/client/src/pages/deep-discussion.tsx`:
- Show ConfigScreen when session is in "configuring" state
- Switch to LiveView when session starts
- Show completed view when session finishes

- [ ] **Step 5: Verify frontend builds**

Run: `cd frontend/AdamChat && npm run build`
Expected: Build succeeds

- [ ] **Step 6: Commit**

```bash
git add frontend/AdamChat/client/src/components/deep-discussion/ frontend/AdamChat/client/src/pages/deep-discussion.tsx
git commit -m "feat: add Deep Discussion live view with SSE streaming and agent cards"
```

---

### Task 9: "Go Deep" Button in Chat

**Files:**
- Modify: `frontend/AdamChat/client/src/components/chat/` (add Go Deep button to chat input)
- Create: `frontend/AdamChat/client/src/components/deep-discussion/go-deep-modal.tsx`

- [ ] **Step 1: Create GoDeepModal component**

Create `frontend/AdamChat/client/src/components/deep-discussion/go-deep-modal.tsx`:
- Modal/drawer that opens from chat
- Reuses ConfigScreen component
- Pre-fills conversation context (last messages from current chat)
- On completion, posts result back into chat via messages API

- [ ] **Step 2: Add "Go Deep" button to chat input bar**

Find the chat input component in `frontend/AdamChat/client/src/components/chat/` and add a brain icon button (🧠) next to the send button. Clicking opens GoDeepModal.

- [ ] **Step 3: Verify frontend builds**

Run: `cd frontend/AdamChat && npm run build`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add frontend/AdamChat/client/src/components/
git commit -m "feat: add Go Deep button in chat for escalating to Deep Discussion"
```

---

### Task 10: Integration Tests and Phase Completion

**Files:**
- Create: `tests/test_deep_discussion_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_deep_discussion_integration.py
"""End-to-end integration tests for Deep Discussion."""
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from adam.api.main import app
from adam.deep_discussion.config import SessionConfig, get_smart_defaults
from adam.deep_discussion.session import DeepDiscussionSession


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
class TestDeepDiscussionE2E:
    async def test_full_session_lifecycle(self, client):
        """Create -> Configure -> Start -> Complete lifecycle."""
        # Create
        resp = await client.post("/api/deep-discussion/sessions", json={
            "project_id": "test-proj",
            "question": "Review my dbt model",
            "pattern": "peer_review",
        })
        assert resp.status_code == 200
        session_id = resp.json()["id"]
        assert resp.json()["status"] == "configuring"

        # Configure
        resp = await client.put(f"/api/deep-discussion/sessions/{session_id}/config", json={
            "model_assignments": {"reasoner": "grok-4.20-multi-agent-0309"},
            "budget": 3.0,
        })
        assert resp.status_code == 200

        # Get
        resp = await client.get(f"/api/deep-discussion/sessions/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["budget"] == 3.0

    async def test_smart_defaults_use_available_providers(self):
        """Smart defaults should return valid model assignments."""
        defaults = get_smart_defaults()
        assert len(defaults) == 4
        assert all(k in defaults for k in ["reasoner", "coder", "critic", "synthesizer"])

    async def test_replay_preserves_config(self, client):
        """Replay should create new session with same config."""
        # Create original
        resp = await client.post("/api/deep-discussion/sessions", json={
            "project_id": "test-proj",
            "question": "original question",
            "pattern": "debate",
        })
        original_id = resp.json()["id"]

        # Replay
        resp = await client.post(f"/api/deep-discussion/sessions/{original_id}/replay")
        assert resp.status_code == 200
        assert resp.json()["id"] != original_id
        assert resp.json()["question"] == "original question"
        assert resp.json()["pattern"] == "debate"
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_deep_discussion_integration.py -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v --timeout=60`
Expected: All tests pass (existing + new)

- [ ] **Step 4: Commit completion**

```bash
git add tests/test_deep_discussion_integration.py
git commit -m "feat: complete Deep Discussion Mode with integration tests

Deep Discussion Mode delivers:
- Peer Review pattern (produce → review → rebuttal → react → synthesize)
- Gemini as 4th LLM provider
- Updated model roster (Grok 4.20, Claude Opus/Sonnet 4.6, GPT 5.4, Gemini 3.1)
- Session management with per-agent model configuration
- React UI: config screen + live discussion view + Go Deep button
- SSE streaming for real-time agent activity
- Smart defaults with manual override
- 7 new API endpoints"
```
