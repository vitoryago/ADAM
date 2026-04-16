"""End-to-end integration tests for the full Deep Discussion system.

These tests verify that all components work together:
  - API endpoints (via httpx AsyncClient + ASGITransport)
  - Session config / smart defaults
  - Replay preserves config
  - Config updates persist
  - All 3 patterns are valid for session creation
  - Peer Review pattern produces 7 entries with correct entry types
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Dict

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def event_loop():
    """Module-scoped event loop so fixtures can be shared across tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="module")
async def client():
    """Async httpx client backed by the ADAM ASGI app.

    follow_redirects=True is needed because FastAPI redirects e.g.
    POST /api/projects → POST /api/projects/ (trailing slash normalisation).
    """
    from adam.api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", follow_redirects=True
    ) as c:
        yield c


@pytest_asyncio.fixture(scope="module")
async def project_id(client: AsyncClient) -> str:
    """Create one project to be reused across all integration tests."""
    resp = await client.post(
        "/api/projects",
        json={
            "name": "Integration Test Project",
            "description": "Project created for deep discussion integration tests",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# 1. Full session lifecycle: Create → Configure → Get → Verify state
# ---------------------------------------------------------------------------


class TestFullSessionLifecycle:
    """Create a session, update its config, then GET it and verify state."""

    @pytest.mark.asyncio
    async def test_create_get_verify_state(
        self, client: AsyncClient, project_id: str
    ) -> None:
        # --- Create ---
        create_resp = await client.post(
            "/api/deep-discussion/sessions",
            json={
                "project_id": project_id,
                "question": "How should we design a distributed cache?",
            },
        )
        assert create_resp.status_code == 201, create_resp.text
        created = create_resp.json()
        session_id = created["id"]

        assert created["status"] == "configuring"
        assert created["question"] == "How should we design a distributed cache?"
        assert created["pattern"] == "peer_review"  # default
        assert created["project_id"] == project_id
        assert created["budget"] == 2.0
        assert created["total_cost"] == 0.0

        # --- Configure (update budget) ---
        config_resp = await client.put(
            f"/api/deep-discussion/sessions/{session_id}/config",
            json={"budget": 3.5},
        )
        assert config_resp.status_code == 200, config_resp.text
        assert config_resp.json()["budget"] == 3.5

        # --- GET ---
        get_resp = await client.get(f"/api/deep-discussion/sessions/{session_id}")
        assert get_resp.status_code == 200, get_resp.text
        fetched = get_resp.json()

        # --- Verify state ---
        assert fetched["id"] == session_id
        assert fetched["question"] == "How should we design a distributed cache?"
        assert fetched["status"] == "configuring"
        assert fetched["budget"] == 3.5
        assert fetched["project_id"] == project_id


# ---------------------------------------------------------------------------
# 2. Smart defaults validation
# ---------------------------------------------------------------------------


class TestSmartDefaultsValidation:
    """get_smart_defaults() must return exactly 4 valid role assignments."""

    def test_returns_four_valid_model_assignments(self) -> None:
        from adam.deep_discussion.config import get_smart_defaults

        defaults = get_smart_defaults()

        assert isinstance(defaults, dict)
        assert len(defaults) == 4

        required_roles = {"reasoner", "coder", "critic", "synthesizer"}
        assert set(defaults.keys()) == required_roles

        for role, model_id in defaults.items():
            assert isinstance(model_id, str) and model_id, (
                f"Role '{role}' has an empty or non-string model id: {model_id!r}"
            )

    @pytest.mark.asyncio
    async def test_api_session_creation_applies_smart_defaults(
        self, client: AsyncClient, project_id: str
    ) -> None:
        from adam.deep_discussion.config import get_smart_defaults

        resp = await client.post(
            "/api/deep-discussion/sessions",
            json={
                "project_id": project_id,
                "question": "Smart defaults validation question",
            },
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()

        expected_defaults = get_smart_defaults()
        assert data["model_assignments"] == expected_defaults


# ---------------------------------------------------------------------------
# 3. Replay preserves config
# ---------------------------------------------------------------------------


class TestReplayPreservesConfig:
    """Replaying a session creates a new session with a different ID but same
    question, pattern, model_assignments, and budget."""

    @pytest.mark.asyncio
    async def test_replay_same_question_pattern_config_different_id(
        self, client: AsyncClient, project_id: str
    ) -> None:
        # Create original
        create_resp = await client.post(
            "/api/deep-discussion/sessions",
            json={
                "project_id": project_id,
                "question": "Replay integrity test question?",
                "pattern": "debate",
            },
        )
        assert create_resp.status_code == 201, create_resp.text
        original = create_resp.json()
        original_id = original["id"]

        # Replay
        replay_resp = await client.post(
            f"/api/deep-discussion/sessions/{original_id}/replay"
        )
        assert replay_resp.status_code == 201, replay_resp.text
        replayed = replay_resp.json()

        # Different ID
        assert replayed["id"] != original_id

        # Same question, pattern, model_assignments, budget
        assert replayed["question"] == original["question"]
        assert replayed["pattern"] == original["pattern"]
        assert replayed["model_assignments"] == original["model_assignments"]
        assert replayed["budget"] == original["budget"]

        # Replayed session starts fresh
        assert replayed["status"] == "configuring"
        assert replayed["total_cost"] == 0.0


# ---------------------------------------------------------------------------
# 4. Session config update — model assignments + budget
# ---------------------------------------------------------------------------


class TestSessionConfigUpdate:
    """Create a session, update model_assignments and budget, then verify
    the changes were persisted (round-tripped through the DB)."""

    @pytest.mark.asyncio
    async def test_update_model_assignments_and_budget(
        self, client: AsyncClient, project_id: str
    ) -> None:
        # Create
        create_resp = await client.post(
            "/api/deep-discussion/sessions",
            json={
                "project_id": project_id,
                "question": "Config update test",
            },
        )
        assert create_resp.status_code == 201, create_resp.text
        session_id = create_resp.json()["id"]

        new_assignments: Dict[str, str] = {
            "reasoner": "grok-4.20-0309-reasoning",
            "coder": "gpt-5.4-mini-2026-03-17",
            "critic": "grok-4.20-0309-non-reasoning",
            "synthesizer": "gpt-5.4-2026-03-05",
        }
        new_budget = 7.5

        # Update
        update_resp = await client.put(
            f"/api/deep-discussion/sessions/{session_id}/config",
            json={"model_assignments": new_assignments, "budget": new_budget},
        )
        assert update_resp.status_code == 200, update_resp.text

        # Verify response
        updated = update_resp.json()
        assert updated["model_assignments"] == new_assignments
        assert updated["budget"] == new_budget

        # Verify persistence via GET
        get_resp = await client.get(f"/api/deep-discussion/sessions/{session_id}")
        assert get_resp.status_code == 200, get_resp.text
        fetched = get_resp.json()
        assert fetched["model_assignments"] == new_assignments
        assert fetched["budget"] == new_budget


# ---------------------------------------------------------------------------
# 5. Pattern enum consistency
# ---------------------------------------------------------------------------


class TestPatternEnumConsistency:
    """All 3 patterns must be accepted by the session creation endpoint."""

    @pytest.mark.asyncio
    async def test_sequential_pattern_valid(
        self, client: AsyncClient, project_id: str
    ) -> None:
        resp = await client.post(
            "/api/deep-discussion/sessions",
            json={
                "project_id": project_id,
                "question": "Sequential pattern test",
                "pattern": "sequential",
            },
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["pattern"] == "sequential"

    @pytest.mark.asyncio
    async def test_debate_pattern_valid(
        self, client: AsyncClient, project_id: str
    ) -> None:
        resp = await client.post(
            "/api/deep-discussion/sessions",
            json={
                "project_id": project_id,
                "question": "Debate pattern test",
                "pattern": "debate",
            },
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["pattern"] == "debate"

    @pytest.mark.asyncio
    async def test_peer_review_pattern_valid(
        self, client: AsyncClient, project_id: str
    ) -> None:
        resp = await client.post(
            "/api/deep-discussion/sessions",
            json={
                "project_id": project_id,
                "question": "Peer review pattern test",
                "pattern": "peer_review",
            },
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["pattern"] == "peer_review"

    def test_all_three_patterns_accepted_by_session_config(self) -> None:
        from adam.deep_discussion.config import SessionConfig

        for pattern in ("sequential", "debate", "peer_review"):
            config = SessionConfig(question="Q", pattern=pattern)
            assert config.pattern == pattern


# ---------------------------------------------------------------------------
# 6. Peer Review pattern unit integration
# ---------------------------------------------------------------------------


# -----------
# Mock helpers (same pattern as test_peer_review_pattern.py)
# -----------


@dataclass
class _MockLLMResponse:
    content: str
    cost: float = 0.01
    total_tokens: int = 100
    model: str = "mock-model"


class _MockLLMClient:
    """Minimal mock that returns canned responses keyed by system prompt substring."""

    def __init__(self, responses: Optional[Dict[str, str]] = None) -> None:
        self.responses = responses or {}
        self.calls: list = []

    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        stream: bool = False,
    ):
        self.calls.append({"system_prompt": system_prompt, "model": model})

        content = "default response"
        for key, value in self.responses.items():
            if key in system_prompt:
                content = value
                break

        if stream:
            return self._stream_chunks(content)

        return _MockLLMResponse(content=content)

    async def _stream_chunks(self, content: str):
        for word in content.split():
            yield word + " "


_REASONER_KEY = "You are a Reasoning Specialist"
_CODER_KEY = "You are a Code Specialist"
_CRITIC_KEY = "You are a Code Review Specialist"
_SYNTHESIZER_KEY = "You are a Synthesis Specialist"


class TestPeerReviewPatternUnitIntegration:
    """Create a DeepDiscussionSession with peer_review pattern + mocked LLM,
    verify it produces all 7 entries with the correct EntryType sequence."""

    @pytest.mark.asyncio
    async def test_session_run_produces_7_entries_with_correct_types(self) -> None:
        from adam.deep_discussion.config import SessionConfig
        from adam.deep_discussion.session import DeepDiscussionSession
        from adam.agents.scratchpad import EntryType

        mock_client = _MockLLMClient(
            responses={
                _REASONER_KEY: "Reasoner plan: step by step approach",
                _CODER_KEY: "Coder review: implementation looks solid",
                _CRITIC_KEY: "Critic review: edge cases need attention",
                _SYNTHESIZER_KEY: "Final synthesis: combined insights",
            }
        )

        config = SessionConfig(
            question="How do we implement distributed tracing?",
            pattern="peer_review",
        )
        session = DeepDiscussionSession(config=config)

        # Verify initial state
        assert session.status == "configuring"

        # Run the session
        result = await session.run(mock_client)

        # Verify session is completed
        assert session.status == "completed"
        assert session.result is not None
        assert isinstance(result, str) and len(result) > 0

        # Verify scratchpad has 7 entries with correct types
        assert session.scratchpad_data is not None
        entries = session.scratchpad_data["entries"]
        assert len(entries) == 7, (
            f"Expected 7 entries, got {len(entries)}: "
            f"{[e['entry_type'] for e in entries]}"
        )

        entry_types = [e["entry_type"] for e in entries]
        assert entry_types[0] == EntryType.PLAN.value       # Step 1: Reasoner
        assert entry_types[1] == EntryType.CODE.value        # Step 2: Coder review
        assert entry_types[2] == EntryType.CRITIQUE.value    # Step 2: Critic review
        assert entry_types[3] == EntryType.REBUTTAL.value    # Step 3: Reasoner rebuttal
        assert entry_types[4] == EntryType.REACTION.value    # Step 4: Coder reaction
        assert entry_types[5] == EntryType.REACTION.value    # Step 4: Critic reaction
        assert entry_types[6] == EntryType.SYNTHESIS.value   # Step 5: Synthesizer

    @pytest.mark.asyncio
    async def test_session_cost_is_nonzero_after_run(self) -> None:
        from adam.deep_discussion.config import SessionConfig
        from adam.deep_discussion.session import DeepDiscussionSession

        mock_client = _MockLLMClient(
            responses={
                _REASONER_KEY: "plan",
                _CODER_KEY: "code review",
                _CRITIC_KEY: "critique",
                _SYNTHESIZER_KEY: "synthesis",
            }
        )
        config = SessionConfig(
            question="Cost tracking test?",
            pattern="peer_review",
        )
        session = DeepDiscussionSession(config=config)
        await session.run(mock_client)

        assert session.total_cost > 0.0, "Total cost should be > 0 after running"

    @pytest.mark.asyncio
    async def test_session_completed_at_is_set_after_run(self) -> None:
        from datetime import datetime
        from adam.deep_discussion.config import SessionConfig
        from adam.deep_discussion.session import DeepDiscussionSession

        mock_client = _MockLLMClient(
            responses={
                _REASONER_KEY: "plan",
                _CODER_KEY: "code",
                _CRITIC_KEY: "critique",
                _SYNTHESIZER_KEY: "synth",
            }
        )
        config = SessionConfig(
            question="Completion timestamp test?",
            pattern="peer_review",
        )
        session = DeepDiscussionSession(config=config)
        assert session.completed_at is None

        await session.run(mock_client)

        assert isinstance(session.completed_at, datetime)


class TestFallbackSynthesisIntegration:
    """Sequential sessions should surface when fallback synthesis was used."""

    @pytest.mark.asyncio
    async def test_session_marks_fallback_synthesis_when_synthesizer_never_runs(self) -> None:
        from adam.deep_discussion.config import SessionConfig
        from adam.deep_discussion.session import DeepDiscussionSession

        mock_client = _MockLLMClient(
            responses={
                _REASONER_KEY: "Reasoner plan only",
                _CODER_KEY: "Coder output",
                _CRITIC_KEY: "Critic notes",
                _SYNTHESIZER_KEY: "Synth output should not be reached",
            }
        )
        config = SessionConfig(
            question="Fallback synthesis test?",
            pattern="sequential",
            budget=0.01,
            model_assignments={
                "reasoner": "mock-reasoner",
                "coder": "mock-coder",
                "critic": "mock-critic",
                "synthesizer": "mock-synthesizer",
            },
        )
        session = DeepDiscussionSession(config=config)

        result = await session.run(mock_client)

        assert session.used_fallback_synthesis is True
        assert "Approach" in result
