"""Integration tests -- verify the full stack works together.

Phase 1: Foundation tests (server startup, project creation, imports)
Phase 2: Conversational core tests (message flow, memory search, sliding window, health)
"""
import os
from fastapi.testclient import TestClient


def _reset_db_engine():
    """Reset the global database engine so the next TestClient gets a fresh one."""
    import adam.database as db_mod
    db_mod._engine = None


# ---------------------------------------------------------------------------
# Phase 1 integration tests
# ---------------------------------------------------------------------------

def test_full_server_startup():
    """Server starts without errors and responds to health check."""
    _reset_db_engine()
    from adam.api.main import app
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "4.0.0"


def test_create_project():
    """Can create a project through the API."""
    _reset_db_engine()
    from adam.api.main import app
    with TestClient(app) as client:
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
    assert config.embedding_model_name is not None


def test_knowledge_services_load():
    """Knowledge services initialize without errors."""
    from adam.knowledge import DBTKnowledgeService
    service = DBTKnowledgeService()
    assert service is not None


def test_no_adam_v2_imports():
    """Verify no code imports from the deleted adam_v2 package."""
    violations = []
    src_adam_dir = os.path.join(os.path.dirname(__file__), "..", "src", "adam")
    src_adam_dir = os.path.normpath(src_adam_dir)

    for root, dirs, files in os.walk(src_adam_dir):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".py"):
                filepath = os.path.join(root, f)
                with open(filepath) as fh:
                    for line_num, line in enumerate(fh, 1):
                        stripped = line.strip()
                        # Skip comments -- provenance notes are OK
                        if stripped.startswith("#"):
                            continue
                        if stripped.startswith('"""') or stripped.startswith("'''"):
                            continue
                        if "adam_v2" in line and (
                            "from adam_v2" in line
                            or "import adam_v2" in line
                            or "adam_v2." in line
                        ):
                            violations.append(
                                f"{filepath}:{line_num}: {stripped}"
                            )

    assert violations == [], (
        f"Files still importing from adam_v2:\n" +
        "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Phase 2 integration tests -- conversational core
# ---------------------------------------------------------------------------

def test_conversation_message_flow():
    """Full flow: create project -> create conversation -> send message -> get response."""
    _reset_db_engine()
    from adam.api.main import app
    with TestClient(app) as client:
        # Create project
        project_resp = client.post("/api/projects/", json={"name": "Integration Test"})
        assert project_resp.status_code in [200, 201]
        project = project_resp.json()
        assert "id" in project

        # Create conversation
        conv_resp = client.post(
            f"/api/projects/{project['id']}/conversations",
            json={"title": "Test Chat"}
        )
        assert conv_resp.status_code in [200, 201]
        conv = conv_resp.json()
        assert "id" in conv
        assert conv["title"] == "Test Chat"

        # Send message (without real LLM -- uses mock if no API keys)
        msg_resp = client.post(
            f"/api/conversations/{conv['id']}/messages",
            json={"content": "Hello ADAM", "use_memory": False}
        )
        # 200/201 = success with real or mock LLM, 500 = LLM unavailable
        assert msg_resp.status_code in [200, 201, 500]

        if msg_resp.status_code in [200, 201]:
            messages = msg_resp.json()
            assert isinstance(messages, list)
            # Should return user message + assistant message
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Hello ADAM"
            assert messages[1]["role"] == "assistant"
            assert len(messages[1]["content"]) > 0


def test_memory_search_integration():
    """Memory search returns structured response."""
    _reset_db_engine()
    from adam.api.main import app
    with TestClient(app) as client:
        resp = client.get("/api/memories/search", params={
            "query": "test query",
            "project_id": "integration-test"
        })
        assert resp.status_code == 200
        data = resp.json()
        # Returns a list (possibly empty if memory system unavailable)
        assert isinstance(data, list)


def test_sliding_window_with_real_messages():
    """Sliding window works with database-loaded messages."""
    from adam.services.llm_service import LLMService
    from dataclasses import dataclass

    @dataclass
    class FakeMsg:
        role: str
        content: str

    service = LLMService.__new__(LLMService)
    history = [FakeMsg("user", f"Turn {i}: {'x' * 200}") for i in range(35)]
    context = service._build_conversation_context(history, max_recent=10, max_summary=20)

    # Should have 1 summary + 10 recent = 11 messages
    assert len(context) == 11
    # Summary should mention the message count
    assert "20" in context[0]["content"] or "summary" in context[0]["content"].lower()


def test_no_adam_v2_references():
    """No code references the deleted adam_v2 package via import/usage.

    Docstring provenance notes (e.g. 'Consolidated from src/adam_v2/...')
    and comments are allowed.  Only real import/usage statements are flagged.
    """
    violations = []
    src_adam_dir = os.path.join(os.path.dirname(__file__), "..", "src", "adam")
    src_adam_dir = os.path.normpath(src_adam_dir)

    for root, dirs, files in os.walk(src_adam_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".py"):
                filepath = os.path.join(root, f)
                in_docstring = False
                with open(filepath) as fh:
                    for i, line in enumerate(fh, 1):
                        stripped = line.strip()
                        # Track multi-line docstrings
                        if '"""' in stripped or "'''" in stripped:
                            # Count triple-quote occurrences to toggle state
                            triple_double = stripped.count('"""')
                            triple_single = stripped.count("'''")
                            total_triples = triple_double + triple_single
                            if total_triples == 1:
                                in_docstring = not in_docstring
                            # If a line both opens and closes a docstring
                            # (e.g. """text"""), it stays outside
                            continue
                        if in_docstring:
                            continue
                        # Skip single-line comments
                        if stripped.startswith("#"):
                            continue
                        # Check for actual import/usage of adam_v2
                        if "adam_v2" in line and (
                            "from adam_v2" in line
                            or "import adam_v2" in line
                            or "adam_v2." in line
                        ):
                            violations.append(f"{filepath}:{i}")

    assert violations == [], f"adam_v2 references found: {violations}"


def test_health_endpoints():
    """Both health check endpoints work."""
    _reset_db_engine()
    from adam.api.main import app
    with TestClient(app) as client:
        resp1 = client.get("/health")
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["version"] == "4.0.0"
        assert data1["status"] == "ok"

        resp2 = client.get("/api/health")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["version"] == "4.0.0"
        assert data2["status"] == "healthy"


def test_conversation_lifecycle():
    """Full lifecycle: create -> list -> get -> update -> delete conversation."""
    _reset_db_engine()
    from adam.api.main import app
    with TestClient(app) as client:
        # Create project first
        proj = client.post("/api/projects/", json={"name": "Lifecycle Test"}).json()

        # Create conversation
        conv = client.post(
            f"/api/projects/{proj['id']}/conversations",
            json={"title": "Original Title"}
        ).json()
        conv_id = conv["id"]

        # List conversations for project
        list_resp = client.get(f"/api/projects/{proj['id']}/conversations")
        assert list_resp.status_code == 200
        convs = list_resp.json()
        assert any(c["id"] == conv_id for c in convs)

        # Get single conversation
        get_resp = client.get(f"/api/conversations/{conv_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["title"] == "Original Title"

        # Update conversation title
        update_resp = client.put(
            f"/api/conversations/{conv_id}",
            json={"title": "Updated Title"}
        )
        assert update_resp.status_code == 200
        assert update_resp.json()["title"] == "Updated Title"

        # Delete conversation
        del_resp = client.delete(f"/api/conversations/{conv_id}")
        assert del_resp.status_code == 204

        # Verify it's gone
        gone_resp = client.get(f"/api/conversations/{conv_id}")
        assert gone_resp.status_code == 404


def test_memory_stats_integration():
    """Memory stats endpoint returns structured data even when memory is unavailable."""
    _reset_db_engine()
    from adam.api.main import app
    with TestClient(app) as client:
        resp = client.get("/api/memories/stats", params={"project_id": "test-stats"})
        assert resp.status_code == 200
        data = resp.json()
        assert "total_memories" in data
        assert "memory_types" in data
        assert isinstance(data["total_memories"], int)


def test_memory_types_endpoint():
    """Memory types endpoint returns available types."""
    _reset_db_engine()
    from adam.api.main import app
    with TestClient(app) as client:
        resp = client.get("/api/memory-types")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data
        assert "types" in data


# ---------------------------------------------------------------------------
# Phase 3 integration tests -- multi-agent reasoning system
# ---------------------------------------------------------------------------

def test_multi_agent_imports():
    """All multi-agent components are importable."""
    from adam.agents import (
        Agent, AgentConfig, Scratchpad, ScratchpadEntry,
        AgentRole, EntryType, Orchestrator, OrchestrationPattern,
        CostGuard, FallbackSynthesizer,
        create_reasoner, create_coder, create_researcher,
        create_critic, create_synthesizer,
    )
    assert Orchestrator is not None
    assert len(AgentRole) == 5
    assert len(EntryType) == 9  # 7 original + REBUTTAL + REACTION


def test_multi_agent_trigger_detection():
    """Orchestrator correctly identifies multi-agent queries."""
    from adam.agents import Orchestrator
    orch = Orchestrator.__new__(Orchestrator)

    # Should trigger
    assert orch.should_use_multi_agent("Design a distributed caching system", "complex")
    assert orch.should_use_multi_agent("Compare React vs Vue for our project", "complex")

    # Should NOT trigger
    assert not orch.should_use_multi_agent("What time is it?", "simple")
    assert not orch.should_use_multi_agent("How do I print in Python?", "moderate")


def test_pattern_selection():
    """Correct pattern selected based on query type."""
    from adam.agents import Orchestrator, OrchestrationPattern
    orch = Orchestrator.__new__(Orchestrator)

    assert orch.select_pattern("Implement a REST API") == OrchestrationPattern.SEQUENTIAL
    assert orch.select_pattern("Compare pros and cons of SQL vs NoSQL") == OrchestrationPattern.DEBATE


def test_scratchpad_visibility_rules():
    """Agents see only what they should see."""
    from adam.agents import Scratchpad, ScratchpadEntry, AgentRole, EntryType

    pad = Scratchpad(query="Design an API")
    pad.add_entry(ScratchpadEntry(
        agent_role=AgentRole.REASONER, agent_name="Reasoner",
        content="Plan: 3 endpoints", entry_type=EntryType.PLAN
    ))
    pad.add_entry(ScratchpadEntry(
        agent_role=AgentRole.CODER, agent_name="Coder",
        content="def endpoint():", entry_type=EntryType.CODE
    ))
    pad.add_entry(ScratchpadEntry(
        agent_role=AgentRole.CRITIC, agent_name="Critic",
        content="Missing error handling", entry_type=EntryType.CRITIQUE
    ))

    # Reasoner sees nothing (works from query only)
    assert len(pad._get_visible_entries(AgentRole.REASONER)) == 0

    # Coder sees plan but not critique
    coder_visible = pad._get_visible_entries(AgentRole.CODER)
    assert any(e.entry_type == EntryType.PLAN for e in coder_visible)
    assert not any(e.entry_type == EntryType.CRITIQUE for e in coder_visible)

    # Synthesizer sees everything
    assert len(pad._get_visible_entries(AgentRole.SYNTHESIZER)) == 3


def test_cost_guard_budget():
    """Cost guard enforces budget limits."""
    from adam.agents import CostGuard
    guard = CostGuard(budget=0.10)
    assert guard.can_afford(0.05)
    guard.record_spend("agent1", 0.08)
    assert not guard.can_afford(0.05)
    assert guard.suggest_action() == "cheap_model"


def test_fallback_synthesis():
    """Fallback synthesizer produces response from scratchpad."""
    from adam.agents import Scratchpad, ScratchpadEntry, AgentRole, EntryType, FallbackSynthesizer

    pad = Scratchpad(query="test")
    pad.add_entry(ScratchpadEntry(
        agent_role=AgentRole.REASONER, agent_name="R",
        content="Step 1: Do X", entry_type=EntryType.PLAN
    ))
    pad.add_entry(ScratchpadEntry(
        agent_role=AgentRole.CODER, agent_name="C",
        content="def x(): pass", entry_type=EntryType.CODE
    ))

    result = FallbackSynthesizer.synthesize(pad)
    assert "Step 1" in result
    assert "def x" in result


def test_llm_service_has_multi_agent():
    """LLMService has generate_with_agents method."""
    from adam.services.llm_service import LLMService
    assert hasattr(LLMService, 'generate_with_agents')
    assert callable(getattr(LLMService, 'generate_with_agents'))
