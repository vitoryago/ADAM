"""Tests for the Deep Discussion API router."""

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """TestClient for the ADAM app (creates DB tables via lifespan)."""
    from adam.api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def project_id(client):
    """Create a test project and return its id."""
    resp = client.post("/api/projects", json={
        "name": "Deep Discussion Test Project",
        "description": "Project for deep discussion API tests",
    })
    assert resp.status_code == 201
    return resp.json()["id"]


@pytest.fixture(scope="module")
def conversation_id(client, project_id):
    """Create a test conversation and return its id."""
    resp = client.post(f"/api/projects/{project_id}/conversations", json={
        "title": "Test Conversation for Deep Discussion",
    })
    assert resp.status_code == 201
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Router Registration
# ---------------------------------------------------------------------------

class TestRouterRegistration:
    """Verify the deep-discussion router is properly included in the app."""

    def test_deep_discussion_router_registered(self, client):
        """The /api/deep-discussion/* routes should be discoverable."""
        from adam.api.main import app
        routes = [r.path for r in app.routes]
        assert any("/api/deep-discussion" in r for r in routes), (
            f"No /api/deep-discussion routes found in {routes}"
        )

    def test_deep_discussion_endpoints_exist(self, client):
        from adam.api.main import app
        routes = [r.path for r in app.routes]
        expected = [
            "/api/deep-discussion/sessions",
            "/api/deep-discussion/sessions/{session_id}",
            "/api/deep-discussion/sessions/{session_id}/config",
            "/api/deep-discussion/sessions/{session_id}/start",
            "/api/deep-discussion/sessions/{session_id}/replay",
            "/api/deep-discussion/sessions/from-conversation/{conversation_id}",
        ]
        for endpoint in expected:
            assert endpoint in routes, f"Endpoint {endpoint} not found in app routes"


# ---------------------------------------------------------------------------
# Create Session
# ---------------------------------------------------------------------------

class TestCreateSession:
    """POST /sessions creates a session with smart defaults."""

    def test_create_session_returns_smart_defaults(self, client, project_id):
        resp = client.post("/api/deep-discussion/sessions", json={
            "project_id": project_id,
            "question": "How should we structure our data pipeline?",
        })
        assert resp.status_code == 201
        data = resp.json()

        assert data["status"] == "configuring"
        assert data["question"] == "How should we structure our data pipeline?"
        assert data["pattern"] == "peer_review"  # default
        assert data["project_id"] == project_id
        assert data["budget"] == 2.0
        assert data["total_cost"] == 0.0

        # Smart defaults should have been applied
        assignments = data["model_assignments"]
        assert "reasoner" in assignments
        assert "coder" in assignments
        assert "critic" in assignments
        assert "synthesizer" in assignments

    def test_create_session_custom_pattern(self, client, project_id):
        resp = client.post("/api/deep-discussion/sessions", json={
            "project_id": project_id,
            "question": "Debate this architecture",
            "pattern": "debate",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["pattern"] == "debate"

    def test_create_session_with_conversation(self, client, project_id, conversation_id):
        resp = client.post("/api/deep-discussion/sessions", json={
            "project_id": project_id,
            "question": "What about this topic?",
            "conversation_id": conversation_id,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["conversation_id"] == conversation_id

    def test_create_session_invalid_project_returns_404(self, client):
        resp = client.post("/api/deep-discussion/sessions", json={
            "project_id": "nonexistent-project-id",
            "question": "This should fail",
        })
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Update Config
# ---------------------------------------------------------------------------

class TestUpdateConfig:
    """PUT /sessions/{session_id}/config updates model assignments and budget."""

    def test_update_config_changes_budget(self, client, project_id):
        # Create a session first
        create_resp = client.post("/api/deep-discussion/sessions", json={
            "project_id": project_id,
            "question": "Budget update test",
        })
        session_id = create_resp.json()["id"]

        # Update budget only
        resp = client.put(f"/api/deep-discussion/sessions/{session_id}/config", json={
            "budget": 5.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["budget"] == 5.0

    def test_update_config_changes_model_assignments(self, client, project_id):
        create_resp = client.post("/api/deep-discussion/sessions", json={
            "project_id": project_id,
            "question": "Model assignment test",
        })
        session_id = create_resp.json()["id"]

        new_assignments = {"reasoner": "gpt-5.4-2026-03-05", "coder": "o3", "critic": "gemini-2.5-pro", "synthesizer": "claude-opus-4-6"}
        resp = client.put(f"/api/deep-discussion/sessions/{session_id}/config", json={
            "model_assignments": new_assignments,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_assignments"] == new_assignments

    def test_update_config_changes_pattern(self, client, project_id):
        create_resp = client.post("/api/deep-discussion/sessions", json={
            "project_id": project_id,
            "question": "Pattern update test",
        })
        session_id = create_resp.json()["id"]

        resp = client.put(f"/api/deep-discussion/sessions/{session_id}/config", json={
            "pattern": "debate",
        })
        assert resp.status_code == 200
        assert resp.json()["pattern"] == "debate"

    def test_update_config_nonexistent_session_returns_404(self, client):
        resp = client.put("/api/deep-discussion/sessions/nonexistent-id/config", json={
            "budget": 5.0,
        })
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Get Session
# ---------------------------------------------------------------------------

class TestGetSession:
    """GET /sessions/{session_id} returns correct data."""

    def test_get_session_returns_data(self, client, project_id):
        create_resp = client.post("/api/deep-discussion/sessions", json={
            "project_id": project_id,
            "question": "Get session test",
        })
        session_id = create_resp.json()["id"]

        resp = client.get(f"/api/deep-discussion/sessions/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == session_id
        assert data["question"] == "Get session test"
        assert data["status"] == "configuring"

    def test_get_session_nonexistent_returns_404(self, client):
        resp = client.get("/api/deep-discussion/sessions/nonexistent-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# List Sessions
# ---------------------------------------------------------------------------

class TestListSessions:
    """GET /sessions returns array of sessions for a project."""

    def test_list_sessions_returns_array(self, client, project_id):
        resp = client.get(f"/api/deep-discussion/sessions?project_id={project_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        # All sessions belong to the same project
        for session in data:
            assert session["project_id"] == project_id

    def test_list_sessions_empty_project(self, client):
        # Create a fresh project with no sessions
        proj_resp = client.post("/api/projects", json={
            "name": "Empty Project",
        })
        empty_project_id = proj_resp.json()["id"]

        resp = client.get(f"/api/deep-discussion/sessions?project_id={empty_project_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 0


# ---------------------------------------------------------------------------
# Replay Session
# ---------------------------------------------------------------------------

class TestReplaySession:
    """POST /sessions/{session_id}/replay creates a clone with new id."""

    def test_replay_creates_new_session_with_same_config(self, client, project_id):
        # Create the original session
        create_resp = client.post("/api/deep-discussion/sessions", json={
            "project_id": project_id,
            "question": "Replay test question",
            "pattern": "debate",
        })
        original = create_resp.json()
        original_id = original["id"]

        # Replay it
        resp = client.post(f"/api/deep-discussion/sessions/{original_id}/replay")
        assert resp.status_code == 201
        cloned = resp.json()

        # New id, same config
        assert cloned["id"] != original_id
        assert cloned["question"] == original["question"]
        assert cloned["pattern"] == original["pattern"]
        assert cloned["model_assignments"] == original["model_assignments"]
        assert cloned["budget"] == original["budget"]
        assert cloned["status"] == "configuring"

    def test_replay_nonexistent_session_returns_404(self, client):
        resp = client.post("/api/deep-discussion/sessions/nonexistent-id/replay")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# From Conversation
# ---------------------------------------------------------------------------

class TestFromConversation:
    """POST /sessions/from-conversation/{conversation_id} creates session from chat."""

    def test_from_conversation_nonexistent_returns_404(self, client):
        resp = client.post(
            "/api/deep-discussion/sessions/from-conversation/nonexistent-conv-id",
            json={"question": "What about this?"},
        )
        assert resp.status_code == 404

    def test_from_conversation_creates_session(self, client, project_id, conversation_id):
        resp = client.post(
            f"/api/deep-discussion/sessions/from-conversation/{conversation_id}",
            json={"question": "Summarize this discussion"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["conversation_id"] == conversation_id
        assert "Summarize this discussion" in data["question"]
        assert data["status"] == "configuring"
        # Smart defaults applied
        assert "reasoner" in data["model_assignments"]
