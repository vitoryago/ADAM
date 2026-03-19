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
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "4.0.0"


def test_api_health_check():
    from adam.api.main import app
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_routers_registered():
    from adam.api.main import app
    routes = [r.path for r in app.routes]
    assert "/health" in routes
    assert any("/api/projects" in r for r in routes)
    assert any("/api/conversations" in r for r in routes)
    assert any("/api/voice" in r for r in routes)
    assert any("/api/lineage" in r for r in routes)


def test_models_importable():
    from adam.api.models import (
        Project, Conversation, Message,
        ProjectCreate, ProjectResponse,
        ConversationCreate, ConversationResponse,
        MessageCreate, MessageResponse,
        MessageRole, Base
    )
    assert Project is not None
    assert MessageRole.USER.value == "user"
    assert Base is not None
