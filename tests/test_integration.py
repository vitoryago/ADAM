"""Integration tests -- verify the full stack works together."""
import os
from fastapi.testclient import TestClient


def _reset_db_engine():
    """Reset the global database engine so the next TestClient gets a fresh one."""
    import adam.database as db_mod
    db_mod._engine = None


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
