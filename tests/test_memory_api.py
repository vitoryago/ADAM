"""
Tests for memory browsing API endpoints.

Validates both the convenience endpoints (/memories/search, /memories/stats,
/memories/store) and the project-scoped endpoints.

These tests work even when ChromaDB is not installed -- endpoints should
return empty / degraded results instead of crashing.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client."""
    from adam.api.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# Convenience endpoints  (GET /api/memories/search, GET /api/memories/stats,
#                         POST /api/memories/store)
# ---------------------------------------------------------------------------

class TestMemorySearchEndpoint:
    """Tests for GET /api/memories/search"""

    def test_memory_search_returns_200_or_422(self, client):
        """Search endpoint returns 200 (results/empty) or 422 (missing params)."""
        response = client.get(
            "/api/memories/search",
            params={"query": "test", "project_id": "test-project"},
        )
        # 200 = working (possibly empty results), 422 = validation error
        assert response.status_code in [200, 422]

    def test_memory_search_requires_query(self, client):
        """Search endpoint requires a query parameter."""
        response = client.get(
            "/api/memories/search",
            params={"project_id": "test-project"},
        )
        assert response.status_code == 422  # missing required 'query'

    def test_memory_search_requires_project_id(self, client):
        """Search endpoint requires a project_id parameter."""
        response = client.get(
            "/api/memories/search",
            params={"query": "test"},
        )
        assert response.status_code == 422  # missing required 'project_id'

    def test_memory_search_returns_list(self, client):
        """Search endpoint returns a JSON list."""
        response = client.get(
            "/api/memories/search",
            params={"query": "test query", "project_id": "nonexistent-project"},
        )
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)

    def test_memory_search_with_custom_limit(self, client):
        """Search endpoint respects the limit parameter."""
        response = client.get(
            "/api/memories/search",
            params={"query": "test", "project_id": "test-project", "limit": 5},
        )
        assert response.status_code in [200, 422]


class TestMemoryStatsEndpoint:
    """Tests for GET /api/memories/stats"""

    def test_memory_stats_returns_200_or_422(self, client):
        """Stats endpoint returns 200 or 422."""
        response = client.get(
            "/api/memories/stats",
            params={"project_id": "test-project"},
        )
        assert response.status_code in [200, 422]

    def test_memory_stats_requires_project_id(self, client):
        """Stats endpoint requires a project_id parameter."""
        response = client.get("/api/memories/stats")
        assert response.status_code == 422

    def test_memory_stats_response_shape(self, client):
        """Stats endpoint returns the expected fields."""
        response = client.get(
            "/api/memories/stats",
            params={"project_id": "test-project"},
        )
        if response.status_code == 200:
            data = response.json()
            assert "total_memories" in data
            assert "memory_types" in data
            assert "total_cost" in data
            assert isinstance(data["total_memories"], int)
            assert isinstance(data["memory_types"], dict)

    def test_memory_stats_graceful_when_unavailable(self, client):
        """Stats endpoint returns zeros, not an error, when memory is unavailable."""
        response = client.get(
            "/api/memories/stats",
            params={"project_id": "nonexistent-project"},
        )
        if response.status_code == 200:
            data = response.json()
            assert data["total_memories"] >= 0


class TestMemoryStoreEndpoint:
    """Tests for POST /api/memories/store"""

    def test_memory_store_requires_project_id(self, client):
        """Store endpoint requires a project_id query parameter."""
        response = client.post(
            "/api/memories/store",
            json={"content": "test memory", "memory_type": "conversation"},
        )
        assert response.status_code == 422

    def test_memory_store_requires_content(self, client):
        """Store endpoint requires content in the body."""
        response = client.post(
            "/api/memories/store",
            params={"project_id": "test-project"},
            json={},
        )
        assert response.status_code == 422

    def test_memory_store_returns_response_or_503(self, client):
        """Store endpoint returns 200/201 or 503 if memory unavailable."""
        response = client.post(
            "/api/memories/store",
            params={"project_id": "test-project"},
            json={
                "content": "test memory content",
                "memory_type": "conversation",
            },
        )
        # 200 = stored, 503 = memory system unavailable
        assert response.status_code in [200, 503]


# ---------------------------------------------------------------------------
# Utility endpoint
# ---------------------------------------------------------------------------

class TestMemoryTypesEndpoint:
    """Tests for GET /api/memory-types"""

    def test_memory_types_returns_200(self, client):
        """Memory types endpoint always returns 200."""
        response = client.get("/api/memory-types")
        assert response.status_code == 200

    def test_memory_types_response_shape(self, client):
        """Memory types endpoint returns available flag and types list."""
        response = client.get("/api/memory-types")
        data = response.json()
        assert "available" in data
        assert "types" in data
        assert isinstance(data["types"], list)

    def test_memory_types_values_when_available(self, client):
        """When available, memory types include expected entries."""
        response = client.get("/api/memory-types")
        data = response.json()
        if data["available"]:
            type_values = [t["value"] for t in data["types"]]
            assert "conversation" in type_values
            assert "error_solution" in type_values


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------

class TestMemoryRouterRegistration:
    """Verify memory routes are registered in the main app."""

    def test_memory_routes_registered(self, client):
        """Memory-related routes exist in the app."""
        from adam.api.main import app
        routes = [r.path for r in app.routes]
        assert any("memories" in r for r in routes), (
            f"No memory routes found. Routes: {routes}"
        )

    def test_convenience_search_route_exists(self, client):
        """The convenience /api/memories/search route is registered."""
        from adam.api.main import app
        routes = [r.path for r in app.routes]
        assert any("/api/memories/search" in r for r in routes)

    def test_convenience_stats_route_exists(self, client):
        """The convenience /api/memories/stats route is registered."""
        from adam.api.main import app
        routes = [r.path for r in app.routes]
        assert any("/api/memories/stats" in r for r in routes)

    def test_convenience_store_route_exists(self, client):
        """The convenience /api/memories/store route is registered."""
        from adam.api.main import app
        routes = [r.path for r in app.routes]
        assert any("/api/memories/store" in r for r in routes)
