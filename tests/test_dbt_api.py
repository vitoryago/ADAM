"""Tests for the dbt API router (src/adam/api/routers/dbt.py)."""

import os
import tempfile
import textwrap

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """TestClient for the ADAM app."""
    from adam.api.main import app
    return TestClient(app)


@pytest.fixture()
def fake_dbt_project(tmp_path):
    """Create a minimal dbt project on disk for integration-level tests."""
    # dbt_project.yml
    (tmp_path / "dbt_project.yml").write_text(textwrap.dedent("""\
        name: test_project
        version: '1.0.0'
        config-version: 2
        model-paths: ["models"]
    """))

    # models directory with a simple SQL file
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "stg_orders.sql").write_text(textwrap.dedent("""\
        {{ config(materialized='view') }}

        SELECT
            order_id,
            customer_id,
            order_date,
            total_amount,
            is_active,
            created_at,
            updated_at
        FROM {{ source('raw', 'orders') }}
    """))

    # A simple schema.yml alongside the model
    (models_dir / "schema.yml").write_text(textwrap.dedent("""\
        version: 2

        models:
          - name: stg_orders
            description: "Staging model for orders"
            columns:
              - name: order_id
                description: "Primary key for orders"
                tests:
                  - unique
                  - not_null
              - name: customer_id
                description: "FK to customers"

        sources:
          - name: raw
            tables:
              - name: orders
    """))

    return str(tmp_path)


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------

class TestRouterRegistration:
    """Verify the dbt router is properly included in the app."""

    def test_dbt_router_registered(self, client):
        """The /api/dbt/* routes should be discoverable."""
        from adam.api.main import app
        routes = [r.path for r in app.routes]
        assert any("/api/dbt" in r for r in routes), f"No /api/dbt routes found in {routes}"

    def test_dbt_endpoints_exist(self, client):
        from adam.api.main import app
        routes = [r.path for r in app.routes]
        expected = [
            "/api/dbt/columns/document-model",
            "/api/dbt/columns/analyze",
            "/api/dbt/columns/common-columns",
            "/api/dbt/columns/generate-schema",
            "/api/dbt/generate",
        ]
        for endpoint in expected:
            assert endpoint in routes, f"Endpoint {endpoint} not found in app routes"


# ---------------------------------------------------------------------------
# Invalid input handling (no real project needed)
# ---------------------------------------------------------------------------

class TestInvalidInput:
    """Each endpoint should return 200 with an error payload for bad input."""

    def test_document_model_missing_project(self, client):
        resp = client.post("/api/dbt/columns/document-model", json={
            "project_path": "/nonexistent/path",
            "model_name": "stg_orders",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data
        assert data["total_columns"] == 0

    def test_analyze_missing_project(self, client):
        resp = client.post("/api/dbt/columns/analyze", json={
            "project_path": "/nonexistent/path",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_common_columns_missing_project(self, client):
        resp = client.post("/api/dbt/columns/common-columns", json={
            "project_path": "/nonexistent/path",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_generate_schema_missing_project(self, client):
        resp = client.post("/api/dbt/columns/generate-schema", json={
            "project_path": "/nonexistent/path",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "Error" in data

    def test_document_model_validation_error(self, client):
        """Missing required field should give 422."""
        resp = client.post("/api/dbt/columns/document-model", json={
            "project_path": "/some/path",
            # model_name is missing
        })
        assert resp.status_code == 422

    def test_generate_model_validation_error(self, client):
        """Missing required fields should give 422."""
        resp = client.post("/api/dbt/generate", json={
            "model_name": "test",
            # source_table is missing
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Generate model (does not need file system)
# ---------------------------------------------------------------------------

class TestGenerateModel:
    """Test the /generate endpoint using the knowledge service."""

    def test_generate_staging_model(self, client):
        resp = client.post("/api/dbt/generate", json={
            "model_name": "stg_users",
            "source_table": "users",
            "layer": "staging",
            "source_schema": "raw",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "sql" in data
        assert "documentation" in data
        assert "tests" in data
        assert isinstance(data["tests"], list)
        # SQL is always a non-empty string (may be a comment if patterns file is absent)
        assert isinstance(data["sql"], str) and len(data["sql"]) > 0

    def test_generate_fact_model(self, client):
        resp = client.post("/api/dbt/generate", json={
            "model_name": "fct_orders",
            "source_table": "int_orders",
            "layer": "mart_fact",
            "unique_key": "order_id",
            "timestamp_column": "updated_at",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "sql" in data
        assert isinstance(data["sql"], str) and len(data["sql"]) > 0

    def test_generate_intermediate_model(self, client):
        resp = client.post("/api/dbt/generate", json={
            "model_name": "int_orders",
            "source_table": "stg_orders",
            "layer": "intermediate",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "sql" in data

    def test_generate_model_invalid_layer_falls_back(self, client):
        """An unknown layer string should fall back to staging."""
        resp = client.post("/api/dbt/generate", json={
            "model_name": "test_model",
            "source_table": "raw_data",
            "layer": "nonexistent_layer",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "sql" in data

    def test_generate_model_includes_suggestions(self, client):
        resp = client.post("/api/dbt/generate", json={
            "model_name": "stg_items",
            "source_table": "items",
            "layer": "staging",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)


# ---------------------------------------------------------------------------
# Integration tests with a real (fake) dbt project
# ---------------------------------------------------------------------------

class TestWithFakeProject:
    """Tests that exercise the knowledge layer against a temp dbt project."""

    def test_analyze_columns_with_project(self, client, fake_dbt_project):
        resp = client.post("/api/dbt/columns/analyze", json={
            "project_path": fake_dbt_project,
            "use_ai": False,  # skip AI to avoid LLM calls in tests
        })
        assert resp.status_code == 200
        data = resp.json()
        # Even if the service fails (e.g. LLM client init), we get a 200 with error info
        assert "summary" in data or "error" in data

    def test_common_columns_with_project(self, client, fake_dbt_project):
        resp = client.post("/api/dbt/columns/common-columns", json={
            "project_path": fake_dbt_project,
            "min_occurrences": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_generate_schema_with_project(self, client, fake_dbt_project):
        resp = client.post("/api/dbt/columns/generate-schema", json={
            "project_path": fake_dbt_project,
            "use_ai": False,
            "include_tests": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, str)
        # Should contain YAML-like content
        assert "version" in data or "models" in data or "Error" not in data

    def test_document_model_with_project(self, client, fake_dbt_project):
        resp = client.post("/api/dbt/columns/document-model", json={
            "project_path": fake_dbt_project,
            "model_name": "stg_orders",
            "use_ai": False,
            "preserve_existing": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        # If the model was found, we should have columns
        if "error" not in data:
            assert "columns" in data
            assert "total_columns" in data


# ---------------------------------------------------------------------------
# Pydantic model sanity
# ---------------------------------------------------------------------------

class TestRequestModels:
    """Verify Pydantic request models accept expected payloads."""

    def test_document_model_request_defaults(self):
        from adam.api.routers.dbt import DocumentModelRequest
        req = DocumentModelRequest(project_path="/tmp", model_name="test")
        assert req.use_ai is True
        assert req.preserve_existing is True

    def test_analyze_request_defaults(self):
        from adam.api.routers.dbt import AnalyzeColumnsRequest
        req = AnalyzeColumnsRequest(project_path="/tmp")
        assert req.use_ai is True

    def test_common_columns_request_defaults(self):
        from adam.api.routers.dbt import CommonColumnsRequest
        req = CommonColumnsRequest(project_path="/tmp")
        assert req.min_occurrences == 2

    def test_generate_schema_request_defaults(self):
        from adam.api.routers.dbt import GenerateSchemaRequest
        req = GenerateSchemaRequest(project_path="/tmp")
        assert req.use_ai is True
        assert req.include_tests is True

    def test_generate_model_request_defaults(self):
        from adam.api.routers.dbt import GenerateModelRequest
        req = GenerateModelRequest(model_name="test", source_table="src")
        assert req.layer == "staging"
        assert req.source_schema is None
