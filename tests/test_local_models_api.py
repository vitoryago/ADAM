"""Tests for GET /api/local-models endpoint."""
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport
from adam.api.main import app
from adam.llm.local_provider import LocalModel


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
class TestLocalModelsAPI:
    async def test_list_local_models_returns_200(self, client):
        resp = await client.get("/api/local-models")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_list_local_models_returns_model_data(self, client):
        mock_models = [
            LocalModel(
                model_id="qwen3.5:72b-q4_K_M",
                display_name="Qwen3.5 72B (Q4_K_M)",
                backend="ollama",
                base_url="http://localhost:11434/v1",
                parameter_count=72,
                quantization="q4_K_M",
                available=True,
            ),
        ]
        with patch(
            "adam.api.routers.local_models.get_local_provider",
            return_value=MagicMock(get_available_models=MagicMock(return_value=mock_models)),
        ):
            resp = await client.get("/api/local-models")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) >= 1
            first = data[0]
            assert first["model_id"] == "qwen3.5:72b-q4_K_M"
            assert first["parameter_count"] == 72
            assert first["available"] is True
