"""Tests for LocalModelProvider: discovery, registry, health checks."""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from adam.llm.local_provider import LocalModelProvider, LocalModel


class TestLocalModelParsing:
    def test_parse_model_id_extracts_parameter_count(self):
        provider = LocalModelProvider(endpoints=[])
        model = provider._parse_model_entry("qwen3.5:72b-q4_K_M", "http://localhost:11434/v1")
        assert model.model_id == "qwen3.5:72b-q4_K_M"
        assert model.parameter_count == 72
        assert model.quantization == "q4_K_M"

    def test_parse_model_id_no_quant(self):
        provider = LocalModelProvider(endpoints=[])
        model = provider._parse_model_entry("llama3:8b", "http://localhost:11434/v1")
        assert model.parameter_count == 8
        assert model.quantization == ""

    def test_parse_model_id_no_size(self):
        provider = LocalModelProvider(endpoints=[])
        model = provider._parse_model_entry("my-custom-model", "http://localhost:5000/v1")
        assert model.parameter_count == 0
        assert model.quantization == ""

    def test_display_name_cleaned(self):
        provider = LocalModelProvider(endpoints=[])
        model = provider._parse_model_entry("qwen3.5:72b-q4_K_M", "http://localhost:11434/v1")
        assert "Qwen3.5" in model.display_name or "qwen3.5" in model.display_name.lower()
        assert "72B" in model.display_name or "72b" in model.display_name.lower()


class TestLocalModelDiscovery:
    @pytest.mark.asyncio
    async def test_discover_from_openai_compatible_endpoint(self):
        """Discovery via /v1/models returns model list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "qwen3.5:72b-q4_K_M"},
                {"id": "deepseek-coder:33b-q4_K_M"},
            ]
        }

        provider = LocalModelProvider(endpoints=["http://localhost:11434"])
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            await provider.discover()

        assert len(provider.models) == 2
        assert "qwen3.5:72b-q4_K_M" in provider.models
        assert "deepseek-coder:33b-q4_K_M" in provider.models

    @pytest.mark.asyncio
    async def test_discover_unreachable_endpoint_returns_empty(self):
        """Unreachable endpoint produces no models, no error."""
        provider = LocalModelProvider(endpoints=["http://localhost:99999"])
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=Exception("Connection refused")):
            await provider.discover()

        assert len(provider.models) == 0

    @pytest.mark.asyncio
    async def test_discover_multiple_endpoints(self):
        """Multiple backends merge into one registry."""
        resp_a = MagicMock()
        resp_a.status_code = 200
        resp_a.json.return_value = {"data": [{"id": "model-a"}]}

        resp_b = MagicMock()
        resp_b.status_code = 200
        resp_b.json.return_value = {"data": [{"id": "model-b"}]}

        call_count = 0

        async def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "11434" in url:
                return resp_a
            return resp_b

        provider = LocalModelProvider(endpoints=[
            "http://localhost:11434",
            "http://localhost:8000",
        ])
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=mock_get):
            await provider.discover()

        assert "model-a" in provider.models
        assert "model-b" in provider.models


class TestLocalModelAvailability:
    def test_get_available_models_filters_unavailable(self):
        provider = LocalModelProvider(endpoints=[])
        provider.models = {
            "good": LocalModel(
                model_id="good", display_name="Good", backend="ollama",
                base_url="http://localhost:11434/v1", parameter_count=72,
                quantization="q4_K_M", available=True,
            ),
            "bad": LocalModel(
                model_id="bad", display_name="Bad", backend="ollama",
                base_url="http://localhost:11434/v1", parameter_count=7,
                quantization="q4_K_M", available=False,
            ),
        }
        available = provider.get_available_models()
        assert len(available) == 1
        assert available[0].model_id == "good"
