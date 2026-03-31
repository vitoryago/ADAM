"""Tests for LOCAL provider integration in AsyncLLMClient."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from adam.llm.async_client import AsyncLLMProvider, AsyncLLMClient
from adam.llm.config import ModelProvider
from adam.llm.local_provider import LocalModelProvider, LocalModel


class TestLocalProviderEnum:
    def test_local_in_async_provider_enum(self):
        assert AsyncLLMProvider.LOCAL.value == "local"

    def test_local_in_model_provider_enum(self):
        assert ModelProvider.LOCAL.value == "local"


class TestLocalModelRouting:
    def test_cloud_prefixes_still_route_correctly(self):
        """Existing cloud models are unaffected."""
        client = AsyncLLMClient()
        assert client._get_provider_for_model("grok-4.20-multi-agent-0309") == AsyncLLMProvider.XAI
        assert client._get_provider_for_model("claude-opus-4-6") == AsyncLLMProvider.ANTHROPIC
        assert client._get_provider_for_model("gpt-5.4-2026-03-05") == AsyncLLMProvider.OPENAI
        assert client._get_provider_for_model("gemini-3.1-pro-preview") == AsyncLLMProvider.GEMINI

    def test_local_model_routes_to_local_provider(self):
        """A model in the local registry routes to LOCAL."""
        client = AsyncLLMClient()
        mock_provider = LocalModelProvider(endpoints=[])
        mock_provider.models = {
            "qwen3.5:72b-q4_K_M": LocalModel(
                model_id="qwen3.5:72b-q4_K_M", display_name="Qwen 3.5 72B",
                backend="ollama", base_url="http://localhost:11434/v1",
                parameter_count=72, quantization="q4_K_M", available=True,
            ),
        }
        client._local_provider = mock_provider
        assert client._get_provider_for_model("qwen3.5:72b-q4_K_M") == AsyncLLMProvider.LOCAL

    def test_unknown_model_without_local_falls_back_to_xai(self):
        """Unknown model with no local match falls back to XAI (existing behavior)."""
        client = AsyncLLMClient()
        client._local_provider = LocalModelProvider(endpoints=[])
        assert client._get_provider_for_model("some-random-model") == AsyncLLMProvider.XAI

    def test_routing_safe_when_local_provider_is_none(self):
        """If _local_provider is None, unknown models still fall back to XAI."""
        client = AsyncLLMClient()
        client._local_provider = None
        assert client._get_provider_for_model("some-random-model") == AsyncLLMProvider.XAI

    def test_routing_safe_when_local_provider_attr_missing(self):
        """Old-style __new__ instantiation (no __init__) still works."""
        client = AsyncLLMClient.__new__(AsyncLLMClient)
        assert client._get_provider_for_model("gemini-3.1-pro-preview") == AsyncLLMProvider.GEMINI
        assert client._get_provider_for_model("some-random-model") == AsyncLLMProvider.XAI


class TestLocalCostTracking:
    def test_local_response_cost_is_zero(self):
        """Local models should report zero cost in AsyncLLMResponse."""
        from adam.llm.async_client import AsyncLLMResponse
        response = AsyncLLMResponse(
            content="test",
            model="qwen3.5:72b-q4_K_M",
            provider="local",
            total_tokens=1500,
            completion_tokens=500,
        )
        # The async client always sets cost=0.0 (no billing for local)
        assert response.cost == 0.0


class TestLocalCompletionDispatch:
    @pytest.mark.asyncio
    async def test_complete_single_routes_local_to_openai_compat(self):
        """_complete_single for LOCAL provider calls _complete_local."""
        client = AsyncLLMClient()

        mock_local_provider = LocalModelProvider(endpoints=[])
        mock_local_provider.models = {
            "qwen3.5:72b-q4_K_M": LocalModel(
                model_id="qwen3.5:72b-q4_K_M", display_name="Qwen 3.5 72B",
                backend="ollama", base_url="http://localhost:11434/v1",
                parameter_count=72, quantization="q4_K_M", available=True,
            ),
        }
        client._local_provider = mock_local_provider

        # Place a mock client at the expected tuple key
        mock_openai = MagicMock()
        client.clients[(AsyncLLMProvider.LOCAL, "http://localhost:11434/v1")] = mock_openai

        from adam.llm.async_client import AsyncLLMResponse
        fake_response = AsyncLLMResponse(
            content="hello", model="qwen3.5:72b-q4_K_M", provider="local"
        )

        with patch.object(client, "_complete_local", new_callable=AsyncMock, return_value=fake_response) as mock_complete:
            result = await client._complete_single(
                AsyncLLMProvider.LOCAL, "qwen3.5:72b-q4_K_M",
                "say hello", None, 0.7, None,
            )
            mock_complete.assert_called_once()
            assert result.content == "hello"
