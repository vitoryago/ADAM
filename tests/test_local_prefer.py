# tests/test_local_prefer.py
"""Tests for prefer_local in Deep Discussion config."""
import pytest
from unittest.mock import patch, MagicMock

from adam.deep_discussion.config import get_smart_defaults
from adam.api.models import DeepDiscussionSessionCreate
from adam.llm.local_provider import LocalModel


class TestSmartDefaultsPreferLocal:
    def test_prefer_local_false_returns_cloud_defaults(self):
        result = get_smart_defaults(prefer_local=False)
        assert result["reasoner"] == "grok-4.20-multi-agent-0309"
        assert result["coder"] == "claude-opus-4-6"

    def test_prefer_local_true_with_no_local_models_returns_cloud(self):
        with patch("adam.deep_discussion.config._get_local_models", return_value=[]):
            result = get_smart_defaults(prefer_local=True)
        assert result["reasoner"] == "grok-4.20-multi-agent-0309"

    def test_prefer_local_true_assigns_best_local_model(self):
        mock_models = [
            LocalModel(
                model_id="small:7b", display_name="Small 7B", backend="ollama",
                base_url="http://localhost:11434/v1", parameter_count=7,
                quantization="q4_K_M", available=True,
            ),
            LocalModel(
                model_id="qwen3.5:72b-q4_K_M", display_name="Qwen 3.5 72B",
                backend="ollama", base_url="http://localhost:11434/v1",
                parameter_count=72, quantization="q4_K_M", available=True,
            ),
        ]
        with patch("adam.deep_discussion.config._get_local_models", return_value=mock_models):
            result = get_smart_defaults(prefer_local=True)
        assert result["reasoner"] == "qwen3.5:72b-q4_K_M"
        assert result["coder"] == "qwen3.5:72b-q4_K_M"
        assert result["critic"] == "qwen3.5:72b-q4_K_M"
        assert result["synthesizer"] == "qwen3.5:72b-q4_K_M"

    def test_backward_compatible_no_args(self):
        result = get_smart_defaults()
        assert "reasoner" in result
        assert "coder" in result


class TestSessionCreateSchema:
    def test_prefer_local_defaults_false(self):
        create = DeepDiscussionSessionCreate(
            project_id="proj-1", question="test", pattern="peer_review",
        )
        assert create.prefer_local is False

    def test_prefer_local_can_be_set(self):
        create = DeepDiscussionSessionCreate(
            project_id="proj-1", question="test", pattern="peer_review",
            prefer_local=True,
        )
        assert create.prefer_local is True
