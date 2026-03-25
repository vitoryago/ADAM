"""Tests for Deep Discussion session management and configuration."""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Optional, AsyncGenerator

from adam.deep_discussion.config import (
    SessionConfig,
    get_smart_defaults,
    AVAILABLE_MODELS,
)
from adam.deep_discussion.session import DeepDiscussionSession


# ---------------------------------------------------------------------------
# SessionConfig tests
# ---------------------------------------------------------------------------


class TestSessionConfigDefaults:
    """SessionConfig should apply smart defaults automatically."""

    def test_budget_default_is_two(self):
        config = SessionConfig(question="What is recursion?", pattern="sequential")
        assert config.budget == 2.0

    def test_smart_defaults_applied_when_model_assignments_empty(self):
        config = SessionConfig(question="Explain sorting", pattern="sequential")
        assert config.model_assignments == get_smart_defaults()

    def test_explicit_model_assignments_not_overridden(self):
        custom = {"reasoner": "custom-model"}
        config = SessionConfig(
            question="Test question",
            pattern="sequential",
            model_assignments=custom,
        )
        assert config.model_assignments == custom

    def test_conversation_id_optional_none_by_default(self):
        config = SessionConfig(question="Q", pattern="sequential")
        assert config.conversation_id is None

    def test_conversation_context_optional_none_by_default(self):
        config = SessionConfig(question="Q", pattern="sequential")
        assert config.conversation_context is None

    def test_conversation_id_can_be_set(self):
        config = SessionConfig(
            question="Q", pattern="sequential", conversation_id="conv-123"
        )
        assert config.conversation_id == "conv-123"

    def test_conversation_context_can_be_set(self):
        config = SessionConfig(
            question="Q",
            pattern="sequential",
            conversation_context="Prior messages here.",
        )
        assert config.conversation_context == "Prior messages here."


class TestGetSmartDefaults:
    """get_smart_defaults() should return exactly 4 role-to-model mappings."""

    def test_returns_four_roles(self):
        defaults = get_smart_defaults()
        assert len(defaults) == 4

    def test_has_required_roles(self):
        defaults = get_smart_defaults()
        assert "reasoner" in defaults
        assert "coder" in defaults
        assert "critic" in defaults
        assert "synthesizer" in defaults

    def test_all_values_are_non_empty_strings(self):
        defaults = get_smart_defaults()
        for role, model_id in defaults.items():
            assert isinstance(model_id, str) and model_id, (
                f"role '{role}' has empty model id"
            )


class TestAvailableModels:
    """AVAILABLE_MODELS should be a non-empty dict with display labels."""

    def test_available_models_is_dict(self):
        assert isinstance(AVAILABLE_MODELS, dict)

    def test_available_models_is_non_empty(self):
        assert len(AVAILABLE_MODELS) > 0

    def test_all_values_are_strings(self):
        for label, model_id in AVAILABLE_MODELS.items():
            assert isinstance(label, str) and isinstance(model_id, str)


class TestPatternValidation:
    """SessionConfig should accept known patterns."""

    def test_sequential_pattern_accepted(self):
        config = SessionConfig(question="Q", pattern="sequential")
        assert config.pattern == "sequential"

    def test_debate_pattern_accepted(self):
        config = SessionConfig(question="Q", pattern="debate")
        assert config.pattern == "debate"

    def test_peer_review_pattern_accepted(self):
        config = SessionConfig(question="Q", pattern="peer_review")
        assert config.pattern == "peer_review"


# ---------------------------------------------------------------------------
# DeepDiscussionSession tests
# ---------------------------------------------------------------------------


class TestDeepDiscussionSessionInit:
    """DeepDiscussionSession should initialise to a known state."""

    def _make_session(self, pattern="sequential") -> DeepDiscussionSession:
        config = SessionConfig(question="What is caching?", pattern=pattern)
        return DeepDiscussionSession(config=config)

    def test_initial_status_is_configuring(self):
        session = self._make_session()
        assert session.status == "configuring"

    def test_id_is_non_empty_string(self):
        session = self._make_session()
        assert isinstance(session.id, str) and session.id

    def test_id_looks_like_uuid(self):
        import re
        session = self._make_session()
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        assert uuid_pattern.match(session.id), f"id {session.id!r} is not a UUID"

    def test_result_is_none_initially(self):
        session = self._make_session()
        assert session.result is None

    def test_scratchpad_data_is_none_initially(self):
        session = self._make_session()
        assert session.scratchpad_data is None

    def test_total_cost_is_zero_initially(self):
        session = self._make_session()
        assert session.total_cost == 0.0

    def test_completed_at_is_none_initially(self):
        session = self._make_session()
        assert session.completed_at is None

    def test_created_at_is_set(self):
        from datetime import datetime
        session = self._make_session()
        assert isinstance(session.created_at, datetime)

    def test_config_stored_on_session(self):
        config = SessionConfig(question="Q", pattern="debate")
        session = DeepDiscussionSession(config=config)
        assert session.config is config


class TestUpdateConfig:
    """update_config should mutate the session's config in-place."""

    def _make_session(self) -> DeepDiscussionSession:
        config = SessionConfig(question="Q", pattern="sequential")
        return DeepDiscussionSession(config=config)

    def test_update_model_assignments(self):
        session = self._make_session()
        new_assignments = {"reasoner": "model-x", "coder": "model-y",
                           "critic": "model-z", "synthesizer": "model-w"}
        session.update_config(model_assignments=new_assignments)
        assert session.config.model_assignments == new_assignments

    def test_update_pattern(self):
        session = self._make_session()
        session.update_config(pattern="debate")
        assert session.config.pattern == "debate"

    def test_update_budget(self):
        session = self._make_session()
        session.update_config(budget=5.0)
        assert session.config.budget == 5.0

    def test_update_none_leaves_field_unchanged(self):
        session = self._make_session()
        original_budget = session.config.budget
        session.update_config(pattern="debate")
        assert session.config.budget == original_budget


class TestGetPatternModelAssignments:
    """get_pattern_model_assignments should remap roles according to the pattern."""

    def _make_session(self, pattern: str) -> DeepDiscussionSession:
        config = SessionConfig(question="Q", pattern=pattern)
        return DeepDiscussionSession(config=config)

    # -- Sequential: pass-through -------

    def test_sequential_pass_through_has_four_keys(self):
        session = self._make_session("sequential")
        result = session.get_pattern_model_assignments()
        assert len(result) == 4

    def test_sequential_pass_through_preserves_reasoner(self):
        session = self._make_session("sequential")
        result = session.get_pattern_model_assignments()
        assert result["reasoner"] == session.config.model_assignments["reasoner"]

    def test_sequential_pass_through_preserves_coder(self):
        session = self._make_session("sequential")
        result = session.get_pattern_model_assignments()
        assert result["coder"] == session.config.model_assignments["coder"]

    def test_sequential_pass_through_preserves_critic(self):
        session = self._make_session("sequential")
        result = session.get_pattern_model_assignments()
        assert result["critic"] == session.config.model_assignments["critic"]

    def test_sequential_pass_through_preserves_synthesizer(self):
        session = self._make_session("sequential")
        result = session.get_pattern_model_assignments()
        assert result["synthesizer"] == session.config.model_assignments["synthesizer"]

    # -- Debate: remapping -------

    def test_debate_has_three_keys(self):
        session = self._make_session("debate")
        result = session.get_pattern_model_assignments()
        assert set(result.keys()) == {"debater_a", "debater_b", "reconciler"}

    def test_debate_debater_a_maps_to_reasoner_model(self):
        session = self._make_session("debate")
        result = session.get_pattern_model_assignments()
        assert result["debater_a"] == session.config.model_assignments["reasoner"]

    def test_debate_debater_b_maps_to_coder_model(self):
        session = self._make_session("debate")
        result = session.get_pattern_model_assignments()
        assert result["debater_b"] == session.config.model_assignments["coder"]

    def test_debate_reconciler_maps_to_critic_model(self):
        session = self._make_session("debate")
        result = session.get_pattern_model_assignments()
        assert result["reconciler"] == session.config.model_assignments["critic"]

    # -- Peer review: pass-through -------

    def test_peer_review_pass_through_has_four_keys(self):
        session = self._make_session("peer_review")
        result = session.get_pattern_model_assignments()
        assert len(result) == 4

    def test_peer_review_pass_through_preserves_reasoner(self):
        session = self._make_session("peer_review")
        result = session.get_pattern_model_assignments()
        assert result["reasoner"] == session.config.model_assignments["reasoner"]

    def test_peer_review_pass_through_preserves_synthesizer(self):
        session = self._make_session("peer_review")
        result = session.get_pattern_model_assignments()
        assert result["synthesizer"] == session.config.model_assignments["synthesizer"]
