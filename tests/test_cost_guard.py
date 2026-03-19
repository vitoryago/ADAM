"""Tests for CostGuard and FallbackSynthesizer."""

import pytest
from adam.agents.cost_guard import CostGuard
from adam.agents.synthesis import FallbackSynthesizer
from adam.agents.scratchpad import (
    Scratchpad,
    ScratchpadEntry,
    AgentRole,
    EntryType,
)


# ---------------------------------------------------------------------------
# CostGuard: budget tracking
# ---------------------------------------------------------------------------

class TestCostGuardBudgetTracking:
    """Test can_afford, record_spend, and remaining."""

    def test_initial_state(self):
        guard = CostGuard(budget=1.00)
        assert guard.budget == 1.00
        assert guard.spent == 0.0
        assert guard.remaining() == 1.00
        assert guard.history == []

    def test_can_afford_within_budget(self):
        guard = CostGuard(budget=0.50)
        assert guard.can_afford(0.10) is True
        assert guard.can_afford(0.50) is True

    def test_can_afford_over_budget(self):
        guard = CostGuard(budget=0.50)
        assert guard.can_afford(0.51) is False

    def test_record_spend_updates_state(self):
        guard = CostGuard(budget=0.50)
        guard.record_spend("reasoner", 0.02)
        assert guard.spent == pytest.approx(0.02)
        assert guard.remaining() == pytest.approx(0.48)
        assert guard.history == [("reasoner", 0.02)]

    def test_multiple_spends(self):
        guard = CostGuard(budget=0.50)
        guard.record_spend("reasoner", 0.02)
        guard.record_spend("coder", 0.05)
        guard.record_spend("critic", 0.02)
        assert guard.spent == pytest.approx(0.09)
        assert guard.remaining() == pytest.approx(0.41)
        assert len(guard.history) == 3

    def test_remaining_never_negative(self):
        guard = CostGuard(budget=0.05)
        guard.record_spend("coder", 0.10)
        assert guard.remaining() == 0.0

    def test_can_afford_after_spending(self):
        guard = CostGuard(budget=0.10)
        guard.record_spend("reasoner", 0.08)
        assert guard.can_afford(0.02) is True
        assert guard.can_afford(0.03) is False


# ---------------------------------------------------------------------------
# CostGuard: suggestion logic
# ---------------------------------------------------------------------------

class TestCostGuardSuggestions:
    """Test suggest_action thresholds."""

    def test_suggest_continue_when_plenty_of_budget(self):
        guard = CostGuard(budget=0.50)
        assert guard.suggest_action() == "continue"

    def test_suggest_cheap_model_when_low_budget(self):
        guard = CostGuard(budget=0.50)
        guard.record_spend("agents", 0.46)  # remaining = 0.04
        assert guard.suggest_action() == "cheap_model"

    def test_suggest_skip_when_no_budget(self):
        guard = CostGuard(budget=0.50)
        guard.record_spend("agents", 0.50)  # remaining = 0.00
        assert guard.suggest_action() == "skip"

    def test_suggest_skip_when_over_budget(self):
        guard = CostGuard(budget=0.50)
        guard.record_spend("agents", 0.55)  # over-spent
        assert guard.suggest_action() == "skip"

    def test_boundary_cheap_model(self):
        # remaining == 0.01 => cheap_model (>= 0.01, < 0.05)
        guard = CostGuard(budget=0.50)
        guard.record_spend("agents", 0.49)
        assert guard.suggest_action() == "cheap_model"

    def test_boundary_continue(self):
        # remaining >= 0.05 => continue
        guard = CostGuard(budget=1.00)
        guard.record_spend("agents", 0.95)  # remaining = 0.05
        assert guard.suggest_action() == "continue"


# ---------------------------------------------------------------------------
# CostGuard: cost estimation
# ---------------------------------------------------------------------------

class TestCostGuardEstimation:
    """Test estimate_agent_cost."""

    def test_known_roles(self):
        guard = CostGuard()
        assert guard.estimate_agent_cost("reasoner") == 0.02
        assert guard.estimate_agent_cost("coder") == 0.05
        assert guard.estimate_agent_cost("researcher") == 0.03
        assert guard.estimate_agent_cost("critic") == 0.02
        assert guard.estimate_agent_cost("synthesizer") == 0.03

    def test_unknown_role_defaults(self):
        guard = CostGuard()
        assert guard.estimate_agent_cost("unknown_role") == 0.03

    def test_model_param_accepted(self):
        guard = CostGuard()
        # model param is accepted but currently unused
        cost = guard.estimate_agent_cost("coder", model="gpt-4")
        assert cost == 0.05


# ---------------------------------------------------------------------------
# CostGuard: report generation
# ---------------------------------------------------------------------------

class TestCostGuardReport:
    """Test get_report."""

    def test_initial_report(self):
        guard = CostGuard(budget=0.50)
        report = guard.get_report()
        assert report["budget"] == 0.50
        assert report["spent"] == 0.0
        assert report["remaining"] == 0.50
        assert report["history"] == []
        assert report["suggestion"] == "continue"

    def test_report_after_spending(self):
        guard = CostGuard(budget=0.10)
        guard.record_spend("reasoner", 0.02)
        guard.record_spend("coder", 0.05)
        report = guard.get_report()
        assert report["spent"] == pytest.approx(0.07)
        assert report["remaining"] == pytest.approx(0.03)
        assert len(report["history"]) == 2
        assert report["suggestion"] == "cheap_model"

    def test_report_over_budget(self):
        guard = CostGuard(budget=0.05)
        guard.record_spend("coder", 0.06)
        report = guard.get_report()
        assert report["remaining"] == 0.0
        assert report["suggestion"] == "skip"


# ---------------------------------------------------------------------------
# FallbackSynthesizer: synthesize
# ---------------------------------------------------------------------------

def _make_entry(role: AgentRole, entry_type: EntryType, content: str) -> ScratchpadEntry:
    """Helper to create a ScratchpadEntry."""
    return ScratchpadEntry(
        agent_role=role,
        agent_name=role.value,
        content=content,
        entry_type=entry_type,
    )


class TestFallbackSynthesizerSynthesize:
    """Test FallbackSynthesizer.synthesize with various scratchpad states."""

    def test_empty_scratchpad(self):
        pad = Scratchpad(query="test")
        result = FallbackSynthesizer.synthesize(pad)
        assert "wasn't able to complete" in result

    def test_plan_only(self):
        pad = Scratchpad(query="test")
        pad.add_entry(_make_entry(AgentRole.REASONER, EntryType.PLAN, "Step 1: Do X"))
        result = FallbackSynthesizer.synthesize(pad)
        assert "## Approach" in result
        assert "Step 1: Do X" in result

    def test_code_only(self):
        pad = Scratchpad(query="test")
        pad.add_entry(_make_entry(AgentRole.CODER, EntryType.CODE, "def foo(): pass"))
        result = FallbackSynthesizer.synthesize(pad)
        assert "## Implementation" in result
        assert "def foo(): pass" in result

    def test_research_only(self):
        pad = Scratchpad(query="test")
        pad.add_entry(_make_entry(AgentRole.RESEARCHER, EntryType.RESEARCH, "Found relevant docs"))
        result = FallbackSynthesizer.synthesize(pad)
        assert "## Research & Context" in result

    def test_critique_only(self):
        pad = Scratchpad(query="test")
        pad.add_entry(_make_entry(AgentRole.CRITIC, EntryType.CRITIQUE, "Looks good overall"))
        result = FallbackSynthesizer.synthesize(pad)
        assert "## Review Notes" in result

    def test_full_scratchpad(self):
        pad = Scratchpad(query="test")
        pad.add_entry(_make_entry(AgentRole.REASONER, EntryType.PLAN, "Plan"))
        pad.add_entry(_make_entry(AgentRole.CODER, EntryType.CODE, "Code"))
        pad.add_entry(_make_entry(AgentRole.RESEARCHER, EntryType.RESEARCH, "Research"))
        pad.add_entry(_make_entry(AgentRole.CRITIC, EntryType.CRITIQUE, "Critique"))
        result = FallbackSynthesizer.synthesize(pad)
        assert "## Approach" in result
        assert "## Implementation" in result
        assert "## Research & Context" in result
        assert "## Review Notes" in result

    def test_uses_latest_entry_per_type(self):
        pad = Scratchpad(query="test")
        pad.add_entry(_make_entry(AgentRole.REASONER, EntryType.PLAN, "Old plan"))
        pad.add_entry(_make_entry(AgentRole.REASONER, EntryType.PLAN, "New plan"))
        result = FallbackSynthesizer.synthesize(pad)
        assert "New plan" in result
        assert "Old plan" not in result


# ---------------------------------------------------------------------------
# FallbackSynthesizer: extract_best_response priority
# ---------------------------------------------------------------------------

class TestFallbackSynthesizerExtractBest:
    """Test extract_best_response priority order."""

    def test_prefers_synthesis(self):
        pad = Scratchpad(query="test")
        pad.add_entry(_make_entry(AgentRole.REASONER, EntryType.PLAN, "Plan"))
        pad.add_entry(_make_entry(AgentRole.CODER, EntryType.CODE, "Code"))
        pad.add_entry(_make_entry(AgentRole.SYNTHESIZER, EntryType.SYNTHESIS, "Final answer"))
        result = FallbackSynthesizer.extract_best_response(pad)
        assert result == "Final answer"

    def test_prefers_code_over_plan(self):
        pad = Scratchpad(query="test")
        pad.add_entry(_make_entry(AgentRole.REASONER, EntryType.PLAN, "Plan"))
        pad.add_entry(_make_entry(AgentRole.CODER, EntryType.CODE, "Code"))
        result = FallbackSynthesizer.extract_best_response(pad)
        assert result == "Code"

    def test_falls_back_to_plan(self):
        pad = Scratchpad(query="test")
        pad.add_entry(_make_entry(AgentRole.REASONER, EntryType.PLAN, "Plan"))
        result = FallbackSynthesizer.extract_best_response(pad)
        assert result == "Plan"

    def test_falls_back_to_last_entry(self):
        pad = Scratchpad(query="test")
        pad.add_entry(_make_entry(AgentRole.RESEARCHER, EntryType.RESEARCH, "Some research"))
        result = FallbackSynthesizer.extract_best_response(pad)
        assert result == "Some research"

    def test_empty_scratchpad(self):
        pad = Scratchpad(query="test")
        result = FallbackSynthesizer.extract_best_response(pad)
        assert result == "No response generated."
