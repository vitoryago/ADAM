"""Tests for new EntryType values (REBUTTAL, REACTION) and PEER_REVIEW pattern."""

import pytest
from adam.agents.scratchpad import EntryType, AgentRole, Scratchpad, ScratchpadEntry
from adam.agents.orchestrator import OrchestrationPattern


class TestNewEntryTypes:
    def test_rebuttal_exists_with_correct_value(self):
        assert EntryType.REBUTTAL.value == "rebuttal"

    def test_reaction_exists_with_correct_value(self):
        assert EntryType.REACTION.value == "reaction"

    def test_rebuttal_is_entry_type_member(self):
        assert EntryType.REBUTTAL in EntryType

    def test_reaction_is_entry_type_member(self):
        assert EntryType.REACTION in EntryType


class TestScratchpadFilterByNewTypes:
    def _make_scratchpad(self):
        pad = Scratchpad(query="test query")
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.CODER,
            agent_name="producer",
            content="initial proposal",
            entry_type=EntryType.CODE,
        ))
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.CRITIC,
            agent_name="reviewer",
            content="here is my critique",
            entry_type=EntryType.CRITIQUE,
        ))
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.CODER,
            agent_name="producer",
            content="here is my rebuttal",
            entry_type=EntryType.REBUTTAL,
        ))
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.CRITIC,
            agent_name="reviewer",
            content="here is my reaction to the rebuttal",
            entry_type=EntryType.REACTION,
        ))
        return pad

    def test_filter_by_rebuttal_returns_correct_entries(self):
        pad = self._make_scratchpad()
        rebuttals = pad.get_entries_by_type(EntryType.REBUTTAL)
        assert len(rebuttals) == 1
        assert rebuttals[0].content == "here is my rebuttal"
        assert rebuttals[0].entry_type == EntryType.REBUTTAL

    def test_filter_by_reaction_returns_correct_entries(self):
        pad = self._make_scratchpad()
        reactions = pad.get_entries_by_type(EntryType.REACTION)
        assert len(reactions) == 1
        assert reactions[0].content == "here is my reaction to the rebuttal"
        assert reactions[0].entry_type == EntryType.REACTION

    def test_filter_by_rebuttal_excludes_other_types(self):
        pad = self._make_scratchpad()
        rebuttals = pad.get_entries_by_type(EntryType.REBUTTAL)
        for entry in rebuttals:
            assert entry.entry_type != EntryType.CODE
            assert entry.entry_type != EntryType.CRITIQUE
            assert entry.entry_type != EntryType.REACTION

    def test_filter_by_reaction_excludes_other_types(self):
        pad = self._make_scratchpad()
        reactions = pad.get_entries_by_type(EntryType.REACTION)
        for entry in reactions:
            assert entry.entry_type != EntryType.CRITIQUE
            assert entry.entry_type != EntryType.REBUTTAL


class TestPeerReviewPattern:
    def test_peer_review_exists_in_orchestration_pattern(self):
        assert OrchestrationPattern.PEER_REVIEW.value == "peer_review"

    def test_peer_review_is_pattern_member(self):
        assert OrchestrationPattern.PEER_REVIEW in OrchestrationPattern


class TestExistingPatternsUnchanged:
    def test_sequential_pattern_unchanged(self):
        assert OrchestrationPattern.SEQUENTIAL.value == "sequential"

    def test_debate_pattern_unchanged(self):
        assert OrchestrationPattern.DEBATE.value == "debate"

    def test_single_pattern_unchanged(self):
        assert OrchestrationPattern.SINGLE.value == "single"

    def test_existing_entry_types_unchanged(self):
        assert EntryType.QUERY.value == "query"
        assert EntryType.PLAN.value == "plan"
        assert EntryType.CODE.value == "code"
        assert EntryType.RESEARCH.value == "research"
        assert EntryType.CRITIQUE.value == "critique"
        assert EntryType.SYNTHESIS.value == "synthesis"
        assert EntryType.NOTE.value == "note"
