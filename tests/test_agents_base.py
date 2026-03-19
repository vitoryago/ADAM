"""Tests for the agent foundation: base class, scratchpad, and roles."""
import pytest
from adam.agents.scratchpad import Scratchpad, ScratchpadEntry, AgentRole, EntryType
from adam.agents.base import Agent, AgentConfig
from adam.agents.roles import (
    create_reasoner, create_coder, create_researcher,
    create_critic, create_synthesizer,
    REASONER_PROMPT, CODER_PROMPT, CRITIC_PROMPT,
)


class TestScratchpad:
    def test_create_empty(self):
        pad = Scratchpad(query="How do I sort a list?")
        assert pad.query == "How do I sort a list?"
        assert len(pad.entries) == 0
        assert pad.total_cost == 0.0

    def test_add_entry(self):
        pad = Scratchpad(query="test")
        entry = ScratchpadEntry(
            agent_role=AgentRole.REASONER,
            agent_name="Reasoner",
            content="Step 1: Analyze. Step 2: Implement.",
            entry_type=EntryType.PLAN,
            cost=0.01,
            tokens=50,
        )
        pad.add_entry(entry)
        assert len(pad.entries) == 1
        assert pad.total_cost == 0.01

    def test_budget_tracking(self):
        pad = Scratchpad(query="test", budget=0.05)
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.REASONER, agent_name="R",
            content="plan", entry_type=EntryType.PLAN, cost=0.03
        ))
        assert pad.budget_remaining() == pytest.approx(0.02)
        assert not pad.is_over_budget()
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.CODER, agent_name="C",
            content="code", entry_type=EntryType.CODE, cost=0.03
        ))
        assert pad.is_over_budget()

    def test_visibility_reasoner_sees_nothing(self):
        pad = Scratchpad(query="test")
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.CODER, agent_name="C",
            content="old code", entry_type=EntryType.CODE
        ))
        visible = pad._get_visible_entries(AgentRole.REASONER)
        assert len(visible) == 0

    def test_visibility_coder_sees_plan(self):
        pad = Scratchpad(query="test")
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.REASONER, agent_name="R",
            content="the plan", entry_type=EntryType.PLAN
        ))
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.CRITIC, agent_name="Cr",
            content="critique", entry_type=EntryType.CRITIQUE
        ))
        visible = pad._get_visible_entries(AgentRole.CODER)
        assert len(visible) == 1
        assert visible[0].entry_type == EntryType.PLAN

    def test_visibility_synthesizer_sees_all(self):
        pad = Scratchpad(query="test")
        for role, etype in [(AgentRole.REASONER, EntryType.PLAN),
                             (AgentRole.CODER, EntryType.CODE),
                             (AgentRole.CRITIC, EntryType.CRITIQUE)]:
            pad.add_entry(ScratchpadEntry(
                agent_role=role, agent_name=role.value,
                content=f"{role.value} output", entry_type=etype
            ))
        visible = pad._get_visible_entries(AgentRole.SYNTHESIZER)
        assert len(visible) == 3

    def test_context_building(self):
        pad = Scratchpad(query="Design an API")
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.REASONER, agent_name="Reasoner",
            content="1. Define endpoints\n2. Add auth",
            entry_type=EntryType.PLAN
        ))
        context = pad.build_context_for_agent(AgentRole.CODER)
        assert "Design an API" in context
        assert "Define endpoints" in context

    def test_to_dict(self):
        pad = Scratchpad(query="test", budget=1.0)
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.REASONER, agent_name="R",
            content="plan", entry_type=EntryType.PLAN, cost=0.01
        ))
        d = pad.to_dict()
        assert d["query"] == "test"
        assert len(d["entries"]) == 1
        assert d["entries"][0]["agent_role"] == "reasoner"


class TestAgentConfig:
    def test_create_config(self):
        config = AgentConfig(
            role=AgentRole.REASONER,
            name="Reasoner",
            system_prompt="You are a reasoner.",
        )
        assert config.role == AgentRole.REASONER
        assert config.temperature == 0.7
        assert config.max_tokens == 4000


class TestRoles:
    def test_all_roles_have_prompts(self):
        assert len(REASONER_PROMPT) > 100
        assert len(CODER_PROMPT) > 100
        assert len(CRITIC_PROMPT) > 100

    def test_create_all_roles(self):
        class MockClient:
            pass
        client = MockClient()

        agents = [
            create_reasoner(client),
            create_coder(client),
            create_researcher(client),
            create_critic(client),
            create_synthesizer(client),
        ]

        roles = [a.config.role for a in agents]
        assert AgentRole.REASONER in roles
        assert AgentRole.CODER in roles
        assert AgentRole.CRITIC in roles
        assert AgentRole.SYNTHESIZER in roles

    def test_agent_temperatures(self):
        class MockClient:
            pass
        reasoner = create_reasoner(MockClient())
        coder = create_coder(MockClient())
        assert coder.config.temperature < reasoner.config.temperature  # Coder more deterministic


class TestAgent:
    def test_build_prompt(self):
        config = AgentConfig(
            role=AgentRole.CODER, name="Coder",
            system_prompt="You write code.",
        )

        class MockClient:
            pass

        agent = Agent(config, MockClient())
        pad = Scratchpad(query="Write a sort function")
        pad.add_entry(ScratchpadEntry(
            agent_role=AgentRole.REASONER, agent_name="Reasoner",
            content="Use quicksort algorithm",
            entry_type=EntryType.PLAN,
        ))

        prompt = agent._build_prompt(pad)
        assert "Write a sort function" in prompt
        assert "quicksort" in prompt
        assert "Coder" in prompt

    def test_build_prompt_with_focus(self):
        config = AgentConfig(
            role=AgentRole.CRITIC, name="Critic",
            system_prompt="You review code.",
        )
        agent = Agent(config, None)
        pad = Scratchpad(query="test")
        prompt = agent._build_prompt(pad, focus="Focus on error handling")
        assert "Focus on error handling" in prompt
