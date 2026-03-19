"""Tests for the sliding window conversation context in LLMService."""

from dataclasses import dataclass


@dataclass
class MockMessage:
    role: str
    content: str


class TestSlidingWindow:
    """Tests for _build_conversation_context sliding window logic."""

    def _make_service(self):
        """Create an LLMService instance without calling __init__."""
        from adam.services.llm_service import LLMService
        return LLMService.__new__(LLMService)

    def test_empty_history(self):
        service = self._make_service()
        result = service._build_conversation_context([])
        assert result == []

    def test_none_history(self):
        service = self._make_service()
        result = service._build_conversation_context(None)
        assert result == []

    def test_short_history(self):
        service = self._make_service()
        history = [MockMessage("user", "Hello"), MockMessage("assistant", "Hi!")]
        result = service._build_conversation_context(history, max_recent=10)
        assert len(result) == 2
        assert result[0]["content"] == "Hello"
        assert result[0]["role"] == "user"
        assert result[1]["content"] == "Hi!"
        assert result[1]["role"] == "assistant"

    def test_long_history_gets_summarized(self):
        service = self._make_service()
        history = [MockMessage("user", f"Message {i} " * 50) for i in range(25)]
        result = service._build_conversation_context(history, max_recent=10, max_summary=10)
        # Should have: 1 summary message + 10 recent = 11
        assert len(result) == 11
        assert result[0]["role"] == "system"
        assert "Summary" in result[0]["content"]

    def test_very_long_history_drops_oldest(self):
        service = self._make_service()
        history = [MockMessage("user", f"Msg {i}") for i in range(50)]
        result = service._build_conversation_context(history, max_recent=10, max_summary=20)
        # 1 summary + 10 recent = 11 (oldest 20 messages dropped)
        assert len(result) == 11

    def test_exact_boundary(self):
        service = self._make_service()
        history = [MockMessage("user", f"Msg {i}") for i in range(10)]
        result = service._build_conversation_context(history, max_recent=10)
        # All fit in recent window, no summary needed
        assert len(result) == 10
        assert all(r["role"] == "user" for r in result)

    def test_summary_truncates_long_content(self):
        service = self._make_service()
        long_content = "A" * 300
        history = [MockMessage("user", long_content)] * 15
        result = service._build_conversation_context(history, max_recent=10, max_summary=10)
        # The summary message should contain truncated previews
        summary_msg = result[0]
        assert summary_msg["role"] == "system"
        assert "..." in summary_msg["content"]
        # Each preview is 150 chars + "..."
        assert "A" * 150 in summary_msg["content"]
        assert "A" * 151 not in summary_msg["content"]

    def test_recent_messages_preserve_order(self):
        service = self._make_service()
        history = [MockMessage("user", f"Msg {i}") for i in range(20)]
        result = service._build_conversation_context(history, max_recent=5, max_summary=5)
        # Recent messages should be the last 5
        recent = [r for r in result if r["role"] != "system"]
        assert len(recent) == 5
        assert recent[0]["content"] == "Msg 15"
        assert recent[4]["content"] == "Msg 19"

    def test_summary_message_count_label(self):
        service = self._make_service()
        history = [MockMessage("user", f"Msg {i}") for i in range(15)]
        result = service._build_conversation_context(history, max_recent=10, max_summary=20)
        summary_msg = result[0]
        assert "5 messages" in summary_msg["content"]

    def test_dict_like_objects(self):
        """Verify fallback works with objects that lack .role/.content attributes."""
        service = self._make_service()
        history = ["plain string message 1", "plain string message 2"]
        result = service._build_conversation_context(history, max_recent=10)
        assert len(result) == 2
        assert result[0]["content"] == "plain string message 1"
        assert result[0]["role"] == "user"

    def test_mixed_roles(self):
        service = self._make_service()
        history = [
            MockMessage("user", "Question"),
            MockMessage("assistant", "Answer"),
            MockMessage("user", "Follow-up"),
            MockMessage("assistant", "More info"),
        ]
        result = service._build_conversation_context(history, max_recent=10)
        assert len(result) == 4
        roles = [r["role"] for r in result]
        assert roles == ["user", "assistant", "user", "assistant"]

    def test_single_message(self):
        service = self._make_service()
        history = [MockMessage("user", "Only one")]
        result = service._build_conversation_context(history, max_recent=10)
        assert len(result) == 1
        assert result[0]["content"] == "Only one"

    def test_no_summary_when_exactly_at_recent_boundary(self):
        """When history length equals max_recent, no summary should be produced."""
        service = self._make_service()
        history = [MockMessage("user", f"Msg {i}") for i in range(10)]
        result = service._build_conversation_context(history, max_recent=10, max_summary=5)
        assert len(result) == 10
        assert all(r["role"] != "system" for r in result)
