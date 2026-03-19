"""Smoke tests for services layer."""


def test_llm_service_importable():
    from adam.services.llm_service import LLMService, LLMResponse, StreamChunk
    assert LLMService is not None
    assert LLMResponse is not None
    assert StreamChunk is not None


def test_llm_response_dataclass():
    from adam.services.llm_service import LLMResponse
    resp = LLMResponse(content="test", model_used="mock", tokens_used=10, cost=0.0)
    assert resp.content == "test"
    assert resp.model_used == "mock"


def test_stream_chunk_dataclass():
    from adam.services.llm_service import StreamChunk
    chunk = StreamChunk(content="hello", model_used="mock")
    assert chunk.content == "hello"
    assert chunk.is_final is False


def test_voice_service_importable():
    from adam.services.voice_service import VoiceService, VoiceConfig, VoiceProvider
    assert VoiceService is not None
    assert VoiceConfig is not None
    assert VoiceProvider is not None


def test_voice_websocket_importable():
    from adam.services.voice_websocket import ElevenLabsWebSocket, WebSocketVoiceConfig
    assert ElevenLabsWebSocket is not None
    assert WebSocketVoiceConfig is not None


def test_voice_conversation_handler_importable():
    from adam.services.voice_conversation_handler import VoiceConversationHandler, ConversationState
    assert VoiceConversationHandler is not None
    assert ConversationState.IDLE.value == "idle"


def test_voice_response_formatter_importable():
    from adam.services.voice_response_formatter import VoiceResponseFormatter, VoiceResponse
    assert VoiceResponseFormatter is not None
    formatter = VoiceResponseFormatter()
    assert formatter is not None


def test_response_style_importable():
    from adam.services.response_style_service import ResponseStyleService, ResponseStyle
    assert ResponseStyleService is not None
    service = ResponseStyleService()
    assert service.get_current_style() == ResponseStyle.NORMAL


def test_response_style_all_styles():
    from adam.services.response_style_service import ResponseStyleService, ResponseStyle
    service = ResponseStyleService()
    styles = service.get_available_styles()
    assert "concise" in styles
    assert "normal" in styles
    assert "explanatory" in styles
    assert len(styles) == 7


def test_cost_monitor_importable():
    from adam.services.cost_monitor import CostMonitor, get_cost_monitor
    assert CostMonitor is not None
    assert get_cost_monitor is not None


def test_pricing_manager_importable():
    from adam.services.pricing_manager import PricingManager, CostOptimizer, get_pricing_manager
    assert PricingManager is not None
    assert CostOptimizer is not None
    assert get_pricing_manager is not None


def test_activity_tracker_importable():
    from adam.services.activity_tracker import ActivityTracker
    assert ActivityTracker is not None


def test_markdown_service_importable():
    from adam.services.markdown_service import MarkdownService, markdown_service
    assert MarkdownService is not None
    assert markdown_service is not None


def test_services_init_exports():
    from adam.services import LLMService, LLMResponse, StreamChunk
    assert LLMService is not None
    assert LLMResponse is not None
    assert StreamChunk is not None
