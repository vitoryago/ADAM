from types import SimpleNamespace
from unittest.mock import patch

import pytest


class _FakeUsage:
    total_tokens = 42
    output_tokens = 11


class _FakeResponse:
    def __init__(self, text: str):
        self.output_text = text
        self.usage = _FakeUsage()

    def model_dump(self):
        return {"output_text": self.output_text}


class _FakeResponsesClient:
    def __init__(self, response):
        self.calls = []
        self._response = response

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeAsyncOpenAI:
    last_instance = None

    def __init__(self, *args, **kwargs):
        self.responses = _FakeResponsesClient(_FakeResponse("search result"))
        _FakeAsyncOpenAI.last_instance = self


class _FakeEvent:
    def __init__(self, event_type: str, delta: str | None = None):
        self.type = event_type
        self.delta = delta


class _FakeAsyncStream:
    def __init__(self, events):
        self._events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _FakeStreamingAsyncOpenAI:
    last_instance = None

    def __init__(self, *args, **kwargs):
        self.responses = _FakeResponsesClient(
            _FakeAsyncStream(
                [
                    _FakeEvent("response.output_text.delta", "fresh "),
                    _FakeEvent("response.output_text.delta", "news"),
                    _FakeEvent("response.completed"),
                ]
            )
        )
        _FakeStreamingAsyncOpenAI.last_instance = self


def _model_config():
    return SimpleNamespace(
        api_name="grok-4.20-0309-reasoning",
        name="Grok Reasoning",
        cost_per_1k_tokens=2.0,
        cost_per_1k_input_tokens=None,
        cost_per_1k_output_tokens=None,
    )


@pytest.mark.asyncio
async def test_grok_search_uses_responses_api_web_search():
    from adam.llm.client import UnifiedLLMClient

    client = UnifiedLLMClient.__new__(UnifiedLLMClient)

    with patch("adam.llm.client.AsyncOpenAI", _FakeAsyncOpenAI):
        response = await client._complete_grok_with_search(
            prompt="latest AI news",
            model_config=_model_config(),
            system_prompt="Be concise.",
            temperature=0.2,
            max_tokens=321,
            stream=False,
            search_mode="web",
        )

    call = _FakeAsyncOpenAI.last_instance.responses.calls[0]
    assert call["input"] == "latest AI news"
    assert call["instructions"] == "Be concise."
    assert call["tools"] == [{"type": "web_search"}]
    assert call["max_output_tokens"] == 321
    assert response.content == "search result"


@pytest.mark.asyncio
async def test_grok_search_uses_x_search_for_x_mode():
    from adam.llm.client import UnifiedLLMClient

    client = UnifiedLLMClient.__new__(UnifiedLLMClient)

    with patch("adam.llm.client.AsyncOpenAI", _FakeAsyncOpenAI):
        await client._complete_grok_with_search(
            prompt="what's happening on X",
            model_config=_model_config(),
            system_prompt=None,
            temperature=0.2,
            max_tokens=128,
            stream=False,
            search_mode="x",
        )

    call = _FakeAsyncOpenAI.last_instance.responses.calls[0]
    assert call["tools"] == [{"type": "x_search"}]


@pytest.mark.asyncio
async def test_grok_search_streaming_reads_response_deltas():
    from adam.llm.client import UnifiedLLMClient

    client = UnifiedLLMClient.__new__(UnifiedLLMClient)

    with patch("adam.llm.client.AsyncOpenAI", _FakeStreamingAsyncOpenAI):
        stream = await client._complete_grok_with_search(
            prompt="latest AI news",
            model_config=_model_config(),
            system_prompt=None,
            temperature=0.2,
            max_tokens=128,
            stream=True,
            search_mode="web",
        )
        chunks = [chunk async for chunk in stream]

    assert chunks == ["fresh ", "news"]
