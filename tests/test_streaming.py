"""
Tests for LLM streaming functionality in ADAM.
Verifies StreamChunk structure and stream_response behavior.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass


def test_stream_chunk_structure():
    """Test that StreamChunk dataclass has correct fields and defaults."""
    from adam.services.llm_service import StreamChunk

    chunk = StreamChunk(content="Hello", model_used="test", tokens_used=0, cost=0.0)
    assert chunk.content == "Hello"
    assert chunk.model_used == "test"
    assert chunk.tokens_used == 0
    assert chunk.cost == 0.0
    assert chunk.is_final is False


def test_stream_chunk_final():
    """Test that a final StreamChunk can be created with metadata."""
    from adam.services.llm_service import StreamChunk

    final = StreamChunk(
        content="", model_used="test", tokens_used=100, cost=0.01, is_final=True
    )
    assert final.is_final is True
    assert final.tokens_used == 100
    assert final.cost == 0.01
    assert final.content == ""


def test_stream_chunk_with_content():
    """Test StreamChunk with incremental content (typical streaming chunk)."""
    from adam.services.llm_service import StreamChunk

    chunk = StreamChunk(content="world", model_used="gpt-4o-mini")
    assert chunk.content == "world"
    assert chunk.tokens_used == 0
    assert chunk.cost == 0.0
    assert chunk.is_final is False


def test_llm_response_structure():
    """Test that LLMResponse dataclass has correct fields."""
    from adam.services.llm_service import LLMResponse

    resp = LLMResponse(
        content="Hello world",
        model_used="gpt-4o-mini",
        tokens_used=15,
        cost=0.001,
    )
    assert resp.content == "Hello world"
    assert resp.model_used == "gpt-4o-mini"
    assert resp.tokens_used == 15
    assert resp.cost == 0.001
    assert resp.metadata is None


def test_llm_response_with_metadata():
    """Test LLMResponse with optional metadata."""
    from adam.services.llm_service import LLMResponse

    resp = LLMResponse(
        content="test",
        model_used="test-model",
        tokens_used=10,
        cost=0.0,
        metadata={"complexity": "low", "has_image": False},
    )
    assert resp.metadata["complexity"] == "low"
    assert resp.metadata["has_image"] is False


@pytest.mark.asyncio
async def test_stream_response_mock_client():
    """Test stream_response yields multiple chunks when LLM client is None (mock mode)."""
    from adam.services.llm_service import LLMService, StreamChunk

    # LLMService without real client falls back to mock streaming
    with patch("adam.services.llm_service.LLMConfig", side_effect=Exception("no config")):
        service = LLMService.__new__(LLMService)
        service.project_settings = {}
        service.project_id = None
        service.default_model = "gpt-4o-mini"
        service.temperature = 0.7
        service.max_tokens = 8000
        service.style_service = None
        service.response_style = "normal"
        service.llm_client = None
        service.query_analyzer = None
        service.dbt_knowledge = None
        service.sql_knowledge = None
        service.fast_router = None
        service.memory_service = None

    chunks = []
    async for chunk in service.stream_response(message="Hello"):
        chunks.append(chunk)
        assert isinstance(chunk, StreamChunk)

    # Should have multiple word chunks plus one final chunk
    assert len(chunks) >= 2
    # Last chunk should be final
    assert chunks[-1].is_final is True
    assert chunks[-1].content == ""
    # Non-final chunks should have content
    for chunk in chunks[:-1]:
        assert chunk.is_final is False
        assert len(chunk.content) > 0


@pytest.mark.asyncio
async def test_stream_response_with_async_generator():
    """Test stream_response correctly handles an async generator from the LLM client."""
    from adam.services.llm_service import LLMService, StreamChunk

    # Create a mock async generator that yields string chunks
    async def mock_stream():
        for word in ["Hello", " ", "world", "!"]:
            yield word

    # Build a minimal LLMService with mocked client
    service = LLMService.__new__(LLMService)
    service.project_settings = {}
    service.project_id = None
    service.default_model = "test-model"
    service.temperature = 0.7
    service.max_tokens = 8000
    service.style_service = None
    service.response_style = "normal"
    service.query_analyzer = None
    service.dbt_knowledge = None
    service.sql_knowledge = None
    service.fast_router = None
    service.memory_service = None

    # Mock the LLM client to return an async generator
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value=mock_stream())
    service.llm_client = mock_client

    chunks = []
    async for chunk in service.stream_response(message="Test streaming"):
        chunks.append(chunk)

    # Should have 4 content chunks + 1 final
    content_chunks = [c for c in chunks if not c.is_final]
    final_chunks = [c for c in chunks if c.is_final]

    assert len(content_chunks) == 4
    assert len(final_chunks) == 1
    assert final_chunks[0].tokens_used > 0
    assert final_chunks[0].cost >= 0

    # Verify accumulated content
    full_content = "".join(c.content for c in content_chunks)
    assert full_content == "Hello world!"


@pytest.mark.asyncio
async def test_stream_response_fallback_on_llm_response():
    """Test stream_response falls back gracefully when client returns LLMResponse instead of generator."""
    from adam.services.llm_service import LLMService, StreamChunk

    # Create a mock LLMResponse object (non-streaming response)
    mock_response = MagicMock()
    mock_response.content = "Complete response text"
    mock_response.total_tokens = 50
    mock_response.cost = 0.005

    service = LLMService.__new__(LLMService)
    service.project_settings = {}
    service.project_id = None
    service.default_model = "test-model"
    service.temperature = 0.7
    service.max_tokens = 8000
    service.style_service = None
    service.response_style = "normal"
    service.query_analyzer = None
    service.dbt_knowledge = None
    service.sql_knowledge = None
    service.fast_router = None
    service.memory_service = None

    # Mock client returns a non-iterable response (LLMResponse object)
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value=mock_response)
    service.llm_client = mock_client

    chunks = []
    async for chunk in service.stream_response(message="Test fallback"):
        chunks.append(chunk)

    # Should have 1 content chunk (full response) + 1 final
    content_chunks = [c for c in chunks if not c.is_final]
    assert len(content_chunks) == 1
    assert content_chunks[0].content == "Complete response text"

    final_chunks = [c for c in chunks if c.is_final]
    assert len(final_chunks) == 1
    assert final_chunks[0].is_final is True


@pytest.mark.asyncio
async def test_stream_response_error_fallback():
    """Test stream_response handles streaming errors by falling back to non-streaming."""
    from adam.services.llm_service import LLMService, StreamChunk

    call_count = 0

    # Mock client that fails on stream=True, succeeds on stream=False
    async def mock_complete(**kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs.get("stream", False):
            raise Exception("Streaming not supported")
        # Return a normal response
        resp = MagicMock()
        resp.content = "Fallback response"
        resp.total_tokens = 20
        resp.cost = 0.001
        return resp

    service = LLMService.__new__(LLMService)
    service.project_settings = {}
    service.project_id = None
    service.default_model = "test-model"
    service.temperature = 0.7
    service.max_tokens = 8000
    service.style_service = None
    service.response_style = "normal"
    service.query_analyzer = None
    service.dbt_knowledge = None
    service.sql_knowledge = None
    service.fast_router = None
    service.memory_service = None

    mock_client = MagicMock()
    mock_client.complete = mock_complete
    service.llm_client = mock_client

    chunks = []
    async for chunk in service.stream_response(message="Test error"):
        chunks.append(chunk)

    # Should have fallback content + final
    content_chunks = [c for c in chunks if not c.is_final]
    assert len(content_chunks) == 1
    assert content_chunks[0].content == "Fallback response"

    # Client should have been called twice: once with stream=True (failed), once with stream=False
    assert call_count == 2
