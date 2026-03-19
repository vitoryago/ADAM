"""Tests for memory integration into the chat pipeline.

These tests validate the memory recall formatting, worthiness evaluation,
and the helper functions used in the messages router -- all without requiring
ChromaDB or any external service.
"""

import pytest


# ---------------------------------------------------------------------------
# 1. Worthiness evaluator tests (no ChromaDB needed)
# ---------------------------------------------------------------------------

def test_worthiness_evaluator_imports():
    """MemoryWorthinessEvaluator and QueryComplexity are importable."""
    from adam.memory.core import MemoryWorthinessEvaluator, QueryComplexity
    evaluator = MemoryWorthinessEvaluator()
    assert evaluator is not None
    assert QueryComplexity.TRIVIAL.value == 1


def test_trivial_query_not_stored():
    """Trivial queries with short responses should NOT be stored."""
    from adam.memory.core import MemoryWorthinessEvaluator, QueryComplexity
    evaluator = MemoryWorthinessEvaluator()

    should_store, reason = evaluator.should_store_memory(
        "What is 2+2?",
        "4",
        0.001,
        QueryComplexity.TRIVIAL,
    )
    assert not should_store, f"Trivial query stored unexpectedly: {reason}"


def test_complex_query_with_code_stored():
    """Complex queries that produce code responses should be stored."""
    from adam.memory.core import MemoryWorthinessEvaluator, QueryComplexity
    evaluator = MemoryWorthinessEvaluator()

    code_response = (
        "```python\ndef binary_search(arr, target):\n"
        "    lo, hi = 0, len(arr) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n```\n"
    ) * 3  # Repeat to ensure substantial length

    should_store, reason = evaluator.should_store_memory(
        "Implement a binary search",
        code_response,
        0.02,
        QueryComplexity.COMPLEX,
    )
    assert should_store, f"Complex code query not stored: {reason}"


def test_expensive_response_always_stored():
    """Responses that cost > $0.01 should always be stored."""
    from adam.memory.core import MemoryWorthinessEvaluator, QueryComplexity
    evaluator = MemoryWorthinessEvaluator()

    should_store, reason = evaluator.should_store_memory(
        "Short question",
        "Short answer",
        0.05,  # 5 cents -- clearly expensive
        QueryComplexity.SIMPLE,
    )
    assert should_store
    assert "xpensive" in reason.lower() or "cost" in reason.lower()


def test_simple_factual_not_stored():
    """Simple factual responses without structure should not be stored."""
    from adam.memory.core import MemoryWorthinessEvaluator, QueryComplexity
    evaluator = MemoryWorthinessEvaluator()

    should_store, reason = evaluator.should_store_memory(
        "Hello",
        "Hi there!",
        0.0001,
        QueryComplexity.SIMPLE,
    )
    assert not should_store


def test_query_complexity_classification():
    """assess_query_complexity returns reasonable complexity levels."""
    from adam.memory.core import MemoryWorthinessEvaluator, QueryComplexity
    evaluator = MemoryWorthinessEvaluator()

    # Trivial
    c = evaluator.assess_query_complexity("What day is today?")
    assert c == QueryComplexity.TRIVIAL

    # Complex -- contains 'implement'
    c = evaluator.assess_query_complexity("Implement a caching layer for the API")
    assert c in (QueryComplexity.COMPLEX, QueryComplexity.EXPERT)

    # Simple -- short, no special patterns
    c = evaluator.assess_query_complexity("Hi")
    assert c == QueryComplexity.SIMPLE


# ---------------------------------------------------------------------------
# 2. Memory context formatting tests
# ---------------------------------------------------------------------------

def test_format_memory_context_with_relevance_scores():
    """Memory results should format into a string with relevance percentages."""
    from adam.api.routers.messages import _format_memory_context

    memories = [
        {"content": "Previous discussion about Python decorators", "relevance_score": 0.85},
        {"content": "Code pattern for error handling", "relevance_score": 0.72},
    ]
    context = _format_memory_context(memories)

    assert "85%" in context
    assert "72%" in context
    assert "decorators" in context
    assert "error handling" in context
    assert "Relevant context from previous conversations:" in context


def test_format_memory_context_with_similarity_key():
    """_format_memory_context should fall back to 'similarity' key."""
    from adam.api.routers.messages import _format_memory_context

    memories = [
        {"content": "Some old memory", "similarity": 0.60},
    ]
    context = _format_memory_context(memories)

    assert "60%" in context
    assert "Some old memory" in context


def test_format_memory_context_with_document_key():
    """_format_memory_context should fall back to 'document' key for content."""
    from adam.api.routers.messages import _format_memory_context

    memories = [
        {"document": "Content from document field", "relevance_score": 0.91},
    ]
    context = _format_memory_context(memories)

    assert "91%" in context
    assert "Content from document field" in context


def test_format_memory_context_truncates_long_content():
    """Long memory content should be truncated to 300 chars."""
    from adam.api.routers.messages import _format_memory_context

    long_content = "A" * 500
    memories = [{"content": long_content, "relevance_score": 0.80}]
    context = _format_memory_context(memories)

    # The content portion (after the [Relevance: ...] prefix) should be at most 300 chars
    # Total line = prefix + truncated content
    lines = context.split("\n")
    # The actual memory line is after the header
    memory_line = [l for l in lines if "[Relevance:" in l][0]
    # Verify the 'A' characters are truncated
    assert memory_line.count("A") == 300


def test_format_memory_context_empty_list():
    """Empty memory list should return empty string."""
    from adam.api.routers.messages import _format_memory_context
    assert _format_memory_context([]) == ""


def test_format_memory_context_separator():
    """Multiple memories should be joined by --- separators."""
    from adam.api.routers.messages import _format_memory_context

    memories = [
        {"content": "First memory", "relevance_score": 0.90},
        {"content": "Second memory", "relevance_score": 0.80},
        {"content": "Third memory", "relevance_score": 0.70},
    ]
    context = _format_memory_context(memories)

    # Should have 2 separators for 3 memories
    assert context.count("---") == 2


# ---------------------------------------------------------------------------
# 3. Module-level availability flags
# ---------------------------------------------------------------------------

def test_memory_availability_flag():
    """ADAM_MEMORY_AVAILABLE flag should be set based on ProjectAwareMemory import."""
    from adam.api.routers.messages import ADAM_MEMORY_AVAILABLE
    # If we got here, the import succeeded
    assert isinstance(ADAM_MEMORY_AVAILABLE, bool)


def test_worthiness_evaluator_availability_flag():
    """WORTHINESS_EVALUATOR_AVAILABLE flag should be set."""
    from adam.api.routers.messages import WORTHINESS_EVALUATOR_AVAILABLE
    assert isinstance(WORTHINESS_EVALUATOR_AVAILABLE, bool)


def test_worthiness_evaluator_instance():
    """The module-level _worthiness_evaluator should be a MemoryWorthinessEvaluator."""
    from adam.api.routers.messages import _worthiness_evaluator, WORTHINESS_EVALUATOR_AVAILABLE
    if WORTHINESS_EVALUATOR_AVAILABLE:
        from adam.memory.core import MemoryWorthinessEvaluator
        assert isinstance(_worthiness_evaluator, MemoryWorthinessEvaluator)


# ---------------------------------------------------------------------------
# 4. _store_memory_if_worthy logic (unit-level, no DB/ChromaDB)
# ---------------------------------------------------------------------------

def test_store_memory_if_worthy_is_async():
    """_store_memory_if_worthy should be an async function."""
    import asyncio
    from adam.api.routers.messages import _store_memory_if_worthy
    assert asyncio.iscoroutinefunction(_store_memory_if_worthy)


def test_get_memory_context_is_async():
    """_get_memory_context should be an async function."""
    import asyncio
    from adam.api.routers.messages import _get_memory_context
    assert asyncio.iscoroutinefunction(_get_memory_context)
