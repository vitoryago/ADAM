"""Smoke tests for consolidated memory system."""


def test_memory_imports():
    """Core memory classes are importable from the package."""
    from adam.memory import ADAMMemoryAdvanced, MemoryNetworkSystem, MemoryConfig
    assert ADAMMemoryAdvanced is not None
    assert MemoryNetworkSystem is not None
    assert MemoryConfig is not None


def test_memory_config():
    """MemoryConfig initializes with a valid embedding model name."""
    from adam.memory.config import MemoryConfig
    config = MemoryConfig()
    assert config.embedding_model_name is not None
    assert config.embedding_config is not None
    assert config.embedding_config.dimension > 0


def test_memory_types():
    """MemoryType enum has expected values."""
    from adam.memory.core import MemoryType
    assert MemoryType.ERROR_SOLUTION.value == "error_solution"
    assert MemoryType.CODE_PATTERN.value == "code_pattern"
    assert MemoryType.CONVERSATION.value == "conversation"


def test_worthiness_evaluator():
    """MemoryWorthinessEvaluator classifies query complexity correctly."""
    from adam.memory.core import MemoryWorthinessEvaluator, QueryComplexity
    evaluator = MemoryWorthinessEvaluator()
    complexity = evaluator.assess_query_complexity("What is Python?")
    assert complexity in [QueryComplexity.TRIVIAL, QueryComplexity.SIMPLE]

    complex_q = evaluator.assess_query_complexity(
        "How do I implement a distributed cache with consistent hashing?"
    )
    assert complex_q in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]


def test_temporal_scorer():
    """TemporalMemoryScorer and config are importable and functional."""
    from adam.memory.core import TemporalMemoryScorer, TemporalScoringConfig
    config = TemporalScoringConfig()
    assert config.temporal_weight == 0.3
    scorer = TemporalMemoryScorer(config)
    assert scorer is not None


def test_search_enhancer():
    """MemorySearchEnhancer is importable and analyzes intent."""
    from adam.memory.core import MemorySearchEnhancer
    enhancer = MemorySearchEnhancer()
    intent = enhancer.analyze_user_intent("remember that code we wrote?")
    assert intent == "recall"
    intent2 = enhancer.analyze_user_intent("what is Python?")
    assert intent2 == "general"


def test_conversation_aware_system():
    """ConversationAwareMemorySystem is importable."""
    from adam.memory.core import ConversationAwareMemorySystem
    assert ConversationAwareMemorySystem is not None


def test_search_context_dataclass():
    """SearchContext dataclass works correctly."""
    from adam.memory.core import SearchContext
    ctx = SearchContext(
        current_query="test",
        conversation_history=[],
        technical_terms=["python"],
        user_intent="general",
    )
    assert ctx.current_query == "test"
    assert ctx.user_intent == "general"


def test_network_classes_importable():
    """Network module classes are importable."""
    from adam.memory.network import MemoryNetworkSystem, MemoryNode, ConversationThread
    assert MemoryNetworkSystem is not None
    assert MemoryNode is not None
    assert ConversationThread is not None


def test_lifecycle_importable():
    """MemoryLifecycleManager is importable."""
    from adam.memory.lifecycle import MemoryLifecycleManager
    assert MemoryLifecycleManager is not None


def test_project_aware_memory_importable():
    """ProjectAwareMemory is importable."""
    from adam.memory.project import ProjectAwareMemory
    assert ProjectAwareMemory is not None


def test_format_memory_for_prompt():
    """format_memory_for_prompt helper works."""
    from adam.memory.core import format_memory_for_prompt, SearchContext
    ctx = SearchContext(
        current_query="test",
        conversation_history=[],
        technical_terms=[],
        user_intent="general",
    )
    memory = {"content": "Query: hello\n\nResponse: world"}
    result = format_memory_for_prompt(memory, ctx)
    assert "hello" in result
    assert "world" in result
