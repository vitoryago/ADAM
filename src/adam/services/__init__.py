"""ADAM Services - Business logic layer."""


def __getattr__(name):
    """Lazy imports to avoid circular dependencies with memory system."""
    if name in ('LLMService', 'LLMResponse', 'StreamChunk'):
        from .llm_service import LLMService, LLMResponse, StreamChunk
        _exports = {'LLMService': LLMService, 'LLMResponse': LLMResponse, 'StreamChunk': StreamChunk}
        return _exports[name]
    raise AttributeError(f"module 'adam.services' has no attribute {name!r}")


__all__ = ['LLMService', 'LLMResponse', 'StreamChunk']
