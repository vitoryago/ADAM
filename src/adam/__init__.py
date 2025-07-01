"""
ADAM - Analytics Data Assistant with Memory
Core components for the intelligent assistant system
"""

from .conversation_system import ConversationSystem, ConversationSession, ConversationExchange
from .conversation_aware_memory import ConversationAwareMemorySystem
from .memory_network import MemoryNetworkSystem, MemoryNode, ConversationThread
from .langgraph_conversation import (
    LangGraphConversationSystem,
    ConversationState,
    QueryComplexityAnalyzer,
    MemoryConfidenceScorer
)
from .integrated_conversation_system import IntegratedADAMSystem

__version__ = "3.0.0"  # Major version bump for LangGraph integration

__all__ = [
    'ConversationSystem',
    'ConversationSession', 
    'ConversationExchange',
    'ConversationAwareMemorySystem',
    'MemoryNetworkSystem',
    'MemoryNode',
    'ConversationThread',
    'LangGraphConversationSystem',
    'ConversationState',
    'QueryComplexityAnalyzer',
    'MemoryConfidenceScorer',
    'IntegratedADAMSystem'
]