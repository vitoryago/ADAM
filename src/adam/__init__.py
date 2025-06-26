"""
ADAM - Analytics Data Assistant with Memory
Core components for the intelligent assistant system
"""

from .conversation_system import ConversationSystem, ConversationSession, ConversationExchange
from .conversation_aware_memory import ConversationAwareMemorySystem
from .memory_network import MemoryNetworkSystem, MemoryNode, ConversationThread

__version__ = "2.0.0"

__all__ = [
    'ConversationSystem',
    'ConversationSession', 
    'ConversationExchange',
    'ConversationAwareMemorySystem',
    'MemoryNetworkSystem',
    'MemoryNode',
    'ConversationThread'
]