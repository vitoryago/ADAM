"""
ADAM - Analytics Data Assistant with Memory
Core components for the intelligent assistant system
"""

from .conversation_system import ConversationSystem, ConversationSession, ConversationExchange
from .memory.conversation import ConversationAwareMemorySystem
from .memory.network import MemoryNetworkSystem, MemoryNode, ConversationThread
from .langgraph_conversation import (
    LangGraphConversationSystem,
    ConversationState,
    QueryComplexityAnalyzer,
    MemoryConfidenceScorer
)
from .integrated_conversation_system import IntegratedADAMSystem

# New coworker components
from .project_manager import ProjectManager, Project
from .screen_capture import ScreenCaptureService, ScreenContextAnalyzer
from .memory.project import ProjectAwareMemory

__version__ = "3.1.0"  # Version bump for coworker features

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
    'IntegratedADAMSystem',
    # New coworker components
    'ProjectManager',
    'Project',
    'ScreenCaptureService',
    'ScreenContextAnalyzer',
    'ProjectAwareMemory'
]