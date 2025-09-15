"""
ADAM Memory System - Consolidated Module

This module provides the complete memory system for ADAM, including:
- Core memory operations (storage, retrieval, updates)
- Memory network connections and relationships
- Conversation-aware context handling
- Project-based memory isolation
- Advanced search and retrieval
- Memory lifecycle management (compression, cleanup)
"""

# Core memory classes
from .core import (
    Memory,
    MemoryType,
    QueryComplexity,
    ADAMMemoryAdvanced,
    ConversationState,
    MemoryWorthinessEvaluator
)

# Configuration
from .config import (
    EmbeddingProvider,
    EmbeddingConfig
)

# Memory network and connections
from .network import (
    MemoryNetworkSystem,
    MemoryNode,
    ConversationThread,
    MemoryConnection,
    ConnectionType
)

# Conversation handling
from .conversation import (
    ConversationAwareMemorySystem
)

# Project isolation
from .project import (
    ProjectAwareMemory
)

# Search and retrieval
from .search import (
    SearchContext
)

# Lifecycle management
from .lifecycle import (
    MemoryLifecycleManager
)

# Compression utilities
from .compressor import (
    MemoryCompressor
)

# Temporal scoring
from .scoring import (
    TemporalScoringConfig
)

__all__ = [
    # Core
    'Memory',
    'MemoryType', 
    'QueryComplexity',
    'ADAMMemoryAdvanced',
    'ConversationState',
    'MemoryWorthinessEvaluator',
    
    # Config
    'EmbeddingProvider',
    'EmbeddingConfig',
    
    # Network
    'MemoryNetworkSystem',
    'MemoryNode',
    'ConversationThread',
    'MemoryConnection',
    'ConnectionType',
    
    # Conversation
    'ConversationAwareMemorySystem',
    
    # Project
    'ProjectAwareMemory',
    
    # Search
    'SearchContext',
    
    # Lifecycle
    'MemoryLifecycleManager',
    'MemoryCompressor',
    
    # Scoring
    'TemporalScoringConfig'
]

# Version info
__version__ = '2.0.0'