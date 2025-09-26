"""
Unified Database Models for ADAM
Consolidates all database schemas into a single, coherent model
"""

from sqlalchemy import (
    Column, String, Text, Boolean, Float, Integer,
    DateTime, JSON, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid
from typing import Optional, Dict, Any, List
from enum import Enum

Base = declarative_base()


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MemoryType(str, Enum):
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    CODE = "code"
    QUERY_RESULT = "query_result"
    USER_PREFERENCE = "user_preference"


# ===== Core Project Management =====

class Project(Base):
    """
    Project model - top level organization unit
    Replaces the old project system with enhanced capabilities
    """
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_archived = Column(Boolean, nullable=False, default=False)
    settings = Column(JSON, nullable=False, default={})

    # Enhanced fields
    owner_id = Column(String, nullable=True)  # For future user management
    team_members = Column(JSON, nullable=False, default=[])  # List of user IDs
    project_type = Column(String(50), nullable=False, default="general")  # general, dbt, sql, etc.

    # Relationships
    conversations = relationship("Conversation", back_populates="project", cascade="all, delete-orphan")
    memories = relationship("Memory", back_populates="project", cascade="all, delete-orphan")

    def __init__(self, **kwargs):
        kwargs.setdefault('is_archived', False)
        kwargs.setdefault('settings', {})
        kwargs.setdefault('team_members', [])
        super().__init__(**kwargs)


class Conversation(Base):
    """
    Conversation model - contains multiple messages
    Enhanced with memory integration
    """
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    title = Column(String(200), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_pinned = Column(Boolean, nullable=False, default=False)

    # Enhanced fields
    tags = Column(JSON, nullable=False, default=[])  # List of tags
    summary = Column(Text, nullable=True)  # AI-generated summary
    total_cost = Column(Float, nullable=False, default=0.0)
    total_tokens = Column(Integer, nullable=False, default=0)

    # Relationships
    project = relationship("Project", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

    def __init__(self, **kwargs):
        kwargs.setdefault('is_pinned', False)
        kwargs.setdefault('tags', [])
        kwargs.setdefault('total_cost', 0.0)
        kwargs.setdefault('total_tokens', 0)
        super().__init__(**kwargs)


class Message(Base):
    """
    Message model - individual message in a conversation
    Enhanced with better metadata tracking
    """
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # LLM metadata
    model = Column(String, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    cost = Column(Float, nullable=True)
    processing_time = Column(Float, nullable=True)  # seconds

    # Media attachments
    has_image = Column(Boolean, nullable=False, default=False)
    image_url = Column(String, nullable=True)
    has_audio = Column(Boolean, nullable=False, default=False)
    audio_url = Column(String, nullable=True)

    # Enhanced metadata (renamed to avoid SQLAlchemy conflict)
    extra_metadata = Column(JSON, nullable=False, default={})
    is_error = Column(Boolean, nullable=False, default=False)
    error_details = Column(Text, nullable=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    def __init__(self, **kwargs):
        kwargs.setdefault('has_image', False)
        kwargs.setdefault('has_audio', False)
        kwargs.setdefault('extra_metadata', {})
        kwargs.setdefault('is_error', False)
        super().__init__(**kwargs)


# ===== Unified Memory System =====

class Memory(Base):
    """
    Unified memory model - replaces ChromaDB for metadata
    Vector embeddings are still stored in ChromaDB, but metadata is here
    """
    __tablename__ = "memories"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)

    # Content
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)  # Short summary for quick reference
    title = Column(String(200), nullable=True)  # Optional title

    # Classification
    memory_type = Column(String, nullable=False, default=MemoryType.CONVERSATION.value)
    tags = Column(JSON, nullable=False, default=[])
    category = Column(String(100), nullable=True)  # e.g., "sql", "python", "dbt"

    # Provenance
    source_type = Column(String(50), nullable=False)  # conversation, document, manual, etc.
    source_id = Column(String, nullable=True)  # ID of source (e.g., conversation_id)
    source_url = Column(String, nullable=True)  # For external sources

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    accessed_at = Column(DateTime(timezone=True), nullable=True)  # Last access time
    access_count = Column(Integer, nullable=False, default=0)

    # Quality and relevance
    importance_score = Column(Float, nullable=False, default=0.5)  # 0.0 to 1.0
    quality_score = Column(Float, nullable=False, default=0.5)    # 0.0 to 1.0
    is_verified = Column(Boolean, nullable=False, default=False)   # Human verified

    # ChromaDB integration
    chromadb_id = Column(String, nullable=True, unique=True)  # ID in ChromaDB
    embedding_model = Column(String, nullable=False, default="all-mpnet-base-v2")

    # Extended metadata
    extra_metadata = Column(JSON, nullable=False, default={})

    # Relationships
    project = relationship("Project", back_populates="memories")
    connections = relationship(
        "MemoryConnection",
        foreign_keys="MemoryConnection.from_memory_id",
        back_populates="from_memory"
    )

    def __init__(self, **kwargs):
        kwargs.setdefault('memory_type', MemoryType.CONVERSATION.value)
        kwargs.setdefault('tags', [])
        kwargs.setdefault('access_count', 0)
        kwargs.setdefault('importance_score', 0.5)
        kwargs.setdefault('quality_score', 0.5)
        kwargs.setdefault('is_verified', False)
        kwargs.setdefault('extra_metadata', {})
        super().__init__(**kwargs)


class MemoryConnection(Base):
    """
    Memory connections - represents relationships between memories
    Replaces the NetworkX graph with database storage
    """
    __tablename__ = "memory_connections"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    from_memory_id = Column(String, ForeignKey("memories.id"), nullable=False)
    to_memory_id = Column(String, ForeignKey("memories.id"), nullable=False)

    # Connection metadata
    connection_type = Column(String(50), nullable=False)  # "similar", "follows", "contradicts", etc.
    strength = Column(Float, nullable=False, default=0.5)  # 0.0 to 1.0
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Optional explanation
    reason = Column(Text, nullable=True)
    extra_metadata = Column(JSON, nullable=False, default={})

    # Relationships
    from_memory = relationship("Memory", foreign_keys=[from_memory_id])
    to_memory = relationship("Memory", foreign_keys=[to_memory_id])

    # Unique constraint to prevent duplicate connections
    __table_args__ = (
        UniqueConstraint('from_memory_id', 'to_memory_id', name='unique_memory_connection'),
        Index('idx_memory_connections_from', 'from_memory_id'),
        Index('idx_memory_connections_to', 'to_memory_id'),
    )

    def __init__(self, **kwargs):
        kwargs.setdefault('strength', 0.5)
        kwargs.setdefault('extra_metadata', {})
        super().__init__(**kwargs)


# ===== Configuration and State =====

class SystemConfiguration(Base):
    """
    System configuration - stores runtime config in database
    Allows for dynamic configuration changes
    """
    __tablename__ = "system_configuration"

    id = Column(String, primary_key=True)  # Configuration key
    value = Column(JSON, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_sensitive = Column(Boolean, nullable=False, default=False)  # Don't log changes


class UserSession(Base):
    """
    User sessions - tracks active sessions and preferences
    Prepares for future multi-user support
    """
    __tablename__ = "user_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_token = Column(String, nullable=False, unique=True)
    user_id = Column(String, nullable=True)  # For future user system

    # Session data
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)

    # Preferences and state
    preferences = Column(JSON, nullable=False, default={})
    current_project_id = Column(String, ForeignKey("projects.id"), nullable=True)

    def __init__(self, **kwargs):
        kwargs.setdefault('is_active', True)
        kwargs.setdefault('preferences', {})
        super().__init__(**kwargs)


# ===== Analytics and Monitoring =====

class UsageMetrics(Base):
    """
    Usage metrics - tracks system usage for analytics
    """
    __tablename__ = "usage_metrics"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # What was used
    metric_type = Column(String(50), nullable=False)  # "api_call", "memory_search", etc.
    resource = Column(String(100), nullable=False)   # specific resource used

    # When and how much
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    value = Column(Float, nullable=False, default=1.0)
    cost = Column(Float, nullable=True)

    # Context
    project_id = Column(String, ForeignKey("projects.id"), nullable=True)
    session_id = Column(String, nullable=True)
    extra_metadata = Column(JSON, nullable=False, default={})

    def __init__(self, **kwargs):
        kwargs.setdefault('value', 1.0)
        kwargs.setdefault('extra_metadata', {})
        super().__init__(**kwargs)


# ===== Indexes for Performance =====

# Add indexes for common queries
Index('idx_messages_conversation_created', Message.conversation_id, Message.created_at)
Index('idx_memories_project_type', Memory.project_id, Memory.memory_type)
Index('idx_memories_created_at', Memory.created_at)
Index('idx_memories_importance', Memory.importance_score.desc())
Index('idx_conversations_project_updated', Conversation.project_id, Conversation.updated_at.desc())
Index('idx_usage_metrics_timestamp', UsageMetrics.timestamp)
Index('idx_usage_metrics_project', UsageMetrics.project_id, UsageMetrics.timestamp)