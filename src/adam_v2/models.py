"""
Database models for ADAM v2.0
Defines the schema for projects, conversations, and messages
"""

from sqlalchemy import Column, String, Text, Boolean, Float, Integer, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any
from enum import Enum

from database import Base


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# SQLAlchemy Models
class Project(Base):
    """Project model - top level organization unit"""
    __tablename__ = "projects"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_archived = Column(Boolean, nullable=False, server_default='0')
    settings = Column(JSON, nullable=False, server_default='{}')
    
    def __init__(self, **kwargs):
        # Set defaults for Python-side usage
        kwargs.setdefault('is_archived', False)
        kwargs.setdefault('settings', {})
        super().__init__(**kwargs)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="project", cascade="all, delete-orphan")
    
    # Note: conversation_count and memory_count are calculated separately
    # to avoid async context issues


class Conversation(Base):
    """Conversation model - contains multiple messages"""
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    title = Column(String(200), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_pinned = Column(Boolean, nullable=False, server_default='0')
    
    def __init__(self, **kwargs):
        # Set defaults for Python-side usage
        kwargs.setdefault('is_pinned', False)
        super().__init__(**kwargs)
    
    # Relationships
    project = relationship("Project", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    # Note: message_count and total_cost are calculated separately
    # to avoid async context issues


class Message(Base):
    """Message model - individual message in a conversation"""
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Optional fields for assistant messages
    model = Column(String, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    cost = Column(Float, nullable=True)
    
    # For image attachments
    has_image = Column(Boolean, nullable=False, server_default='0')
    image_url = Column(String, nullable=True)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def __init__(self, **kwargs):
        # Set defaults for Python-side usage
        kwargs.setdefault('has_image', False)
        super().__init__(**kwargs)


# Pydantic models for API request/response
class ProjectBase(BaseModel):
    """Base project schema"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    settings: Dict[str, Any] = Field(default_factory=dict)


class ProjectCreate(ProjectBase):
    """Request model for creating a project"""
    pass


class ProjectUpdate(BaseModel):
    """Request model for updating a project"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    is_archived: Optional[bool] = None


class ProjectResponse(ProjectBase):
    """Response model for project"""
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    created_at: datetime
    updated_at: datetime
    is_archived: bool
    conversation_count: int = 0
    memory_count: int = 0


class ConversationBase(BaseModel):
    """Base conversation schema"""
    title: str = Field(..., min_length=1, max_length=200)


class ConversationCreate(ConversationBase):
    """Request model for creating a conversation"""
    pass


class ConversationUpdate(BaseModel):
    """Request model for updating a conversation"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    is_pinned: Optional[bool] = None


class ConversationResponse(ConversationBase):
    """Response model for conversation"""
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    project_id: str
    created_at: datetime
    updated_at: datetime
    is_pinned: bool
    message_count: int = 0
    total_cost: float = 0.0


class MessageBase(BaseModel):
    """Base message schema"""
    content: str = Field(..., min_length=1)
    role: str = "user"


class MessageCreate(BaseModel):
    """Request model for sending a message"""
    content: str = Field(..., min_length=1)
    use_memory: bool = True
    model: Optional[str] = None  # If None, use automatic routing
    
    # For image attachments
    has_image: bool = False
    image_data: Optional[str] = None  # Base64 encoded image
    
    # For workspace context from VSCode
    workspace_context: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    
    # For live search
    use_search: bool = False
    search_mode: Optional[str] = None  # auto, web, x, news, rss


class MessageResponse(MessageBase):
    """Response model for message"""
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    conversation_id: str
    created_at: datetime
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    has_image: bool = False
    image_url: Optional[str] = None


class MemorySearchRequest(BaseModel):
    """Request model for searching memories"""
    query: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=50)


class MemorySearchResponse(BaseModel):
    """Response model for memory search"""
    memories: List[Dict[str, Any]]
    total_results: int