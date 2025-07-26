"""
Database models for ADAM v2.0
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

# Pydantic models for API
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict)

class Project(BaseModel):
    id: str
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    settings: Dict[str, Any]
    is_archived: bool = False
    conversation_count: Optional[int] = 0
    memory_count: Optional[int] = 0

class ConversationCreate(BaseModel):
    project_id: str
    title: str = "New Conversation"

class Conversation(BaseModel):
    id: str
    project_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    last_message_at: Optional[datetime]
    message_count: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    is_pinned: bool = False

class MessageCreate(BaseModel):
    conversation_id: str
    content: str
    role: MessageRole = MessageRole.USER

class Message(BaseModel):
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    model: Optional[str]
    tokens_used: Optional[int]
    cost: Optional[float]
    created_at: datetime
    metadata: Optional[Dict[str, Any]]

class ProjectMemory(BaseModel):
    id: str
    project_id: str
    memory_type: str
    query: str
    response: str
    created_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime]
    importance_score: float = 0.5

# Response models
class ProjectWithStats(Project):
    recent_conversations: List[Conversation] = []
    
class ConversationWithMessages(Conversation):
    recent_messages: List[Message] = []
    
class ProjectListResponse(BaseModel):
    projects: List[Project]
    total: int