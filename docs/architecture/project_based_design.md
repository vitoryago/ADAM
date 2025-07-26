# ADAM 2.0: Project-Based Memory Architecture

## Overview
Transform ADAM from a single-memory system to a project-based architecture with isolated memory contexts, similar to Claude Projects or ChatGPT's conversation organization.

## Core Concepts

### 1. Projects
- **Definition**: A container for related conversations and memories
- **Features**:
  - Custom name and description
  - Isolated memory space
  - Shared context across conversations
  - Project-specific settings (models, temperature, etc.)

### 2. Conversations
- **Definition**: Individual chat sessions within a project
- **Features**:
  - Belong to exactly one project
  - Can be renamed
  - Share project memory pool
  - Maintain their own message history

### 3. Memory Isolation
- **Definition**: Each project has its own ChromaDB collection
- **Benefits**:
  - No cross-contamination between projects
  - Faster searches (smaller collections)
  - Better relevance (domain-specific context)

## Database Schema

```sql
-- Projects table
CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settings JSON,  -- Model preferences, etc.
    is_archived BOOLEAN DEFAULT FALSE
);

-- Conversations table
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_message_at TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost REAL DEFAULT 0.0,
    is_pinned BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- Messages table
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    model TEXT,
    tokens_used INTEGER,
    cost REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,  -- Store routing decisions, etc.
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

-- Project memories (metadata only, actual vectors in ChromaDB)
CREATE TABLE project_memories (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    memory_type TEXT,
    query TEXT,
    response TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    importance_score REAL DEFAULT 0.5,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
```

## Memory Architecture

### ChromaDB Collections
Each project gets its own ChromaDB collection:
- Collection name: `adam_project_{project_id}`
- Isolation: Complete separation between projects
- Metadata includes: conversation_id, timestamp, importance

### Memory Flow
1. User sends message in a conversation
2. System searches project-specific ChromaDB collection
3. Relevant memories included in context
4. Response generated
5. Important exchanges stored back to project memory

## API Design

### FastAPI Endpoints

```python
# Project Management
POST   /api/projects                 # Create new project
GET    /api/projects                 # List all projects
GET    /api/projects/{id}           # Get project details
PUT    /api/projects/{id}           # Update project
DELETE /api/projects/{id}           # Delete project (and all data)

# Conversation Management
POST   /api/projects/{id}/conversations      # Create conversation
GET    /api/projects/{id}/conversations      # List conversations
GET    /api/conversations/{id}               # Get conversation
PUT    /api/conversations/{id}               # Update (rename)
DELETE /api/conversations/{id}               # Delete conversation

# Messaging
POST   /api/conversations/{id}/messages      # Send message
GET    /api/conversations/{id}/messages      # Get message history

# Memory Management
GET    /api/projects/{id}/memories           # Browse project memories
DELETE /api/projects/{id}/memories/{mid}     # Delete specific memory
POST   /api/projects/{id}/memories/search    # Search memories
```

## UI/UX Design

### Layout Structure
```
+------------------+----------------------+
|     Sidebar      |    Main Content      |
|                  |                      |
| [+ New Project]  |  Conversation View   |
|                  |                      |
| My Projects:     |  [User]: Hello!      |
| ▼ DBT Optimize   |  [ADAM]: Hi there... |
|   - Initial Debug|                      |
|   - Performance  |  [Input box____]     |
| ▶ React Dev      |                      |
| ▶ General        |                      |
+------------------+----------------------+
```

### Key Features
1. **Sidebar Navigation**
   - Collapsible project list
   - Conversation management
   - Quick project switching

2. **Conversation View**
   - Message history
   - Model selector
   - Cost tracker
   - Memory indicator

3. **Project Settings**
   - Default model selection
   - Temperature settings
   - Memory preferences
   - Export options

## Implementation Plan

### Phase 1: Backend Infrastructure (Week 1)
- Set up FastAPI project structure
- Create SQLite database with schema
- Implement project/conversation CRUD APIs
- Integrate with existing ADAM memory system

### Phase 2: Memory Isolation (Week 2)
- Modify ChromaDB to use project-specific collections
- Update memory search to respect project boundaries
- Implement memory migration tools
- Add project-aware memory lifecycle

### Phase 3: Frontend Development (Week 3)
- Build HTMX-based UI components
- Implement project sidebar
- Create conversation interface
- Add real-time updates

### Phase 4: Features & Polish (Week 4)
- Conversation search
- Memory management UI
- Export functionality
- Settings and preferences

## Migration Strategy

### From Current System
1. Create "General" project for existing memories
2. Migrate current conversations to new schema
3. Preserve all memory connections
4. Maintain backward compatibility

## Benefits

1. **Better Organization**: Clear separation of concerns
2. **Improved Performance**: Smaller, focused memory searches
3. **Enhanced Privacy**: Project isolation
4. **Scalability**: Can handle many projects efficiently
5. **User Experience**: Familiar interface pattern

## Example Usage Scenarios

### Scenario 1: Software Development
- Project: "E-commerce Platform"
- Conversations:
  - "Database Schema Design"
  - "API Development"
  - "Frontend Implementation"
- Memories: Code snippets, decisions, patterns specific to this project

### Scenario 2: Learning
- Project: "Python Mastery"
- Conversations:
  - "Basic Concepts"
  - "Advanced Patterns"
  - "Project Exercises"
- Memories: Learning progress, code examples, explanations

### Scenario 3: Business Analysis
- Project: "Q4 Strategy"
- Conversations:
  - "Market Analysis"
  - "Competitor Research"
  - "Action Plans"
- Memories: Data points, insights, decisions

This architecture will transform ADAM into a truly powerful knowledge management system!