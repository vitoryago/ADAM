"""
Project management endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
import uuid
from datetime import datetime

from adam_v2.models import Project, ProjectCreate, ProjectWithStats
from adam_v2.database import get_db
from adam_v2.memory_manager import ProjectMemoryManager

router = APIRouter()

@router.post("/", response_model=Project)
async def create_project(project: ProjectCreate, db=Depends(get_db)):
    """Create a new project with isolated memory space"""
    project_id = str(uuid.uuid4())
    
    # Create project in database
    db.execute("""
        INSERT INTO projects (id, name, description, settings)
        VALUES (?, ?, ?, json(?))
    """, (project_id, project.name, project.description, 
          json.dumps(project.settings or {})))
    
    # Create ChromaDB collection for this project
    memory_manager = ProjectMemoryManager(project_id)
    memory_manager.initialize_collection()
    
    # Create default conversation
    conv_id = str(uuid.uuid4())
    db.execute("""
        INSERT INTO conversations (id, project_id, title)
        VALUES (?, ?, ?)
    """, (conv_id, project_id, "General Discussion"))
    
    db.commit()
    
    return Project(
        id=project_id,
        name=project.name,
        description=project.description,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        settings=project.settings or {},
        is_archived=False
    )

@router.get("/", response_model=List[Project])
async def list_projects(
    archived: bool = False,
    db=Depends(get_db)
):
    """List all projects"""
    cursor = db.execute("""
        SELECT p.*, 
               COUNT(DISTINCT c.id) as conversation_count,
               COUNT(DISTINCT m.id) as memory_count
        FROM projects p
        LEFT JOIN conversations c ON p.id = c.project_id
        LEFT JOIN project_memories m ON p.id = m.project_id
        WHERE p.is_archived = ?
        GROUP BY p.id
        ORDER BY p.updated_at DESC
    """, (archived,))
    
    projects = []
    for row in cursor.fetchall():
        projects.append(Project(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            settings=json.loads(row['settings']),
            is_archived=row['is_archived'],
            conversation_count=row['conversation_count'],
            memory_count=row['memory_count']
        ))
    
    return projects

@router.get("/{project_id}", response_model=ProjectWithStats)
async def get_project(project_id: str, db=Depends(get_db)):
    """Get project details with recent conversations"""
    # Get project
    cursor = db.execute(
        "SELECT * FROM projects WHERE id = ?", 
        (project_id,)
    )
    project_row = cursor.fetchone()
    
    if not project_row:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get recent conversations
    cursor = db.execute("""
        SELECT * FROM conversations 
        WHERE project_id = ?
        ORDER BY last_message_at DESC
        LIMIT 5
    """, (project_id,))
    
    conversations = []
    for row in cursor.fetchall():
        conversations.append(Conversation(
            id=row['id'],
            project_id=row['project_id'],
            title=row['title'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            last_message_at=row['last_message_at'],
            message_count=row['message_count'],
            total_cost=row['total_cost']
        ))
    
    return ProjectWithStats(
        id=project_row['id'],
        name=project_row['name'],
        description=project_row['description'],
        created_at=project_row['created_at'],
        updated_at=project_row['updated_at'],
        settings=json.loads(project_row['settings']),
        is_archived=project_row['is_archived'],
        recent_conversations=conversations
    )

@router.put("/{project_id}")
async def update_project(
    project_id: str,
    update: ProjectCreate,
    db=Depends(get_db)
):
    """Update project details"""
    db.execute("""
        UPDATE projects 
        SET name = ?, description = ?, settings = json(?), updated_at = ?
        WHERE id = ?
    """, (update.name, update.description, json.dumps(update.settings or {}),
          datetime.now(), project_id))
    
    db.commit()
    return {"status": "updated"}

@router.delete("/{project_id}")
async def delete_project(project_id: str, db=Depends(get_db)):
    """Delete project and all associated data"""
    # Delete ChromaDB collection
    memory_manager = ProjectMemoryManager(project_id)
    memory_manager.delete_collection()
    
    # Delete from database (cascade will handle related data)
    db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    db.commit()
    
    return {"status": "deleted"}