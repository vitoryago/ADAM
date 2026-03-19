"""
Project Service for ADAM - Handles project-based memory isolation
"""
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class Project:
    """Represents an ADAM project with isolated memory"""
    id: str
    name: str
    description: str = ""
    created_at: datetime = None
    last_accessed: datetime = None
    settings: Dict[str, Any] = None
    memory_stats: Dict[str, int] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
        if self.settings is None:
            self.settings = {}
        if self.memory_stats is None:
            self.memory_stats = {
                "total_memories": 0,
                "conversations": 0,
                "screen_captures": 0
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)


class ProjectManager:
    """
    Manages ADAM projects with isolated memory spaces.
    Each project has its own ChromaDB collection and settings.
    """

    def __init__(self, base_path: str = "./data/adam_projects"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.projects_file = self.base_path / "projects.json"
        self.active_project_file = self.base_path / "active_project.txt"
        self.projects: Dict[str, Project] = self._load_projects()
        self._active_project_id: Optional[str] = self._load_active_project()

    def _load_projects(self) -> Dict[str, Project]:
        """Load projects from disk"""
        if self.projects_file.exists():
            try:
                with open(self.projects_file, 'r') as f:
                    data = json.load(f)
                return {
                    pid: Project.from_dict(pdata)
                    for pid, pdata in data.items()
                }
            except Exception as e:
                print(f"Error loading projects: {e}")
        return {}

    def _save_projects(self):
        """Save projects to disk"""
        data = {
            pid: project.to_dict()
            for pid, project in self.projects.items()
        }
        with open(self.projects_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_active_project(self) -> Optional[str]:
        """Load the active project ID"""
        if self.active_project_file.exists():
            return self.active_project_file.read_text().strip()
        return None

    def _save_active_project(self):
        """Save the active project ID"""
        if self._active_project_id:
            self.active_project_file.write_text(self._active_project_id)
        elif self.active_project_file.exists():
            self.active_project_file.unlink()

    def create_project(self, name: str, description: str = "") -> Project:
        """Create a new project"""
        # Generate unique ID
        project_id = hashlib.md5(f"{name}{datetime.now()}".encode()).hexdigest()[:12]

        # Create project
        project = Project(
            id=project_id,
            name=name,
            description=description
        )

        # Save
        self.projects[project_id] = project
        self._save_projects()

        # If no active project, make this one active
        if not self._active_project_id:
            self.set_active_project(project_id)

        return project

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID"""
        return self.projects.get(project_id)

    def list_projects(self) -> List[Project]:
        """List all projects"""
        return list(self.projects.values())

    def set_active_project(self, project_id: str) -> bool:
        """Set the active project"""
        if project_id in self.projects:
            self._active_project_id = project_id
            self.projects[project_id].last_accessed = datetime.now()
            self._save_active_project()
            self._save_projects()
            return True
        return False

    def get_active_project(self) -> Optional[Project]:
        """Get the currently active project"""
        if self._active_project_id:
            return self.projects.get(self._active_project_id)
        return None

    def update_project_stats(self, project_id: str, stats: Dict[str, int]):
        """Update project statistics"""
        if project_id in self.projects:
            self.projects[project_id].memory_stats.update(stats)
            self._save_projects()

    def delete_project(self, project_id: str) -> bool:
        """Delete a project (does not delete memories)"""
        if project_id in self.projects:
            del self.projects[project_id]
            if self._active_project_id == project_id:
                self._active_project_id = None
                self._save_active_project()
            self._save_projects()
            return True
        return False

    def get_or_create_default(self) -> Project:
        """Get or create a default project"""
        default_id = "default"
        if default_id not in self.projects:
            return self.create_project("Default", "Default ADAM workspace")
        return self.projects[default_id]
