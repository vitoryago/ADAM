"""
Project Manager for ADAM
Handles project-specific memory and context isolation
"""
import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

class ProjectManager:
    """Manage projects and their isolated memory spaces"""
    
    def __init__(self, config):
        self.config = config
        self.projects_file = Path(config.project_storage_path) / 'projects.json'
        self.projects_data = self._load_projects_data()
        
        # Ensure projects directory exists
        Path(config.project_storage_path).mkdir(parents=True, exist_ok=True)
    
    def _load_projects_data(self) -> Dict[str, Any]:
        """Load projects data from file"""
        if self.projects_file.exists():
            try:
                with open(self.projects_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        return {
            'projects': {},
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_projects_data(self):
        """Save projects data to file"""
        self.projects_file.parent.mkdir(parents=True, exist_ok=True)
        self.projects_data['last_updated'] = datetime.now().isoformat()
        
        with open(self.projects_file, 'w') as f:
            json.dump(self.projects_data, f, indent=2)
    
    async def ensure_project(self, project_id: str) -> Dict[str, Any]:
        """Ensure a project exists and return its data"""
        if project_id not in self.projects_data['projects']:
            # Create new project
            project_data = {
                'id': project_id,
                'name': f'Project {project_id[:8]}',
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'memory_collection': f'project_{project_id}',
                'conversation_count': 0,
                'total_queries': 0,
                'total_cost': 0.0,
                'settings': {
                    'default_model': self.config.default_simple_model,
                    'memory_enabled': True,
                    'rag_enabled': True
                },
                'metadata': {}
            }
            
            self.projects_data['projects'][project_id] = project_data
            self._save_projects_data()
            
            # Initialize project-specific memory storage
            await self._initialize_project_memory(project_id)
        
        else:
            # Update last accessed time
            self.projects_data['projects'][project_id]['last_accessed'] = datetime.now().isoformat()
            self._save_projects_data()
        
        return self.projects_data['projects'][project_id]
    
    async def _initialize_project_memory(self, project_id: str):
        """Initialize project-specific memory storage"""
        try:
            # Initialize ChromaDB collection for this project
            memory_collection = f'project_{project_id}'
            
            # Create project memory directory
            project_memory_dir = Path(self.config.memory_storage_path) / project_id
            project_memory_dir.mkdir(parents=True, exist_ok=True)
            
            # Create initial memory index file
            memory_index = {
                'project_id': project_id,
                'collection_name': memory_collection,
                'created_at': datetime.now().isoformat(),
                'memory_count': 0,
                'last_updated': datetime.now().isoformat()
            }
            
            memory_index_file = project_memory_dir / 'memory_index.json'
            with open(memory_index_file, 'w') as f:
                json.dump(memory_index, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to initialize memory for project {project_id}: {e}")
    
    async def get_project_stats(self, project_id: str) -> Dict[str, Any]:
        """Get statistics for a specific project"""
        project_data = self.projects_data['projects'].get(project_id, {})
        
        if not project_data:
            return {'exists': False}
        
        return {
            'exists': True,
            'name': project_data.get('name', f'Project {project_id[:8]}'),
            'created_at': project_data.get('created_at'),
            'last_accessed': project_data.get('last_accessed'),
            'conversation_count': project_data.get('conversation_count', 0),
            'total_queries': project_data.get('total_queries', 0),
            'total_cost': project_data.get('total_cost', 0.0),
            'memory_enabled': project_data.get('settings', {}).get('memory_enabled', True),
            'default_model': project_data.get('settings', {}).get('default_model', self.config.default_simple_model)
        }
    
    async def update_project_stats(self, project_id: str, query_cost: float = 0.0):
        """Update project statistics after a query"""
        if project_id in self.projects_data['projects']:
            project = self.projects_data['projects'][project_id]
            project['total_queries'] = project.get('total_queries', 0) + 1
            project['total_cost'] = project.get('total_cost', 0.0) + query_cost
            project['last_accessed'] = datetime.now().isoformat()
            
            self._save_projects_data()
    
    async def get_all_projects(self) -> List[Dict[str, Any]]:
        """Get all projects summary"""
        projects = []
        
        for project_id, project_data in self.projects_data['projects'].items():
            projects.append({
                'id': project_id,
                'name': project_data.get('name', f'Project {project_id[:8]}'),
                'created_at': project_data.get('created_at'),
                'last_accessed': project_data.get('last_accessed'),
                'total_queries': project_data.get('total_queries', 0),
                'total_cost': project_data.get('total_cost', 0.0)
            })
        
        # Sort by last accessed (most recent first)
        projects.sort(key=lambda x: x.get('last_accessed', ''), reverse=True)
        
        return projects
    
    async def delete_project(self, project_id: str) -> bool:
        """Delete a project and its associated data"""
        try:
            if project_id in self.projects_data['projects']:
                # Remove from projects data
                del self.projects_data['projects'][project_id]
                self._save_projects_data()
                
                # Clean up project memory directory
                project_memory_dir = Path(self.config.memory_storage_path) / project_id
                if project_memory_dir.exists():
                    import shutil
                    shutil.rmtree(project_memory_dir)
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Error deleting project {project_id}: {e}")
            return False
    
    def get_project_memory_path(self, project_id: str) -> str:
        """Get the memory storage path for a specific project"""
        return str(Path(self.config.memory_storage_path) / project_id)
    
    def get_project_collection_name(self, project_id: str) -> str:
        """Get the ChromaDB collection name for a project"""
        return f'project_{project_id}'