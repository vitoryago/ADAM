"""
File Watcher API Router
Provides endpoints for managing file watching and project tracking
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Optional, List
from pydantic import BaseModel
import asyncio
from pathlib import Path

from database import get_db
from services.file_watcher import FileWatcherService
from services.memory_service import ProjectMemoryService
from crud import crud_project
from sqlalchemy.orm import Session

router = APIRouter()

# Global file watcher service instance
file_watcher_service = None

def get_file_watcher() -> FileWatcherService:
    """Get or create file watcher service"""
    global file_watcher_service
    if file_watcher_service is None:
        # Create a stub memory service for now
        # In production, this would use the actual memory service
        file_watcher_service = FileWatcherService(None)
    return file_watcher_service


class WatchRequest(BaseModel):
    """Request to start watching a directory"""
    directory: str
    ignored_patterns: Optional[List[str]] = None


class WatchResponse(BaseModel):
    """Response for watch operations"""
    success: bool
    message: str
    status: Optional[Dict] = None


@router.post("/projects/{project_id}/watch", response_model=WatchResponse)
async def start_watching(
    project_id: str,
    request: WatchRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start watching a directory for a project"""
    # Verify project exists
    project = crud_project.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Verify directory exists
    path = Path(request.directory)
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail="Directory does not exist")
    
    # Get memory service for project
    memory_service = ProjectMemoryService(project_id, project.name)
    
    # Create file watcher with memory service
    watcher = FileWatcherService(memory_service)
    
    # Start watching
    success = watcher.start_watching(
        project_id,
        request.directory,
        request.ignored_patterns
    )
    
    if success:
        # Store watcher in global service
        global file_watcher_service
        file_watcher_service = watcher
        
        return WatchResponse(
            success=True,
            message=f"Started watching {request.directory}",
            status=watcher.get_status(project_id)
        )
    else:
        return WatchResponse(
            success=False,
            message="Failed to start watching"
        )


@router.delete("/projects/{project_id}/watch", response_model=WatchResponse)
async def stop_watching(
    project_id: str,
    db: Session = Depends(get_db)
):
    """Stop watching for a project"""
    # Verify project exists
    project = crud_project.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    watcher = get_file_watcher()
    success = watcher.stop_watching(project_id)
    
    if success:
        return WatchResponse(
            success=True,
            message="Stopped watching"
        )
    else:
        return WatchResponse(
            success=False,
            message="No active watcher for this project"
        )


@router.get("/projects/{project_id}/watch/status", response_model=Dict)
async def get_watch_status(
    project_id: str,
    db: Session = Depends(get_db)
):
    """Get watching status for a project"""
    # Verify project exists
    project = crud_project.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    watcher = get_file_watcher()
    return watcher.get_status(project_id)


@router.post("/projects/{project_id}/scan", response_model=WatchResponse)
async def scan_directory(
    project_id: str,
    request: WatchRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Perform one-time scan of directory without continuous watching"""
    # Verify project exists
    project = crud_project.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Verify directory exists
    path = Path(request.directory)
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail="Directory does not exist")
    
    # Get memory service for project
    memory_service = ProjectMemoryService(project_id, project.name)
    
    # Perform scan in background
    background_tasks.add_task(
        scan_directory_task,
        project_id,
        request.directory,
        memory_service,
        request.ignored_patterns
    )
    
    return WatchResponse(
        success=True,
        message=f"Started scanning {request.directory}"
    )


async def scan_directory_task(
    project_id: str,
    directory: str,
    memory_service: ProjectMemoryService,
    ignored_patterns: Optional[List[str]] = None
):
    """Background task to scan directory"""
    from services.file_watcher import ADAMFileHandler
    
    handler = ADAMFileHandler(project_id, memory_service, ignored_patterns)
    path = Path(directory)
    files_processed = 0
    
    for file_path in path.rglob('*'):
        if file_path.is_file() and not handler.should_ignore(str(file_path)):
            file_hash = handler.get_file_hash(str(file_path))
            if file_hash:
                handler.file_hashes[str(file_path)] = file_hash
                await handler._update_file_memory(str(file_path))
                files_processed += 1
                
                # Batch processing
                if files_processed % 10 == 0:
                    await asyncio.sleep(0.1)
    
    print(f"Scan complete: {files_processed} files indexed for project {project_id}")