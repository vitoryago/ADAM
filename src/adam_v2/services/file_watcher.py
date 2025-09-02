"""
File Watcher Service for ADAM
Tracks file changes and updates project knowledge base
"""

import asyncio
import hashlib
from pathlib import Path
from typing import Dict, Set, Optional, List, Tuple
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
import logging

logger = logging.getLogger(__name__)

class ADAMFileHandler(FileSystemEventHandler):
    """Handles file system events for ADAM project tracking"""
    
    def __init__(self, project_id: str, memory_service, ignored_patterns: Optional[List[str]] = None):
        self.project_id = project_id
        self.memory_service = memory_service
        self.ignored_patterns = ignored_patterns or [
            '.git', '__pycache__', 'node_modules', '.env', '.venv',
            '*.pyc', '*.log', '*.tmp', '.DS_Store', 'dist', 'build'
        ]
        self.file_hashes: Dict[str, str] = {}
        self.pending_changes: Set[str] = set()
        self.batch_interval = 5  # seconds
        self._batch_task = None
    
    def should_ignore(self, path: str) -> bool:
        """Check if file/path should be ignored"""
        path_obj = Path(path)
        
        for pattern in self.ignored_patterns:
            if pattern.startswith('*'):
                # File extension pattern
                if path_obj.suffix == pattern[1:]:
                    return True
            elif pattern in path_obj.parts:
                # Directory name in path
                return True
            elif path_obj.name == pattern:
                # Exact filename match
                return True
        
        return False
    
    def get_file_hash(self, filepath: str) -> Optional[str]:
        """Calculate hash of file content"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except:
            return None
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events"""
        if event.is_directory or self.should_ignore(event.src_path):
            return
        
        # Check if content actually changed
        new_hash = self.get_file_hash(event.src_path)
        old_hash = self.file_hashes.get(event.src_path)
        
        if new_hash and new_hash != old_hash:
            self.file_hashes[event.src_path] = new_hash
            self.pending_changes.add(event.src_path)
            logger.info(f"File modified: {event.src_path}")
            self._schedule_batch_update()
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation events"""
        if event.is_directory or self.should_ignore(event.src_path):
            return
        
        file_hash = self.get_file_hash(event.src_path)
        if file_hash:
            self.file_hashes[event.src_path] = file_hash
            self.pending_changes.add(event.src_path)
            logger.info(f"File created: {event.src_path}")
            self._schedule_batch_update()
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion events"""
        if event.is_directory or self.should_ignore(event.src_path):
            return
        
        if event.src_path in self.file_hashes:
            del self.file_hashes[event.src_path]
            logger.info(f"File deleted: {event.src_path}")
            # Mark for removal from memory
            asyncio.create_task(self._remove_from_memory(event.src_path))
    
    def _schedule_batch_update(self):
        """Schedule batch update of pending changes"""
        if self._batch_task:
            self._batch_task.cancel()
        
        self._batch_task = asyncio.create_task(self._batch_update())
    
    async def _batch_update(self):
        """Process pending file changes in batch"""
        await asyncio.sleep(self.batch_interval)
        
        if not self.pending_changes:
            return
        
        files_to_process = list(self.pending_changes)
        self.pending_changes.clear()
        
        for filepath in files_to_process:
            await self._update_file_memory(filepath)
    
    async def _update_file_memory(self, filepath: str):
        """Update file content in memory"""
        try:
            path_obj = Path(filepath)
            
            # Determine file type and read content accordingly
            if path_obj.suffix in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h']:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract meaningful chunks (functions, classes)
                chunks = self._extract_code_chunks(content, path_obj.suffix)
                
                for chunk_name, chunk_content in chunks:
                    memory_entry = {
                        "file_path": str(path_obj),
                        "chunk_name": chunk_name,
                        "content": chunk_content,
                        "file_type": path_obj.suffix[1:],
                        "timestamp": datetime.now().isoformat(),
                        "project_id": self.project_id
                    }
                    
                    await self.memory_service.add_file_knowledge(
                        self.project_id,
                        memory_entry
                    )
            
            logger.info(f"Updated memory for: {filepath}")
            
        except Exception as e:
            logger.error(f"Error updating memory for {filepath}: {e}")
    
    async def _remove_from_memory(self, filepath: str):
        """Remove file from memory"""
        try:
            await self.memory_service.remove_file_knowledge(
                self.project_id,
                filepath
            )
            logger.info(f"Removed from memory: {filepath}")
        except Exception as e:
            logger.error(f"Error removing from memory {filepath}: {e}")
    
    def _extract_code_chunks(self, content: str, extension: str) -> List[Tuple[str, str]]:
        """Extract meaningful code chunks (functions, classes)"""
        chunks = []
        lines = content.split('\n')
        
        if extension == '.py':
            # Python: Extract functions and classes
            current_chunk = []
            current_name = None
            indent_level = 0
            
            for line in lines:
                stripped = line.lstrip()
                
                if stripped.startswith('def ') or stripped.startswith('class '):
                    # Save previous chunk
                    if current_name and current_chunk:
                        chunks.append((current_name, '\n'.join(current_chunk)))
                    
                    # Start new chunk
                    current_name = stripped.split('(')[0].replace('def ', '').replace('class ', '').strip()
                    current_chunk = [line]
                    indent_level = len(line) - len(stripped)
                
                elif current_name:
                    # Continue current chunk if same or deeper indentation
                    if line.strip() and (len(line) - len(line.lstrip())) > indent_level:
                        current_chunk.append(line)
                    elif not line.strip():
                        current_chunk.append(line)
                    else:
                        # End of chunk
                        chunks.append((current_name, '\n'.join(current_chunk)))
                        current_name = None
                        current_chunk = []
            
            # Save last chunk
            if current_name and current_chunk:
                chunks.append((current_name, '\n'.join(current_chunk)))
        
        elif extension in ['.js', '.ts', '.jsx', '.tsx']:
            # JavaScript/TypeScript: Extract functions and classes
            # Simple extraction based on function/class keywords
            import_section = []
            in_imports = True
            current_chunk = []
            current_name = None
            
            for line in lines:
                stripped = line.strip()
                
                # Collect imports
                if in_imports and (stripped.startswith('import ') or stripped.startswith('export ') or 
                                  stripped.startswith('const ') or stripped.startswith('let ')):
                    import_section.append(line)
                elif in_imports:
                    in_imports = False
                
                # Look for functions and classes
                if ('function ' in stripped or 'class ' in stripped or 
                    '= () =>' in stripped or '= function' in stripped):
                    
                    if current_name and current_chunk:
                        chunks.append((current_name, '\n'.join(import_section + current_chunk)))
                    
                    # Extract name
                    if 'function ' in stripped:
                        current_name = stripped.split('function ')[1].split('(')[0].strip()
                    elif 'class ' in stripped:
                        current_name = stripped.split('class ')[1].split(' ')[0].strip()
                    elif 'const ' in stripped or 'let ' in stripped:
                        current_name = stripped.split(' ')[1].split('=')[0].strip()
                    else:
                        current_name = "anonymous"
                    
                    current_chunk = [line]
                
                elif current_name:
                    current_chunk.append(line)
            
            # Save last chunk
            if current_name and current_chunk:
                chunks.append((current_name, '\n'.join(import_section + current_chunk)))
        
        # If no chunks extracted, return whole file as single chunk
        if not chunks:
            chunks.append(("full_content", content[:5000]))  # Limit size
        
        return chunks


class FileWatcherService:
    """Service to manage file watchers for projects"""
    
    def __init__(self, memory_service):
        self.memory_service = memory_service
        self.observers: Dict[str, Observer] = {}
        self.handlers: Dict[str, ADAMFileHandler] = {}
    
    def start_watching(self, project_id: str, directory: str, 
                       ignored_patterns: Optional[List[str]] = None) -> bool:
        """Start watching a directory for a project"""
        if project_id in self.observers:
            logger.warning(f"Already watching for project {project_id}")
            return False
        
        try:
            handler = ADAMFileHandler(project_id, self.memory_service, ignored_patterns)
            observer = Observer()
            observer.schedule(handler, directory, recursive=True)
            observer.start()
            
            self.observers[project_id] = observer
            self.handlers[project_id] = handler
            
            logger.info(f"Started watching {directory} for project {project_id}")
            
            # Initial scan of existing files
            asyncio.create_task(self._initial_scan(project_id, directory))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start watching: {e}")
            return False
    
    def stop_watching(self, project_id: str) -> bool:
        """Stop watching for a project"""
        if project_id not in self.observers:
            return False
        
        try:
            self.observers[project_id].stop()
            self.observers[project_id].join()
            del self.observers[project_id]
            del self.handlers[project_id]
            
            logger.info(f"Stopped watching for project {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop watching: {e}")
            return False
    
    def stop_all(self):
        """Stop all watchers"""
        for project_id in list(self.observers.keys()):
            self.stop_watching(project_id)
    
    async def _initial_scan(self, project_id: str, directory: str):
        """Perform initial scan of directory"""
        handler = self.handlers.get(project_id)
        if not handler:
            return
        
        path = Path(directory)
        files_processed = 0
        
        for file_path in path.rglob('*'):
            if file_path.is_file() and not handler.should_ignore(str(file_path)):
                file_hash = handler.get_file_hash(str(file_path))
                if file_hash:
                    handler.file_hashes[str(file_path)] = file_hash
                    await handler._update_file_memory(str(file_path))
                    files_processed += 1
                    
                    # Batch processing to avoid overwhelming
                    if files_processed % 10 == 0:
                        await asyncio.sleep(0.1)
        
        logger.info(f"Initial scan complete: {files_processed} files indexed for project {project_id}")
    
    def get_status(self, project_id: str) -> Dict:
        """Get watcher status for a project"""
        if project_id not in self.observers:
            return {"watching": False}
        
        handler = self.handlers[project_id]
        return {
            "watching": True,
            "files_tracked": len(handler.file_hashes),
            "pending_changes": len(handler.pending_changes)
        }