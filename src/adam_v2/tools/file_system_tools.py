"""
File System Tools for ADAM Backend
Provides file reading, directory listing, and search capabilities
"""

import os
import glob
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FileSystemTools:
    """Tools for file system operations"""
    
    def __init__(self, workspace_path: Optional[str] = None):
        """
        Initialize with optional workspace path
        
        Args:
            workspace_path: Base path for file operations
        """
        self.workspace_path = workspace_path or os.getcwd()
        self.max_file_size = 1024 * 1024  # 1MB max for safety
        
    def list_directory(self, path: str = "", pattern: str = "*") -> Dict[str, Any]:
        """
        List files and directories
        
        Args:
            path: Relative path from workspace
            pattern: Glob pattern for filtering
            
        Returns:
            Dict with files and directories
        """
        try:
            full_path = os.path.join(self.workspace_path, path)
            
            if not os.path.exists(full_path):
                return {"error": f"Path does not exist: {path}"}
            
            if not os.path.isdir(full_path):
                return {"error": f"Path is not a directory: {path}"}
            
            # Get all items matching pattern
            items = []
            for item in glob.glob(os.path.join(full_path, pattern)):
                item_path = os.path.relpath(item, self.workspace_path)
                is_dir = os.path.isdir(item)
                
                items.append({
                    "name": os.path.basename(item),
                    "path": item_path,
                    "type": "directory" if is_dir else "file",
                    "size": os.path.getsize(item) if not is_dir else None
                })
            
            return {
                "path": path,
                "pattern": pattern,
                "items": items,
                "count": len(items)
            }
            
        except Exception as e:
            logger.error(f"Error listing directory: {e}")
            return {"error": str(e)}
    
    def read_file(self, file_path: str, max_lines: int = 1000) -> Dict[str, Any]:
        """
        Read a file's contents
        
        Args:
            file_path: Path to file relative to workspace
            max_lines: Maximum lines to read
            
        Returns:
            Dict with file content or error
        """
        try:
            full_path = os.path.join(self.workspace_path, file_path)
            
            if not os.path.exists(full_path):
                return {"error": f"File does not exist: {file_path}"}
            
            if not os.path.isfile(full_path):
                return {"error": f"Path is not a file: {file_path}"}
            
            # Check file size
            file_size = os.path.getsize(full_path)
            if file_size > self.max_file_size:
                return {
                    "error": f"File too large: {file_size} bytes",
                    "partial": True,
                    "content": self._read_partial_file(full_path, max_lines)
                }
            
            # Read file
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            if len(lines) > max_lines:
                content = ''.join(lines[:max_lines])
                return {
                    "path": file_path,
                    "content": content,
                    "truncated": True,
                    "total_lines": len(lines),
                    "lines_shown": max_lines
                }
            else:
                content = ''.join(lines)
                return {
                    "path": file_path,
                    "content": content,
                    "truncated": False,
                    "total_lines": len(lines)
                }
                
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return {"error": str(e)}
    
    def search_files(self, 
                    pattern: str, 
                    path: str = "",
                    file_pattern: str = "*",
                    max_results: int = 50) -> Dict[str, Any]:
        """
        Search for files containing a pattern
        
        Args:
            pattern: Text pattern to search for
            path: Directory to search in
            file_pattern: File name pattern
            max_results: Maximum results to return
            
        Returns:
            Dict with matching files
        """
        try:
            full_path = os.path.join(self.workspace_path, path)
            
            if not os.path.exists(full_path):
                return {"error": f"Path does not exist: {path}"}
            
            matches = []
            files_searched = 0
            
            # Search files
            for file_path in Path(full_path).rglob(file_pattern):
                if len(matches) >= max_results:
                    break
                    
                if file_path.is_file():
                    files_searched += 1
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if pattern.lower() in content.lower():
                                rel_path = os.path.relpath(file_path, self.workspace_path)
                                
                                # Find line numbers with matches
                                lines_with_matches = []
                                for i, line in enumerate(content.split('\n'), 1):
                                    if pattern.lower() in line.lower():
                                        lines_with_matches.append({
                                            "line_number": i,
                                            "content": line[:200]  # First 200 chars
                                        })
                                        if len(lines_with_matches) >= 3:  # Max 3 line previews
                                            break
                                
                                matches.append({
                                    "path": rel_path,
                                    "matches": lines_with_matches
                                })
                    except:
                        continue  # Skip files that can't be read
            
            return {
                "pattern": pattern,
                "path": path,
                "file_pattern": file_pattern,
                "matches": matches,
                "files_searched": files_searched,
                "results_count": len(matches)
            }
            
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return {"error": str(e)}
    
    def find_files(self, 
                  name_pattern: str,
                  path: str = "",
                  max_results: int = 100) -> Dict[str, Any]:
        """
        Find files by name pattern
        
        Args:
            name_pattern: File name pattern (supports wildcards)
            path: Directory to search in
            max_results: Maximum results
            
        Returns:
            Dict with matching file paths
        """
        try:
            full_path = os.path.join(self.workspace_path, path)
            
            if not os.path.exists(full_path):
                return {"error": f"Path does not exist: {path}"}
            
            matches = []
            for file_path in Path(full_path).rglob(name_pattern):
                if len(matches) >= max_results:
                    break
                    
                rel_path = os.path.relpath(file_path, self.workspace_path)
                matches.append({
                    "path": rel_path,
                    "type": "directory" if file_path.is_dir() else "file",
                    "size": file_path.stat().st_size if file_path.is_file() else None
                })
            
            return {
                "pattern": name_pattern,
                "path": path,
                "matches": matches,
                "count": len(matches)
            }
            
        except Exception as e:
            logger.error(f"Error finding files: {e}")
            return {"error": str(e)}
    
    def _read_partial_file(self, file_path: str, max_lines: int) -> str:
        """Read partial file content"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line)
                return ''.join(lines)
        except:
            return ""
    
    def get_workspace_info(self) -> Dict[str, Any]:
        """Get information about the workspace"""
        try:
            return {
                "workspace_path": self.workspace_path,
                "exists": os.path.exists(self.workspace_path),
                "is_directory": os.path.isdir(self.workspace_path),
                "files_count": len(list(Path(self.workspace_path).rglob('*'))),
            }
        except Exception as e:
            return {"error": str(e)}