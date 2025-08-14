"""
File Operation Tools for ADAM
Provides file system access similar to Claude Code
"""

import os
import re
import glob
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import aiofiles
import difflib
from .base import Tool, ToolResult, ToolStatus

class ReadFileTool(Tool):
    """Read contents of a file"""
    
    def __init__(self):
        super().__init__(
            name="read_file",
            description="Read the contents of a file"
        )
    
    async def execute(self, file_path: str, encoding: str = "utf-8") -> ToolResult:
        """Read file contents"""
        try:
            path = Path(file_path).resolve()
            
            if not path.exists():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"File not found: {file_path}"
                )
            
            if not path.is_file():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Path is not a file: {file_path}"
                )
            
            # Read file
            async with aiofiles.open(path, 'r', encoding=encoding) as f:
                content = await f.read()
            
            # Add line numbers for better readability
            lines = content.split('\n')
            numbered_content = '\n'.join([f"{i+1:4d}→{line}" for i, line in enumerate(lines)])
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=numbered_content,
                message=f"Read {len(lines)} lines from {path.name}",
                metadata={
                    "file_path": str(path),
                    "line_count": len(lines),
                    "size_bytes": path.stat().st_size
                }
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to read file: {str(e)}"
            )
    
    def validate_params(self, **kwargs) -> bool:
        return 'file_path' in kwargs
    
    def _get_param_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file to read"},
                "encoding": {"type": "string", "description": "File encoding (default: utf-8)"}
            },
            "required": ["file_path"]
        }

class WriteFileTool(Tool):
    """Write content to a file"""
    
    def __init__(self):
        super().__init__(
            name="write_file",
            description="Write content to a file (creates or overwrites)"
        )
        self.requires_confirmation = True  # Ask before overwriting
    
    async def execute(self, file_path: str, content: str, create_dirs: bool = True) -> ToolResult:
        """Write content to file"""
        try:
            path = Path(file_path).resolve()
            
            # Create directories if needed
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file exists
            exists = path.exists()
            
            # Write file
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            action = "Updated" if exists else "Created"
            lines = content.count('\n') + 1
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=content,
                message=f"{action} {path.name} with {lines} lines",
                metadata={
                    "file_path": str(path),
                    "action": action.lower(),
                    "line_count": lines,
                    "size_bytes": len(content.encode('utf-8'))
                }
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to write file: {str(e)}"
            )
    
    def validate_params(self, **kwargs) -> bool:
        return 'file_path' in kwargs and 'content' in kwargs
    
    def _get_param_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to write the file"},
                "content": {"type": "string", "description": "Content to write"},
                "create_dirs": {"type": "boolean", "description": "Create parent directories if needed"}
            },
            "required": ["file_path", "content"]
        }

class EditFileTool(Tool):
    """Edit specific parts of a file"""
    
    def __init__(self):
        super().__init__(
            name="edit_file",
            description="Replace specific text in a file"
        )
    
    async def execute(self, 
                     file_path: str, 
                     old_text: str, 
                     new_text: str,
                     occurrence: Union[int, str] = "all") -> ToolResult:
        """Edit file by replacing text"""
        try:
            path = Path(file_path).resolve()
            
            if not path.exists():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"File not found: {file_path}"
                )
            
            # Read current content
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Count occurrences
            count = content.count(old_text)
            if count == 0:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Text not found in file: {old_text[:50]}..."
                )
            
            # Replace text
            if occurrence == "all":
                new_content = content.replace(old_text, new_text)
                replaced = count
            elif isinstance(occurrence, int):
                # Replace specific occurrence
                parts = content.split(old_text)
                if occurrence > len(parts) - 1:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        data=None,
                        message=f"Occurrence {occurrence} not found (only {count} occurrences)"
                    )
                parts[occurrence] = new_text + parts[occurrence]
                new_content = old_text.join(parts[:occurrence+1]) + old_text.join(parts[occurrence+1:])
                replaced = 1
            else:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Invalid occurrence parameter: {occurrence}"
                )
            
            # Write back
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.write(new_content)
            
            # Generate diff for review
            diff = list(difflib.unified_diff(
                content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{path.name}",
                tofile=f"b/{path.name}",
                n=3
            ))
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=''.join(diff),
                message=f"Replaced {replaced} occurrence(s) in {path.name}",
                metadata={
                    "file_path": str(path),
                    "replacements": replaced,
                    "total_occurrences": count
                }
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to edit file: {str(e)}"
            )
    
    def validate_params(self, **kwargs) -> bool:
        return all(k in kwargs for k in ['file_path', 'old_text', 'new_text'])
    
    def _get_param_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file to edit"},
                "old_text": {"type": "string", "description": "Text to replace"},
                "new_text": {"type": "string", "description": "New text"},
                "occurrence": {"type": ["integer", "string"], "description": "Which occurrence to replace (number or 'all')"}
            },
            "required": ["file_path", "old_text", "new_text"]
        }

class DeleteFileTool(Tool):
    """Delete a file or directory"""
    
    def __init__(self):
        super().__init__(
            name="delete_file",
            description="Delete a file or directory"
        )
        self.requires_confirmation = True
    
    async def execute(self, path: str, recursive: bool = False) -> ToolResult:
        """Delete file or directory"""
        try:
            target = Path(path).resolve()
            
            if not target.exists():
                return ToolResult(
                    status=ToolStatus.WARNING,
                    data=None,
                    message=f"Path does not exist: {path}"
                )
            
            if target.is_file():
                target.unlink()
                action = "Deleted file"
            elif target.is_dir():
                if recursive:
                    shutil.rmtree(target)
                    action = "Deleted directory recursively"
                else:
                    target.rmdir()  # Only works if empty
                    action = "Deleted empty directory"
            else:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Unknown path type: {path}"
                )
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=None,
                message=f"{action}: {target.name}"
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to delete: {str(e)}"
            )
    
    def validate_params(self, **kwargs) -> bool:
        return 'path' in kwargs
    
    def _get_param_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to delete"},
                "recursive": {"type": "boolean", "description": "Delete directories recursively"}
            },
            "required": ["path"]
        }

class ListFilesTool(Tool):
    """List files in a directory"""
    
    def __init__(self):
        super().__init__(
            name="list_files",
            description="List files and directories"
        )
    
    async def execute(self, 
                     path: str = ".", 
                     pattern: Optional[str] = None,
                     recursive: bool = False) -> ToolResult:
        """List files in directory"""
        try:
            target = Path(path).resolve()
            
            if not target.exists():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Path not found: {path}"
                )
            
            if target.is_file():
                # Single file
                files = [target]
            else:
                # Directory listing
                if pattern:
                    if recursive:
                        files = list(target.glob(f"**/{pattern}"))
                    else:
                        files = list(target.glob(pattern))
                else:
                    if recursive:
                        files = list(target.rglob("*"))
                    else:
                        files = list(target.iterdir())
            
            # Format output
            file_list = []
            for f in sorted(files):
                rel_path = f.relative_to(target.parent if target.is_file() else target)
                if f.is_dir():
                    file_list.append(f"📁 {rel_path}/")
                else:
                    size = f.stat().st_size
                    size_str = self._format_size(size)
                    file_list.append(f"📄 {rel_path} ({size_str})")
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data='\n'.join(file_list),
                message=f"Found {len(files)} items",
                metadata={
                    "path": str(target),
                    "count": len(files),
                    "pattern": pattern
                }
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to list files: {str(e)}"
            )
    
    def _format_size(self, size: int) -> str:
        """Format file size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"
    
    def validate_params(self, **kwargs) -> bool:
        return True  # All params are optional
    
    def _get_param_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: current)"},
                "pattern": {"type": "string", "description": "Glob pattern to filter files"},
                "recursive": {"type": "boolean", "description": "List recursively"}
            }
        }

class SearchFilesTool(Tool):
    """Search for text in files"""
    
    def __init__(self):
        super().__init__(
            name="search_files",
            description="Search for text patterns in files"
        )
    
    async def execute(self,
                     pattern: str,
                     path: str = ".",
                     file_pattern: str = "*",
                     regex: bool = False) -> ToolResult:
        """Search for text in files"""
        try:
            target = Path(path).resolve()
            matches = []
            
            # Get files to search
            if target.is_file():
                files = [target]
            else:
                files = list(target.rglob(file_pattern))
            
            # Compile regex if needed
            if regex:
                import re
                search_pattern = re.compile(pattern)
            
            # Search files
            for file_path in files:
                if not file_path.is_file():
                    continue
                    
                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        lines = content.split('\n')
                        
                        for i, line in enumerate(lines, 1):
                            if regex:
                                if search_pattern.search(line):
                                    matches.append({
                                        "file": str(file_path.relative_to(target.parent if target.is_file() else target)),
                                        "line": i,
                                        "text": line.strip()
                                    })
                            else:
                                if pattern in line:
                                    matches.append({
                                        "file": str(file_path.relative_to(target.parent if target.is_file() else target)),
                                        "line": i,
                                        "text": line.strip()
                                    })
                except:
                    # Skip binary files or encoding errors
                    continue
            
            # Format results
            if matches:
                result_text = []
                current_file = None
                for match in matches:
                    if match["file"] != current_file:
                        current_file = match["file"]
                        result_text.append(f"\n📄 {current_file}:")
                    result_text.append(f"  Line {match['line']}: {match['text'][:100]}")
                
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data='\n'.join(result_text),
                    message=f"Found {len(matches)} matches in {len(set(m['file'] for m in matches))} files",
                    metadata={
                        "total_matches": len(matches),
                        "files_with_matches": len(set(m['file'] for m in matches)),
                        "pattern": pattern
                    }
                )
            else:
                return ToolResult(
                    status=ToolStatus.INFO,
                    data="",
                    message="No matches found"
                )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Search failed: {str(e)}"
            )
    
    def validate_params(self, **kwargs) -> bool:
        return 'pattern' in kwargs
    
    def _get_param_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Text or regex pattern to search"},
                "path": {"type": "string", "description": "Directory or file to search"},
                "file_pattern": {"type": "string", "description": "Glob pattern for files to search"},
                "regex": {"type": "boolean", "description": "Use regex pattern matching"}
            },
            "required": ["pattern"]
        }