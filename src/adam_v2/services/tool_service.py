"""
Tool Service - Integrates ADAM tools with the backend
Gives ADAM full capabilities like Claude Code
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adam.tools import (
    ReadFileTool, WriteFileTool, EditFileTool,
    DeleteFileTool, ListFilesTool, SearchFilesTool
)
from adam.tools.system_tools import (
    RunCommandTool, GitOperationsTool, EnvironmentInfoTool
)
from adam.tools.base import ToolResult, ToolStatus

logger = logging.getLogger(__name__)

class ToolService:
    """Service for executing ADAM tools"""
    
    def __init__(self):
        """Initialize with all available tools"""
        self.tools = {
            # File operations
            'read_file': ReadFileTool(),
            'write_file': WriteFileTool(),
            'edit_file': EditFileTool(),
            'delete_file': DeleteFileTool(),
            'list_files': ListFilesTool(),
            'search_files': SearchFilesTool(),
            
            # System operations
            'run_command': RunCommandTool(),
            'git_operations': GitOperationsTool(),
            'environment_info': EnvironmentInfoTool(),
        }
        
        logger.info(f"Initialized ToolService with {len(self.tools)} tools")
    
    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tools"""
        return [
            {
                'name': name,
                'description': tool.description if hasattr(tool, 'description') else f"Execute {name}"
            }
            for name, tool in self.tools.items()
        ]
    
    def execute_tool(self, tool_name: str, **params) -> Dict[str, Any]:
        """Execute a tool by name with parameters"""
        
        if tool_name not in self.tools:
            return {
                'status': 'error',
                'message': f"Tool '{tool_name}' not found",
                'available_tools': list(self.tools.keys())
            }
        
        try:
            tool = self.tools[tool_name]
            result = tool.execute(**params)
            
            # Convert ToolResult to dict
            if isinstance(result, ToolResult):
                return {
                    'status': result.status.value,
                    'output': result.data if hasattr(result, 'data') else result.output,
                    'message': result.message if hasattr(result, 'message') else None,
                    'metadata': result.metadata if hasattr(result, 'metadata') else {}
                }
            else:
                return {
                    'status': 'success',
                    'output': str(result)
                }
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {
                'status': 'error',
                'message': f"Tool execution failed: {str(e)}"
            }
    
    def process_tool_request(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Process a message to determine if it requires tool usage
        Returns tool execution result or None if no tool needed
        """
        
        # Simple pattern matching for tool detection
        # This could be enhanced with AI-based intent detection
        
        tool_patterns = {
            'read_file': ['read file', 'show file', 'cat ', 'open '],
            'write_file': ['write file', 'create file', 'save to'],
            'edit_file': ['edit file', 'modify file', 'change in file'],
            'list_files': ['list files', 'ls ', 'dir ', 'show directory'],
            'run_command': ['run command', 'execute', 'bash ', 'shell '],
            'git_operations': ['git ', 'commit', 'push', 'pull', 'branch'],
        }
        
        message_lower = message.lower()
        
        for tool_name, patterns in tool_patterns.items():
            for pattern in patterns:
                if pattern in message_lower:
                    # Extract parameters from message (simplified)
                    # In production, use better parsing or LLM
                    params = self._extract_params(message, tool_name)
                    if params:
                        return self.execute_tool(tool_name, **params)
        
        return None
    
    def _extract_params(self, message: str, tool_name: str) -> Optional[Dict[str, Any]]:
        """Extract parameters for a tool from message"""
        
        # Simplified parameter extraction
        # In production, use regex or LLM for better extraction
        
        if tool_name == 'read_file':
            # Look for file path in quotes or after keywords
            import re
            match = re.search(r'["\']([^"\']+)["\']|(?:file|read|show|cat)\s+(\S+)', message)
            if match:
                file_path = match.group(1) or match.group(2)
                return {'file_path': file_path}
        
        elif tool_name == 'list_files':
            # Default to current directory
            return {'directory': '.'}
        
        elif tool_name == 'run_command':
            # Extract command after keywords
            import re
            match = re.search(r'(?:run|execute|bash)\s+(.+)', message, re.IGNORECASE)
            if match:
                return {'command': match.group(1)}
        
        # Add more parameter extraction logic for other tools
        
        return None


# Singleton instance
_tool_service = None

def get_tool_service() -> ToolService:
    """Get or create the tool service singleton"""
    global _tool_service
    if _tool_service is None:
        _tool_service = ToolService()
    return _tool_service