"""
Base Tool Framework for ADAM
Provides foundation for all tools similar to Claude Code
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

class ToolStatus(Enum):
    """Status of tool execution"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ToolResult:
    """Result from tool execution"""
    status: ToolStatus
    data: Any
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "status": self.status.value,
            "data": self.data,
            "message": self.message,
            "metadata": self.metadata or {}
        }
    
    def __str__(self) -> str:
        """Human-readable representation"""
        if self.status == ToolStatus.SUCCESS:
            return f"✓ {self.message or 'Success'}\n{self.data if isinstance(self.data, str) else json.dumps(self.data, indent=2)}"
        elif self.status == ToolStatus.ERROR:
            return f"✗ Error: {self.message}\n{self.data if self.data else ''}"
        else:
            return f"ℹ {self.message}: {self.data}"

class Tool(ABC):
    """Base class for all ADAM tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.requires_confirmation = False
        self.memory_relevant = True  # Should this tool's usage be saved to memory?
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        pass
    
    @abstractmethod
    def validate_params(self, **kwargs) -> bool:
        """Validate parameters before execution"""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for this tool's parameters"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_param_schema()
        }
    
    @abstractmethod
    def _get_param_schema(self) -> Dict[str, Any]:
        """Get parameter schema for this tool"""
        pass
    
    async def safe_execute(self, **kwargs) -> ToolResult:
        """Safely execute tool with validation and error handling"""
        try:
            # Validate parameters
            if not self.validate_params(**kwargs):
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Invalid parameters for {self.name}"
                )
            
            # Log execution
            logger.info(f"Executing tool: {self.name} with params: {kwargs}")
            
            # Execute
            result = await self.execute(**kwargs)
            
            # Log result
            logger.info(f"Tool {self.name} completed with status: {result.status}")
            
            return result
            
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {str(e)}")
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Tool execution failed: {str(e)}"
            )

class ToolExecutor:
    """Manages and executes tools"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    def register_tool(self, tool: Tool):
        """Register a tool for use"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def register_tools(self, tools: List[Tool]):
        """Register multiple tools"""
        for tool in tools:
            self.register_tool(tool)
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        return [tool.get_schema() for tool in self.tools.values()]
    
    async def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name"""
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Tool '{name}' not found"
            )
        
        # Execute the tool
        result = await tool.safe_execute(**kwargs)
        
        # Record in history
        self.execution_history.append({
            "tool": name,
            "params": kwargs,
            "result": result.to_dict(),
            "timestamp": self._get_timestamp()
        })
        
        return result
    
    async def execute_tool_chain(self, chain: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute a chain of tools in sequence"""
        results = []
        
        for step in chain:
            tool_name = step.get("tool")
            params = step.get("params", {})
            
            # Check if we should use result from previous step
            if step.get("use_previous_result") and results:
                # Pass previous result as input
                params["previous_result"] = results[-1].data
            
            result = await self.execute_tool(tool_name, **params)
            results.append(result)
            
            # Stop chain if error
            if result.status == ToolStatus.ERROR and step.get("stop_on_error", True):
                break
        
        return results
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:]
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history = []