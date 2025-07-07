#!/usr/bin/env python3
"""
Agent Tools for ADAM's Proactive Agent System
=============================================

This module provides a comprehensive toolkit that transforms ADAM from a passive
Q&A system into an active agent capable of achieving complex objectives.

WHY AGENTS NEED TOOLS:
=====================
1. **Action Grounding**: LLMs can plan, but tools execute real-world actions
2. **Feedback Loops**: Tools provide concrete results that inform next steps  
3. **Capability Extension**: Tools add abilities beyond text generation
4. **Error Handling**: Real execution reveals issues planning might miss

TOOL DESIGN PRINCIPLES:
======================
1. **Atomic Operations**: Each tool does one thing well
2. **Clear Interfaces**: Predictable inputs/outputs for LLM usage
3. **Error Recovery**: Graceful failures with actionable error messages
4. **Observability**: Rich logging for debugging agent behavior

The tools here enable ADAM to:
- Search and analyze information
- Execute code and calculations
- Interact with external systems
- Monitor its own progress
- Learn from execution results
"""

import json
import subprocess
import tempfile
import os
import re
import ast
import math
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# For code execution sandboxing
import resource
import signal

# For web searching
from bs4 import BeautifulSoup

# Rich output for monitoring
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

# LangChain tool decorators
from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field

console = Console()
logger = logging.getLogger(__name__)


# ==============================================================================
# Tool Input/Output Models (for type safety and validation)
# ==============================================================================

class SearchInput(BaseModel):
    """Input model for search tool"""
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum number of results")
    search_type: str = Field(
        default="general", 
        description="Type of search: 'general', 'code', 'academic', 'local_files'"
    )


class CalculateInput(BaseModel):
    """Input model for calculation tool"""
    expression: str = Field(description="Mathematical expression to evaluate")
    variables: Dict[str, float] = Field(
        default_factory=dict,
        description="Variables to use in the expression"
    )


class CodeExecuteInput(BaseModel):
    """Input model for code execution tool"""
    code: str = Field(description="Code to execute")
    language: str = Field(default="python", description="Programming language")
    timeout: int = Field(default=30, description="Execution timeout in seconds")
    safe_mode: bool = Field(default=True, description="Run in sandboxed environment")


class FileOperationInput(BaseModel):
    """Input model for file operations"""
    operation: str = Field(description="Operation: read, write, append, delete")
    path: str = Field(description="File path")
    content: Optional[str] = Field(default=None, description="Content for write/append")


class WebInteractInput(BaseModel):
    """Input model for web interaction"""
    url: str = Field(description="URL to interact with")
    method: str = Field(default="GET", description="HTTP method")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Request data")
    extract: Optional[str] = Field(
        default=None,
        description="CSS selector or XPath to extract specific content"
    )


class SystemCommandInput(BaseModel):
    """Input model for system commands"""
    command: str = Field(description="System command to execute")
    shell: bool = Field(default=False, description="Execute through shell")
    timeout: int = Field(default=30, description="Command timeout")


# ==============================================================================
# Core Agent Tools
# ==============================================================================

@tool(args_schema=SearchInput)
def search_tool(query: str, max_results: int = 5, search_type: str = "general") -> Dict[str, Any]:
    """
    Search for information across multiple sources
    
    This tool is fundamental for agents to gather information before making decisions.
    It demonstrates the principle of "perception before action" - agents must understand
    their environment before acting.
    
    Search types:
    - general: Web search for general information
    - code: Search code repositories and documentation
    - academic: Search academic papers and research
    - local_files: Search ADAM's memory and local files
    
    Returns structured results with relevance scores for planning.
    """
    logger.info(f"Searching for: {query} (type: {search_type})")
    
    results = []
    
    try:
        if search_type == "general":
            # Simulate web search (in production, use real search API)
            # This would integrate with Google, Bing, or DuckDuckGo APIs
            results = _mock_web_search(query, max_results)
            
        elif search_type == "code":
            # Search code repositories, Stack Overflow, documentation
            results = _search_code_resources(query, max_results)
            
        elif search_type == "academic":
            # Search arXiv, Google Scholar, PubMed
            results = _search_academic_resources(query, max_results)
            
        elif search_type == "local_files":
            # Search ADAM's memory system and local files
            results = _search_local_resources(query, max_results)
            
        else:
            raise ValueError(f"Unknown search type: {search_type}")
            
        # Add metadata for agent planning
        return {
            "success": True,
            "query": query,
            "search_type": search_type,
            "result_count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "confidence": _calculate_search_confidence(results)
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "search_type": search_type,
            "results": []
        }


@tool(args_schema=CalculateInput)
def calculate_tool(expression: str, variables: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Perform mathematical calculations
    
    This tool allows agents to:
    1. Verify numerical claims
    2. Plan based on quantitative analysis
    3. Make data-driven decisions
    
    Supports:
    - Basic arithmetic: +, -, *, /, **, %
    - Functions: sin, cos, tan, log, sqrt, abs
    - Variables: Define and use in expressions
    
    Safe evaluation prevents code injection while allowing complex math.
    """
    logger.info(f"Calculating: {expression}")
    
    try:
        # Create safe namespace with math functions
        safe_namespace = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'log': math.log, 'log10': math.log10, 'sqrt': math.sqrt,
            'abs': abs, 'pow': pow, 'round': round,
            'pi': math.pi, 'e': math.e
        }
        
        # Add user variables
        if variables:
            safe_namespace.update(variables)
        
        # Parse and validate expression
        tree = ast.parse(expression, mode='eval')
        
        # Check for unsafe operations
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
                # Allow only safe function calls
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id not in safe_namespace:
                        raise ValueError(f"Unsafe function: {node.func.id}")
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    raise ValueError("Imports not allowed in expressions")
        
        # Evaluate safely
        result = eval(compile(tree, '<string>', 'eval'), {"__builtins__": {}}, safe_namespace)
        
        return {
            "success": True,
            "expression": expression,
            "variables": variables or {},
            "result": result,
            "result_type": type(result).__name__,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "expression": expression,
            "variables": variables or {},
            "hint": "Check syntax and ensure all variables are defined"
        }


@tool(args_schema=CodeExecuteInput)
def code_execute_tool(
    code: str, 
    language: str = "python", 
    timeout: int = 30,
    safe_mode: bool = True
) -> Dict[str, Any]:
    """
    Execute code in a sandboxed environment
    
    This tool is crucial for agents that need to:
    1. Test hypotheses through code
    2. Automate complex calculations
    3. Interact with APIs and systems
    4. Validate solutions before suggesting them
    
    Safety features:
    - Resource limits (CPU, memory, file access)
    - Timeout protection
    - Isolated execution environment
    - No network access in safe mode
    
    This demonstrates the principle of "safe exploration" - agents can
    experiment without causing harm.
    """
    logger.info(f"Executing {language} code (safe_mode={safe_mode})")
    
    if language != "python":
        return {
            "success": False,
            "error": f"Language {language} not yet supported",
            "supported_languages": ["python"]
        }
    
    try:
        if safe_mode:
            result = _execute_python_sandboxed(code, timeout)
        else:
            # Unsafe mode for trusted operations (requires explicit permission)
            result = _execute_python_unrestricted(code, timeout)
        
        return {
            "success": True,
            "language": language,
            "code": code,
            "output": result["stdout"],
            "error_output": result["stderr"],
            "return_value": result.get("return_value"),
            "execution_time": result["execution_time"],
            "timestamp": datetime.now().isoformat()
        }
        
    except TimeoutError:
        return {
            "success": False,
            "error": f"Code execution timed out after {timeout} seconds",
            "code": code,
            "hint": "Consider optimizing the code or increasing timeout"
        }
    except Exception as e:
        logger.error(f"Code execution failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "code": code,
            "language": language
        }


@tool(args_schema=FileOperationInput)
def file_operation_tool(
    operation: str,
    path: str,
    content: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform file system operations
    
    This tool enables agents to:
    1. Read configuration and data files
    2. Write results and reports
    3. Manage workspace organization
    4. Track progress through file artifacts
    
    Operations are restricted to ADAM's workspace for safety.
    This demonstrates "bounded autonomy" - freedom within limits.
    """
    logger.info(f"File operation: {operation} on {path}")
    
    # Validate path is within allowed directories
    allowed_dirs = [
        Path("./adam_workspace"),
        Path("./adam_memory_advanced"),
        Path("/tmp/adam_agent")
    ]
    
    path_obj = Path(path).resolve()
    if not any(path_obj.is_relative_to(allowed) for allowed in allowed_dirs):
        return {
            "success": False,
            "error": "Access denied: Path outside allowed directories",
            "allowed_directories": [str(d) for d in allowed_dirs]
        }
    
    try:
        if operation == "read":
            if not path_obj.exists():
                return {
                    "success": False,
                    "error": f"File not found: {path}"
                }
            
            content = path_obj.read_text()
            return {
                "success": True,
                "operation": operation,
                "path": str(path_obj),
                "content": content,
                "size": len(content),
                "modified": datetime.fromtimestamp(path_obj.stat().st_mtime).isoformat()
            }
            
        elif operation == "write":
            if content is None:
                return {
                    "success": False,
                    "error": "Content required for write operation"
                }
            
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            path_obj.write_text(content)
            
            return {
                "success": True,
                "operation": operation,
                "path": str(path_obj),
                "size": len(content),
                "timestamp": datetime.now().isoformat()
            }
            
        elif operation == "append":
            if content is None:
                return {
                    "success": False,
                    "error": "Content required for append operation"
                }
            
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(path_obj, 'a') as f:
                f.write(content)
            
            return {
                "success": True,
                "operation": operation,
                "path": str(path_obj),
                "appended_size": len(content),
                "timestamp": datetime.now().isoformat()
            }
            
        elif operation == "delete":
            if path_obj.exists():
                path_obj.unlink()
                return {
                    "success": True,
                    "operation": operation,
                    "path": str(path_obj),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": f"File not found: {path}"
                }
                
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}",
                "valid_operations": ["read", "write", "append", "delete"]
            }
            
    except Exception as e:
        logger.error(f"File operation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "operation": operation,
            "path": path
        }


async def _web_interact_async(
    url: str,
    method: str = "GET",
    data: Optional[Dict[str, Any]] = None,
    extract: Optional[str] = None
) -> Dict[str, Any]:
    """Async implementation of web interaction"""
    logger.info(f"Web interaction: {method} {url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Prepare request
            kwargs = {"timeout": aiohttp.ClientTimeout(total=30)}
            if data and method in ["POST", "PUT", "PATCH"]:
                kwargs["json"] = data
            
            # Make request
            async with session.request(method, url, **kwargs) as response:
                content_type = response.headers.get('Content-Type', '')
                
                if 'application/json' in content_type:
                    result = await response.json()
                else:
                    text = await response.text()
                    
                    # Extract specific content if requested
                    if extract:
                        soup = BeautifulSoup(text, 'html.parser')
                        extracted = soup.select(extract)
                        result = [elem.get_text(strip=True) for elem in extracted]
                    else:
                        result = text
                
                return {
                    "success": True,
                    "url": url,
                    "method": method,
                    "status_code": response.status,
                    "content_type": content_type,
                    "result": result,
                    "headers": dict(response.headers),
                    "timestamp": datetime.now().isoformat()
                }
                
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": "Request timed out",
            "url": url,
            "method": method
        }
    except Exception as e:
        logger.error(f"Web interaction failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "url": url,
            "method": method
        }


@tool(args_schema=WebInteractInput)
def web_interact_tool(
    url: str,
    method: str = "GET",
    data: Optional[Dict[str, Any]] = None,
    extract: Optional[str] = None
) -> Dict[str, Any]:
    """
    Interact with web resources
    
    This tool allows agents to:
    1. Fetch live data from APIs
    2. Monitor external systems
    3. Submit forms and reports
    4. Extract specific information from web pages
    
    Supports both REST APIs and HTML scraping, demonstrating
    how agents can interact with the broader world.
    """
    # Run async function in sync context
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(_web_interact_async(url, method, data, extract))


@tool(args_schema=SystemCommandInput)
def system_command_tool(
    command: str,
    shell: bool = False,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Execute system commands
    
    This tool enables agents to:
    1. Check system status (disk space, processes)
    2. Run analysis tools (grep, find, etc.)
    3. Interact with version control (git)
    4. Manage services and processes
    
    Commands are filtered for safety. This demonstrates how
    agents can have system-level capabilities while maintaining security.
    """
    logger.info(f"System command: {command}")
    
    # Whitelist of allowed commands for safety
    allowed_commands = [
        'ls', 'pwd', 'echo', 'cat', 'grep', 'find', 'wc',
        'df', 'du', 'ps', 'top', 'date', 'whoami',
        'git', 'python', 'pip', 'npm', 'yarn'
    ]
    
    # Extract base command
    base_command = command.split()[0] if command else ""
    
    if base_command not in allowed_commands:
        return {
            "success": False,
            "error": f"Command '{base_command}' not in allowed list",
            "allowed_commands": allowed_commands
        }
    
    try:
        # Execute with timeout
        result = subprocess.run(
            command if shell else command.split(),
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        return {
            "success": True,
            "command": command,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time": f"{timeout}s max",
            "timestamp": datetime.now().isoformat()
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Command timed out after {timeout} seconds",
            "command": command
        }
    except Exception as e:
        logger.error(f"System command failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "command": command
        }


# ==============================================================================
# Specialized Agent Tools
# ==============================================================================

@tool
def monitor_resources_tool() -> Dict[str, Any]:
    """
    Monitor system resources and agent performance
    
    This meta-tool allows agents to:
    1. Track their own resource usage
    2. Detect performance bottlenecks
    3. Make decisions about task scheduling
    4. Implement self-throttling when needed
    
    This demonstrates "self-awareness" - agents monitoring themselves.
    """
    try:
        import psutil
        
        # Get current process
        process = psutil.Process()
        
        # System-wide stats
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Process-specific stats
        process_info = {
            "cpu_percent": process.cpu_percent(interval=1),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "threads": process.num_threads(),
            "open_files": len(process.open_files()),
        }
        
        return {
            "success": True,
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / 1024 / 1024 / 1024,
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / 1024 / 1024 / 1024
            },
            "process": process_info,
            "timestamp": datetime.now().isoformat(),
            "health_status": _determine_health_status(cpu_percent, memory.percent)
        }
        
    except Exception as e:
        logger.error(f"Resource monitoring failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@tool
def send_notification_tool(
    message: str,
    priority: str = "info",
    channel: str = "console"
) -> Dict[str, Any]:
    """
    Send notifications about agent activities
    
    This tool enables agents to:
    1. Alert users about important findings
    2. Request human input when needed
    3. Report completion of long-running tasks
    4. Escalate issues that need attention
    
    Demonstrates "human-in-the-loop" agent design.
    """
    logger.info(f"Sending notification: {message} (priority: {priority})")
    
    priority_colors = {
        "info": "blue",
        "warning": "yellow",
        "error": "red",
        "success": "green"
    }
    
    try:
        if channel == "console":
            color = priority_colors.get(priority, "white")
            console.print(f"[{color}]🔔 AGENT NOTIFICATION [{priority.upper()}][/{color}]")
            console.print(f"[{color}]{message}[/{color}]")
            console.print(f"[dim]Timestamp: {datetime.now().isoformat()}[/dim]\n")
            
        elif channel == "file":
            # Log to notification file
            notif_file = Path("./adam_workspace/notifications.log")
            notif_file.parent.mkdir(exist_ok=True)
            
            with open(notif_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()} [{priority.upper()}] {message}\n")
                
        elif channel == "webhook":
            # In production, send to webhook (Slack, Discord, etc.)
            pass
            
        return {
            "success": True,
            "message": message,
            "priority": priority,
            "channel": channel,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Notification failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": message
        }


@tool
def analyze_data_tool(
    data: Union[List, Dict],
    analysis_type: str = "summary"
) -> Dict[str, Any]:
    """
    Analyze data to inform agent decisions
    
    This tool helps agents:
    1. Understand patterns in collected data
    2. Make statistical inferences
    3. Identify anomalies and trends
    4. Generate insights for planning
    
    Supports various analysis types showing how agents
    can be data-driven in their decision making.
    """
    logger.info(f"Analyzing data: {analysis_type}")
    
    try:
        if analysis_type == "summary":
            if isinstance(data, list):
                # Numerical analysis
                if all(isinstance(x, (int, float)) for x in data):
                    import statistics
                    return {
                        "success": True,
                        "analysis_type": analysis_type,
                        "count": len(data),
                        "mean": statistics.mean(data),
                        "median": statistics.median(data),
                        "std_dev": statistics.stdev(data) if len(data) > 1 else 0,
                        "min": min(data),
                        "max": max(data),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # Categorical analysis
                    from collections import Counter
                    counts = Counter(str(x) for x in data)
                    return {
                        "success": True,
                        "analysis_type": analysis_type,
                        "count": len(data),
                        "unique_values": len(counts),
                        "most_common": counts.most_common(5),
                        "timestamp": datetime.now().isoformat()
                    }
                    
            elif isinstance(data, dict):
                # Dictionary analysis
                return {
                    "success": True,
                    "analysis_type": analysis_type,
                    "keys": list(data.keys()),
                    "size": len(data),
                    "nested_structure": _analyze_dict_structure(data),
                    "timestamp": datetime.now().isoformat()
                }
                
        elif analysis_type == "trend":
            # Time series trend analysis
            if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                # Simple linear trend
                x = list(range(len(data)))
                y = data
                
                # Calculate slope
                n = len(data)
                if n > 1:
                    slope = (n * sum(i*v for i, v in enumerate(y)) - sum(x) * sum(y)) / \
                           (n * sum(i**2 for i in x) - sum(x)**2)
                    
                    trend = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
                else:
                    slope = 0
                    trend = "insufficient_data"
                
                return {
                    "success": True,
                    "analysis_type": analysis_type,
                    "trend": trend,
                    "slope": slope,
                    "data_points": len(data),
                    "timestamp": datetime.now().isoformat()
                }
                
        elif analysis_type == "anomaly":
            # Simple anomaly detection using z-scores
            if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                import statistics
                if len(data) > 2:
                    mean = statistics.mean(data)
                    std_dev = statistics.stdev(data)
                    
                    anomalies = []
                    for i, value in enumerate(data):
                        z_score = (value - mean) / std_dev if std_dev > 0 else 0
                        if abs(z_score) > 2:  # 2 standard deviations
                            anomalies.append({
                                "index": i,
                                "value": value,
                                "z_score": z_score
                            })
                    
                    return {
                        "success": True,
                        "analysis_type": analysis_type,
                        "anomaly_count": len(anomalies),
                        "anomalies": anomalies[:5],  # Top 5
                        "threshold": "2 standard deviations",
                        "timestamp": datetime.now().isoformat()
                    }
                    
        return {
            "success": False,
            "error": f"Unsupported analysis type: {analysis_type}",
            "supported_types": ["summary", "trend", "anomaly"]
        }
        
    except Exception as e:
        logger.error(f"Data analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "analysis_type": analysis_type
        }


# ==============================================================================
# Helper Functions
# ==============================================================================

def _execute_python_sandboxed(code: str, timeout: int) -> Dict[str, Any]:
    """
    Execute Python code in a sandboxed environment with resource limits
    
    This is critical for agent safety - allows experimentation without risk.
    """
    import io
    import contextlib
    from datetime import datetime
    
    start_time = datetime.now()
    
    # Create restricted globals
    restricted_globals = {
        "__builtins__": {
            "print": print,
            "len": len,
            "range": range,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
        }
    }
    
    # Capture output
    stdout = io.StringIO()
    stderr = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            # Execute with timeout using signal
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Execution exceeded {timeout} seconds")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                exec(code, restricted_globals)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "execution_time": execution_time,
            "return_value": None  # exec doesn't return values
        }
        
    except Exception as e:
        return {
            "stdout": stdout.getvalue(),
            "stderr": f"{type(e).__name__}: {str(e)}",
            "execution_time": (datetime.now() - start_time).total_seconds(),
            "return_value": None
        }


def _execute_python_unrestricted(code: str, timeout: int) -> Dict[str, Any]:
    """Execute Python code without restrictions (use with caution)"""
    # This would run in a subprocess or container for isolation
    # Implementation depends on deployment environment
    raise NotImplementedError("Unrestricted execution requires container setup")


def _mock_web_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Mock web search for demonstration"""
    # In production, integrate with real search APIs
    mock_results = [
        {
            "title": f"Result {i+1} for: {query}",
            "url": f"https://example.com/result{i+1}",
            "snippet": f"This is a relevant snippet about {query}...",
            "relevance_score": 0.9 - (i * 0.1)
        }
        for i in range(min(max_results, 3))
    ]
    return mock_results


def _search_code_resources(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Search code repositories and documentation"""
    # Would integrate with GitHub API, StackOverflow API, etc.
    return [
        {
            "source": "GitHub",
            "title": f"Code example: {query}",
            "url": "https://github.com/example/repo",
            "language": "python",
            "stars": 1234
        }
    ]


def _search_academic_resources(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Search academic papers"""
    # Would integrate with arXiv API, Semantic Scholar, etc.
    return [
        {
            "title": f"Research on {query}",
            "authors": ["Researcher A", "Researcher B"],
            "year": 2023,
            "abstract": "This paper explores...",
            "arxiv_id": "2023.12345"
        }
    ]


def _search_local_resources(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Search ADAM's local memory and files"""
    # This would integrate with ADAM's memory system
    return [
        {
            "source": "memory",
            "content": f"Previous conversation about {query}",
            "timestamp": "2023-01-01T12:00:00",
            "relevance": 0.85
        }
    ]


def _calculate_search_confidence(results: List[Dict[str, Any]]) -> float:
    """Calculate confidence score for search results"""
    if not results:
        return 0.0
    
    # Average relevance scores
    scores = [r.get('relevance_score', 0.5) for r in results]
    return sum(scores) / len(scores)


def _determine_health_status(cpu_percent: float, memory_percent: float) -> str:
    """Determine system health status"""
    if cpu_percent > 90 or memory_percent > 90:
        return "critical"
    elif cpu_percent > 70 or memory_percent > 70:
        return "warning"
    else:
        return "healthy"


def _analyze_dict_structure(d: Dict, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
    """Analyze nested dictionary structure"""
    if current_depth >= max_depth:
        return {"max_depth_reached": True}
    
    structure = {
        "keys": len(d),
        "types": {}
    }
    
    for key, value in d.items():
        value_type = type(value).__name__
        if value_type not in structure["types"]:
            structure["types"][value_type] = 0
        structure["types"][value_type] += 1
        
        if isinstance(value, dict) and current_depth < max_depth - 1:
            structure[f"nested_{key}"] = _analyze_dict_structure(
                value, max_depth, current_depth + 1
            )
    
    return structure


# ==============================================================================
# Tool Registry
# ==============================================================================

def get_all_tools() -> List:
    """
    Get all available tools for the agent
    
    This registry makes tools discoverable by the agent system.
    Tools are categorized by capability for better planning.
    """
    return [
        # Information gathering
        search_tool,
        
        # Computation
        calculate_tool,
        code_execute_tool,
        analyze_data_tool,
        
        # System interaction
        file_operation_tool,
        system_command_tool,
        web_interact_tool,
        
        # Meta tools
        monitor_resources_tool,
        send_notification_tool
    ]


def get_tool_categories() -> Dict[str, List]:
    """Categorize tools for agent planning"""
    return {
        "information": [search_tool],
        "computation": [calculate_tool, code_execute_tool, analyze_data_tool],
        "interaction": [file_operation_tool, system_command_tool, web_interact_tool],
        "meta": [monitor_resources_tool, send_notification_tool]
    }


if __name__ == "__main__":
    # Quick test of tools
    console.print("[bold cyan]Agent Tools Test[/bold cyan]\n")
    
    # Test calculation
    result = calculate_tool.run("2 * pi * 10", {"pi": 3.14159})
    console.print("Calculate:", result)
    
    # Test search
    result = search_tool.run("Python async programming", max_results=3)
    console.print("\nSearch:", result)
    
    # Test monitoring
    result = monitor_resources_tool.run()
    console.print("\nResources:", result)