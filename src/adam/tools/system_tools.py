"""
System Tools for ADAM
Provides system operations like running commands, git operations, etc.
"""

import os
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from .base import Tool, ToolResult, ToolStatus
import platform
import json

class RunCommandTool(Tool):
    """Run shell commands"""
    
    def __init__(self):
        super().__init__(
            name="run_command",
            description="Execute shell commands"
        )
        self.requires_confirmation = True  # Safety
    
    async def execute(self,
                     command: str,
                     cwd: Optional[str] = None,
                     timeout: int = 30) -> ToolResult:
        """Execute a shell command"""
        try:
            # Security check - block dangerous commands
            dangerous = ['rm -rf /', 'format', 'del /f', 'shutdown']
            if any(d in command.lower() for d in dangerous):
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message="Command blocked for safety reasons"
                )
            
            # Run command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Command timed out after {timeout} seconds"
                )
            
            # Decode output
            stdout_text = stdout.decode('utf-8', errors='replace')
            stderr_text = stderr.decode('utf-8', errors='replace')
            
            if process.returncode == 0:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data=stdout_text,
                    message=f"Command executed successfully",
                    metadata={
                        "command": command,
                        "return_code": process.returncode,
                        "stderr": stderr_text if stderr_text else None
                    }
                )
            else:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=stderr_text,
                    message=f"Command failed with return code {process.returncode}",
                    metadata={
                        "command": command,
                        "return_code": process.returncode,
                        "stdout": stdout_text if stdout_text else None
                    }
                )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Command execution failed: {str(e)}"
            )
    
    def validate_params(self, **kwargs) -> bool:
        return 'command' in kwargs
    
    def _get_param_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "cwd": {"type": "string", "description": "Working directory"},
                "timeout": {"type": "integer", "description": "Timeout in seconds"}
            },
            "required": ["command"]
        }

class GitOperationsTool(Tool):
    """Perform Git operations"""
    
    def __init__(self):
        super().__init__(
            name="git_operations",
            description="Perform Git operations"
        )
    
    async def execute(self,
                     operation: str,
                     repo_path: str = ".",
                     **kwargs) -> ToolResult:
        """Execute Git operations"""
        try:
            repo = Path(repo_path).resolve()
            
            # Check if it's a git repo
            if not (repo / ".git").exists():
                if operation == "init":
                    # Initialize new repo
                    result = await self._run_git_command("init", repo)
                    return ToolResult(
                        status=ToolStatus.SUCCESS,
                        data=result,
                        message="Git repository initialized"
                    )
                else:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        data=None,
                        message="Not a git repository"
                    )
            
            # Handle different operations
            if operation == "status":
                result = await self._run_git_command("status --short", repo)
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data=result,
                    message="Git status retrieved"
                )
            
            elif operation == "add":
                files = kwargs.get("files", ".")
                result = await self._run_git_command(f"add {files}", repo)
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data=result,
                    message=f"Files added to staging"
                )
            
            elif operation == "commit":
                message = kwargs.get("message", "Update")
                result = await self._run_git_command(f'commit -m "{message}"', repo)
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data=result,
                    message="Changes committed"
                )
            
            elif operation == "branch":
                branch_name = kwargs.get("name")
                if branch_name:
                    result = await self._run_git_command(f"checkout -b {branch_name}", repo)
                    return ToolResult(
                        status=ToolStatus.SUCCESS,
                        data=result,
                        message=f"Created and switched to branch '{branch_name}'"
                    )
                else:
                    result = await self._run_git_command("branch", repo)
                    return ToolResult(
                        status=ToolStatus.SUCCESS,
                        data=result,
                        message="Listed branches"
                    )
            
            elif operation == "diff":
                result = await self._run_git_command("diff", repo)
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data=result,
                    message="Git diff retrieved"
                )
            
            elif operation == "log":
                limit = kwargs.get("limit", 10)
                result = await self._run_git_command(f"log --oneline -n {limit}", repo)
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data=result,
                    message=f"Retrieved last {limit} commits"
                )
            
            elif operation == "clone":
                url = kwargs.get("url")
                if not url:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        data=None,
                        message="Clone requires 'url' parameter"
                    )
                result = await self._run_git_command(f"clone {url}", Path.cwd())
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data=result,
                    message=f"Repository cloned from {url}"
                )
            
            else:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Unknown git operation: {operation}"
                )
                
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Git operation failed: {str(e)}"
            )
    
    async def _run_git_command(self, command: str, cwd: Path) -> str:
        """Run a git command"""
        process = await asyncio.create_subprocess_shell(
            f"git {command}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd)
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(stderr.decode('utf-8'))
        
        return stdout.decode('utf-8')
    
    def validate_params(self, **kwargs) -> bool:
        return 'operation' in kwargs
    
    def _get_param_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "description": "Git operation (status, add, commit, branch, diff, log, clone, init)"},
                "repo_path": {"type": "string", "description": "Repository path"},
                "files": {"type": "string", "description": "Files for add operation"},
                "message": {"type": "string", "description": "Commit message"},
                "name": {"type": "string", "description": "Branch name"},
                "url": {"type": "string", "description": "Repository URL for clone"},
                "limit": {"type": "integer", "description": "Limit for log"}
            },
            "required": ["operation"]
        }

class EnvironmentInfoTool(Tool):
    """Get environment and system information"""
    
    def __init__(self):
        super().__init__(
            name="environment_info",
            description="Get system and environment information"
        )
    
    async def execute(self, info_type: str = "all") -> ToolResult:
        """Get environment information"""
        try:
            info = {}
            
            if info_type in ["all", "system"]:
                info["system"] = {
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "python_version": platform.python_version(),
                    "hostname": platform.node(),
                    "os": platform.system(),
                    "os_release": platform.release()
                }
            
            if info_type in ["all", "environment"]:
                # Get selected environment variables
                env_vars = {
                    "PATH": os.environ.get("PATH", "").split(os.pathsep)[:5],  # First 5 PATH entries
                    "USER": os.environ.get("USER"),
                    "HOME": os.environ.get("HOME"),
                    "PWD": os.environ.get("PWD"),
                    "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
                    "PYTHON_PATH": os.environ.get("PYTHONPATH")
                }
                info["environment"] = {k: v for k, v in env_vars.items() if v}
            
            if info_type in ["all", "packages"]:
                # Get installed packages
                try:
                    import pkg_resources
                    packages = [
                        f"{d.project_name}=={d.version}"
                        for d in pkg_resources.working_set
                    ]
                    info["packages"] = packages[:20]  # First 20 packages
                except:
                    info["packages"] = "Unable to retrieve package list"
            
            if info_type in ["all", "disk"]:
                import shutil
                total, used, free = shutil.disk_usage("/")
                info["disk"] = {
                    "total_gb": round(total / (1024**3), 2),
                    "used_gb": round(used / (1024**3), 2),
                    "free_gb": round(free / (1024**3), 2),
                    "percent_used": round((used / total) * 100, 1)
                }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=json.dumps(info, indent=2),
                message="Environment information retrieved",
                metadata=info
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to get environment info: {str(e)}"
            )
    
    def validate_params(self, **kwargs) -> bool:
        return True  # All params optional
    
    def _get_param_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "info_type": {"type": "string", "description": "Type of info (all, system, environment, packages, disk)"}
            }
        }