"""
Workflow Service - Integrates LangGraph workflows with ADAM
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class WorkflowService:
    """Service for executing LangGraph workflows"""
    
    def __init__(self, workspace_path: str = None):
        self.workspace_path = workspace_path or str(Path.cwd())
        self.workflow = None
        self._initialize_workflow()
    
    def _initialize_workflow(self):
        """Initialize the tool workflow"""
        try:
            from workflows.tool_workflow import ToolWorkflow
            self.workflow = ToolWorkflow(self.workspace_path)
            logger.info("Tool workflow initialized successfully")
        except ImportError as e:
            logger.warning(f"Could not import ToolWorkflow: {e}")
            logger.info("Workflow features will be limited")
        except Exception as e:
            logger.error(f"Error initializing workflow: {e}")
    
    async def process_with_workflow(self, message: str, workspace_context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Process a message through the workflow if applicable
        
        Returns:
            Dict with tool execution results or None if no tool needed
        """
        
        if not self.workflow:
            return None
        
        # Extract current location from workspace context
        current_location = self.workspace_path
        if workspace_context:
            workspace = workspace_context.get('workspace', {})
            if workspace and not workspace.get('error'):
                folders = workspace.get('folders', [])
                if folders and folders[0].get('path'):
                    current_location = folders[0]['path']
        
        try:
            # Process through workflow
            result = await self.workflow.process_request(
                user_request=message,
                current_location=current_location
            )
            
            if result and result != "I understand your request. How can I help you?":
                return {
                    'status': 'success',
                    'output': result,
                    'workflow_used': True
                }
            
        except Exception as e:
            logger.error(f"Error in workflow processing: {e}")
        
        return None
    
    def update_workspace(self, workspace_path: str):
        """Update the workspace path"""
        self.workspace_path = workspace_path
        if self.workflow:
            self.workflow.workspace_path = workspace_path

# Singleton instance
_workflow_service = None

def get_workflow_service(workspace_path: str = None) -> WorkflowService:
    """Get or create the workflow service singleton"""
    global _workflow_service
    if _workflow_service is None:
        _workflow_service = WorkflowService(workspace_path)
    elif workspace_path:
        _workflow_service.update_workspace(workspace_path)
    return _workflow_service