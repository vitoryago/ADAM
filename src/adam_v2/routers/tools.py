"""
Tools API endpoints for ADAM v2.0
Provides direct access to ADAM's tool capabilities
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, List
from pydantic import BaseModel
import logging

from services.tool_service import get_tool_service

router = APIRouter()
logger = logging.getLogger(__name__)
tool_service = get_tool_service()


class ToolExecuteRequest(BaseModel):
    """Request model for tool execution"""
    tool_name: str
    parameters: Dict[str, Any]


class ToolResponse(BaseModel):
    """Response model for tool execution"""
    status: str
    output: Any
    message: str = None
    metadata: Dict[str, Any] = {}


@router.get("/tools", response_model=List[Dict[str, str]])
async def list_tools():
    """List all available tools"""
    return tool_service.list_tools()


@router.post("/tools/execute", response_model=ToolResponse)
async def execute_tool(request: ToolExecuteRequest):
    """Execute a specific tool with parameters"""
    
    try:
        result = tool_service.execute_tool(
            tool_name=request.tool_name,
            **request.parameters
        )
        
        return ToolResponse(
            status=result.get('status', 'unknown'),
            output=result.get('output', ''),
            message=result.get('message'),
            metadata=result.get('metadata', {})
        )
        
    except Exception as e:
        logger.error(f"Error executing tool {request.tool_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool execution failed: {str(e)}"
        )


@router.post("/tools/process", response_model=ToolResponse)
async def process_message_for_tools(message: str):
    """Process a message to determine if it requires tool usage"""
    
    result = tool_service.process_tool_request(message)
    
    if result:
        return ToolResponse(
            status=result.get('status', 'unknown'),
            output=result.get('output', ''),
            message=result.get('message'),
            metadata=result.get('metadata', {})
        )
    else:
        return ToolResponse(
            status='info',
            output='',
            message='No tool operation detected in message',
            metadata={}
        )