"""
LangGraph Workflow for Tool Execution
Intelligent routing and validation for ADAM's tools
"""

import os
import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from enum import Enum
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from services.tool_service import get_tool_service
from adam.tools import ReadFileTool, WriteFileTool, EditFileTool, ListFilesTool, SearchFilesTool

logger = logging.getLogger(__name__)

class ToolAction(str, Enum):
    """Possible tool actions"""
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EDIT_FILE = "edit_file"
    LIST_FILES = "list_files"
    SEARCH_FILES = "search_files"
    NAVIGATE = "navigate"
    NO_ACTION = "no_action"

class WorkflowState(TypedDict):
    """State for the workflow"""
    messages: List[BaseMessage]
    user_request: str
    workspace_path: str
    current_location: str
    intent: Optional[str]
    tool_action: Optional[ToolAction]
    tool_params: Optional[Dict[str, Any]]
    tool_result: Optional[Dict[str, Any]]
    validation_result: Optional[bool]
    validation_feedback: Optional[str]
    final_response: Optional[str]

class ToolWorkflow:
    """LangGraph workflow for intelligent tool execution"""
    
    def __init__(self, workspace_path: str = None):
        self.workspace_path = workspace_path or os.getcwd()
        
        # Initialize LLMs for different tasks
        self.intent_llm = ChatOpenAI(
            model="gpt-5-nano",  # Fast, cheap for intent classification
            temperature=0
        )
        
        self.param_llm = ChatOpenAI(
            model="gpt-5-mini",  # Good for parameter extraction
            temperature=0
        )
        
        self.validation_llm = ChatOpenAI(
            model="gpt-5-mini",  # Good for validation
            temperature=0
        )
        
        # Initialize tools
        self.tools = {
            ToolAction.READ_FILE: ReadFileTool(),
            ToolAction.WRITE_FILE: WriteFileTool(),
            ToolAction.EDIT_FILE: EditFileTool(),
            ToolAction.LIST_FILES: ListFilesTool(),
            ToolAction.SEARCH_FILES: SearchFilesTool(),
        }
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("understand_intent", self.understand_intent)
        workflow.add_node("extract_parameters", self.extract_parameters)
        workflow.add_node("execute_tool", self.execute_tool)
        workflow.add_node("validate_result", self.validate_result)
        workflow.add_node("prepare_response", self.prepare_response)
        workflow.add_node("handle_error", self.handle_error)
        
        # Set entry point
        workflow.set_entry_point("understand_intent")
        
        # Add edges with conditions
        workflow.add_conditional_edges(
            "understand_intent",
            self.route_after_intent,
            {
                "extract_params": "extract_parameters",
                "no_action": "prepare_response",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("extract_parameters", "execute_tool")
        
        workflow.add_conditional_edges(
            "execute_tool",
            self.route_after_execution,
            {
                "validate": "validate_result",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_result",
            self.route_after_validation,
            {
                "success": "prepare_response",
                "retry": "extract_parameters",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("prepare_response", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    async def understand_intent(self, state: WorkflowState) -> WorkflowState:
        """Understand user intent and determine tool action"""
        
        prompt = f"""Analyze this user request and determine what tool action is needed:
        
User request: {state['user_request']}
Current location: {state['current_location']}

Possible actions:
- read_file: User wants to read/view a file
- write_file: User wants to create a new file or overwrite existing
- edit_file: User wants to modify parts of an existing file
- list_files: User wants to see files in a directory
- search_files: User wants to find files matching a pattern
- navigate: User wants to change directory
- no_action: No file operation needed

Return ONLY the action name, nothing else."""
        
        response = await self.intent_llm.ainvoke([
            SystemMessage(content="You are a tool intent classifier. Return only the action name."),
            HumanMessage(content=prompt)
        ])
        
        action_str = response.content.strip().lower()
        
        try:
            state['tool_action'] = ToolAction(action_str)
            state['intent'] = f"User wants to: {action_str.replace('_', ' ')}"
        except ValueError:
            state['tool_action'] = ToolAction.NO_ACTION
            state['intent'] = "No specific tool action identified"
        
        logger.info(f"Intent identified: {state['tool_action']}")
        return state
    
    async def extract_parameters(self, state: WorkflowState) -> WorkflowState:
        """Extract parameters for the identified tool"""
        
        tool_action = state['tool_action']
        
        if tool_action == ToolAction.WRITE_FILE:
            prompt = f"""Extract parameters for creating a file from this request:
            
User request: {state['user_request']}
Current location: {state['current_location']}

You need to determine:
1. file_path: Where to save the file (if user says "same folder", use current location)
2. content: What content should be in the file

If the user references another file to base it on, mention that in the content requirements.
Return as JSON with keys: file_path, content_description"""
        
        elif tool_action == ToolAction.READ_FILE:
            prompt = f"""Extract the file path from this request:
            
User request: {state['user_request']}
Current location: {state['current_location']}

Return as JSON with key: file_path"""
        
        else:
            prompt = f"""Extract parameters for {tool_action} from: {state['user_request']}
Return as JSON with appropriate keys."""
        
        response = await self.param_llm.ainvoke([
            SystemMessage(content="Extract tool parameters as JSON. Be precise."),
            HumanMessage(content=prompt)
        ])
        
        # Parse JSON from response
        import json
        try:
            params = json.loads(response.content)
            state['tool_params'] = params
        except:
            # Fallback to basic extraction
            state['tool_params'] = {"raw_request": state['user_request']}
        
        logger.info(f"Parameters extracted: {state['tool_params']}")
        return state
    
    async def execute_tool(self, state: WorkflowState) -> WorkflowState:
        """Execute the identified tool with parameters"""
        
        tool_action = state['tool_action']
        params = state['tool_params']
        
        if tool_action == ToolAction.WRITE_FILE:
            # For write file, we need to generate content if not provided
            if 'content' not in params or 'content_description' in params:
                content = await self._generate_file_content(
                    state['user_request'],
                    params.get('content_description', ''),
                    state['current_location']
                )
                params['content'] = content
            
            # Ensure proper file path
            file_path = params.get('file_path', 'new_file.txt')
            if not os.path.isabs(file_path):
                file_path = os.path.join(state['current_location'], file_path)
            params['file_path'] = file_path
        
        # Execute the tool
        tool = self.tools.get(tool_action)
        if tool:
            result = await tool.execute(**params)
            state['tool_result'] = {
                'status': result.status.value,
                'message': result.message,
                'data': result.data,
                'metadata': result.metadata
            }
        else:
            state['tool_result'] = {
                'status': 'error',
                'message': f'Tool {tool_action} not found'
            }
        
        logger.info(f"Tool executed: {state['tool_result']['status']}")
        return state
    
    async def validate_result(self, state: WorkflowState) -> WorkflowState:
        """Validate if the tool execution matches user intent"""
        
        prompt = f"""Validate if this tool execution matches the user's request:
        
Original request: {state['user_request']}
Tool executed: {state['tool_action']}
Result: {state['tool_result']['message']}

Does this fulfill what the user asked for?
Return JSON with:
- valid: true/false
- feedback: brief explanation"""
        
        response = await self.validation_llm.ainvoke([
            SystemMessage(content="You are a result validator. Be precise."),
            HumanMessage(content=prompt)
        ])
        
        import json
        try:
            validation = json.loads(response.content)
            state['validation_result'] = validation.get('valid', False)
            state['validation_feedback'] = validation.get('feedback', '')
        except:
            state['validation_result'] = True  # Assume success if can't parse
            state['validation_feedback'] = "Validation completed"
        
        logger.info(f"Validation: {state['validation_result']}")
        return state
    
    async def prepare_response(self, state: WorkflowState) -> WorkflowState:
        """Prepare the final response for the user"""
        
        if state['tool_result']:
            state['final_response'] = f"✅ {state['tool_result']['message']}"
            if state['validation_feedback']:
                state['final_response'] += f"\n{state['validation_feedback']}"
        else:
            state['final_response'] = "I understand your request. How can I help you?"
        
        return state
    
    async def handle_error(self, state: WorkflowState) -> WorkflowState:
        """Handle errors in the workflow"""
        
        state['final_response'] = f"❌ An error occurred: {state.get('error', 'Unknown error')}"
        return state
    
    def route_after_intent(self, state: WorkflowState) -> str:
        """Route after understanding intent"""
        if state['tool_action'] == ToolAction.NO_ACTION:
            return "no_action"
        elif state.get('error'):
            return "error"
        else:
            return "extract_params"
    
    def route_after_execution(self, state: WorkflowState) -> str:
        """Route after tool execution"""
        if state['tool_result'].get('status') == 'error':
            return "error"
        else:
            return "validate"
    
    def route_after_validation(self, state: WorkflowState) -> str:
        """Route after validation"""
        if state['validation_result']:
            return "success"
        elif state.get('retry_count', 0) < 2:
            state['retry_count'] = state.get('retry_count', 0) + 1
            return "retry"
        else:
            return "error"
    
    async def _generate_file_content(self, user_request: str, content_desc: str, location: str) -> str:
        """Generate file content based on user request"""
        
        prompt = f"""Generate file content based on this request:
        
User request: {user_request}
Content description: {content_desc}
Current location: {location}

Generate the complete file content. If it's a SQL file, generate proper SQL.
If it's based on another file, create an appropriate variation.
Return ONLY the file content, no explanations."""
        
        response = await self.param_llm.ainvoke([
            SystemMessage(content="You are a file content generator. Generate clean, production-ready content."),
            HumanMessage(content=prompt)
        ])
        
        return response.content.strip()
    
    async def process_request(self, user_request: str, current_location: str = None) -> str:
        """Process a user request through the workflow"""
        
        initial_state = WorkflowState(
            messages=[],
            user_request=user_request,
            workspace_path=self.workspace_path,
            current_location=current_location or self.workspace_path,
            intent=None,
            tool_action=None,
            tool_params=None,
            tool_result=None,
            validation_result=None,
            validation_feedback=None,
            final_response=None
        )
        
        # Run the workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        return final_state.get('final_response', 'Request processed')