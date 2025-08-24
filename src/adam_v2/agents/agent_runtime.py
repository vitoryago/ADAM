"""
Agent Runtime System
Real agent execution that runs tasks in the background
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
import json

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool as LangChainTool
from langchain_core.messages import HumanMessage, AIMessage

# Import ADAM's actual tools
from adam.tools import (
    ReadFileTool, WriteFileTool, EditFileTool,
    DeleteFileTool, ListFilesTool, SearchFilesTool
)
from adam.tools.system_tools import RunCommandTool

logger = logging.getLogger(__name__)

class AgentRuntime:
    """
    Real agent runtime that executes tasks
    Similar to how Claude Code actually runs tools
    """
    
    def __init__(self, workspace_path: str = None):
        self.workspace_path = workspace_path
        self.agents = {}
        self.initialize_agents()
        
    def initialize_agents(self):
        """Initialize all agent types"""
        
        # File Explorer Agent - navigates and understands repositories
        self.agents['explorer'] = self._create_explorer_agent()
        
        # File Operations Agent - reads, writes, edits files
        self.agents['file_ops'] = self._create_file_ops_agent()
        
        # Analysis Agent - understands code structure
        self.agents['analyzer'] = self._create_analyzer_agent()
        
        # Orchestrator Agent - routes to other agents
        self.agents['orchestrator'] = self._create_orchestrator_agent()
    
    def _create_explorer_agent(self):
        """Agent that explores repositories"""
        
        # Convert ADAM tools to LangChain tools
        tools = [
            self._wrap_tool(ListFilesTool(), "list_files", "List files in a directory"),
            self._wrap_tool(SearchFilesTool(), "search_files", "Search for files by pattern"),
            self._wrap_tool(ReadFileTool(), "read_file", "Read file contents"),
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a file explorer agent. Your job is to:
            1. Navigate through directories
            2. Find specific files or patterns
            3. Understand repository structure
            
            Always execute the actual tools to get real results.
            Don't make assumptions - actually explore."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        llm = ChatOpenAI(model="gpt-5-mini", temperature=1.0)  # GPT-5 only supports default temperature
        agent = create_openai_tools_agent(llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=5
        )
    
    def _create_file_ops_agent(self):
        """Agent that performs file operations"""
        
        tools = [
            self._wrap_tool(ReadFileTool(), "read_file", "Read file contents"),
            self._wrap_tool(WriteFileTool(), "write_file", "Write content to file"),
            self._wrap_tool(EditFileTool(), "edit_file", "Edit parts of a file"),
            self._wrap_tool(DeleteFileTool(), "delete_file", "Delete a file"),
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a file operations agent. Your job is to:
            1. Read files when asked
            2. Create new files with appropriate content
            3. Edit existing files
            4. Delete files when needed
            
            When creating files, generate appropriate content based on context.
            Always execute the actual tools - don't just describe what you would do."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        llm = ChatOpenAI(model="gpt-5", temperature=1.0)  # GPT-5 only supports default temperature
        agent = create_openai_tools_agent(llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=3
        )
    
    def _create_analyzer_agent(self):
        """Agent that analyzes code and structure"""
        
        tools = [
            self._wrap_tool(ReadFileTool(), "read_file", "Read file contents"),
            self._wrap_tool(SearchFilesTool(), "search_files", "Search for patterns"),
            self._wrap_tool(RunCommandTool(), "run_command", "Run analysis commands"),
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a code analysis agent. Your job is to:
            1. Understand code structure and patterns
            2. Identify relationships between files
            3. Analyze dependencies and imports
            4. Summarize functionality
            
            Always read actual files to understand them - don't guess."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        llm = ChatOpenAI(model="gpt-5-mini", temperature=1.0)  # GPT-5 only supports default temperature
        agent = create_openai_tools_agent(llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True
        )
    
    def _create_orchestrator_agent(self):
        """Master agent that routes to other agents"""
        
        # Tools that call other agents
        tools = [
            LangChainTool(
                name="explorer_agent",
                func=lambda x: asyncio.run(self._run_agent('explorer', x)),
                description="Use this to explore directories and find files"
            ),
            LangChainTool(
                name="file_ops_agent",
                func=lambda x: asyncio.run(self._run_agent('file_ops', x)),
                description="Use this to read, write, or edit files"
            ),
            LangChainTool(
                name="analyzer_agent",
                func=lambda x: asyncio.run(self._run_agent('analyzer', x)),
                description="Use this to analyze code structure and patterns"
            ),
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the master orchestrator agent. Your job is to:
            1. Understand what the user wants
            2. Break it down into tasks
            3. Route to the appropriate specialized agents
            4. Combine their results into a coherent response
            
            Available agents:
            - explorer_agent: For navigating directories and finding files
            - file_ops_agent: For reading, writing, editing files
            - analyzer_agent: For understanding code structure
            
            Always use the appropriate agent for each task.
            Don't try to do everything yourself - delegate to specialists."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        llm = ChatOpenAI(model="gpt-5", temperature=1.0)  # GPT-5 only supports default temperature
        agent = create_openai_tools_agent(llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=5
        )
    
    def _wrap_tool(self, adam_tool, name: str, description: str) -> LangChainTool:
        """Wrap ADAM tool as LangChain tool"""
        
        async def tool_func(**kwargs):
            """Execute ADAM tool and return result"""
            try:
                result = await adam_tool.execute(**kwargs)
                if hasattr(result, 'data') and result.data:
                    return result.data
                elif hasattr(result, 'message'):
                    return result.message
                else:
                    return str(result)
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Create sync wrapper
        def sync_tool_func(**kwargs):
            return asyncio.run(tool_func(**kwargs))
        
        return LangChainTool(
            name=name,
            func=sync_tool_func,
            description=description,
            args_schema=adam_tool._get_param_schema() if hasattr(adam_tool, '_get_param_schema') else None
        )
    
    async def _run_agent(self, agent_name: str, task: str) -> str:
        """Run a specific agent with a task"""
        
        agent = self.agents.get(agent_name)
        if not agent:
            return f"Agent {agent_name} not found"
        
        try:
            result = await agent.ainvoke({
                "input": task,
                "chat_history": []
            })
            
            # Extract the actual output
            if isinstance(result, dict):
                return result.get('output', str(result))
            return str(result)
            
        except Exception as e:
            logger.error(f"Error running agent {agent_name}: {e}")
            return f"Error: {str(e)}"
    
    async def process_request(self, user_request: str) -> Dict[str, Any]:
        """
        Process a user request through the agent system
        This is the main entry point
        """
        
        logger.info(f"Processing request: {user_request}")
        
        try:
            # Run through orchestrator
            result = await self.agents['orchestrator'].ainvoke({
                "input": user_request,
                "chat_history": []
            })
            
            # Format the response
            if isinstance(result, dict):
                return {
                    'status': 'success',
                    'output': result.get('output', ''),
                    'steps': result.get('intermediate_steps', []),
                    'agents_used': self._extract_agents_used(result)
                }
            
            return {
                'status': 'success',
                'output': str(result)
            }
            
        except Exception as e:
            logger.error(f"Error in agent runtime: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _extract_agents_used(self, result: Dict) -> List[str]:
        """Extract which agents were used"""
        agents_used = []
        
        if 'intermediate_steps' in result:
            for step in result['intermediate_steps']:
                if isinstance(step, tuple) and len(step) > 0:
                    action = step[0]
                    if hasattr(action, 'tool'):
                        agents_used.append(action.tool)
        
        return agents_used


# Global runtime instance
_agent_runtime = None

def get_agent_runtime(workspace_path: str = None) -> AgentRuntime:
    """Get or create the agent runtime"""
    global _agent_runtime
    
    if _agent_runtime is None:
        _agent_runtime = AgentRuntime(workspace_path)
    elif workspace_path:
        _agent_runtime.workspace_path = workspace_path
    
    return _agent_runtime