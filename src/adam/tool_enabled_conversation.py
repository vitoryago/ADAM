"""
Tool-Enabled Conversation System for ADAM
Integrates tools with the conversation flow
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .conversation_system import ConversationSystem
from .llm.client import UnifiedLLMClient
from .memory import ADAMMemoryAdvanced as MemorySystem
from .tools import (
    ToolExecutor,
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    DeleteFileTool,
    ListFilesTool,
    SearchFilesTool,
    GenerateCodeTool,
    CreateDAGTool,
    OptimizeSQLTool,
    CreateProjectStructureTool,
    RunCommandTool,
    GitOperationsTool,
    EnvironmentInfoTool
)

logger = logging.getLogger(__name__)

class ToolUseDecision(Enum):
    """Decision about tool usage"""
    USE_TOOLS = "use_tools"
    NO_TOOLS = "no_tools"
    ASK_CLARIFICATION = "ask_clarification"

@dataclass
class ToolPlan:
    """Plan for using tools"""
    tools_to_use: List[Dict[str, Any]]
    explanation: str
    requires_confirmation: bool = False

class ToolEnabledConversationSystem(ConversationSystem):
    """
    Enhanced conversation system with tool capabilities
    Like Claude Code but for ADAM
    """
    
    def __init__(self, 
                 llm_client: Optional[UnifiedLLMClient] = None,
                 memory_system: Optional[MemorySystem] = None,
                 storage_path: str = "./adam_memory_advanced/conversations"):
        super().__init__(storage_path)
        
        # Store LLM client and memory system
        self.llm_client = llm_client or UnifiedLLMClient()
        self.memory_system = memory_system
        
        # Initialize tool executor
        self.tool_executor = ToolExecutor()
        self._register_all_tools()
        
        # Tool usage patterns
        self.tool_patterns = {
            "read": r"(read|show|display|cat|view|look at|examine)\s+(file|code|document)",
            "write": r"(write|create|save|generate)\s+(file|code|script|document)",
            "edit": r"(edit|modify|change|update|replace|fix)\s+(file|code|in)",
            "delete": r"(delete|remove|rm|del)\s+(file|directory|folder)",
            "list": r"(list|ls|show|display)\s+(files|directories|folders|contents)",
            "search": r"(search|find|grep|look for)\s+.*(in files|in code|in project)",
            "generate": r"(generate|create|write|make)\s+.*(code|function|class|module|script)",
            "dag": r"(create|generate|write)\s+.*(dag|airflow|pipeline|workflow)",
            "sql": r"(optimize|improve|fix)\s+.*(sql|query|database)",
            "project": r"(create|setup|initialize)\s+.*(project|app|application)",
            "command": r"(run|execute|exec)\s+(command|script|bash|shell)",
            "git": r"(git|commit|branch|push|pull|clone|checkout)"
        }
    
    def _register_all_tools(self):
        """Register all available tools"""
        # File tools
        self.tool_executor.register_tools([
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            DeleteFileTool(),
            ListFilesTool(),
            SearchFilesTool()
        ])
        
        # Code tools (they need LLM client)
        self.tool_executor.register_tools([
            GenerateCodeTool(self.llm_client),
            CreateDAGTool(self.llm_client),
            OptimizeSQLTool(self.llm_client),
            CreateProjectStructureTool()
        ])
        
        # System tools
        self.tool_executor.register_tools([
            RunCommandTool(),
            GitOperationsTool(),
            EnvironmentInfoTool()
        ])
        
        logger.info(f"Registered {len(self.tool_executor.tools)} tools")
    
    async def process_message(self, message: str, **kwargs) -> str:
        """
        Process message with tool capabilities
        """
        # First, analyze if tools are needed
        decision, tool_plan = await self._analyze_tool_need(message)
        
        if decision == ToolUseDecision.USE_TOOLS:
            # Execute tools and get results
            tool_results = await self._execute_tool_plan(tool_plan)
            
            # Generate response incorporating tool results
            response = await self._generate_response_with_tools(
                message, tool_results, tool_plan
            )
            
            # Save to memory if important
            if self._should_save_to_memory(message, response, tool_results):
                await self._save_tool_interaction_to_memory(
                    message, response, tool_plan, tool_results
                )
            
            return response
            
        elif decision == ToolUseDecision.ASK_CLARIFICATION:
            # Ask for more details
            return await self._ask_for_clarification(message, tool_plan)
        
        else:
            # Regular conversation without tools - using LLM directly
            response = await self.llm_client.complete(
                prompt=message,
                **kwargs
            )
            return response.content
    
    async def _analyze_tool_need(self, message: str) -> Tuple[ToolUseDecision, Optional[ToolPlan]]:
        """
        Analyze if tools are needed for this message
        """
        message_lower = message.lower()
        
        # Check for explicit tool indicators
        tool_indicators = []
        for tool_type, pattern in self.tool_patterns.items():
            if re.search(pattern, message_lower):
                tool_indicators.append(tool_type)
        
        if not tool_indicators:
            return ToolUseDecision.NO_TOOLS, None
        
        # Use LLM to understand intent and create tool plan
        prompt = f"""Analyze this user request and determine what tools to use:

User Request: {message}

Available Tools:
{json.dumps(self.tool_executor.list_tools(), indent=2)}

Detected possible tool types: {tool_indicators}

Create a detailed plan of which tools to use and their parameters.
If the request is unclear, note what clarification is needed.

Respond in JSON format:
{{
    "needs_tools": true/false,
    "needs_clarification": true/false,
    "clarification_needed": "what to clarify if needed",
    "tool_plan": [
        {{
            "tool": "tool_name",
            "params": {{"param": "value"}},
            "purpose": "what this accomplishes"
        }}
    ],
    "explanation": "brief explanation of the plan"
}}"""
        
        response = await self.llm_client.complete(
            prompt=prompt,
            model="grok-4",
            temperature=0.2,
            max_tokens=1000
        )
        
        try:
            plan_data = json.loads(response.content)
            
            if plan_data.get("needs_clarification"):
                return ToolUseDecision.ASK_CLARIFICATION, ToolPlan(
                    tools_to_use=[],
                    explanation=plan_data.get("clarification_needed", "")
                )
            
            if plan_data.get("needs_tools") and plan_data.get("tool_plan"):
                return ToolUseDecision.USE_TOOLS, ToolPlan(
                    tools_to_use=plan_data["tool_plan"],
                    explanation=plan_data.get("explanation", ""),
                    requires_confirmation=any(
                        self.tool_executor.get_tool(t["tool"]).requires_confirmation
                        for t in plan_data["tool_plan"]
                        if self.tool_executor.get_tool(t["tool"])
                    )
                )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse tool plan: {e}")
        
        return ToolUseDecision.NO_TOOLS, None
    
    async def _execute_tool_plan(self, plan: ToolPlan) -> List[Any]:
        """
        Execute the tool plan
        """
        results = []
        
        for step in plan.tools_to_use:
            tool_name = step["tool"]
            params = step.get("params", {})
            
            logger.info(f"Executing tool: {tool_name} with params: {params}")
            
            # Execute tool
            result = await self.tool_executor.execute_tool(tool_name, **params)
            results.append({
                "tool": tool_name,
                "params": params,
                "result": result,
                "purpose": step.get("purpose", "")
            })
            
            # Check if we should stop on error
            if result.status.value == "error":
                logger.error(f"Tool {tool_name} failed: {result.message}")
                if step.get("stop_on_error", True):
                    break
        
        return results
    
    async def _generate_response_with_tools(self, 
                                           original_message: str,
                                           tool_results: List[Any],
                                           plan: ToolPlan) -> str:
        """
        Generate response incorporating tool results
        """
        # Format tool results for presentation
        results_summary = []
        for result in tool_results:
            tool_result = result["result"]
            if tool_result.status.value == "success":
                results_summary.append(
                    f"✓ {result['purpose'] or result['tool']}: {tool_result.message}"
                )
                if tool_result.data and len(str(tool_result.data)) < 500:
                    results_summary.append(f"  Result: {tool_result.data}")
            else:
                results_summary.append(
                    f"✗ {result['tool']} failed: {tool_result.message}"
                )
        
        # Create response prompt
        prompt = f"""Generate a helpful response based on these tool execution results:

Original Request: {original_message}

Tool Execution Plan: {plan.explanation}

Tool Results:
{chr(10).join(results_summary)}

Provide a clear, helpful response that:
1. Confirms what was accomplished
2. Shows relevant results or output
3. Suggests next steps if applicable
4. Is concise and professional"""
        
        response = await self.llm_client.complete(
            prompt=prompt,
            model="grok-4",
            temperature=0.5,
            max_tokens=1000
        )
        
        # Format the final response
        formatted_response = f"{response.content}\n\n"
        
        # Add detailed results if needed
        for result in tool_results:
            if result["result"].status.value == "success":
                tool_name = result["tool"]
                if tool_name in ["read_file", "generate_code", "create_dag"]:
                    # Show full content for these tools
                    formatted_response += f"\n```\n{result['result'].data}\n```\n"
        
        return formatted_response
    
    async def _ask_for_clarification(self, message: str, plan: ToolPlan) -> str:
        """
        Ask user for clarification
        """
        return f"""I understand you want me to help with something, but I need more details:

{plan.explanation}

For example, you could say:
- "Read the file src/main.py"
- "Create a Python script that processes CSV files"
- "Edit config.json to change the port to 8080"
- "Generate a DAG for daily data processing with 3 tasks"

What specifically would you like me to do?"""
    
    def _should_save_to_memory(self, 
                               message: str, 
                               response: str,
                               tool_results: List[Any]) -> bool:
        """
        Determine if this interaction should be saved to memory
        """
        # Save if:
        # 1. Code was generated
        # 2. Important files were created/modified
        # 3. Project structure was created
        
        important_tools = [
            "generate_code", "create_dag", "create_project",
            "write_file", "edit_file"
        ]
        
        for result in tool_results:
            if result["tool"] in important_tools and result["result"].status.value == "success":
                return True
        
        return False
    
    async def _save_tool_interaction_to_memory(self,
                                              message: str,
                                              response: str,
                                              plan: ToolPlan,
                                              results: List[Any]):
        """
        Save tool interaction to memory for future reference
        """
        # Create memory entry
        memory_content = f"""Tool Interaction:
Request: {message}
Plan: {plan.explanation}
Tools Used: {', '.join(r['tool'] for r in results)}
Outcome: {response[:500]}"""
        
        # Extract keywords
        keywords = []
        for result in results:
            if "file_path" in result.get("params", {}):
                keywords.append(result["params"]["file_path"])
            keywords.append(result["tool"])
        
        # Save to memory
        if self.memory_system:
            await self.memory_system.add_memory(
                content=memory_content,
                memory_type="tool_interaction",
                keywords=keywords,
                metadata={
                    "tools_used": [r["tool"] for r in results],
                    "success": all(r["result"].status.value == "success" for r in results)
                }
            )
            logger.info("Tool interaction saved to memory")