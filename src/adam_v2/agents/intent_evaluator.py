"""
Intent Evaluator using GPT-5-nano for fast, efficient routing
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import json
from adam.llm.client import UnifiedLLMClient

class IntentType(Enum):
    """Types of intents ADAM can handle"""
    FILE_OPERATION = "file_operation"
    CODE_ANALYSIS = "code_analysis"
    CONVERSATION = "conversation"
    SYSTEM_COMMAND = "system_command"
    MEMORY_QUERY = "memory_query"
    PROJECT_MANAGEMENT = "project_management"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    SEARCH = "search"
    UNKNOWN = "unknown"

class ToolType(Enum):
    """Available tools for ADAM"""
    FILE_READER = "file_reader"
    FILE_WRITER = "file_writer"
    DIRECTORY_EXPLORER = "directory_explorer"
    CODE_ANALYZER = "code_analyzer"
    MEMORY_SEARCH = "memory_search"
    MEMORY_SAVE = "memory_save"
    WEB_SEARCH = "web_search"
    TERMINAL = "terminal"
    NONE = "none"

class IntentEvaluator:
    """
    Evaluates user intent using GPT-5-nano for fast routing
    """
    
    def __init__(self):
        self.llm_client = UnifiedLLMClient()
        self.evaluation_model = "gpt-5-nano"  # Fast, cheap evaluation
        
        # Intent patterns for training the evaluator
        self.intent_examples = {
            IntentType.FILE_OPERATION: [
                "read the file", "show me", "what's in", "open", 
                "look at", "go through folder", "list files"
            ],
            IntentType.CODE_ANALYSIS: [
                "analyze this", "explain this code", "what does this do",
                "review", "understand this function"
            ],
            IntentType.DEBUGGING: [
                "debug", "error", "fix", "bug", "not working", 
                "exception", "crash"
            ],
            IntentType.MEMORY_QUERY: [
                "remember", "did we", "last time", "previously",
                "what did I ask", "history"
            ],
            IntentType.PROJECT_MANAGEMENT: [
                "create project", "setup", "initialize", "structure",
                "organize"
            ]
        }
    
    async def evaluate_intent(self, 
                             query: str, 
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate user intent using GPT-5-nano
        
        Returns:
            {
                "intent": IntentType,
                "tools_needed": List[ToolType],
                "parameters": Dict,
                "confidence": float,
                "reasoning": str,
                "next_steps": List[str]
            }
        """
        
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(query, context)
        
        try:
            # Use GPT-5-nano for fast evaluation
            response = await self.llm_client.generate(
                model=self.evaluation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Lower temperature for consistent routing
                max_tokens=500
            )
            
            # Parse the structured response
            evaluation = self._parse_evaluation(response.content)
            
            # Add to memory for learning
            await self._save_evaluation_to_memory(query, evaluation)
            
            return evaluation
            
        except Exception as e:
            # Fallback to rule-based evaluation
            return self._fallback_evaluation(query)
    
    def _build_evaluation_prompt(self, query: str, context: Optional[Dict] = None) -> str:
        """Build the evaluation prompt for GPT-5-nano"""
        
        context_str = ""
        if context:
            context_str = f"\nContext:\n- Workspace: {context.get('workspace', 'unknown')}\n"
            context_str += f"- Active file: {context.get('active_file', 'none')}\n"
            context_str += f"- Previous intent: {context.get('previous_intent', 'none')}\n"
        
        return f"""Analyze this user query and determine the intent and required tools.
        
Query: "{query}"{context_str}

Respond in JSON format:
{{
    "intent": "file_operation|code_analysis|debugging|memory_query|conversation|etc",
    "tools_needed": ["file_reader", "code_analyzer", etc],
    "parameters": {{
        "target": "specific file/folder if mentioned",
        "action": "read|write|analyze|search|etc",
        "scope": "file|directory|workspace"
    }},
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "next_steps": ["step1", "step2"]
}}

Examples:
- "go through our marketing folder" -> file_operation, directory_explorer
- "debug this error" -> debugging, code_analyzer
- "what did we discuss yesterday" -> memory_query, memory_search

Be concise and accurate."""
    
    def _parse_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse GPT-5-nano's evaluation response"""
        try:
            # Try to parse JSON response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                evaluation = json.loads(json_str)
                
                # Convert strings to enums
                evaluation['intent'] = IntentType[evaluation['intent'].upper()]
                evaluation['tools_needed'] = [
                    ToolType[tool.upper()] for tool in evaluation.get('tools_needed', [])
                ]
                
                return evaluation
        except:
            pass
        
        # Fallback parsing
        return self._fallback_evaluation(response)
    
    def _fallback_evaluation(self, query: str) -> Dict[str, Any]:
        """Fallback rule-based evaluation"""
        query_lower = query.lower()
        
        # Simple pattern matching as fallback
        intent = IntentType.CONVERSATION
        tools = []
        
        if any(word in query_lower for word in ['file', 'folder', 'directory', 'read', 'show']):
            intent = IntentType.FILE_OPERATION
            tools = [ToolType.FILE_READER, ToolType.DIRECTORY_EXPLORER]
        elif any(word in query_lower for word in ['debug', 'error', 'fix', 'bug']):
            intent = IntentType.DEBUGGING
            tools = [ToolType.CODE_ANALYZER]
        elif any(word in query_lower for word in ['remember', 'previously', 'last time']):
            intent = IntentType.MEMORY_QUERY
            tools = [ToolType.MEMORY_SEARCH]
        
        return {
            "intent": intent,
            "tools_needed": tools,
            "parameters": {},
            "confidence": 0.5,
            "reasoning": "Fallback evaluation",
            "next_steps": ["Process query with standard handler"]
        }
    
    async def _save_evaluation_to_memory(self, query: str, evaluation: Dict[str, Any]):
        """Save evaluation results to memory for learning"""
        # This will help ADAM learn patterns over time
        # Implementation depends on memory system
        pass