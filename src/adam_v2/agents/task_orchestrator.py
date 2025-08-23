"""
Task Orchestrator using LangGraph for complex task execution
"""

from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import Graph, StateGraph, END
from langgraph.checkpoint import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import json
import asyncio

from adam_v2.agents.intent_evaluator import IntentEvaluator, IntentType, ToolType
from adam_v2.services.llm_service import LLMService
from adam_v2.memory_manager import MemoryManager

class AgentState(TypedDict):
    """State passed between agents in the graph"""
    messages: List[BaseMessage]
    intent: Optional[IntentType]
    tools_needed: List[ToolType]
    parameters: Dict[str, Any]
    workspace_context: Dict[str, Any]
    memory_context: List[Dict]
    current_task: str
    next_steps: List[str]
    results: List[Dict[str, Any]]
    final_response: Optional[str]

class TaskOrchestrator:
    """
    Orchestrates complex tasks using LangGraph
    """
    
    def __init__(self, project_id: str, conversation_id: str):
        self.project_id = project_id
        self.conversation_id = conversation_id
        self.intent_evaluator = IntentEvaluator()
        self.llm_service = LLMService(project_id=project_id)
        self.memory_manager = MemoryManager(project_id=project_id)
        
        # Build the task graph
        self.graph = self._build_graph()
        
        # Memory for conversation continuity
        self.checkpointer = MemorySaver()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("evaluate_intent", self.evaluate_intent_node)
        workflow.add_node("search_memory", self.search_memory_node)
        workflow.add_node("file_operations", self.file_operations_node)
        workflow.add_node("code_analysis", self.code_analysis_node)
        workflow.add_node("generate_response", self.generate_response_node)
        workflow.add_node("save_to_memory", self.save_to_memory_node)
        
        # Define the edges (transitions)
        workflow.set_entry_point("evaluate_intent")
        
        # Conditional routing based on intent
        workflow.add_conditional_edges(
            "evaluate_intent",
            self.route_by_intent,
            {
                "memory": "search_memory",
                "files": "file_operations",
                "code": "code_analysis",
                "response": "generate_response"
            }
        )
        
        # All paths lead to response generation
        workflow.add_edge("search_memory", "generate_response")
        workflow.add_edge("file_operations", "generate_response")
        workflow.add_edge("code_analysis", "generate_response")
        
        # Save to memory and end
        workflow.add_edge("generate_response", "save_to_memory")
        workflow.add_edge("save_to_memory", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def evaluate_intent_node(self, state: AgentState) -> AgentState:
        """Evaluate user intent using GPT-5-nano"""
        
        # Get the latest user message
        user_message = state["messages"][-1].content if state["messages"] else ""
        
        # Evaluate intent
        evaluation = await self.intent_evaluator.evaluate_intent(
            user_message,
            context=state.get("workspace_context", {})
        )
        
        # Update state
        state["intent"] = evaluation["intent"]
        state["tools_needed"] = evaluation["tools_needed"]
        state["parameters"] = evaluation["parameters"]
        state["next_steps"] = evaluation["next_steps"]
        
        return state
    
    def route_by_intent(self, state: AgentState) -> str:
        """Route to appropriate agent based on intent"""
        
        intent = state.get("intent")
        
        if intent == IntentType.MEMORY_QUERY:
            return "memory"
        elif intent == IntentType.FILE_OPERATION:
            return "files"
        elif intent in [IntentType.CODE_ANALYSIS, IntentType.DEBUGGING]:
            return "code"
        else:
            return "response"
    
    async def search_memory_node(self, state: AgentState) -> AgentState:
        """Search memory for relevant context"""
        
        query = state["messages"][-1].content
        
        # Search memory
        memories = await self.memory_manager.search(
            query=query,
            limit=5
        )
        
        state["memory_context"] = memories
        state["results"].append({
            "type": "memory_search",
            "data": memories
        })
        
        return state
    
    async def file_operations_node(self, state: AgentState) -> AgentState:
        """Handle file operations"""
        
        params = state.get("parameters", {})
        target = params.get("target", "")
        action = params.get("action", "read")
        
        # This would integrate with the FileSystemTool
        # For now, we'll add a placeholder
        result = {
            "type": "file_operation",
            "action": action,
            "target": target,
            "data": f"Would {action} {target}"
        }
        
        state["results"].append(result)
        
        return state
    
    async def code_analysis_node(self, state: AgentState) -> AgentState:
        """Analyze code using appropriate model"""
        
        # Use grok-4-reasoning for code analysis
        model = "grok-4-reasoning"
        
        # Get code context from state
        code_context = state.get("workspace_context", {}).get("active_file_content", "")
        query = state["messages"][-1].content
        
        # Analyze with specialized model
        response = await self.llm_service.generate_response(
            message=f"Analyze this code:\n{code_context}\n\nQuery: {query}",
            model=model
        )
        
        state["results"].append({
            "type": "code_analysis",
            "model": model,
            "analysis": response["content"]
        })
        
        return state
    
    async def generate_response_node(self, state: AgentState) -> AgentState:
        """Generate final response using appropriate model"""
        
        # Select model based on complexity
        intent = state.get("intent")
        if intent in [IntentType.CODE_ANALYSIS, IntentType.DEBUGGING]:
            model = "grok-4-reasoning"
        elif intent == IntentType.MEMORY_QUERY:
            model = "gpt-5"
        else:
            model = "gpt-5-mini"
        
        # Build context from results
        context_parts = []
        
        if state.get("memory_context"):
            context_parts.append("Relevant memories:\n" + 
                               json.dumps(state["memory_context"], indent=2))
        
        for result in state.get("results", []):
            context_parts.append(f"{result['type']}: {result.get('data', '')}")
        
        # Generate response
        system_prompt = f"""You are ADAM, an AI assistant with memory and file access.
        Context from your analysis:
        {chr(10).join(context_parts)}
        
        Respond naturally to the user's query."""
        
        response = await self.llm_service.generate_response(
            message=state["messages"][-1].content,
            model=model,
            system_prompt=system_prompt
        )
        
        state["final_response"] = response["content"]
        
        return state
    
    async def save_to_memory_node(self, state: AgentState) -> AgentState:
        """Save important information to memory"""
        
        # Determine if this interaction should be saved
        should_save = state.get("intent") != IntentType.CONVERSATION
        
        if should_save:
            await self.memory_manager.save(
                query=state["messages"][-1].content,
                response=state.get("final_response", ""),
                metadata={
                    "intent": str(state.get("intent")),
                    "tools_used": [str(tool) for tool in state.get("tools_needed", [])],
                    "conversation_id": self.conversation_id
                }
            )
        
        return state
    
    async def process_message(self, 
                             message: str, 
                             workspace_context: Optional[Dict] = None) -> str:
        """
        Process a message through the task graph
        """
        
        # Initialize state
        initial_state = AgentState(
            messages=[HumanMessage(content=message)],
            intent=None,
            tools_needed=[],
            parameters={},
            workspace_context=workspace_context or {},
            memory_context=[],
            current_task="",
            next_steps=[],
            results=[],
            final_response=None
        )
        
        # Run the graph
        config = {"configurable": {"thread_id": self.conversation_id}}
        final_state = await self.graph.ainvoke(initial_state, config)
        
        return final_state.get("final_response", "I'm processing your request...")