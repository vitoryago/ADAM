"""
AI-Powered Routing System for ADAM
Uses LLM to intelligently route queries to appropriate models
"""
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Types of tasks the system can handle"""
    GREETING = "greeting"
    SIMPLE_QA = "simple_qa"
    COMPLEX_QA = "complex_qa"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    MATHEMATICAL = "mathematical"
    CREATIVE_WRITING = "creative_writing"
    DATA_ANALYSIS = "data_analysis"
    REASONING = "reasoning"
    CONVERSATION = "conversation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    VISION_ANALYSIS = "vision_analysis"
    SYSTEM_ARCHITECTURE = "system_architecture"
    DEBUGGING = "debugging"

@dataclass
class RoutingDecision:
    """Represents a routing decision made by the AI"""
    primary_model: str
    fallback_models: List[str]
    task_type: TaskType
    complexity: str  # low, medium, high, very_high
    confidence: float
    reasoning: str
    estimated_tokens: int
    estimated_cost: float
    special_requirements: List[str]  # e.g., ["vision", "reasoning", "streaming"]
    metadata: Dict[str, Any]

class AIRouter:
    """AI-powered routing engine using LLM for intelligent model selection"""
    
    def __init__(self, routing_model: str = "gpt-5-mini", enable_caching: bool = True):
        """
        Initialize the AI Router
        
        Args:
            routing_model: The model to use for routing decisions (gpt-5-nano, gpt-5-mini, or gpt-5)
            enable_caching: Whether to cache routing decisions for similar queries
        """
        self.routing_model = routing_model
        self.enable_caching = enable_caching
        self.routing_cache = {} if enable_caching else None
        
        # Model capabilities database
        self.model_capabilities = {
            # Claude models
            "claude-opus-4.1": {
                "strengths": ["deep reasoning", "complex analysis", "architecture", "research"],
                "supports": ["vision", "extended_thinking", "long_context"],
                "cost_tier": "premium",
                "speed": "slow",
                "context_window": 200000,
                "best_for": ["system design", "complex debugging", "research papers", "deep analysis"]
            },
            "claude-sonnet-4": {
                "strengths": ["balanced performance", "code generation", "analysis"],
                "supports": ["vision", "extended_thinking"],
                "cost_tier": "medium",
                "speed": "medium",
                "context_window": 200000,
                "best_for": ["general coding", "documentation", "moderate complexity tasks"]
            },
            "claude-3.5-sonnet": {
                "strengths": ["code generation", "fast responses", "good reasoning"],
                "supports": ["vision"],
                "cost_tier": "medium",
                "speed": "fast",
                "context_window": 200000,
                "best_for": ["coding tasks", "API development", "quick analysis"]
            },
            "claude-3.5-haiku": {
                "strengths": ["speed", "simple tasks", "classification"],
                "supports": [],
                "cost_tier": "cheap",
                "speed": "very_fast",
                "context_window": 200000,
                "best_for": ["simple Q&A", "classification", "routing"]
            },
            
            # GPT models
            "gpt-5": {
                "strengths": ["general intelligence", "reasoning", "vision", "versatility"],
                "supports": ["vision", "reasoning_effort"],
                "cost_tier": "medium",
                "speed": "medium",
                "context_window": 128000,
                "best_for": ["general tasks", "creative writing", "complex Q&A"]
            },
            "gpt-5-mini": {
                "strengths": ["speed", "cost-effective", "good general performance"],
                "supports": ["reasoning_effort"],
                "cost_tier": "cheap",
                "speed": "fast",
                "context_window": 128000,
                "best_for": ["simple tasks", "quick responses", "basic Q&A"]
            },
            "gpt-5-nano": {
                "strengths": ["ultra-fast", "very cheap", "basic tasks"],
                "supports": ["reasoning_effort"],
                "cost_tier": "very_cheap",
                "speed": "very_fast",
                "context_window": 128000,
                "best_for": ["greetings", "simple classification", "basic math"]
            },
            
            # Grok models
            "grok-4-reasoning": {
                "strengths": ["reasoning", "analysis", "vision"],
                "supports": ["vision", "live_search"],
                "cost_tier": "medium",
                "speed": "medium",
                "context_window": 131072,
                "best_for": ["reasoning tasks", "current events", "web search"]
            },
            "grok-4": {
                "strengths": ["general tasks", "vision", "search"],
                "supports": ["vision", "live_search"],
                "cost_tier": "medium",
                "speed": "medium",
                "context_window": 131072,
                "best_for": ["general Q&A with search", "current events"]
            },
            "grok-3-mini-high": {
                "strengths": ["speed", "basic reasoning"],
                "supports": ["reasoning_effort"],
                "cost_tier": "cheap",
                "speed": "fast",
                "context_window": 131072,
                "best_for": ["simple tasks", "quick answers"]
            }
        }
        
        self.routing_prompt_template = """Select the best AI model for this query.

Query: {query}

Models:
- gpt-5-mini: Simple tasks, greetings, basic Q&A
- gpt-5: Medium complexity, general analysis, standard coding
- grok-4-reasoning: Complex tasks, deep reasoning, architecture

Guidelines:
- Use gpt-5-mini for simple/quick queries
- Use gpt-5 for medium complexity tasks
- Use grok-4-reasoning for very complex tasks requiring deep reasoning

Return JSON:
{{
    "primary_model": "model_name",
    "complexity": "low|medium|high",
    "confidence": 0.0-1.0,
    "task_type": "greeting|simple_qa|code_generation|complex_analysis|reasoning",
    "reasoning": "Why this model"
}}"""

    def _get_model_descriptions(self) -> str:
        """Generate model descriptions for the routing prompt"""
        descriptions = []
        for model, caps in self.model_capabilities.items():
            desc = f"- {model}: {', '.join(caps['strengths'][:3])}. "
            desc += f"Cost: {caps['cost_tier']}, Speed: {caps['speed']}. "
            desc += f"Best for: {', '.join(caps['best_for'][:2])}"
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    def _get_task_types(self) -> str:
        """Get list of task types for the prompt"""
        return ", ".join([t.value for t in TaskType])
    
    async def route(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        available_models: Optional[List[str]] = None
    ) -> RoutingDecision:
        """
        Route a query to the appropriate model using AI analysis
        
        Args:
            query: The user's query
            context: Optional context (conversation history, etc.)
            user_preferences: User preferences (speed vs quality, cost limits, etc.)
            available_models: List of available models (if None, uses all)
            
        Returns:
            RoutingDecision object with model selection and metadata
        """
        # Check cache if enabled
        if self.enable_caching and query in self.routing_cache:
            logger.info(f"Using cached routing decision for query")
            return self.routing_cache[query]
        
        # Prepare the routing prompt (simplified)
        prompt = self.routing_prompt_template.format(
            query=query
        )
        
        try:
            # Import here to avoid circular dependency
            try:
                from .client import UnifiedLLMClient
            except ImportError:
                from adam.llm.client import UnifiedLLMClient
            
            # Use the routing model to analyze
            client = UnifiedLLMClient()
            response = await client.complete(
                prompt=prompt,
                model=self.routing_model,
                temperature=0.3,  # Lower temperature for more consistent routing
                max_tokens=500
            )
            
            # Check response
            if not response or not response.content:
                logger.error(f"Empty response from routing model {self.routing_model}")
                raise ValueError("Empty response from routing model")
            
            logger.debug(f"Routing model response: {response.content[:200]}")
            
            # Parse the response
            routing_data = self._parse_routing_response(response.content)
            
            # Map simplified task types to enum
            task_type_map = {
                "greeting": TaskType.GREETING,
                "simple_qa": TaskType.SIMPLE_QA,
                "code_generation": TaskType.CODE_GENERATION,
                "complex_analysis": TaskType.DATA_ANALYSIS,
                "reasoning": TaskType.REASONING
            }
            
            # Create routing decision
            estimated_tokens = 500 if routing_data.get("complexity") == "low" else 1500 if routing_data.get("complexity") == "medium" else 3000
            primary_model = routing_data.get("primary_model", "gpt-5-mini")
            
            decision = RoutingDecision(
                primary_model=primary_model,
                fallback_models=routing_data.get("fallback_models", ["gpt-5", "gpt-5-mini"]),
                task_type=task_type_map.get(routing_data.get("task_type", "simple_qa"), TaskType.SIMPLE_QA),
                complexity=routing_data.get("complexity", "medium"),
                confidence=routing_data.get("confidence", 0.7),
                reasoning=routing_data.get("reasoning", ""),
                estimated_tokens=estimated_tokens,
                estimated_cost=self._calculate_cost(primary_model, estimated_tokens),
                special_requirements=[],
                metadata={}
            )
            
            # Cache the decision if enabled
            if self.enable_caching:
                self.routing_cache[query] = decision
            
            # Log the routing decision
            logger.info(f"Routed to {decision.primary_model} (complexity: {decision.complexity}, confidence: {decision.confidence:.2f})")
            logger.debug(f"Routing reasoning: {decision.reasoning}")
            
            return decision
            
        except Exception as e:
            logger.error(f"AI routing failed: {e}, falling back to default")
            # Fallback to simple routing
            return self._fallback_routing(query)
    
    def _parse_routing_response(self, response: str) -> Dict:
        """Parse the JSON response from the routing model"""
        try:
            # Log raw response for debugging
            logger.debug(f"Raw routing response: {response[:500]}")
            
            # Handle empty responses
            if not response or response.strip() == "":
                logger.warning("Empty routing response received")
                raise ValueError("Empty response")
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                # Try to find JSON object in the response
                import re
                # More robust JSON extraction that handles nested objects
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    json_str = response
            
            parsed = json.loads(json_str.strip())
            
            # Validate required fields
            required_fields = ["primary_model", "task_type", "complexity", "confidence"]
            for field in required_fields:
                if field not in parsed:
                    logger.warning(f"Missing required field in routing response: {field}")
                    parsed[field] = self._get_default_value(field)
            
            return parsed
        except Exception as e:
            logger.error(f"Failed to parse routing response: {e}")
            logger.debug(f"Response that failed to parse: {response[:200]}")
            # Return default routing
            return {
                "primary_model": "gpt-5-mini",
                "fallback_models": ["gpt-5", "claude-3.5-sonnet"],
                "task_type": "simple_qa",
                "complexity": "medium",
                "confidence": 0.5,
                "reasoning": "Failed to parse AI routing, using defaults"
            }
    
    def _get_default_value(self, field: str):
        """Get default value for a missing field"""
        defaults = {
            "primary_model": "gpt-5-mini",
            "fallback_models": ["gpt-5", "claude-3.5-sonnet"],
            "task_type": "simple_qa",
            "complexity": "medium",
            "confidence": 0.5,
            "reasoning": "Field was missing, using default",
            "estimated_tokens": 500,
            "special_requirements": [],
            "analysis": {}
        }
        return defaults.get(field, None)
    
    def _calculate_cost(self, model: str, tokens: int) -> float:
        """Calculate estimated cost for the model and tokens"""
        # Cost per 1k tokens (rough estimates)
        cost_map = {
            "claude-opus-4.1": 0.075,  # $75/1M output
            "claude-sonnet-4": 0.015,   # $15/1M output
            "claude-3.5-sonnet": 0.015,
            "claude-3.5-haiku": 0.005,
            "gpt-5": 0.020,
            "gpt-5-mini": 0.004,
            "gpt-5-nano": 0.0015,
            "grok-4-reasoning": 0.015,
            "grok-4": 0.015,
            "grok-3-mini-high": 0.006
        }
        
        cost_per_token = cost_map.get(model, 0.01) / 1000
        return tokens * cost_per_token
    
    def _fallback_routing(self, query: str) -> RoutingDecision:
        """Simple fallback routing based on query length"""
        query_len = len(query)
        
        if query_len < 50:
            model = "gpt-5-nano"
            complexity = "low"
        elif query_len < 200:
            model = "gpt-5-mini"
            complexity = "medium"
        elif query_len < 500:
            model = "gpt-5"
            complexity = "high"
        else:
            model = "claude-3.5-sonnet"
            complexity = "very_high"
        
        return RoutingDecision(
            primary_model=model,
            fallback_models=["gpt-5-mini"],
            task_type=TaskType.SIMPLE_QA,
            complexity=complexity,
            confidence=0.3,
            reasoning="Fallback routing based on query length",
            estimated_tokens=query_len * 3,
            estimated_cost=self._calculate_cost(model, query_len * 3),
            special_requirements=[],
            metadata={}
        )
    
    async def analyze_routing_patterns(self, history: List[Tuple[str, RoutingDecision]]) -> Dict:
        """Analyze routing patterns from history to improve future routing"""
        if not history:
            return {}
        
        analysis = {
            "total_queries": len(history),
            "model_usage": {},
            "task_distribution": {},
            "complexity_distribution": {},
            "average_confidence": 0,
            "special_requirements": {}
        }
        
        total_confidence = 0
        
        for query, decision in history:
            # Track model usage
            model = decision.primary_model
            analysis["model_usage"][model] = analysis["model_usage"].get(model, 0) + 1
            
            # Track task types
            task = decision.task_type.value
            analysis["task_distribution"][task] = analysis["task_distribution"].get(task, 0) + 1
            
            # Track complexity
            analysis["complexity_distribution"][decision.complexity] = \
                analysis["complexity_distribution"].get(decision.complexity, 0) + 1
            
            # Track confidence
            total_confidence += decision.confidence
            
            # Track special requirements
            for req in decision.special_requirements:
                analysis["special_requirements"][req] = \
                    analysis["special_requirements"].get(req, 0) + 1
        
        analysis["average_confidence"] = total_confidence / len(history)
        
        return analysis


class SmartRoutingEngine:
    """
    Enhanced routing engine that combines AI routing with fallback mechanisms
    """
    
    def __init__(self, use_ai: bool = True, routing_model: str = "gpt-5-mini"):
        """
        Initialize the smart routing engine
        
        Args:
            use_ai: Whether to use AI routing or fall back to rules
            routing_model: Model to use for AI routing
        """
        self.use_ai = use_ai
        self.ai_router = AIRouter(routing_model=routing_model) if use_ai else None
        
        # Import the old query analyzer as fallback
        try:
            from .query_analyzer import QueryAnalyzer
        except ImportError:
            from adam.llm.query_analyzer import QueryAnalyzer
        self.fallback_analyzer = QueryAnalyzer()
    
    async def route(
        self,
        query: str,
        context: Optional[Dict] = None,
        available_models: Optional[List[str]] = None
    ) -> Dict:
        """
        Route a query using AI or fallback to rules
        
        Returns:
            Dict with routing information including model, reasoning, etc.
        """
        if self.use_ai and self.ai_router:
            try:
                decision = await self.ai_router.route(query, context, available_models=available_models)
                return {
                    "model": decision.primary_model,
                    "fallback_models": decision.fallback_models,
                    "reasoning": decision.reasoning,
                    "confidence": decision.confidence,
                    "complexity": decision.complexity,
                    "task_type": decision.task_type.value,
                    "estimated_cost": decision.estimated_cost,
                    "special_requirements": decision.special_requirements,
                    "method": "ai_routing"
                }
            except Exception as e:
                logger.error(f"AI routing failed: {e}, using fallback")
        
        # Fallback to rule-based routing
        complexity, analysis = self.fallback_analyzer.analyze_query(query)
        recommended_model = self.fallback_analyzer.recommend_model(
            complexity, 
            available_models or ["gpt-5", "gpt-5-mini", "claude-3.5-sonnet"]
        )
        
        return {
            "model": recommended_model,
            "fallback_models": [],
            "reasoning": f"Rule-based routing: {analysis.get('reasoning', [])}",
            "confidence": analysis.get('confidence', 0.5),
            "complexity": complexity.value,
            "task_type": "unknown",
            "estimated_cost": 0.001,
            "special_requirements": [],
            "method": "rule_based"
        }


# Example usage
if __name__ == "__main__":
    async def test_routing():
        router = AIRouter(routing_model="gpt-5-mini")
        
        test_queries = [
            "Hello, how are you?",
            "Write a Python function to implement quicksort with detailed comments",
            "Explain quantum computing in simple terms",
            "What's 2+2?",
            "Design a distributed system for handling 1M requests per second",
            "Analyze this image and tell me what you see",
            "Debug this code: def factorial(n): return n * factorial(n-1)"
        ]
        
        for query in test_queries:
            decision = await router.route(query)
            print(f"\nQuery: {query[:50]}...")
            print(f"  Model: {decision.primary_model}")
            print(f"  Complexity: {decision.complexity}")
            print(f"  Task: {decision.task_type.value}")
            print(f"  Confidence: {decision.confidence:.2f}")
            print(f"  Cost: ${decision.estimated_cost:.4f}")
            print(f"  Reasoning: {decision.reasoning}")
    
    # Run test
    import asyncio
    asyncio.run(test_routing())