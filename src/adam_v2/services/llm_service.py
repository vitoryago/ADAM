"""
LLM Service for ADAM v2.0
Integrates with ADAM's existing LLM client while adding streaming support
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import logging

# Add parent directory to path to import ADAM modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from adam.llm.client import UnifiedLLMClient
    from adam.llm.query_analyzer import QueryAnalyzer, QueryComplexity
    from adam.llm.config import MODEL_CONFIGS
    ADAM_LLM_AVAILABLE = True
except ImportError:
    ADAM_LLM_AVAILABLE = False
    UnifiedLLMClient = None
    QueryAnalyzer = None
    QueryComplexity = None
    MODEL_CONFIGS = {}

try:
    from adam.memory import MemoryType
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    class MemoryType:
        CONVERSATION = "conversation"
        CODE_PATTERN = "code_pattern"
        CONCEPT_EXPLANATION = "concept_explanation"

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM"""
    content: str
    model_used: str
    tokens_used: int
    cost: float
    metadata: Dict[str, Any] = None


@dataclass
class StreamChunk:
    """A chunk of streaming response"""
    content: str
    model_used: str
    tokens_used: int = 0
    cost: float = 0.0
    is_final: bool = False


class LLMService:
    """Service for LLM interactions with streaming support"""
    
    def __init__(self, project_settings: Optional[Dict[str, Any]] = None, project_id: Optional[str] = None):
        self.project_settings = project_settings or {}
        self.project_id = project_id
        self.default_model = self.project_settings.get("model", None)
        self.temperature = self.project_settings.get("temperature", 0.7)
        self.max_tokens = self.project_settings.get("max_tokens", 2000)
        
        # Initialize ADAM's LLM client if available
        if ADAM_LLM_AVAILABLE:
            self.llm_client = UnifiedLLMClient()
            self.query_analyzer = QueryAnalyzer()
        else:
            self.llm_client = None
            self.query_analyzer = None
            logger.warning("ADAM LLM client not available, using mock responses")
        
        # Initialize memory service if available
        self.memory_service = None
        if MEMORY_AVAILABLE and project_id:
            # Try to use advanced memory service first
            try:
                from .advanced_memory_service import AdvancedMemoryService
                project_name = self.project_settings.get("name", "Unnamed Project")
                self.memory_service = AdvancedMemoryService(project_id, project_name)
                logger.info("Using AdvancedMemoryService with BM25 and evaluation")
            except ImportError:
                from .memory_service import ProjectMemoryService
                project_name = self.project_settings.get("name", "Unnamed Project")
                self.memory_service = ProjectMemoryService(project_id, project_name)
                logger.info("Using basic ProjectMemoryService")
    
    async def generate_response(
        self,
        message: str,
        history: List[Any] = None,
        memory_context: str = "",
        model: Optional[str] = None,
        image_data: Optional[str] = None
    ) -> LLMResponse:
        """Generate a response using ADAM's LLM client"""
        
        if not self.llm_client:
            # Mock response for testing
            return LLMResponse(
                content="This is a mock response. ADAM LLM client is not available.",
                model_used="mock",
                tokens_used=10,
                cost=0.0
            )
        
        # Build conversation history
        messages = []
        if history:
            for msg in history[-10:]:  # Last 10 messages
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add memory context if available
        full_prompt = message
        if memory_context:
            full_prompt = f"{memory_context}\n\nUser: {message}"
        
        # Analyze query complexity if no model specified
        if not model:
            complexity, _ = self.query_analyzer.analyze_query(message)
            model = self._select_model_by_complexity(complexity)
        
        # Use the specified model or default
        final_model = model or self.default_model or "grok-3-mini-high"
        
        try:
            # Check if model supports vision
            model_config = MODEL_CONFIGS.get(final_model)
            
            if image_data and model_config and model_config.supports_vision:
                # Use vision-capable model
                response = self.llm_client.query(
                    query=full_prompt,
                    model_name=final_model,
                    messages=messages,
                    image_data=image_data,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            else:
                # Text-only query
                response = self.llm_client.query(
                    query=full_prompt,
                    model_name=final_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            
            llm_response = LLMResponse(
                content=response.content,
                model_used=response.model,
                tokens_used=response.total_tokens,
                cost=response.cost,
                metadata={
                    "complexity": complexity.value if 'complexity' in locals() else None,
                    "has_image": bool(image_data)
                }
            )
            
            # Store valuable responses in memory
            if self.memory_service:
                try:
                    memory_type = self._determine_memory_type(message, response.content)
                    
                    # Use advanced evaluation if available
                    if hasattr(self.memory_service, 'store_memory_with_evaluation'):
                        memory_id = await self.memory_service.store_memory_with_evaluation(
                            query=message,
                            response=response.content,
                            memory_type=memory_type,
                            metadata={
                                "model": response.model,
                                "cost": response.cost,
                                "tokens": response.total_tokens,
                                "has_image": bool(image_data),
                                "complexity": complexity.value if 'complexity' in locals() else None
                            },
                            conversation_id=None,  # Will be set by message router
                            cost=response.cost,
                            model=response.model
                        )
                        if memory_id:
                            logger.info(f"Stored memory {memory_id} using advanced evaluation")
                    # Fallback to simple cost-based storage
                    elif response.cost > 0.001:
                        await self.memory_service.store_memory(
                            content=f"Q: {message}\n\nA: {response.content}",
                            memory_type=memory_type,
                            metadata={
                                "model": response.model,
                                "cost": response.cost,
                                "tokens": response.total_tokens,
                                "has_image": bool(image_data)
                            },
                            conversation_id=None,  # Will be set by message router
                            cost=response.cost
                        )
                except Exception as e:
                    logger.warning(f"Failed to store memory: {e}")
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def stream_response(
        self,
        message: str,
        history: List[Any] = None,
        memory_context: str = "",
        model: Optional[str] = None,
        image_data: Optional[str] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a response using ADAM's LLM client"""
        
        if not self.llm_client:
            # Mock streaming response
            mock_response = "This is a mock streaming response. "
            for word in mock_response.split():
                yield StreamChunk(
                    content=word + " ",
                    model_used="mock",
                    tokens_used=1,
                    cost=0.0
                )
            yield StreamChunk(
                content="",
                model_used="mock",
                tokens_used=len(mock_response.split()),
                cost=0.0,
                is_final=True
            )
            return
        
        # Build conversation history
        messages = []
        if history:
            for msg in history[-10:]:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add memory context
        full_prompt = message
        if memory_context:
            full_prompt = f"{memory_context}\n\nUser: {message}"
        
        # Analyze query complexity if no model specified
        if not model:
            complexity, _ = self.query_analyzer.analyze_query(message)
            model = self._select_model_by_complexity(complexity)
        
        final_model = model or self.default_model or "grok-3-mini-high"
        
        try:
            # Check if model supports vision
            model_config = MODEL_CONFIGS.get(final_model)
            
            # For now, we'll simulate streaming by breaking up the response
            # In a real implementation, you'd use the actual streaming API
            if image_data and model_config and model_config.supports_vision:
                response = self.llm_client.query(
                    query=full_prompt,
                    model_name=final_model,
                    messages=messages,
                    image_data=image_data,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            else:
                response = self.llm_client.query(
                    query=full_prompt,
                    model_name=final_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            
            # Simulate streaming by yielding words
            words = response.content.split()
            for i, word in enumerate(words):
                yield StreamChunk(
                    content=word + " ",
                    model_used=response.model,
                    tokens_used=0,  # Will be set in final chunk
                    cost=0.0
                )
            
            # Final chunk with metadata
            yield StreamChunk(
                content="",
                model_used=response.model,
                tokens_used=response.total_tokens,
                cost=response.cost,
                is_final=True
            )
            
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield StreamChunk(
                content=f"\n\nError: {str(e)}",
                model_used=final_model,
                tokens_used=0,
                cost=0.0,
                is_final=True
            )
    
    def _select_model_by_complexity(self, complexity: 'QueryComplexity') -> str:
        """Select model based on query complexity"""
        if not QueryComplexity:
            return "grok-3-mini-high"
            
        if complexity == QueryComplexity.HIGH:
            return "grok-4-reasoning"
        elif complexity == QueryComplexity.MEDIUM:
            return "grok-4"
        else:
            return "grok-3-mini-high"
    
    def estimate_cost(
        self,
        message: str,
        model: Optional[str] = None,
        has_image: bool = False
    ) -> float:
        """Estimate the cost of a query"""
        if not self.llm_client:
            return 0.0
        
        # Rough token estimation
        estimated_tokens = len(message.split()) * 1.5  # 1.5 tokens per word average
        
        model_name = model or self.default_model or "grok-3-mini-high"
        model_config = MODEL_CONFIGS.get(model_name)
        
        if not model_config:
            return 0.001  # Default estimate
        
        # Calculate based on model pricing
        input_cost = (estimated_tokens / 1_000_000) * model_config.cost_per_million_input_tokens
        output_cost = (self.max_tokens / 1_000_000) * model_config.cost_per_million_output_tokens
        
        # Add image cost if applicable
        image_cost = 0.0
        if has_image and model_config.supports_vision:
            # Estimate ~1280 tokens for image
            image_tokens = 1280
            image_cost = (image_tokens / 1_000_000) * model_config.cost_per_million_input_tokens
        
        return input_cost + output_cost + image_cost
    
    def _determine_memory_type(self, query: str, response: str) -> MemoryType:
        """Determine the appropriate memory type based on content"""
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Check for error patterns
        if any(word in query_lower for word in ["error", "exception", "bug", "issue", "problem"]):
            return MemoryType.ERROR_SOLUTION
        
        # Check for code patterns
        if "```" in response or any(word in query_lower for word in ["code", "function", "implement", "write"]):
            return MemoryType.CODE_PATTERN
        
        # Check for explanations
        if any(word in query_lower for word in ["explain", "what is", "how does", "why"]):
            return MemoryType.CONCEPT_EXPLANATION
        
        # Check for screen/image analysis
        if "image" in query_lower or "screen" in query_lower:
            return MemoryType.SCREEN_ANALYSIS
        
        # Default to conversation
        return MemoryType.CONVERSATION