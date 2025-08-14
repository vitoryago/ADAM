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
import asyncio

logger = logging.getLogger(__name__)

# Add parent directory to path to import ADAM modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from adam.llm.client import UnifiedLLMClient
    from adam.llm.query_analyzer import QueryAnalyzer, QueryComplexity
    from adam.llm.config import LLMConfig
    ADAM_LLM_AVAILABLE = True
    logger.info("Successfully imported ADAM LLM client")
    # Get model configs from LLMConfig instance
    llm_config = LLMConfig()
    MODEL_CONFIGS = llm_config.models
except ImportError as e:
    ADAM_LLM_AVAILABLE = False
    UnifiedLLMClient = None
    QueryAnalyzer = None
    QueryComplexity = None
    MODEL_CONFIGS = {}
    llm_config = None
    logger.error(f"Failed to import ADAM LLM client: {e}")

try:
    from adam.memory import MemoryType
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    class MemoryType:
        CONVERSATION = "conversation"
        CODE_PATTERN = "code_pattern"
        CONCEPT_EXPLANATION = "concept_explanation"


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
        # Use project model, then env default, then hardcoded default
        env_default = os.getenv("DEFAULT_MODEL", "gpt-5-mini")
        self.default_model = self.project_settings.get("model", env_default)
        self.temperature = self.project_settings.get("temperature", 0.7)
        self.max_tokens = self.project_settings.get("max_tokens", None)  # No limit by default
        
        # Initialize ADAM's LLM client if available
        if ADAM_LLM_AVAILABLE:
            try:
                self.llm_client = UnifiedLLMClient()
                self.query_analyzer = QueryAnalyzer()
                logger.info("LLM client initialized successfully")
            except Exception as e:
                self.llm_client = None
                self.query_analyzer = None
                logger.error(f"Failed to initialize LLM client: {e}")
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
        image_data: Optional[str] = None,
        use_search: bool = False,
        search_mode: Optional[str] = None,
        system_prompt: Optional[str] = None
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
        final_model = model or self.default_model or "gpt-5-mini"
        
        try:
            # Check if model supports vision
            model_config = MODEL_CONFIGS.get(final_model)
            
            # Use provided system prompt or build default
            if system_prompt:
                final_system_prompt = system_prompt
            else:
                final_system_prompt = "You are a helpful AI assistant for software development and data analysis. Be concise and direct. Do not introduce yourself or explain what you are unless specifically asked."
                
            if messages and len(messages) > 1:
                # Create a conversation context from history
                history_lines = []
                for msg in messages[:-1]:  # Exclude the current message
                    role = msg['role'].capitalize()
                    history_lines.append(f"{role}: {msg['content']}")
                if history_lines:
                    final_system_prompt += "\n\nPrevious conversation:\n" + "\n".join(history_lines)
            
            # Build kwargs for complete call
            complete_kwargs = {
                "prompt": full_prompt,
                "model": final_model,
                "system_prompt": final_system_prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # Add search parameters if enabled
            if use_search and final_model in ["grok-3-mini-high", "grok-4", "grok-4-reasoning"]:
                # Simple boolean flag - the API will handle the search internally
                complete_kwargs["search_parameters"] = True
                logger.info(f"Live search enabled for model: {final_model}")
            
            if image_data and model_config and model_config.supports_vision:
                # Convert base64 string to bytes if needed
                if isinstance(image_data, str):
                    import base64
                    try:
                        # Remove data URL prefix if present
                        if image_data.startswith('data:'):
                            image_data = image_data.split(',')[1]
                        # Decode base64 to bytes
                        image_bytes = base64.b64decode(image_data)
                    except Exception as e:
                        logger.error(f"Failed to decode base64 image: {e}")
                        raise ValueError("Invalid image data format")
                else:
                    image_bytes = image_data
                
                complete_kwargs["image_data"] = image_bytes
            
            # Make the API call
            response = await self.llm_client.complete(**complete_kwargs)
            
            # Calculate cost based on tokens
            # Simple cost estimation - you may want to adjust these rates
            model_config = MODEL_CONFIGS.get(final_model)
            if model_config and hasattr(model_config, 'cost_per_1k_tokens'):
                cost_per_1k = model_config.cost_per_1k_tokens
            else:
                cost_per_1k = 0.002  # Default fallback
            estimated_cost = (response.total_tokens / 1000) * cost_per_1k
            
            # Check if response includes citations from search
            metadata = {
                "complexity": complexity.value if 'complexity' in locals() else None,
                "has_image": bool(image_data)
            }
            
            # Add citations if present
            if hasattr(response, 'raw_response') and response.raw_response and 'citations' in response.raw_response:
                metadata["citations"] = response.raw_response['citations']
                logger.info(f"Response includes {len(response.raw_response['citations'])} citations")
            
            llm_response = LLMResponse(
                content=response.content,
                model_used=response.model,
                tokens_used=response.total_tokens,
                cost=response.cost if hasattr(response, 'cost') else estimated_cost,
                metadata=metadata
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
        image_data: Optional[str] = None,
        use_search: bool = False,
        search_mode: Optional[str] = None,
        system_prompt: Optional[str] = None
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
        
        final_model = model or self.default_model or "gpt-5-mini"
        
        try:
            # Check if model supports vision
            model_config = MODEL_CONFIGS.get(final_model)
            
            # For now, we'll simulate streaming by breaking up the response
            # In a real implementation, you'd use the actual streaming API
            # Use provided system prompt or build default
            if system_prompt:
                final_system_prompt = system_prompt
            else:
                final_system_prompt = "You are a helpful AI assistant for software development and data analysis. Be concise and direct. Do not introduce yourself or explain what you are unless specifically asked."
                
            if messages and len(messages) > 1:
                # Create a conversation context from history
                history_lines = []
                for msg in messages[:-1]:  # Exclude the current message
                    role = msg['role'].capitalize()
                    history_lines.append(f"{role}: {msg['content']}")
                if history_lines:
                    final_system_prompt += "\n\nPrevious conversation:\n" + "\n".join(history_lines)
            
            # Build kwargs for complete call
            complete_kwargs = {
                "prompt": full_prompt,
                "model": final_model,
                "system_prompt": final_system_prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True  # Enable streaming
            }
            
            # Add search parameters if enabled
            if use_search and final_model in ["grok-3-mini-high", "grok-4", "grok-4-reasoning"]:
                # Simple boolean flag - the API will handle the search internally
                complete_kwargs["search_parameters"] = True
                logger.info(f"Live search enabled for streaming with model: {final_model}")
            
            if image_data and model_config and model_config.supports_vision:
                # Convert base64 string to bytes if needed
                if isinstance(image_data, str):
                    import base64
                    try:
                        # Remove data URL prefix if present
                        if image_data.startswith('data:'):
                            image_data = image_data.split(',')[1]
                        # Decode base64 to bytes
                        image_bytes = base64.b64decode(image_data)
                    except Exception as e:
                        logger.error(f"Failed to decode base64 image: {e}")
                        raise ValueError("Invalid image data format")
                else:
                    image_bytes = image_data
                
                complete_kwargs["image_data"] = image_bytes
            
            # Make the API call
            stream_response = await self.llm_client.complete(**complete_kwargs)
            
            # Stream the actual chunks from the LLM
            accumulated_content = ""
            token_count = 0
            chunk_count = 0
            
            logger.info(f"Starting streaming for model {final_model}")
            
            # Check if we got an async generator or a regular response
            if hasattr(stream_response, '__aiter__'):
                # It's an async generator, use real streaming
                async for chunk in stream_response:
                    if chunk:  # Only process non-empty chunks
                        chunk_count += 1
                        accumulated_content += chunk
                        # Estimate tokens (rough approximation)
                        token_count = len(accumulated_content.split()) * 1.3
                        
                        logger.debug(f"Streaming chunk {chunk_count}: {len(chunk)} chars")
                        
                        yield StreamChunk(
                            content=chunk,
                            model_used=final_model,
                            tokens_used=0,  # Will be set in final chunk
                            cost=0.0
                        )
            else:
                # Fallback: Got a regular response, simulate streaming
                logger.warning("Streaming not supported, falling back to chunking")
                content = stream_response.content if hasattr(stream_response, 'content') else str(stream_response)
                
                # Use word-based chunking for more natural streaming
                words = content.split(' ')
                words_per_chunk = 3  # Send 3 words at a time
                
                for i in range(0, len(words), words_per_chunk):
                    chunk_count += 1
                    chunk_words = words[i:i + words_per_chunk]
                    chunk = ' '.join(chunk_words)
                    if i + words_per_chunk < len(words):
                        chunk += ' '  # Add space if not last chunk
                    
                    accumulated_content += chunk
                    token_count = len(accumulated_content.split()) * 1.3
                    
                    yield StreamChunk(
                        content=chunk,
                        model_used=final_model,
                        tokens_used=0,
                        cost=0.0
                    )
                    
                    await asyncio.sleep(0.015)  # 15ms delay for natural feel
            
            logger.info(f"Streaming complete: {chunk_count} chunks, {len(accumulated_content)} total chars")
            
            # Calculate cost based on accumulated content
            model_config = MODEL_CONFIGS.get(final_model)
            if model_config and hasattr(model_config, 'cost_per_1k_tokens'):
                cost_per_1k = model_config.cost_per_1k_tokens
            else:
                cost_per_1k = 0.002  # Default fallback
            
            # Estimate final token count
            final_token_count = int(token_count * 1.1)  # Add 10% for more accurate estimation
            estimated_cost = (final_token_count / 1000) * cost_per_1k
            
            # Final chunk with metadata
            yield StreamChunk(
                content="",
                model_used=final_model,
                tokens_used=final_token_count,
                cost=estimated_cost,
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
            return "gpt-5-mini"
            
        if complexity == QueryComplexity.HIGH:
            return "claude-opus-4.1"  # Claude Opus for most complex tasks
        elif complexity == QueryComplexity.MEDIUM:
            return "gpt-5"  # GPT-5 for medium complexity
        else:
            return "gpt-5-mini"  # GPT-5-mini for simple queries
    
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
        
        model_name = model or self.default_model or "gpt-5-mini"
        model_config = MODEL_CONFIGS.get(model_name)
        
        if not model_config:
            return 0.001  # Default estimate
        
        # Calculate based on model pricing (convert from per-1k to per-million)
        cost_per_million_input = (model_config.cost_per_1k_input_tokens or model_config.cost_per_1k_tokens) * 1000
        cost_per_million_output = (model_config.cost_per_1k_output_tokens or model_config.cost_per_1k_tokens) * 1000
        
        input_cost = (estimated_tokens / 1_000_000) * cost_per_million_input
        output_cost = (self.max_tokens / 1_000_000) * cost_per_million_output
        
        # Add image cost if applicable
        image_cost = 0.0
        if has_image and model_config.supports_vision:
            # Estimate ~1280 tokens for image
            image_tokens = 1280
            image_cost = (image_tokens / 1_000_000) * cost_per_million_input
        
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