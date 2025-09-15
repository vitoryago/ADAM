"""
LLM Service for ADAM v2.0
Integrates with ADAM's existing LLM client while adding streaming support
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import logging
import asyncio
import time

logger = logging.getLogger(__name__)

# Try to import DBT knowledge service
try:
    from services.dbt_knowledge_service import DBTKnowledgeService
    DBT_KNOWLEDGE_AVAILABLE = True
    logger.info("DBT Knowledge Service available")
except ImportError:
    DBT_KNOWLEDGE_AVAILABLE = False
    DBTKnowledgeService = None
    logger.debug("DBT Knowledge Service not available")

# Try to import SQL knowledge service
try:
    from services.sql_knowledge_service import SQLKnowledgeService
    SQL_KNOWLEDGE_AVAILABLE = True
    logger.info("SQL Knowledge Service available")
except ImportError:
    SQL_KNOWLEDGE_AVAILABLE = False
    SQLKnowledgeService = None
    logger.debug("SQL Knowledge Service not available")

# Try to import Fast Routing service
try:
    from services.fast_routing_service import FastRoutingService
    FAST_ROUTING_AVAILABLE = True
    logger.info("Fast Routing Service available")
except ImportError:
    FAST_ROUTING_AVAILABLE = False
    FastRoutingService = None
    logger.debug("Fast Routing Service not available")

# Try to import Response Style service
try:
    from services.response_style_service import ResponseStyleService, ResponseStyle
    RESPONSE_STYLE_AVAILABLE = True
    logger.info("Response Style Service available")
except ImportError:
    RESPONSE_STYLE_AVAILABLE = False
    ResponseStyleService = None
    ResponseStyle = None
    logger.debug("Response Style Service not available")

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
    from adam.memory.core import MemoryType
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
        env_default = os.getenv("DEFAULT_MODEL", "gpt-4.1-mini-2025-04-14")  # Latest OpenAI mini for instant responses
        self.default_model = self.project_settings.get("model", env_default)
        self.temperature = self.project_settings.get("temperature", 0.7)
        self.max_tokens = self.project_settings.get("max_tokens", 8000)  # Increased for much longer responses with larger context windows
        
        # Initialize response style service with normal as default
        self.style_service = ResponseStyleService() if RESPONSE_STYLE_AVAILABLE else None
        self.response_style = self.project_settings.get("response_style", "normal")
        
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
        
        # Initialize DBT knowledge service if available
        self.dbt_knowledge = None
        if DBT_KNOWLEDGE_AVAILABLE:
            try:
                self.dbt_knowledge = DBTKnowledgeService()
                logger.info("DBT Knowledge Service initialized")
            except Exception as e:
                self.dbt_knowledge = None
                logger.warning(f"Failed to initialize DBT Knowledge Service: {e}")
        
        # Initialize SQL knowledge service if available
        self.sql_knowledge = None
        if SQL_KNOWLEDGE_AVAILABLE:
            try:
                self.sql_knowledge = SQLKnowledgeService()
                logger.info("SQL Knowledge Service initialized")
            except Exception as e:
                self.sql_knowledge = None
                logger.warning(f"Failed to initialize SQL Knowledge Service: {e}")
        
        # Initialize fast routing service if available
        self.fast_router = None
        if FAST_ROUTING_AVAILABLE:
            try:
                self.fast_router = FastRoutingService()
                logger.info("Fast Routing Service initialized with Claude Haiku")
            except Exception as e:
                self.fast_router = None
                logger.warning(f"Failed to initialize Fast Routing Service: {e}")
        
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
        system_prompt: Optional[str] = None,
        response_style: Optional[str] = None
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
            for msg in history[-30:]:  # Last 30 messages - increased for better context
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add memory context if available
        full_prompt = message
        if memory_context:
            full_prompt = f"{memory_context}\n\nUser: {message}"
        
        # Add DBT context if detected (using intelligent detection)
        if self.dbt_knowledge:
            try:
                # Try intelligent detection first (async)
                if asyncio.iscoroutinefunction(self.dbt_knowledge.intelligent_dbt_detection):
                    needs_dbt, confidence = await self.dbt_knowledge.intelligent_dbt_detection(message)
                else:
                    # Fallback to keyword detection if not async
                    needs_dbt = self.dbt_knowledge.detect_dbt_context(message)
                    confidence = 0.5
                
                if needs_dbt and confidence > 0.3:  # Only enhance if confident
                    dbt_enhanced_prompt = self.dbt_knowledge.enhance_query_with_dbt_context(full_prompt)
                    full_prompt = dbt_enhanced_prompt
                    logger.info(f"Enhanced prompt with DBT knowledge (confidence: {confidence:.2f})")
            except Exception as e:
                # Silent fallback to keyword detection
                if self.dbt_knowledge.detect_dbt_context(message):
                    try:
                        dbt_enhanced_prompt = self.dbt_knowledge.enhance_query_with_dbt_context(full_prompt)
                        full_prompt = dbt_enhanced_prompt
                        logger.info("Enhanced prompt with DBT knowledge (keyword fallback)")
                    except:
                        pass
        
        # Add SQL context if detected (using intelligent detection)
        if self.sql_knowledge:
            try:
                # Try intelligent detection first
                if asyncio.iscoroutinefunction(self.sql_knowledge.intelligent_sql_detection):
                    needs_sql, confidence = await self.sql_knowledge.intelligent_sql_detection(message)
                else:
                    # Fallback to keyword detection
                    needs_sql = self.sql_knowledge.detect_sql_context(message)
                    confidence = 0.5
                
                if needs_sql and confidence > 0.3:  # Only enhance if confident
                    sql_enhanced_prompt = self.sql_knowledge.enhance_query_with_sql_context(full_prompt)
                    full_prompt = sql_enhanced_prompt
                    logger.info(f"Enhanced prompt with SQL knowledge (confidence: {confidence:.2f})")
                    
                    # If SQL detected, also format any SQL in the query to uppercase
                    if 'select' in full_prompt.lower() or 'from' in full_prompt.lower():
                        # Extract and format SQL blocks
                        sql_pattern = r'```sql(.*?)```'
                        matches = re.findall(sql_pattern, full_prompt, re.DOTALL | re.IGNORECASE)
                        for sql_block in matches:
                            formatted_sql = self.sql_knowledge.format_sql_uppercase(sql_block)
                            full_prompt = full_prompt.replace(sql_block, formatted_sql)
            except Exception as e:
                # Silent fallback
                if self.sql_knowledge.detect_sql_context(message):
                    try:
                        sql_enhanced_prompt = self.sql_knowledge.enhance_query_with_sql_context(full_prompt)
                        full_prompt = sql_enhanced_prompt
                        logger.info("Enhanced prompt with SQL knowledge (keyword fallback)")
                    except:
                        pass
        
        # Try to use intelligent router if available
        routing_config = None
        try:
            from .intelligent_routing_service import IntelligentRoutingService
            router = IntelligentRoutingService()
            routing_config = await router.route_and_configure(
                query=message,
                project_id=self.project_id,
                conversation_id=None  # Could pass this if available
            )
            logger.info(f"Intelligent routing: {routing_config['model']} (confidence: {routing_config['confidence']:.2f})")
            
            # Override model and get specialized prompt
            if not model:  # Only use router if no model explicitly specified
                model = routing_config['model']
                
            # Update memory depth if available
            if routing_config['memory_config']['enabled'] and memory_context:
                # Memory already retrieved, but we can log the recommendation
                logger.info(f"Router suggests memory depth: {routing_config['memory_config']['depth']}")
                
        except (ImportError, Exception) as e:
            logger.debug(f"Intelligent router not available, falling back to analyzer: {e}")
            # Fallback to original complexity analyzer
            if not model:
                complexity, _ = self.query_analyzer.analyze_query(message)
                model = self._select_model_by_complexity(complexity)
        
        # Use the specified model or default
        final_model = model or self.default_model or "gpt-4o-mini"
        
        try:
            # Check if model supports vision
            model_config = MODEL_CONFIGS.get(final_model)
            
            # Apply response style if available
            style_temperature = self.temperature
            style_max_tokens = self.max_tokens
            
            if self.style_service and RESPONSE_STYLE_AVAILABLE:
                # Use provided style or default
                style_to_use = response_style or self.response_style
                try:
                    # Convert string to ResponseStyle enum
                    style_enum = ResponseStyle(style_to_use)
                    self.style_service.set_style(style_enum)
                    
                    # Get style configuration
                    style_prompt, style_temp = self.style_service.get_style_prompt(style_enum)
                    style_params = self.style_service.adjust_model_parameters(style_enum, self.max_tokens)
                    
                    # Apply style settings
                    style_temperature = style_params['temperature']
                    style_max_tokens = style_params['max_tokens']
                    
                    # Enhance the user prompt for style
                    full_prompt = self.style_service.enhance_prompt_for_style(full_prompt, style_enum)
                    
                    # Use style-specific system prompt if no custom prompt provided
                    if not system_prompt:
                        final_system_prompt = style_prompt
                    else:
                        final_system_prompt = system_prompt
                        
                    logger.info(f"Applied response style: {style_to_use} (temp: {style_temperature}, tokens: {style_max_tokens})")
                except Exception as e:
                    logger.warning(f"Failed to apply response style: {e}")
                    # Fall back to defaults
                    final_system_prompt = system_prompt or "You are ADAM (Advanced Data Analytics Model), an AI assistant specializing in software development, data analysis, and problem-solving."
            else:
                # Use provided system prompt or build default
                if system_prompt:
                    final_system_prompt = system_prompt
                else:
                    final_system_prompt = "You are ADAM (Advanced Data Analytics Model), an AI assistant specializing in software development, data analysis, and problem-solving."
                
            # Add specialized prompt from router if available
            if routing_config and routing_config.get('system_prompt_addon'):
                final_system_prompt += "\n\n" + routing_config['system_prompt_addon']
                
            if messages and len(messages) > 1:
                # Create a conversation context from history
                history_lines = []
                for msg in messages[:-1]:  # Exclude the current message
                    role = msg['role'].capitalize()
                    history_lines.append(f"{role}: {msg['content']}")
                if history_lines:
                    final_system_prompt += "\n\nPrevious conversation:\n" + "\n".join(history_lines)
            
            # Build kwargs for complete call with style parameters
            complete_kwargs = {
                "prompt": full_prompt,
                "model": final_model,
                "system_prompt": final_system_prompt,
                "temperature": style_temperature,
                "max_tokens": style_max_tokens
            }
            
            # Add search parameters if enabled
            if use_search and final_model in ["grok-3-mini", "grok-3-mini-high", "grok-4", "grok-4-reasoning"]:
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
                "has_image": bool(image_data)
            }
            
            # Only add complexity if it exists and is not None
            if 'complexity' in locals() and complexity is not None:
                metadata["complexity"] = complexity.value
            
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
                        # Build metadata without None values
                        clean_metadata = {
                            "model": response.model,
                            "cost": response.cost,
                            "tokens": response.total_tokens,
                            "has_image": bool(image_data)
                        }
                        # Only add complexity if it exists
                        if 'complexity' in locals() and complexity is not None:
                            clean_metadata["complexity"] = complexity.value
                        
                        memory_id = await self.memory_service.store_memory_with_evaluation(
                            query=message,
                            response=response.content,
                            memory_type=memory_type,
                            metadata=clean_metadata,
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
        system_prompt: Optional[str] = None,
        response_style: Optional[str] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a response using ADAM's LLM client"""
        import time
        start_time = time.time()
        logger.info(f"Starting stream_response for message: '{message[:50]}...'")
        
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
        
        # Build conversation history with smart truncation
        messages = []
        if history:
            # Smart context management for long conversations
            total_history = len(history)
            
            # Keep last 10 messages in full detail (more efficient)
            recent_messages = history[-10:]
            
            # If there are older messages, create a summary (11-30)
            if total_history > 10:
                older_messages = history[-30:-10] if total_history > 30 else history[:-10]
                if older_messages:
                    # Add a condensed summary of older context
                    summary = "Previous conversation context:\n"
                    for i, msg in enumerate(older_messages):
                        # Truncate long messages in summary to save tokens
                        content_preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                        summary += f"{msg.role.upper()}: {content_preview}\n"
                    
                    messages.append({
                        "role": "system",
                        "content": summary
                    })
            
            # Add recent 10 messages in full for immediate context
            for msg in recent_messages:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add memory context
        full_prompt = message
        if memory_context:
            full_prompt = f"{memory_context}\n\nUser: {message}"
        
        # Use fast routing if available and no model specified
        routing_decision = None
        if not model and self.fast_router:
            try:
                routing_start = time.time()
                routing_decision = await self.fast_router.route_query(message)
                routing_time = time.time() - routing_start
                
                model = routing_decision["model"]
                logger.info(f"Fast routing took {routing_time:.2f}s, selected: {model}")
                
                # Add DBT context if routing says it's needed
                if routing_decision.get("needs_dbt") and self.dbt_knowledge:
                    try:
                        dbt_enhanced_prompt = self.dbt_knowledge.enhance_query_with_dbt_context(full_prompt)
                        full_prompt = dbt_enhanced_prompt
                        logger.info("Enhanced with DBT knowledge (Haiku routing decision)")
                    except:
                        pass
                
                # Add SQL context if routing says it's needed
                if routing_decision.get("needs_sql") and self.sql_knowledge:
                    try:
                        sql_enhanced_prompt = self.sql_knowledge.enhance_query_with_sql_context(full_prompt)
                        full_prompt = sql_enhanced_prompt
                        logger.info("Enhanced with SQL knowledge (Haiku routing decision)")
                    except:
                        pass
                        
            except Exception as e:
                logger.warning(f"Fast routing failed: {e}, falling back to analyzer")
                # Fallback to old method
                complexity, _ = self.query_analyzer.analyze_query(message)
                model = self._select_model_by_complexity(complexity)
        elif not model:
            # Fallback if fast router not available
            complexity, _ = self.query_analyzer.analyze_query(message)
            model = self._select_model_by_complexity(complexity)
        
        final_model = model or self.default_model or "gpt-4o-mini"
        
        try:
            # Check if model supports vision
            model_config = MODEL_CONFIGS.get(final_model)
            
            # For now, we'll simulate streaming by breaking up the response
            # In a real implementation, you'd use the actual streaming API
            # Use provided system prompt or build default
            if system_prompt:
                final_system_prompt = system_prompt
            else:
                final_system_prompt = "You are ADAM (Advanced Data Analytics Model), an AI assistant specializing in software development, data analysis, and problem-solving."
                
            if messages and len(messages) > 1:
                # Create a conversation context from history
                history_lines = []
                for msg in messages[:-1]:  # Exclude the current message
                    role = msg['role'].capitalize()
                    history_lines.append(f"{role}: {msg['content']}")
                if history_lines:
                    final_system_prompt += "\n\nPrevious conversation:\n" + "\n".join(history_lines)
            
            # Apply response style if available
            style_temperature = self.temperature
            style_max_tokens = self.max_tokens
            
            if self.style_service and RESPONSE_STYLE_AVAILABLE:
                # Use provided style or default
                style_to_use = response_style or self.response_style
                try:
                    # Convert string to ResponseStyle enum
                    style_enum = ResponseStyle(style_to_use)
                    self.style_service.set_style(style_enum)
                    
                    # Get style configuration
                    style_prompt, style_temp = self.style_service.get_style_prompt(style_enum)
                    style_params = self.style_service.adjust_model_parameters(style_enum, self.max_tokens)
                    
                    # Apply style settings
                    style_temperature = style_params['temperature']
                    style_max_tokens = style_params['max_tokens']
                    
                    # Enhance the user prompt for style
                    full_prompt = self.style_service.enhance_prompt_for_style(full_prompt, style_enum)
                    
                    # Use style-specific system prompt if no custom prompt provided
                    if not final_system_prompt:
                        final_system_prompt = style_prompt
                        
                    logger.info(f"Applied response style (streaming): {style_to_use} (temp: {style_temperature}, tokens: {style_max_tokens})")
                except Exception as e:
                    logger.warning(f"Failed to apply response style in streaming: {e}")
            
            # Build kwargs for complete call with style parameters
            complete_kwargs = {
                "prompt": full_prompt,
                "model": final_model,
                "system_prompt": final_system_prompt,
                "temperature": style_temperature,
                "max_tokens": style_max_tokens,
                "stream": False  # Disabled streaming for simpler processing
            }
            
            # Add search parameters if enabled
            if use_search and final_model in ["grok-3-mini", "grok-3-mini-high", "grok-4", "grok-4-reasoning"]:
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
            
            logger.info(f"Getting response from model {final_model} (streaming disabled)")
            
            # Since streaming is disabled, we get the full response at once
            content = stream_response.content if hasattr(stream_response, 'content') else str(stream_response)
            accumulated_content = content
            token_count = len(accumulated_content.split()) * 1.3
            
            # Return the entire response in one chunk (no streaming)
            yield StreamChunk(
                content=content,
                model_used=final_model,
                tokens_used=int(token_count),
                cost=0.0  # Will be calculated below
            )
            
            logger.info(f"Response complete: {len(accumulated_content)} total chars")
            
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
            return "gpt-4o-mini"
            
        if complexity == QueryComplexity.HIGH:
            return "grok-4-reasoning"
        elif complexity == QueryComplexity.MEDIUM:
            return "grok-4"
        else:
            return "gpt-4o-mini"
    
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
        
        model_name = model or self.default_model or "gpt-4o-mini"
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