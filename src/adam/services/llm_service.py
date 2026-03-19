"""
LLM Service for ADAM
Integrates with ADAM's LLM client with streaming support, knowledge enhancement,
intelligent routing, and memory storage.
"""

import os
import re
import inspect
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import logging
import asyncio
import time

from adam.llm.client import UnifiedLLMClient
from adam.llm.query_analyzer import QueryAnalyzer, QueryComplexity
from adam.llm.config import LLMConfig
from adam.knowledge.dbt_knowledge import DBTKnowledgeService
from adam.knowledge.sql_knowledge import SQLKnowledgeService
from adam.services.response_style_service import ResponseStyleService, ResponseStyle
from adam.memory.core import MemoryType

logger = logging.getLogger(__name__)

# Get model configs from LLMConfig instance
try:
    _llm_config = LLMConfig()
    MODEL_CONFIGS = _llm_config.models
except Exception:
    MODEL_CONFIGS = {}
    _llm_config = None


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
        env_default = os.getenv("DEFAULT_MODEL", "gpt-4.1-mini-2025-04-14")
        self.default_model = self.project_settings.get("model", env_default)
        self.temperature = self.project_settings.get("temperature", 0.7)
        self.max_tokens = self.project_settings.get("max_tokens", 8000)

        # Initialize response style service with normal as default
        self.style_service = ResponseStyleService()
        self.response_style = self.project_settings.get("response_style", "normal")

        # Initialize ADAM's LLM client
        try:
            self.llm_client = UnifiedLLMClient()
            self.query_analyzer = QueryAnalyzer()
            logger.info("LLM client initialized successfully")
        except Exception as e:
            self.llm_client = None
            self.query_analyzer = None
            logger.error(f"Failed to initialize LLM client: {e}")

        # Initialize DBT knowledge service
        self.dbt_knowledge = None
        try:
            self.dbt_knowledge = DBTKnowledgeService()
            logger.info("DBT Knowledge Service initialized")
        except Exception as e:
            self.dbt_knowledge = None
            logger.warning(f"Failed to initialize DBT Knowledge Service: {e}")

        # Initialize SQL knowledge service
        self.sql_knowledge = None
        try:
            self.sql_knowledge = SQLKnowledgeService()
            logger.info("SQL Knowledge Service initialized")
        except Exception as e:
            self.sql_knowledge = None
            logger.warning(f"Failed to initialize SQL Knowledge Service: {e}")

        # Initialize routing via LLMRouter
        self.fast_router = None
        try:
            from adam.llm.router import LLMRouter
            self.fast_router = LLMRouter()
            logger.info("LLM Router initialized")
        except Exception as e:
            self.fast_router = None
            logger.warning(f"Failed to initialize LLM Router: {e}")

        # Initialize memory service if available
        self.memory_service = None
        if project_id:
            try:
                from adam.memory.project import ProjectAwareMemory
                project_name = self.project_settings.get("name", "Unnamed Project")
                self.memory_service = ProjectAwareMemory(project_id, project_name)
                logger.info("Using ProjectAwareMemory")
            except Exception as e:
                logger.warning(f"Failed to initialize memory service: {e}")

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
            for msg in history[-30:]:  # Last 30 messages
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
                if asyncio.iscoroutinefunction(self.dbt_knowledge.intelligent_dbt_detection):
                    needs_dbt, confidence = await self.dbt_knowledge.intelligent_dbt_detection(message)
                else:
                    needs_dbt = self.dbt_knowledge.detect_dbt_context(message)
                    confidence = 0.5

                if needs_dbt and confidence > 0.3:
                    dbt_enhanced_prompt = self.dbt_knowledge.enhance_query_with_dbt_context(full_prompt)
                    full_prompt = dbt_enhanced_prompt
                    logger.info(f"Enhanced prompt with DBT knowledge (confidence: {confidence:.2f})")
            except Exception as e:
                if self.dbt_knowledge.detect_dbt_context(message):
                    try:
                        dbt_enhanced_prompt = self.dbt_knowledge.enhance_query_with_dbt_context(full_prompt)
                        full_prompt = dbt_enhanced_prompt
                        logger.info("Enhanced prompt with DBT knowledge (keyword fallback)")
                    except Exception:
                        pass

        # Add SQL context if detected (using intelligent detection)
        if self.sql_knowledge:
            try:
                if asyncio.iscoroutinefunction(self.sql_knowledge.intelligent_sql_detection):
                    needs_sql, confidence = await self.sql_knowledge.intelligent_sql_detection(message)
                else:
                    needs_sql = self.sql_knowledge.detect_sql_context(message)
                    confidence = 0.5

                if needs_sql and confidence > 0.3:
                    sql_enhanced_prompt = self.sql_knowledge.enhance_query_with_sql_context(full_prompt)
                    full_prompt = sql_enhanced_prompt
                    logger.info(f"Enhanced prompt with SQL knowledge (confidence: {confidence:.2f})")

                    if 'select' in full_prompt.lower() or 'from' in full_prompt.lower():
                        sql_pattern = r'```sql(.*?)```'
                        matches = re.findall(sql_pattern, full_prompt, re.DOTALL | re.IGNORECASE)
                        for sql_block in matches:
                            formatted_sql = self.sql_knowledge.format_sql_uppercase(sql_block)
                            full_prompt = full_prompt.replace(sql_block, formatted_sql)
            except Exception as e:
                if self.sql_knowledge.detect_sql_context(message):
                    try:
                        sql_enhanced_prompt = self.sql_knowledge.enhance_query_with_sql_context(full_prompt)
                        full_prompt = sql_enhanced_prompt
                        logger.info("Enhanced prompt with SQL knowledge (keyword fallback)")
                    except Exception:
                        pass

        # Try to use intelligent router if available
        routing_config = None
        if not model and self.fast_router:
            try:
                routing_config = await self.fast_router.route_and_configure(
                    query=message,
                    project_id=self.project_id,
                    conversation_id=None
                )
                logger.info(f"Intelligent routing: {routing_config['model']} (confidence: {routing_config['confidence']:.2f})")
                model = routing_config['model']

                if routing_config['memory_config']['enabled'] and memory_context:
                    logger.info(f"Router suggests memory depth: {routing_config['memory_config']['depth']}")

            except Exception as e:
                logger.debug(f"Intelligent router not available, falling back to analyzer: {e}")
                if not model and self.query_analyzer:
                    complexity, _ = self.query_analyzer.analyze_query(message)
                    model = self._select_model_by_complexity(complexity)
        elif not model and self.query_analyzer:
            complexity, _ = self.query_analyzer.analyze_query(message)
            model = self._select_model_by_complexity(complexity)

        # Use the specified model or default
        final_model = model or self.default_model or "gpt-4o-mini"

        try:
            model_config = MODEL_CONFIGS.get(final_model)

            # Apply response style if available
            style_temperature = self.temperature
            style_max_tokens = self.max_tokens

            if self.style_service:
                style_to_use = response_style or self.response_style
                try:
                    style_enum = ResponseStyle(style_to_use)
                    self.style_service.set_style(style_enum)

                    style_prompt, style_temp = self.style_service.get_style_prompt(style_enum)
                    style_params = self.style_service.adjust_model_parameters(style_enum, self.max_tokens)

                    style_temperature = style_params['temperature']
                    style_max_tokens = style_params['max_tokens']

                    full_prompt = self.style_service.enhance_prompt_for_style(full_prompt, style_enum)

                    if not system_prompt:
                        final_system_prompt = style_prompt
                    else:
                        final_system_prompt = system_prompt

                    logger.info(f"Applied response style: {style_to_use} (temp: {style_temperature}, tokens: {style_max_tokens})")
                except Exception as e:
                    logger.warning(f"Failed to apply response style: {e}")
                    final_system_prompt = system_prompt or "You are ADAM (Advanced Data Analytics Model), an AI assistant specializing in software development, data analysis, and problem-solving."
            else:
                if system_prompt:
                    final_system_prompt = system_prompt
                else:
                    final_system_prompt = "You are ADAM (Advanced Data Analytics Model), an AI assistant specializing in software development, data analysis, and problem-solving."

            # Add specialized prompt from router if available
            if routing_config and routing_config.get('system_prompt_addon'):
                final_system_prompt += "\n\n" + routing_config['system_prompt_addon']

            if messages and len(messages) > 1:
                history_lines = []
                for msg in messages[:-1]:
                    role = msg['role'].capitalize()
                    history_lines.append(f"{role}: {msg['content']}")
                if history_lines:
                    final_system_prompt += "\n\nPrevious conversation:\n" + "\n".join(history_lines)

            complete_kwargs = {
                "prompt": full_prompt,
                "model": final_model,
                "system_prompt": final_system_prompt,
                "temperature": style_temperature,
                "max_tokens": style_max_tokens
            }

            if use_search and final_model in ["grok-3-mini", "grok-3-mini-high", "grok-4", "grok-4-reasoning"]:
                complete_kwargs["search_parameters"] = True
                logger.info(f"Live search enabled for model: {final_model}")

            if image_data and model_config and model_config.supports_vision:
                if isinstance(image_data, str):
                    import base64
                    try:
                        if image_data.startswith('data:'):
                            image_data = image_data.split(',')[1]
                        image_bytes = base64.b64decode(image_data)
                    except Exception as e:
                        logger.error(f"Failed to decode base64 image: {e}")
                        raise ValueError("Invalid image data format")
                else:
                    image_bytes = image_data

                complete_kwargs["image_data"] = image_bytes

            response = await self.llm_client.complete(**complete_kwargs)

            model_config = MODEL_CONFIGS.get(final_model)
            if model_config and hasattr(model_config, 'cost_per_1k_tokens'):
                cost_per_1k = model_config.cost_per_1k_tokens
            else:
                cost_per_1k = 0.002
            estimated_cost = (response.total_tokens / 1000) * cost_per_1k

            metadata = {
                "has_image": bool(image_data)
            }

            if 'complexity' in locals() and complexity is not None:
                metadata["complexity"] = complexity.value

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

                    if hasattr(self.memory_service, 'store_memory_with_evaluation'):
                        clean_metadata = {
                            "model": response.model,
                            "cost": response.cost,
                            "tokens": response.total_tokens,
                            "has_image": bool(image_data)
                        }
                        if 'complexity' in locals() and complexity is not None:
                            clean_metadata["complexity"] = complexity.value

                        memory_id = await self.memory_service.store_memory_with_evaluation(
                            query=message,
                            response=response.content,
                            memory_type=memory_type,
                            metadata=clean_metadata,
                            conversation_id=None,
                            cost=response.cost,
                            model=response.model
                        )
                        if memory_id:
                            logger.info(f"Stored memory {memory_id} using advanced evaluation")
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
                            conversation_id=None,
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
        start_time = time.time()
        logger.info(f"Starting stream_response for message: '{message[:50]}...'")

        if not self.llm_client:
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
            total_history = len(history)
            recent_messages = history[-10:]

            if total_history > 10:
                older_messages = history[-30:-10] if total_history > 30 else history[:-10]
                if older_messages:
                    summary = "Previous conversation context:\n"
                    for i, msg in enumerate(older_messages):
                        content_preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                        summary += f"{msg.role.upper()}: {content_preview}\n"

                    messages.append({
                        "role": "system",
                        "content": summary
                    })

            for msg in recent_messages:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # Add memory context
        full_prompt = message
        if memory_context:
            full_prompt = f"{memory_context}\n\nUser: {message}"

        # Use routing if available and no model specified
        routing_decision = None
        if not model and self.fast_router:
            try:
                routing_start = time.time()
                routing_decision = await self.fast_router.route_query(message)
                routing_time = time.time() - routing_start

                model = routing_decision["model"]
                logger.info(f"Fast routing took {routing_time:.2f}s, selected: {model}")

                if routing_decision.get("needs_dbt") and self.dbt_knowledge:
                    try:
                        dbt_enhanced_prompt = self.dbt_knowledge.enhance_query_with_dbt_context(full_prompt)
                        full_prompt = dbt_enhanced_prompt
                        logger.info("Enhanced with DBT knowledge (routing decision)")
                    except Exception:
                        pass

                if routing_decision.get("needs_sql") and self.sql_knowledge:
                    try:
                        sql_enhanced_prompt = self.sql_knowledge.enhance_query_with_sql_context(full_prompt)
                        full_prompt = sql_enhanced_prompt
                        logger.info("Enhanced with SQL knowledge (routing decision)")
                    except Exception:
                        pass

            except Exception as e:
                logger.warning(f"Routing failed: {e}, falling back to analyzer")
                if self.query_analyzer:
                    complexity, _ = self.query_analyzer.analyze_query(message)
                    model = self._select_model_by_complexity(complexity)
        elif not model and self.query_analyzer:
            complexity, _ = self.query_analyzer.analyze_query(message)
            model = self._select_model_by_complexity(complexity)

        final_model = model or self.default_model or "gpt-4o-mini"

        try:
            model_config = MODEL_CONFIGS.get(final_model)

            if system_prompt:
                final_system_prompt = system_prompt
            else:
                final_system_prompt = "You are ADAM (Advanced Data Analytics Model), an AI assistant specializing in software development, data analysis, and problem-solving."

            if messages and len(messages) > 1:
                history_lines = []
                for msg in messages[:-1]:
                    role = msg['role'].capitalize()
                    history_lines.append(f"{role}: {msg['content']}")
                if history_lines:
                    final_system_prompt += "\n\nPrevious conversation:\n" + "\n".join(history_lines)

            # Apply response style if available
            style_temperature = self.temperature
            style_max_tokens = self.max_tokens

            if self.style_service:
                style_to_use = response_style or self.response_style
                try:
                    style_enum = ResponseStyle(style_to_use)
                    self.style_service.set_style(style_enum)

                    style_prompt, style_temp = self.style_service.get_style_prompt(style_enum)
                    style_params = self.style_service.adjust_model_parameters(style_enum, self.max_tokens)

                    style_temperature = style_params['temperature']
                    style_max_tokens = style_params['max_tokens']

                    full_prompt = self.style_service.enhance_prompt_for_style(full_prompt, style_enum)

                    if not final_system_prompt:
                        final_system_prompt = style_prompt

                    logger.info(f"Applied response style (streaming): {style_to_use} (temp: {style_temperature}, tokens: {style_max_tokens})")
                except Exception as e:
                    logger.warning(f"Failed to apply response style in streaming: {e}")

            complete_kwargs = {
                "prompt": full_prompt,
                "model": final_model,
                "system_prompt": final_system_prompt,
                "temperature": style_temperature,
                "max_tokens": style_max_tokens,
                "stream": True
            }

            if use_search and final_model in ["grok-3-mini", "grok-3-mini-high", "grok-4", "grok-4-reasoning"]:
                complete_kwargs["search_parameters"] = True
                logger.info(f"Live search enabled for streaming with model: {final_model}")

            if image_data and model_config and model_config.supports_vision:
                if isinstance(image_data, str):
                    import base64
                    try:
                        if image_data.startswith('data:'):
                            image_data = image_data.split(',')[1]
                        image_bytes = base64.b64decode(image_data)
                    except Exception as e:
                        logger.error(f"Failed to decode base64 image: {e}")
                        raise ValueError("Invalid image data format")
                else:
                    image_bytes = image_data

                complete_kwargs["image_data"] = image_bytes

            accumulated_content = ""
            chunk_count = 0

            try:
                stream = await self.llm_client.complete(**complete_kwargs)

                # Check if the client returned an async generator (real streaming)
                # or an LLMResponse object (streaming not supported for this model)
                if inspect.isasyncgen(stream):
                    # Real streaming: iterate over async generator of string chunks
                    logger.info(f"Streaming response from model {final_model}")
                    async for chunk_text in stream:
                        if chunk_text:
                            chunk_count += 1
                            accumulated_content += chunk_text
                            yield StreamChunk(
                                content=chunk_text,
                                model_used=final_model,
                                tokens_used=0,
                                cost=0.0
                            )
                    logger.info(f"Streaming complete: {chunk_count} chunks, {len(accumulated_content)} total chars")
                else:
                    # Fallback: client returned a complete LLMResponse (no streaming support)
                    logger.info(f"Model {final_model} returned non-streaming response, yielding as single chunk")
                    content = stream.content if hasattr(stream, 'content') else str(stream)
                    accumulated_content = content
                    yield StreamChunk(
                        content=content,
                        model_used=final_model,
                        tokens_used=getattr(stream, 'total_tokens', 0),
                        cost=getattr(stream, 'cost', 0.0)
                    )

            except Exception as stream_error:
                logger.warning(f"Streaming failed for {final_model}, falling back to non-streaming: {stream_error}")
                # Fallback: retry without streaming
                complete_kwargs["stream"] = False
                fallback_response = await self.llm_client.complete(**complete_kwargs)
                content = fallback_response.content if hasattr(fallback_response, 'content') else str(fallback_response)
                accumulated_content = content
                yield StreamChunk(
                    content=content,
                    model_used=final_model,
                    tokens_used=getattr(fallback_response, 'total_tokens', 0),
                    cost=getattr(fallback_response, 'cost', 0.0)
                )

            # Calculate final token count and cost
            model_config = MODEL_CONFIGS.get(final_model)
            if model_config and hasattr(model_config, 'cost_per_1k_tokens'):
                cost_per_1k = model_config.cost_per_1k_tokens
            else:
                cost_per_1k = 0.002

            token_count = int(len(accumulated_content.split()) * 1.3)
            estimated_cost = (token_count / 1000) * cost_per_1k

            # Final chunk with metadata
            yield StreamChunk(
                content="",
                model_used=final_model,
                tokens_used=token_count,
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

        estimated_tokens = len(message.split()) * 1.5

        model_name = model or self.default_model or "gpt-4o-mini"
        model_config = MODEL_CONFIGS.get(model_name)

        if not model_config:
            return 0.001

        cost_per_million_input = (model_config.cost_per_1k_input_tokens or model_config.cost_per_1k_tokens) * 1000
        cost_per_million_output = (model_config.cost_per_1k_output_tokens or model_config.cost_per_1k_tokens) * 1000

        input_cost = (estimated_tokens / 1_000_000) * cost_per_million_input
        output_cost = (self.max_tokens / 1_000_000) * cost_per_million_output

        image_cost = 0.0
        if has_image and model_config.supports_vision:
            image_tokens = 1280
            image_cost = (image_tokens / 1_000_000) * cost_per_million_input

        return input_cost + output_cost + image_cost

    def _determine_memory_type(self, query: str, response: str) -> MemoryType:
        """Determine the appropriate memory type based on content"""
        query_lower = query.lower()
        response_lower = response.lower()

        if any(word in query_lower for word in ["error", "exception", "bug", "issue", "problem"]):
            return MemoryType.ERROR_SOLUTION

        if "```" in response or any(word in query_lower for word in ["code", "function", "implement", "write"]):
            return MemoryType.CODE_PATTERN

        if any(word in query_lower for word in ["explain", "what is", "how does", "why"]):
            return MemoryType.CONCEPT_EXPLANATION

        if "image" in query_lower or "screen" in query_lower:
            return MemoryType.SCREEN_ANALYSIS

        return MemoryType.CONVERSATION
