"""
Unified LLM Client for ADAM
Handles different APIs: xAI (Grok) and OpenAI
"""
import os
import asyncio
import logging
from typing import Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# Import model-specific SDKs
try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user, system, image
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    print("Warning: xai_sdk not installed. Run: pip install xai-sdk")

try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. Run: pip install openai")

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic not installed. Run: pip install anthropic")

from .config import LLMConfig, ModelProvider, ModelConfig
from .query_analyzer import QueryAnalyzer, QueryComplexity

@dataclass
class LLMResponse:
    """Unified response format"""
    content: str
    model: str
    reasoning_content: Optional[str] = None
    total_tokens: int = 0
    reasoning_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    raw_response: Optional[Dict] = None

class UnifiedLLMClient:
    """
    Unified client that abstracts differences between LLM providers
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.clients = {}
        self.query_analyzer = QueryAnalyzer()
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize API clients for available providers"""
        # Initialize xAI client with timeout for streaming
        if XAI_AVAILABLE and self.config.get_api_key(ModelProvider.GROK):
            self.clients[ModelProvider.GROK] = XAIClient(
                api_host="api.x.ai",
                api_key=self.config.get_api_key(ModelProvider.GROK),
                timeout=3600  # 1 hour timeout for streaming and reasoning models
            )
        
        # Initialize OpenAI client
        if OPENAI_AVAILABLE and self.config.get_api_key(ModelProvider.OPENAI):
            self.clients[ModelProvider.OPENAI] = AsyncOpenAI(
                api_key=self.config.get_api_key(ModelProvider.OPENAI)
            )
        
        # Initialize Anthropic client for Claude models
        if ANTHROPIC_AVAILABLE and self.config.get_api_key(ModelProvider.ANTHROPIC):
            self.clients[ModelProvider.ANTHROPIC] = AsyncAnthropic(
                api_key=self.config.get_api_key(ModelProvider.ANTHROPIC)
            )

    def _get_grok_reasoning_param_value(
        self,
        model_config: ModelConfig,
        reasoning_effort: Optional[str],
    ) -> Optional[tuple[str, str]]:
        """Map unified reasoning effort only for xAI models that still support it."""
        if not reasoning_effort or not model_config.reasoning_param:
            return None

        # xAI's current SDK only accepts reasoning_effort on grok-3-mini.
        if "grok-3-mini" not in model_config.api_name:
            logger.info(
                "Ignoring reasoning_effort for Grok model %s; xAI no longer accepts it",
                model_config.api_name,
            )
            return None

        effort_map = {"low": "low", "medium": "high", "high": "high"}
        return model_config.reasoning_param, effort_map.get(reasoning_effort, "high")

    def _normalize_openai_chat_create_kwargs(
        self,
        model_name: str,
        create_kwargs: Dict,
    ) -> Dict:
        """Use the correct token-limit parameter for newer OpenAI chat models."""
        normalized = dict(create_kwargs)
        max_tokens = normalized.pop("max_tokens", None)
        if max_tokens is None:
            return normalized

        if model_name.startswith("gpt-5"):
            normalized["max_completion_tokens"] = max_tokens
        else:
            normalized["max_tokens"] = max_tokens
        return normalized
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,  # 'low', 'medium', 'high'
        stream: bool = False,
        image_data: Optional[bytes] = None,
        search_parameters: Optional[Dict] = None,  # For live search
        search_mode: Optional[str] = None  # "web" or "x"
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """
        Get completion from any configured model
        
        Args:
            prompt: User prompt
            model: Model name (if None, auto-selects)
            system_prompt: System message (optional)
            temperature: Response randomness (0-1)
            max_tokens: Maximum response length
            reasoning_effort: How hard to think ('low', 'medium', 'high')
            stream: Whether to stream response
            image_data: Optional image bytes for vision models
            search_parameters: Optional dict for live search (e.g., {"enabled": True, "include_sources": ["web"]})
            
        Returns:
            LLMResponse object or async generator if streaming
        """
        # Handle automatic model selection
        actual_model = model
        routing_decision = None
        
        if model == "automatic":
            # Check if we need a vision model first
            if image_data:
                # Select a vision-capable model
                vision_models = [m for m in self.config.get_available_models() 
                               if self.config.get_model_config(m).supports_vision]
                if vision_models:
                    # Prefer grok-2-vision for cost efficiency
                    if "grok-2-vision-1212" in vision_models:
                        actual_model = "grok-2-vision-1212"
                    elif "grok-4" in vision_models:
                        actual_model = "grok-4"
                    elif "gpt-4" in vision_models:
                        actual_model = "gpt-4"
                    else:
                        actual_model = vision_models[0]
                    
                    routing_decision = {
                        "requested_model": "automatic",
                        "selected_model": actual_model,
                        "complexity": "vision",
                        "confidence": 1.0,
                        "reasoning": ["Image input detected, selected vision-capable model"],
                        "indicators": ["image_data"]
                    }
                else:
                    raise ValueError("Image provided but no vision models available")
            else:
                # Use intelligent routing to select best model
                actual_model = self._auto_select_model(prompt, reasoning_effort)
                
                # Get routing decision details for transparency
                complexity, analysis = self.query_analyzer.analyze_query(prompt)
                routing_decision = {
                    "requested_model": "automatic",
                    "selected_model": actual_model,
                    "complexity": complexity.value,
                    "confidence": analysis.get('confidence', 0.0),
                    "reasoning": analysis.get('reasoning', [])[:3],  # Top 3 reasons
                    "indicators": analysis.get('indicators_found', [])
                }
                
                # Set reasoning effort if not specified
                if not reasoning_effort:
                    reasoning_effort = self.query_analyzer.get_reasoning_effort(complexity)
                    
        elif not model:
            # Traditional auto-select when no model specified
            actual_model = self._auto_select_model(prompt, reasoning_effort)
            
            # If no explicit reasoning effort, determine from query analysis
            if not reasoning_effort and actual_model:
                complexity, _ = self.query_analyzer.analyze_query(prompt)
                reasoning_effort = self.query_analyzer.get_reasoning_effort(complexity)
        
        # Use actual_model for the rest of the process
        model = actual_model
        
        if not model:
            raise ValueError("No available models. Please set API keys.")
        
        # Get model configuration
        model_config = self.config.get_model_config(model)
        if not model_config:
            raise ValueError(f"Unknown model: {model}")
        
        # Route to appropriate provider
        if model_config.provider == ModelProvider.GROK:
            return await self._complete_grok(
                prompt, model_config, system_prompt, temperature,
                max_tokens, reasoning_effort, stream, image_data, routing_decision, search_parameters, search_mode
            )
        elif model_config.provider == ModelProvider.OPENAI:
            return await self._complete_openai(
                prompt, model_config, system_prompt, temperature,
                max_tokens, reasoning_effort, stream, image_data, routing_decision, search_parameters
            )
        elif model_config.provider == ModelProvider.ANTHROPIC:
            return await self._complete_anthropic(
                prompt, model_config, system_prompt, temperature,
                max_tokens, stream
            )
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
    
    async def _complete_grok(
        self,
        prompt: str,
        model_config: ModelConfig,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        reasoning_effort: Optional[str],
        stream: bool,
        image_data: Optional[bytes] = None,
        routing_decision: Optional[Dict] = None,
        search_parameters: Optional[Dict] = None,
        search_mode: Optional[str] = None
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Handle Grok model completion"""

        # For search requests, use the OpenAI-compatible REST API (tools-based)
        # The old xai_sdk SearchParameters is deprecated
        if search_parameters:
            return await self._complete_grok_with_search(
                prompt, model_config, system_prompt, temperature,
                max_tokens, stream, search_mode
            )

        client = self.clients[ModelProvider.GROK]

        # Create chat session
        chat_params = {
            "model": model_config.api_name,
            "temperature": temperature
        }
        
        reasoning_param = self._get_grok_reasoning_param_value(
            model_config, reasoning_effort
        )
        if reasoning_param:
            param_name, param_value = reasoning_param
            chat_params[param_name] = param_value
        
        chat = client.chat.create(**chat_params)
        
        # Add messages
        if system_prompt:
            chat.append(system(system_prompt))
        
        # Handle image data for vision-enabled Grok models
        if image_data and model_config.supports_vision:
            # Encode image as base64 for Grok vision models
            import base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Use the proper xAI SDK format with image function
            chat.append(
                user(
                    prompt,
                    image(image_url=f"data:image/jpeg;base64,{base64_image}", detail="high")
                )
            )
            logger.info(f"Image provided for {model_config.name} - using vision format")
        elif image_data:
            logger.warning(f"Image provided but {model_config.name} doesn't support vision")
            chat.append(user(prompt))
        else:
            chat.append(user(prompt))
        
        # Get response
        if stream:
            # Return async generator for streaming
            async def stream_generator():
                try:
                    # Use threading to handle sync streaming API asynchronously
                    import threading
                    import queue
                    import asyncio
                    
                    logger.info(f"Starting streaming for {model_config.name}")
                    
                    # Queue for communication between threads
                    chunk_queue = queue.Queue()
                    error_event = threading.Event()
                    error_msg = None
                    
                    def stream_worker():
                        """Worker thread that handles sync streaming"""
                        nonlocal error_msg
                        try:
                            for response, chunk in chat.stream():
                                if chunk.content:
                                    chunk_queue.put(chunk.content)
                            chunk_queue.put(None)  # Signal completion
                        except Exception as e:
                            error_msg = str(e)
                            error_event.set()
                            chunk_queue.put(None)  # Signal to stop
                    
                    # Start streaming in background thread
                    thread = threading.Thread(target=stream_worker, daemon=True)
                    thread.start()
                    
                    # Consume chunks asynchronously
                    chunk_count = 0
                    accumulated_content = ""
                    
                    while True:
                        # Check for errors
                        if error_event.is_set():
                            logger.error(f"Grok streaming error: {error_msg}")
                            # Fallback to non-streaming
                            response = chat.sample()
                            yield response.content
                            break
                        
                        try:
                            # Non-blocking check for chunks
                            chunk = chunk_queue.get_nowait()
                            
                            if chunk is None:
                                # Streaming complete
                                break
                            else:
                                # Got a chunk
                                chunk_count += 1
                                accumulated_content += chunk
                                logger.debug(f"Streaming chunk {chunk_count}: {len(chunk)} chars")
                                yield chunk
                                
                        except queue.Empty:
                            # No chunk available yet, yield control
                            await asyncio.sleep(0.001)
                    
                    # Ensure thread completes
                    thread.join(timeout=1)
                    logger.info(f"Streaming complete: {chunk_count} chunks, {len(accumulated_content)} total chars")
                    
                except Exception as e:
                    logger.error(f"Grok streaming error: {e}", exc_info=True)
                    # Fallback to non-streaming
                    response = chat.sample()
                    yield response.content
            return stream_generator()
        else:
            # Get complete response
            response = chat.sample()
            
            # Build unified response with routing info
            raw_response_data = {
                'prompt_image_tokens': getattr(response.usage, 'prompt_image_tokens', 0) if hasattr(response.usage, 'prompt_image_tokens') else 0
            }
            
            # Add routing decision if automatic model was used
            if routing_decision:
                raw_response_data['routing_decision'] = routing_decision
            
            # Add citations if search was used
            if hasattr(response, 'citations') and response.citations:
                raw_response_data['citations'] = response.citations
                logger.info(f"Found {len(response.citations)} citations in response")
            
            return LLMResponse(
                content=response.content,
                model=model_config.name,
                reasoning_content=getattr(response, 'reasoning_content', None),
                total_tokens=getattr(response.usage, 'total_tokens', 0),
                reasoning_tokens=getattr(response.usage, 'reasoning_tokens', 0),
                completion_tokens=getattr(response.usage, 'completion_tokens', 0),
                cost=self._calculate_cost(model_config, response),
                raw_response=raw_response_data
            )
    
    async def _complete_grok_with_search(
        self,
        prompt: str,
        model_config: ModelConfig,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        stream: bool,
        search_mode: Optional[str] = None
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Handle Grok search via xAI's Responses API."""
        import os

        xai_client = AsyncOpenAI(
            base_url="https://api.x.ai/v1",
            api_key=os.getenv("XAI_API_KEY"),
            timeout=300.0,
        )

        search_tool = {"type": "x_search" if search_mode == "x" else "web_search"}
        create_kwargs = {
            "model": model_config.api_name,
            "input": prompt,
            "temperature": temperature,
            "max_output_tokens": max_tokens or 4096,
            "tools": [search_tool],
        }
        if system_prompt:
            create_kwargs["instructions"] = system_prompt

        logger.info(
            "Grok search enabled via Responses API: tool=%s for model %s",
            search_tool["type"],
            model_config.api_name,
        )

        if stream:
            async def stream_generator():
                response_stream = await xai_client.responses.create(
                    stream=True,
                    **create_kwargs,
                )
                async for event in response_stream:
                    if getattr(event, "type", None) == "response.output_text.delta" and getattr(event, "delta", None):
                        yield event.delta
            return stream_generator()
        else:
            response = await xai_client.responses.create(**create_kwargs)
            usage = getattr(response, "usage", None)
            return LLMResponse(
                content=response.output_text,
                model=model_config.name,
                total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
                completion_tokens=(
                    getattr(usage, "output_tokens", None)
                    or getattr(usage, "completion_tokens", 0)
                ) if usage else 0,
                cost=self._calculate_cost(model_config, response) if usage else 0.0,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else {}
            )

    async def _complete_openai(
        self,
        prompt: str,
        model_config: ModelConfig,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        reasoning_effort: Optional[str],
        stream: bool,
        image_data: Optional[bytes] = None,
        routing_decision: Optional[Dict] = None,
        search_parameters: Optional[Dict] = None
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Handle OpenAI model completion"""
        client = self.clients[ModelProvider.OPENAI]
        
        # Build input array for responses API
        input_messages = []
        if system_prompt:
            input_messages.append({"role": "system", "content": system_prompt})
        input_messages.append({"role": "user", "content": prompt})
        
        # For o4-mini, use the responses API
        if model_config.api_name.startswith("o"):
            # Build reasoning parameter with correct format
            reasoning_params = {}
            if model_config.reasoning_param:
                # Map our effort levels to OpenAI's format
                effort_map = {"low": "low", "medium": "medium", "high": "high"}
                reasoning_params = {"effort": effort_map.get(reasoning_effort, "medium")}
            
            try:
                # Make request to responses API with correct format
                response = await client.responses.create(
                    model=model_config.api_name,
                    input=input_messages,
                    reasoning=reasoning_params,
                    max_output_tokens=max_tokens or model_config.max_tokens
                )
                
                if stream:
                    # Handle streaming for responses API
                    async def stream_generator():
                        # Note: OpenAI responses API streaming might work differently
                        yield response.output_text
                    return stream_generator()
                else:
                    # Build unified response
                    # Build response with routing info for OpenAI responses API
                    raw_response_data = {}
                    if routing_decision:
                        raw_response_data['routing_decision'] = routing_decision
                    
                    return LLMResponse(
                        content=response.output_text,
                        model=model_config.name,
                        reasoning_content=None,  # OpenAI doesn't expose reasoning content
                        total_tokens=response.usage.total_tokens,
                        reasoning_tokens=response.usage.output_tokens_details.reasoning_tokens,
                        completion_tokens=response.usage.output_tokens,
                        cost=self._calculate_openai_cost(model_config, response.usage),
                        raw_response=raw_response_data
                    )
            except Exception as e:
                # If o4-mini fails, it might be an access issue
                if "401" in str(e) or "invalid_api_key" in str(e):
                    raise Exception(f"OpenAI API key issue: {str(e)}")
                elif "404" in str(e):
                    raise Exception(f"Model {model_config.api_name} not available. You may need organization verification for o4-mini.")
                else:
                    raise
        else:
            # Standard chat completion for GPT models
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Handle image data for vision models
            if image_data and "gpt-4" in model_config.api_name:
                # For GPT-4 vision, format message with image
                import base64
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                })
            else:
                messages.append({"role": "user", "content": prompt})
            
            create_kwargs = {
                "model": model_config.api_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }

            if search_parameters:
                create_kwargs["extra_body"] = {
                    "tools": [{"type": "web_search_preview"}],
                }
                logger.info(f"OpenAI web search enabled for model: {model_config.api_name}")

            create_kwargs = self._normalize_openai_chat_create_kwargs(
                model_config.api_name,
                create_kwargs,
            )

            response = await client.chat.completions.create(**create_kwargs)
            
            if stream:
                # Return async generator
                async def stream_generator():
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return stream_generator()
            else:
                # Build response with routing info for OpenAI
                raw_response_data = {}
                if routing_decision:
                    raw_response_data['routing_decision'] = routing_decision
                
                return LLMResponse(
                    content=response.choices[0].message.content,
                    model=model_config.name,
                    total_tokens=response.usage.total_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    cost=self._calculate_openai_cost(model_config, response.usage),
                    raw_response=raw_response_data
                )
    
    async def _complete_anthropic(
        self,
        prompt: str,
        model_config: ModelConfig,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        stream: bool
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Handle Anthropic Claude model completion"""
        client = self.clients[ModelProvider.ANTHROPIC]
        
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": model_config.api_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 1000,
            "stream": stream
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        if stream:
            # Return async generator for streaming
            async def stream_generator():
                stream_response = await client.messages.create(**kwargs)
                async for chunk in stream_response:
                    if chunk.type == "content_block_delta" and chunk.delta.text:
                        yield chunk.delta.text
            return stream_generator()
        else:
            # Get complete response
            response = await client.messages.create(**kwargs)
            
            # Calculate cost
            input_tokens = response.usage.input_tokens if hasattr(response, 'usage') else 0
            output_tokens = response.usage.output_tokens if hasattr(response, 'usage') else 0
            cost = 0.0
            if model_config.cost_per_1k_input_tokens and model_config.cost_per_1k_output_tokens:
                cost = (input_tokens * model_config.cost_per_1k_input_tokens / 1000 +
                        output_tokens * model_config.cost_per_1k_output_tokens / 1000)
            
            return LLMResponse(
                content=response.content[0].text if response.content else "",
                model=model_config.name,
                total_tokens=input_tokens + output_tokens,
                completion_tokens=output_tokens,
                cost=cost
            )
    
    def _auto_select_model(self, prompt: str, reasoning_effort: Optional[str]) -> Optional[str]:
        """Auto-select best model based on prompt and requirements using intelligent analysis"""
        available_models = self.config.get_available_models()
        if not available_models:
            return None
        
        # Analyze query complexity
        complexity, analysis = self.query_analyzer.analyze_query(prompt)
        
        # Override with explicit reasoning effort if provided
        if reasoning_effort:
            if reasoning_effort == "high":
                complexity = QueryComplexity.HIGH
            elif reasoning_effort == "low":
                complexity = QueryComplexity.LOW
        
        # Get recommended model
        recommended_model = self.query_analyzer.recommend_model(complexity, available_models, prompt)
        
        # Log the decision for transparency
        logger.debug(f"Query complexity: {complexity.value}")
        logger.debug(f"Analysis confidence: {analysis['confidence']:.2f}")
        logger.debug(f"Selected model: {recommended_model}")
        logger.debug(f"Reasoning: {analysis['reasoning'][:2]}")
        
        return recommended_model
    
    def analyze_query(self, query: str) -> Dict:
        """
        Analyze a query and return detailed analysis including model recommendation
        
        Args:
            query: The query to analyze
            
        Returns:
            Dictionary with analysis results
        """
        available_models = self.config.get_available_models()
        complexity, analysis = self.query_analyzer.analyze_query(query)
        recommended_model = self.query_analyzer.recommend_model(complexity, available_models)
        
        return {
            'complexity': complexity.value,
            'recommended_model': recommended_model,
            'reasoning_effort': self.query_analyzer.get_reasoning_effort(complexity),
            'confidence': analysis['confidence'],
            'indicators': analysis['indicators_found'],
            'reasoning': analysis['reasoning'],
            'available_models': available_models
        }
    
    def _calculate_cost(self, model_config: ModelConfig, response) -> float:
        """Calculate cost for Grok models"""
        # Check if we have separate input/output pricing
        if hasattr(response.usage, 'prompt_tokens') and hasattr(response.usage, 'completion_tokens'):
            prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
            completion_tokens = getattr(response.usage, 'completion_tokens', 0)
            
            if model_config.cost_per_1k_input_tokens and model_config.cost_per_1k_output_tokens:
                # Use separate pricing
                input_cost = (prompt_tokens / 1000) * model_config.cost_per_1k_input_tokens
                output_cost = (completion_tokens / 1000) * model_config.cost_per_1k_output_tokens
                return input_cost + output_cost
        
        # Fallback to simple calculation
        total_tokens = getattr(response.usage, 'total_tokens', 0)
        return (total_tokens / 1000) * model_config.cost_per_1k_tokens
    
    def _calculate_openai_cost(self, model_config: ModelConfig, usage) -> float:
        """Calculate cost for OpenAI models"""
        # Check if we have separate input/output pricing
        if hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            
            if model_config.cost_per_1k_input_tokens and model_config.cost_per_1k_output_tokens:
                # Use separate pricing
                input_cost = (prompt_tokens / 1000) * model_config.cost_per_1k_input_tokens
                output_cost = (completion_tokens / 1000) * model_config.cost_per_1k_output_tokens
                return input_cost + output_cost
        
        # Fallback to simple calculation
        total_tokens = usage.total_tokens
        return (total_tokens / 1000) * model_config.cost_per_1k_tokens


# Convenience functions
async def quick_complete(prompt: str, model: Optional[str] = None) -> str:
    """Quick completion helper"""
    client = UnifiedLLMClient()
    response = await client.complete(prompt, model=model)
    return response.content

async def reasoning_complete(prompt: str, effort: str = "medium") -> Dict:
    """Get a reasoning response with full details"""
    client = UnifiedLLMClient()
    response = await client.complete(prompt, reasoning_effort=effort)
    return {
        "answer": response.content,
        "reasoning": response.reasoning_content,
        "tokens": {
            "total": response.total_tokens,
            "reasoning": response.reasoning_tokens,
            "completion": response.completion_tokens
        },
        "model": response.model
    }
