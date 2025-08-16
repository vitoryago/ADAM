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
        # Initialize xAI client
        if XAI_AVAILABLE and self.config.get_api_key(ModelProvider.GROK):
            self.clients[ModelProvider.GROK] = XAIClient(
                api_host="api.x.ai",
                api_key=self.config.get_api_key(ModelProvider.GROK)
            )
        
        # Initialize OpenAI client
        if OPENAI_AVAILABLE and self.config.get_api_key(ModelProvider.OPENAI):
            self.clients[ModelProvider.OPENAI] = AsyncOpenAI(
                api_key=self.config.get_api_key(ModelProvider.OPENAI)
            )
        
        # Initialize Anthropic client
        if ANTHROPIC_AVAILABLE and self.config.get_api_key(ModelProvider.ANTHROPIC):
            self.clients[ModelProvider.ANTHROPIC] = AsyncAnthropic(
                api_key=self.config.get_api_key(ModelProvider.ANTHROPIC)
            )
    
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
        search_parameters: Optional[Dict] = None  # For live search
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
                    # Prefer GPT-5 for vision, then grok-2-vision for cost efficiency
                    if "gpt-5" in vision_models:
                        actual_model = "gpt-5"
                    elif "grok-2-vision-1212" in vision_models:
                        actual_model = "grok-2-vision-1212"
                    elif "grok-4" in vision_models:
                        actual_model = "grok-4"
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
                max_tokens, reasoning_effort, stream, image_data, routing_decision, search_parameters
            )
        elif model_config.provider == ModelProvider.OPENAI:
            return await self._complete_openai(
                prompt, model_config, system_prompt, temperature,
                max_tokens, reasoning_effort, stream, image_data, routing_decision
            )
        elif model_config.provider == ModelProvider.ANTHROPIC:
            return await self._complete_anthropic(
                prompt, model_config, system_prompt, temperature,
                max_tokens, stream, image_data, routing_decision, reasoning_effort
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
        search_parameters: Optional[Dict] = None
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Handle Grok model completion"""
        client = self.clients[ModelProvider.GROK]
        
        # Create chat session
        chat_params = {
            "model": model_config.api_name,
            "temperature": temperature
        }
        
        # Add search parameters if provided
        if search_parameters:
            # Import SearchParameters from xai_sdk
            from xai_sdk.search import SearchParameters
            # Create SearchParameters object with mode="on"
            search_params = SearchParameters(mode="on")
            chat_params["search_parameters"] = search_params
            logger.info(f"Search enabled for model {model_config.api_name}")
        
        # Add reasoning effort for models that support it
        if reasoning_effort and model_config.reasoning_param:
            # Check if this is grok-4 (which doesn't support reasoning_effort)
            if model_config.api_name == "grok-4":
                logger.warning(f"grok-4 doesn't support reasoning_effort, ignoring parameter")
                # Don't add reasoning_effort to chat_params
            else:
                # Map our unified effort levels to model-specific values
                # grok-3-mini only supports: low, high (no medium)
                if "grok-3-mini" in model_config.api_name:
                    effort_map = {"low": "low", "medium": "high", "high": "high", "minimal": "low"}
                else:
                    # grok-4-reasoning supports: low, medium, high
                    effort_map = {"low": "low", "medium": "medium", "high": "high", "minimal": "low"}
                chat_params[model_config.reasoning_param] = effort_map.get(reasoning_effort, "high")
        
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
                    # Grok uses sample_stream() method
                    if hasattr(chat, 'sample_stream'):
                        response = chat.sample_stream()
                        for chunk in response:
                            if hasattr(chunk, 'delta'):
                                yield chunk.delta
                    else:
                        # Fallback to non-streaming if method not available
                        response = chat.sample()
                        yield response.content
                except Exception as e:
                    logger.error(f"Grok streaming error: {e}")
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
        routing_decision: Optional[Dict] = None
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
            # Standard chat completion for GPT models (including GPT-5)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Handle image data for vision models
            if image_data and (model_config.supports_vision or "gpt-4" in model_config.api_name):
                # For GPT-4/GPT-5 vision, format message with image
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
            
            # Build request parameters
            request_params = {
                "model": model_config.api_name,
                "messages": messages,
                "stream": stream
            }
            
            # GPT-5 specific parameters
            if "gpt-5" in model_config.api_name:
                # GPT-5 only supports temperature=1 (default), so we omit it
                request_params["max_completion_tokens"] = max_tokens or 2000
                # Don't add temperature for GPT-5
                logger.info(f"Using default temperature for {model_config.api_name} (GPT-5 doesn't support custom temperature)")
            else:
                # Other models support temperature
                request_params["temperature"] = temperature
                request_params["max_tokens"] = max_tokens
            
            # Add reasoning_effort for GPT-5 models
            if "gpt-5" in model_config.api_name and reasoning_effort:
                # GPT-5 uses reasoning_effort parameter
                effort_map = {"low": "low", "medium": "medium", "high": "high", "minimal": "minimal"}
                request_params["reasoning_effort"] = effort_map.get(reasoning_effort, "medium")
                logger.info(f"Using reasoning_effort={request_params['reasoning_effort']} for {model_config.api_name}")
            
            response = await client.chat.completions.create(**request_params)
            
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
        stream: bool,
        image_data: Optional[bytes] = None,
        routing_decision: Optional[Dict] = None,
        reasoning_effort: Optional[str] = None
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Handle Claude model completion"""
        client = self.clients[ModelProvider.ANTHROPIC]
        
        # Build messages for Claude
        messages = []
        
        # Handle image data for vision models
        if image_data and model_config.supports_vision:
            import base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
        # Build request parameters for Claude
        request_params = {
            "model": model_config.api_name,
            "messages": messages,
            "max_tokens": max_tokens or model_config.max_tokens,
            "stream": stream
        }
        
        # Add extended thinking for Claude 4 models (Opus 4.1, Opus 4, Sonnet 4)
        if model_config.api_name in ["claude-opus-4-1-20250805", "claude-opus-4-20250514", "claude-sonnet-4-20250514"]:
            # Determine thinking budget based on reasoning effort
            if reasoning_effort:
                effort_to_budget = {
                    "minimal": 4000,
                    "low": 8000,
                    "medium": 16000,
                    "high": 32000
                }
                budget_tokens = effort_to_budget.get(reasoning_effort, 16000)
            else:
                # Default budget for complex tasks
                budget_tokens = 16000
            
            # Add thinking parameter
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens
            }
            
            # Always force streaming for extended thinking (required by Claude API)
            request_params["stream"] = True
            # Extended thinking requires temperature=1
            request_params["temperature"] = 1.0
            logger.info(f"Forcing streaming and temperature=1 for extended thinking with {budget_tokens} tokens")
            
            logger.info(f"Enabling extended thinking for {model_config.api_name} with {budget_tokens} budget tokens")
        else:
            # Add temperature for non-extended-thinking models
            request_params["temperature"] = temperature
        
        # Add system prompt if provided
        if system_prompt:
            request_params["system"] = system_prompt
        
        try:
            # Check if we're forcing streaming for extended thinking
            forced_streaming = request_params.get("stream", False) and not stream
            
            if stream or forced_streaming:
                # Stream response (either requested or forced for extended thinking)
                # For Anthropic, use the messages.stream method instead of create with stream=True
                request_params.pop('stream', None)  # Remove stream parameter
                stream_response = client.messages.stream(**request_params)
                
                if stream:
                    # Return actual stream for streaming requests
                    async def stream_generator():
                        async with stream_response as stream:
                            async for chunk in stream:
                                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                                    yield chunk.delta.text
                                elif hasattr(chunk, 'content_block') and hasattr(chunk.content_block, 'text'):
                                    yield chunk.content_block.text
                    return stream_generator()
                else:
                    # Collect stream for non-streaming requests with forced streaming
                    content = ""
                    thinking_content = None
                    usage_data = None
                    
                    async with stream_response as stream:
                        async for chunk in stream:
                            if hasattr(chunk, 'type'):
                                if chunk.type == 'content_block_delta':
                                    if hasattr(chunk, 'delta'):
                                        if hasattr(chunk.delta, 'type'):
                                            if chunk.delta.type == 'thinking_delta' and hasattr(chunk.delta, 'thinking'):
                                                if thinking_content is None:
                                                    thinking_content = ""
                                                thinking_content += chunk.delta.thinking
                                            elif chunk.delta.type == 'text_delta' and hasattr(chunk.delta, 'text'):
                                                content += chunk.delta.text
                                elif chunk.type == 'message_delta':
                                    if hasattr(chunk, 'usage'):
                                        usage_data = chunk.usage
                                elif chunk.type == 'message_stop':
                                    # Final message data
                                    if hasattr(chunk, 'message') and hasattr(chunk.message, 'usage'):
                                        usage_data = chunk.message.usage
                    
                    # Calculate cost
                    cost = 0.0
                    if usage_data:
                        input_tokens = getattr(usage_data, 'input_tokens', 0)
                        output_tokens = getattr(usage_data, 'output_tokens', 0)
                        
                        if model_config.cost_per_1k_input_tokens and model_config.cost_per_1k_output_tokens:
                            input_cost = (input_tokens / 1000) * model_config.cost_per_1k_input_tokens
                            output_cost = (output_tokens / 1000) * model_config.cost_per_1k_output_tokens
                            cost = input_cost + output_cost
                        else:
                            total_tokens = input_tokens + output_tokens
                            cost = (total_tokens / 1000) * model_config.cost_per_1k_tokens
                    
                    # Build response with routing info
                    raw_response_data = {}
                    if routing_decision:
                        raw_response_data['routing_decision'] = routing_decision
                    
                    return LLMResponse(
                        content=content,
                        model=model_config.name,
                        reasoning_content=thinking_content,
                        total_tokens=usage_data.input_tokens + usage_data.output_tokens if usage_data else 0,
                        completion_tokens=usage_data.output_tokens if usage_data else 0,
                        cost=cost,
                        raw_response=raw_response_data
                    )
            else:
                # Get complete response (non-streaming, non-extended-thinking)
                response = await client.messages.create(**request_params)
                
                # Extract content from Claude response
                content = ""
                thinking_content = None
                if hasattr(response, 'content'):
                    if isinstance(response.content, list):
                        # Process all blocks, separating thinking from text
                        for block in response.content:
                            if hasattr(block, 'type'):
                                if block.type == 'thinking':
                                    # Capture thinking content (summarized in Claude 4)
                                    if hasattr(block, 'thinking'):
                                        thinking_content = block.thinking
                                elif block.type == 'text':
                                    content += block.text if hasattr(block, 'text') else str(block)
                            elif hasattr(block, 'text'):
                                content += block.text
                    elif isinstance(response.content, str):
                        content = response.content
                    else:
                        content = str(response.content)
                
                # Calculate cost
                usage = response.usage if hasattr(response, 'usage') else None
                if usage:
                    input_tokens = usage.input_tokens if hasattr(usage, 'input_tokens') else 0
                    output_tokens = usage.output_tokens if hasattr(usage, 'output_tokens') else 0
                    
                    if model_config.cost_per_1k_input_tokens and model_config.cost_per_1k_output_tokens:
                        input_cost = (input_tokens / 1000) * model_config.cost_per_1k_input_tokens
                        output_cost = (output_tokens / 1000) * model_config.cost_per_1k_output_tokens
                        cost = input_cost + output_cost
                    else:
                        total_tokens = input_tokens + output_tokens
                        cost = (total_tokens / 1000) * model_config.cost_per_1k_tokens
                else:
                    cost = 0.0
                
                # Build response with routing info
                raw_response_data = {}
                if routing_decision:
                    raw_response_data['routing_decision'] = routing_decision
                
                return LLMResponse(
                    content=content,
                    model=model_config.name,
                    reasoning_content=thinking_content,  # Extended thinking content if available
                    total_tokens=usage.input_tokens + usage.output_tokens if usage else 0,
                    completion_tokens=usage.output_tokens if usage else 0,
                    cost=cost,
                    raw_response=raw_response_data
                )
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise
    
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
        recommended_model = self.query_analyzer.recommend_model(complexity, available_models)
        
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