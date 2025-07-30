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
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,  # 'low', 'medium', 'high'
        stream: bool = False,
        image_data: Optional[bytes] = None
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
                max_tokens, reasoning_effort, stream, image_data, routing_decision
            )
        elif model_config.provider == ModelProvider.OPENAI:
            return await self._complete_openai(
                prompt, model_config, system_prompt, temperature,
                max_tokens, reasoning_effort, stream, image_data, routing_decision
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
        routing_decision: Optional[Dict] = None
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Handle Grok model completion"""
        client = self.clients[ModelProvider.GROK]
        
        # Create chat session
        chat_params = {
            "model": model_config.api_name,
            "temperature": temperature
        }
        
        # Add reasoning effort for models that support it
        if reasoning_effort and model_config.reasoning_param:
            # Check if this is grok-4 (which doesn't support reasoning_effort)
            if model_config.api_name == "grok-4":
                logger.warning(f"grok-4 doesn't support reasoning_effort, ignoring parameter")
                # Don't add reasoning_effort to chat_params
            else:
                # Map our unified effort levels to model-specific values
                # grok-3-mini and grok-3-mini-fast only support: low, high (no medium)
                if "grok-3-mini" in model_config.api_name:
                    effort_map = {"low": "low", "medium": "high", "high": "high"}
                else:
                    # grok-4-reasoning supports: low, medium, high
                    effort_map = {"low": "low", "medium": "medium", "high": "high"}
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
            
            response = await client.chat.completions.create(
                model=model_config.api_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
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