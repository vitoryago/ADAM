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
    from xai_sdk.chat import user, system
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
        stream: bool = False
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
            
        Returns:
            LLMResponse object or async generator if streaming
        """
        # Auto-select model if not specified
        if not model:
            model = self._auto_select_model(prompt, reasoning_effort)
        
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
                max_tokens, reasoning_effort, stream
            )
        elif model_config.provider == ModelProvider.OPENAI:
            return await self._complete_openai(
                prompt, model_config, system_prompt, temperature,
                max_tokens, reasoning_effort, stream
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
        stream: bool
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
            # Map our unified effort levels to Grok's
            effort_map = {"low": "low", "medium": "low", "high": "high"}
            chat_params[model_config.reasoning_param] = effort_map.get(reasoning_effort, "low")
        
        chat = client.chat.create(**chat_params)
        
        # Add messages
        if system_prompt:
            chat.append(system(system_prompt))
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
            
            # Build unified response
            return LLMResponse(
                content=response.content,
                model=model_config.name,
                reasoning_content=getattr(response, 'reasoning_content', None),
                total_tokens=getattr(response.usage, 'total_tokens', 0),
                reasoning_tokens=getattr(response.usage, 'reasoning_tokens', 0),
                completion_tokens=getattr(response.usage, 'completion_tokens', 0),
                cost=self._calculate_cost(model_config, response)
            )
    
    async def _complete_openai(
        self,
        prompt: str,
        model_config: ModelConfig,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        reasoning_effort: Optional[str],
        stream: bool
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
                    return LLMResponse(
                        content=response.output_text,
                        model=model_config.name,
                        reasoning_content=None,  # OpenAI doesn't expose reasoning content
                        total_tokens=response.usage.total_tokens,
                        reasoning_tokens=response.usage.output_tokens_details.reasoning_tokens,
                        completion_tokens=response.usage.output_tokens,
                        cost=self._calculate_openai_cost(model_config, response.usage)
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
                return LLMResponse(
                    content=response.choices[0].message.content,
                    model=model_config.name,
                    total_tokens=response.usage.total_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    cost=self._calculate_openai_cost(model_config, response.usage)
                )
    
    def _auto_select_model(self, prompt: str, reasoning_effort: Optional[str]) -> Optional[str]:
        """Auto-select best model based on prompt and requirements"""
        prompt_lower = prompt.lower()
        
        # If reasoning is requested, prefer reasoning models
        if reasoning_effort:
            # For high effort, prefer o4-mini
            if reasoning_effort == "high":
                if "o4-mini-high" in self.config.get_available_models():
                    return "o4-mini-high"
            # For low effort, prefer grok-3-mini
            elif reasoning_effort == "low":
                if "grok-3-mini" in self.config.get_available_models():
                    return "grok-3-mini"
        
        # Check for SQL/analytics keywords
        sql_keywords = ["sql", "query", "database", "dbt", "snowflake", "optimize"]
        if any(keyword in prompt_lower for keyword in sql_keywords):
            # Prefer grok-4 for complex SQL analysis
            if "grok-4" in self.config.get_available_models():
                return "grok-4"
        
        # Check for reasoning indicators
        reasoning_keywords = ["analyze", "explain", "debug", "why", "how does"]
        if any(keyword in prompt_lower for keyword in reasoning_keywords):
            if "o4-mini-high" in self.config.get_available_models():
                return "o4-mini-high"
        
        # Default to any available model
        available = self.config.get_available_models()
        return available[0] if available else None
    
    def _calculate_cost(self, model_config: ModelConfig, response) -> float:
        """Calculate cost for Grok models"""
        # Add your pricing calculation here
        total_tokens = getattr(response.usage, 'total_tokens', 0)
        return (total_tokens / 1000) * model_config.cost_per_1k_tokens
    
    def _calculate_openai_cost(self, model_config: ModelConfig, usage) -> float:
        """Calculate cost for OpenAI models"""
        # Add your pricing calculation here
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