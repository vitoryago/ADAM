"""
LLM Configuration System for ADAM
Supports: grok-4, grok-3-mini, and openai o4-mini-high
"""
import os
from typing import Dict, Optional, List, Literal
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file with override
load_dotenv(override=True)

class ModelProvider(Enum):
    GROK = "grok"
    OPENAI = "openai"

class ModelCapability(Enum):
    BASIC_QA = "basic_qa"
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    COMPLEX_ANALYSIS = "complex_analysis"
    FAST_RESPONSE = "fast_response"

@dataclass
class ModelConfig:
    """Configuration for each LLM model"""
    name: str
    provider: ModelProvider
    api_name: str  # The actual name to use in API calls
    capabilities: List[ModelCapability]
    supports_reasoning: bool
    reasoning_param: Optional[str]  # 'reasoning_effort' for Grok, 'effort' for OpenAI
    max_tokens: int
    temperature_range: tuple = (0.0, 1.0)
    supports_streaming: bool = True
    cost_per_1k_tokens: float = 0.0  # Add your pricing here
    supports_vision: bool = False  # Whether model supports image input
    # Separate input/output pricing (optional, defaults to cost_per_1k_tokens)
    cost_per_1k_input_tokens: Optional[float] = None
    cost_per_1k_output_tokens: Optional[float] = None

class LLMConfig:
    """Central configuration for LLM models"""
    
    def __init__(self):
        # API Keys - Set these as environment variables
        self.api_keys = {
            ModelProvider.GROK: os.getenv("XAI_API_KEY"),
            ModelProvider.OPENAI: os.getenv("OPENAI_API_KEY"),
        }
        
        # Model configurations
        self.models = {
            # Virtual automatic model for intelligent routing
            "automatic": ModelConfig(
                name="automatic",
                provider=ModelProvider.GROK,  # Placeholder
                api_name="automatic",
                capabilities=[
                    ModelCapability.REASONING,
                    ModelCapability.COMPLEX_ANALYSIS,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FAST_RESPONSE,
                    ModelCapability.BASIC_QA
                ],
                supports_reasoning=True,
                reasoning_param=None,
                max_tokens=8192,
                supports_streaming=True,
                supports_vision=True,  # Can route to vision models
                cost_per_1k_tokens=0.002  # Average cost estimate
            ),
            "grok-4-reasoning": ModelConfig(
                name="grok-4-reasoning",
                provider=ModelProvider.GROK,
                api_name="grok-4",
                capabilities=[
                    ModelCapability.REASONING,
                    ModelCapability.COMPLEX_ANALYSIS,
                    ModelCapability.CODE_GENERATION
                ],
                supports_reasoning=False,  # grok-4 doesn't have reasoning_effort param
                reasoning_param=None,  # Use as high-power model without param
                max_tokens=8192,
                supports_streaming=True,
                supports_vision=True  # grok-4 supports image input
            ),
            
            "grok-4": ModelConfig(
                name="grok-4",
                provider=ModelProvider.GROK,
                api_name="grok-4",
                capabilities=[
                    ModelCapability.REASONING,
                    ModelCapability.COMPLEX_ANALYSIS,
                    ModelCapability.CODE_GENERATION
                ],
                supports_reasoning=True,
                reasoning_param=None,  # grok-4 standard mode
                max_tokens=4096,
                supports_streaming=True,
                supports_vision=True  # grok-4 supports image input
            ),
            
            "grok-3-mini-high": ModelConfig(
                name="grok-3-mini-high",
                provider=ModelProvider.GROK,
                api_name="grok-3-mini",
                capabilities=[
                    ModelCapability.REASONING,
                    ModelCapability.FAST_RESPONSE,
                    ModelCapability.BASIC_QA
                ],
                supports_reasoning=True,
                reasoning_param="reasoning_effort",
                max_tokens=4096,
                supports_streaming=True,
                cost_per_1k_tokens=0.004,  # Average of input/output
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.006
            ),
            
            "grok-3-mini-fast": ModelConfig(
                name="grok-3-mini-fast",
                provider=ModelProvider.GROK,
                api_name="grok-3-mini-fast",
                capabilities=[
                    ModelCapability.FAST_RESPONSE,
                    ModelCapability.BASIC_QA,
                    ModelCapability.REASONING
                ],
                supports_reasoning=True,
                reasoning_param="reasoning_effort",
                max_tokens=4096,
                supports_streaming=True,
                cost_per_1k_tokens=0.0023,  # Average of input/output
                cost_per_1k_input_tokens=0.0006,  # $0.60/1M
                cost_per_1k_output_tokens=0.004    # $4.00/1M
            ),
            
            "grok-2-vision-1212": ModelConfig(
                name="grok-2-vision-1212",
                provider=ModelProvider.GROK,
                api_name="grok-2-vision-1212",
                capabilities=[
                    ModelCapability.COMPLEX_ANALYSIS,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.REASONING
                ],
                supports_reasoning=False,
                reasoning_param=None,
                max_tokens=8192,
                supports_streaming=True,
                supports_vision=True,  # This is a vision model
                cost_per_1k_tokens=0.006,  # Average of input/output for backward compatibility
                cost_per_1k_input_tokens=0.002,  # $2 per million input tokens
                cost_per_1k_output_tokens=0.010   # $10 per million output tokens
            ),
            
            "o4-mini-high": ModelConfig(
                name="o4-mini-high",
                provider=ModelProvider.OPENAI,
                api_name="o4-mini",
                capabilities=[
                    ModelCapability.REASONING,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.COMPLEX_ANALYSIS
                ],
                supports_reasoning=True,
                reasoning_param="effort",
                max_tokens=25000,  # Reserve space for reasoning tokens
                supports_streaming=True
            ),
            
            # Standard OpenAI models that work without verification
            "gpt-4": ModelConfig(
                name="gpt-4",
                provider=ModelProvider.OPENAI,
                api_name="gpt-4",
                capabilities=[
                    ModelCapability.COMPLEX_ANALYSIS,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.REASONING
                ],
                supports_reasoning=False,
                reasoning_param=None,
                max_tokens=8192,
                supports_streaming=True,
                cost_per_1k_tokens=0.03,
                supports_vision=True  # GPT-4 vision is supported
            ),
            
            "gpt-3.5-turbo": ModelConfig(
                name="gpt-3.5-turbo",
                provider=ModelProvider.OPENAI,
                api_name="gpt-3.5-turbo",
                capabilities=[
                    ModelCapability.FAST_RESPONSE,
                    ModelCapability.BASIC_QA,
                    ModelCapability.CODE_GENERATION
                ],
                supports_reasoning=False,
                reasoning_param=None,
                max_tokens=4096,
                supports_streaming=True,
                cost_per_1k_tokens=0.001
            )
        }
        
        # Default model selection rules
        self.default_models = {
            "complex": "grok-4-reasoning",    # For code generation, deep analysis
            "medium": "grok-4",               # For standard queries
            "fast": "grok-3-mini-high",       # For memory recaps, simple queries
            "reasoning": "grok-4-reasoning"   # Explicit reasoning tasks
        }
    
    def get_api_key(self, provider: ModelProvider) -> Optional[str]:
        """Get API key for a provider"""
        return self.api_keys.get(provider)
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        return self.models.get(model_name)
    
    def get_available_models(self) -> List[str]:
        """Get list of models that have API keys configured"""
        available = []
        for model_name, config in self.models.items():
            if self.get_api_key(config.provider):
                available.append(model_name)
        return available
    
    def select_model_for_task(
        self, 
        task_type: Literal["fast", "smart", "reasoning", "code"],
        required_capabilities: Optional[List[ModelCapability]] = None
    ) -> Optional[str]:
        """Select the best available model for a task"""
        # First try default selection
        if task_type in self.default_models:
            model_name = self.default_models[task_type]
            if model_name in self.get_available_models():
                return model_name
        
        # Then try to find any model with required capabilities
        if required_capabilities:
            for model_name, config in self.models.items():
                if self.get_api_key(config.provider):
                    if all(cap in config.capabilities for cap in required_capabilities):
                        return model_name
        
        # Return any available model
        available = self.get_available_models()
        return available[0] if available else None

# Environment setup instructions
SETUP_INSTRUCTIONS = """
To use ADAM's LLM capabilities, you need to set up your API keys:

1. For Grok models (grok-4, grok-3-mini):
   export XAI_API_KEY="your-xai-api-key-here"
   
2. For OpenAI models (o4-mini):
   export OPENAI_API_KEY="your-openai-api-key-here"

You can add these to your shell profile (~/.bashrc, ~/.zshrc) or create a .env file:

# .env file in ADAM root directory
XAI_API_KEY=your-xai-api-key-here
OPENAI_API_KEY=your-openai-api-key-here

Then load with python-dotenv:
from dotenv import load_dotenv
load_dotenv()
"""

if __name__ == "__main__":
    # Test configuration
    config = LLMConfig()
    
    print("=== ADAM LLM Configuration ===")
    print(f"\nConfigured models: {list(config.models.keys())}")
    print(f"\nAvailable models (with API keys): {config.get_available_models()}")
    
    if not config.get_available_models():
        print("\n⚠️  No API keys found!")
        print(SETUP_INSTRUCTIONS)