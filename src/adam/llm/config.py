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
    ANTHROPIC = "anthropic"

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
        # Check both XAI_API_KEY and GROK_API_KEY for Grok models
        self.api_keys = {
            ModelProvider.GROK: os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY"),
            ModelProvider.OPENAI: os.getenv("OPENAI_API_KEY"),
            ModelProvider.ANTHROPIC: os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY"),
        }
        
        # Model configurations
        self.models = {
            # Virtual automatic model for intelligent routing
            "automatic": ModelConfig(
                name="automatic",
                provider=ModelProvider.ANTHROPIC,  # Now defaulting to Claude for complex tasks
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
            
            # Claude Models (New!)
            "claude-opus-4.1": ModelConfig(
                name="claude-opus-4.1",
                provider=ModelProvider.ANTHROPIC,
                api_name="claude-opus-4-1-20250805",
                capabilities=[
                    ModelCapability.REASONING,
                    ModelCapability.COMPLEX_ANALYSIS,
                    ModelCapability.CODE_GENERATION
                ],
                supports_reasoning=True,
                reasoning_param="thinking",  # Claude 4 uses extended thinking
                max_tokens=32000,  # Claude Opus 4.1 supports up to 32k output tokens
                supports_streaming=True,
                supports_vision=True,
                cost_per_1k_tokens=0.015,  # $15/1M input, $75/1M output
                cost_per_1k_input_tokens=0.015,
                cost_per_1k_output_tokens=0.075
            ),
            
            "claude-opus-4": ModelConfig(
                name="claude-opus-4",
                provider=ModelProvider.ANTHROPIC,
                api_name="claude-opus-4-20250514",
                capabilities=[
                    ModelCapability.REASONING,
                    ModelCapability.COMPLEX_ANALYSIS,
                    ModelCapability.CODE_GENERATION
                ],
                supports_reasoning=True,
                reasoning_param="thinking",  # Claude 4 uses extended thinking
                max_tokens=32000,
                supports_streaming=True,
                supports_vision=True,
                cost_per_1k_tokens=0.015,  # $15/1M input, $75/1M output
                cost_per_1k_input_tokens=0.015,
                cost_per_1k_output_tokens=0.075
            ),
            
            "claude-sonnet-4": ModelConfig(
                name="claude-sonnet-4",
                provider=ModelProvider.ANTHROPIC,
                api_name="claude-sonnet-4-20250514",
                capabilities=[
                    ModelCapability.REASONING,
                    ModelCapability.COMPLEX_ANALYSIS,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FAST_RESPONSE
                ],
                supports_reasoning=True,
                reasoning_param="thinking",  # Claude 4 uses extended thinking
                max_tokens=16000,
                supports_streaming=True,
                supports_vision=True,
                cost_per_1k_tokens=0.003,  # $3/1M input, $15/1M output
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.015
            ),
            
            "claude-sonnet-3.7": ModelConfig(
                name="claude-sonnet-3.7",
                provider=ModelProvider.ANTHROPIC,
                api_name="claude-3-7-sonnet-20250219",
                capabilities=[
                    ModelCapability.REASONING,
                    ModelCapability.COMPLEX_ANALYSIS,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FAST_RESPONSE
                ],
                supports_reasoning=True,
                reasoning_param="thinking",  # Claude 3.7 also supports extended thinking
                max_tokens=8192,
                supports_streaming=True,
                supports_vision=True,
                cost_per_1k_tokens=0.003,  # $3/1M input, $15/1M output
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.015
            ),
            
            "claude-3.5-sonnet": ModelConfig(
                name="claude-3.5-sonnet",
                provider=ModelProvider.ANTHROPIC,
                api_name="claude-3-5-sonnet-20241022",
                capabilities=[
                    ModelCapability.REASONING,
                    ModelCapability.COMPLEX_ANALYSIS,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FAST_RESPONSE
                ],
                supports_reasoning=False,
                reasoning_param=None,
                max_tokens=8192,  # Claude 3.5 Sonnet max
                supports_streaming=True,
                supports_vision=True,
                cost_per_1k_tokens=0.003,  # $3/1M input, $15/1M output
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.015
            ),
            
            "claude-3.5-haiku": ModelConfig(
                name="claude-3.5-haiku",
                provider=ModelProvider.ANTHROPIC,
                api_name="claude-3-5-haiku-20241022",
                capabilities=[
                    ModelCapability.FAST_RESPONSE,
                    ModelCapability.BASIC_QA,
                    ModelCapability.CODE_GENERATION
                ],
                supports_reasoning=False,
                reasoning_param=None,
                max_tokens=8192,  # Claude 3.5 Haiku max
                supports_streaming=True,
                supports_vision=False,
                cost_per_1k_tokens=0.001,  # $1/1M input, $5/1M output
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.005
            ),
            
            # GPT-5 Models (New!)
            "gpt-5": ModelConfig(
                name="gpt-5",
                provider=ModelProvider.OPENAI,
                api_name="gpt-5-2025-08-07",
                capabilities=[
                    ModelCapability.REASONING,
                    ModelCapability.COMPLEX_ANALYSIS,
                    ModelCapability.CODE_GENERATION
                ],
                supports_reasoning=True,
                reasoning_param="reasoning_effort",  # Uses "minimal", "low", "medium", "high"
                max_tokens=32768,  # GPT-5 supports up to 32k
                supports_streaming=True,
                supports_vision=True,
                # GPT-5: $1.25 per 1M input, $10.00 per 1M output
                cost_per_1k_tokens=0.005625,  # Average
                cost_per_1k_input_tokens=0.00125,  # $1.25 / 1000
                cost_per_1k_output_tokens=0.010     # $10.00 / 1000
            ),
            
            "gpt-5-mini": ModelConfig(
                name="gpt-5-mini",
                provider=ModelProvider.OPENAI,
                api_name="gpt-5-mini-2025-08-07",
                capabilities=[
                    ModelCapability.FAST_RESPONSE,
                    ModelCapability.BASIC_QA,
                    ModelCapability.REASONING
                ],
                supports_reasoning=True,
                reasoning_param="reasoning_effort",
                max_tokens=8192,
                supports_streaming=True,
                supports_vision=False,
                # GPT-5-mini: $0.25 per 1M input, $2.00 per 1M output
                cost_per_1k_tokens=0.001125,  # Average
                cost_per_1k_input_tokens=0.00025,  # $0.25 / 1000
                cost_per_1k_output_tokens=0.002    # $2.00 / 1000
            ),
            
            "gpt-5-nano": ModelConfig(
                name="gpt-5-nano",
                provider=ModelProvider.OPENAI,
                api_name="gpt-5-nano-2025-08-07",
                capabilities=[
                    ModelCapability.FAST_RESPONSE,
                    ModelCapability.BASIC_QA
                ],
                supports_reasoning=True,
                reasoning_param="reasoning_effort",
                max_tokens=4096,
                supports_streaming=True,
                supports_vision=False,
                # GPT-5-nano: $0.05 per 1M input, $0.40 per 1M output
                cost_per_1k_tokens=0.000225,  # Average
                cost_per_1k_input_tokens=0.00005,  # $0.05 / 1000
                cost_per_1k_output_tokens=0.0004   # $0.40 / 1000
            ),
            # GPT-5 Models
            "gpt-5": ModelConfig(
                name="gpt-5",
                provider=ModelProvider.OPENAI,
                api_name="gpt-5-2025-08-07",
                capabilities=[
                    ModelCapability.REASONING,
                    ModelCapability.COMPLEX_ANALYSIS,
                    ModelCapability.CODE_GENERATION
                ],
                supports_reasoning=True,
                reasoning_param="reasoning_effort",
                max_tokens=8192,
                supports_streaming=False,  # Responses API doesn't support streaming yet
                supports_vision=False,  # Update when vision support is added
                cost_per_1k_tokens=0.002,  # $2 per 1M tokens average
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.003
            ),
            
            "gpt-5-mini": ModelConfig(
                name="gpt-5-mini",
                provider=ModelProvider.OPENAI,
                api_name="gpt-5-mini-2025-08-07",
                capabilities=[
                    ModelCapability.REASONING,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FAST_RESPONSE,
                    ModelCapability.BASIC_QA
                ],
                supports_reasoning=True,
                reasoning_param="reasoning_effort",
                max_tokens=4096,
                supports_streaming=False,
                supports_vision=False,
                cost_per_1k_tokens=0.0005,
                cost_per_1k_input_tokens=0.0002,
                cost_per_1k_output_tokens=0.0008
            ),
            
            "gpt-5-nano": ModelConfig(
                name="gpt-5-nano",
                provider=ModelProvider.OPENAI,
                api_name="gpt-5-nano-2025-08-07",
                capabilities=[
                    ModelCapability.FAST_RESPONSE,
                    ModelCapability.BASIC_QA
                ],
                supports_reasoning=True,
                reasoning_param="reasoning_effort",
                max_tokens=2000,
                supports_streaming=False,
                supports_vision=False,
                cost_per_1k_tokens=0.000225,
                cost_per_1k_input_tokens=0.00005,
                cost_per_1k_output_tokens=0.0004
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
                supports_vision=True,  # grok-4 supports image input
                cost_per_1k_tokens=0.009,  # Average of input/output for backward compatibility
                cost_per_1k_input_tokens=0.003,  # $3.00/1M input tokens
                cost_per_1k_output_tokens=0.015   # $15.00/1M output tokens
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
            
        }
        
        # Default model selection rules
        self.default_models = {
            "complex": "grok-4-reasoning",    # Grok-4-reasoning for hardest problems
            "medium": "gpt-5",                # GPT-5 for medium complexity
            "fast": "gpt-5-mini",             # GPT-5 Mini for simple queries
            "reasoning": "grok-4-reasoning"   # Grok-4-reasoning for deep reasoning
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

1. For Claude models (claude-opus-4.1, claude-3.5-sonnet, claude-3.5-haiku):
   export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
   # OR
   export CLAUDE_API_KEY="your-claude-api-key-here"

2. For GPT-5 models (gpt-5, gpt-5-mini, gpt-5-nano):
   export OPENAI_API_KEY="your-openai-api-key-here"
   
3. For Grok models (grok-4, grok-3-mini):
   export XAI_API_KEY="your-xai-api-key-here"
   # OR
   export GROK_API_KEY="your-xai-api-key-here"

You can add these to your shell profile (~/.bashrc, ~/.zshrc) or create a .env file:

# .env file in ADAM root directory
ANTHROPIC_API_KEY=your-anthropic-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
XAI_API_KEY=your-xai-api-key-here

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