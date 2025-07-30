#!/usr/bin/env python3
"""
Configuration management for ADAM
Safely loads environment variables and provides defaults
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class ADAMConfig:
    """Configuration manager for ADAM system"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            env_file: Path to .env file (defaults to .env in project root)
        """
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            # Try multiple locations
            for env_path in ['.env', '.env.local', '../.env', '../../.env']:
                if Path(env_path).exists():
                    load_dotenv(env_path)
                    logger.info(f"Loaded environment from {env_path}")
                    break
        
        # API Keys (never log these!)
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.xai_api_key = os.getenv('XAI_API_KEY')
        
        # Model Configuration
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
        self.default_simple_model = os.getenv('DEFAULT_SIMPLE_MODEL', 'grok-3-mini-reasoning-high')
        self.default_complex_model = os.getenv('DEFAULT_COMPLEX_MODEL', 'o1-mini-high')
        self.default_coding_model = os.getenv('DEFAULT_CODING_MODEL', 'claude-opus-4')
        
        # Voice Configuration
        self.enable_voice = os.getenv('ENABLE_VOICE', 'true').lower() == 'true'
        self.voice_speed = int(os.getenv('VOICE_SPEED', '150'))
        
        # Storage Paths
        self.memory_storage_path = Path(os.getenv('MEMORY_STORAGE_PATH', './adam_memory_advanced'))
        self.conversation_storage_path = Path(os.getenv('CONVERSATION_STORAGE_PATH', './conversations'))
        
        # Cost Limits
        self.daily_cost_limit = float(os.getenv('DAILY_COST_LIMIT', '1.00'))
        self.monthly_cost_limit = float(os.getenv('MONTHLY_COST_LIMIT', '30.00'))
        
        # Memory Thresholds
        self.memory_confidence_threshold = float(os.getenv('MEMORY_CONFIDENCE_THRESHOLD', '0.7'))
        self.memory_high_confidence_threshold = float(os.getenv('MEMORY_HIGH_CONFIDENCE_THRESHOLD', '0.9'))
        
        # Logging
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Model costs (per 1K tokens)
        self.model_costs = {
            'grok-3-mini-reasoning-high': 0.0002,
            'o1-mini-high': 1.5,
            'claude-opus-4': 2.5
        }
    
    def validate(self) -> bool:
        """
        Validate that required configuration is present
        
        Returns:
            True if configuration is valid
        """
        errors = []
        
        # Check required API keys based on models
        if self.default_simple_model.startswith('grok') and not self.xai_api_key:
            errors.append("XAI_API_KEY is required for Grok models")
        
        if self.default_complex_model.startswith('o1') and not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required for O1 models")
            
        if self.default_coding_model.startswith('claude') and not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required for Claude models")
        
        # Check paths exist
        if not self.memory_storage_path.exists():
            self.memory_storage_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created memory storage directory: {self.memory_storage_path}")
            
        if not self.conversation_storage_path.exists():
            self.conversation_storage_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created conversation storage directory: {self.conversation_storage_path}")
        
        if errors:
            for error in errors:
                logger.error(error)
            return False
        
        return True
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'simple': self.default_simple_model,
            'complex': self.default_complex_model,
            'coding': self.default_coding_model,
            'costs': self.model_costs
        }
    
    def get_api_headers(self, model: str) -> Dict[str, str]:
        """
        Get API headers for a specific model
        
        Args:
            model: Model name
            
        Returns:
            Headers dict with authentication
        """
        if model.startswith('grok'):
            return {'Authorization': f'Bearer {self.xai_api_key}'}
        elif model.startswith('o1'):
            return {'Authorization': f'Bearer {self.openai_api_key}'}
        elif model.startswith('claude'):
            return {'X-API-Key': self.anthropic_api_key}
        else:
            return {}
    
    def mask_api_key(self, key: Optional[str]) -> str:
        """Mask API key for logging"""
        if not key:
            return "not_set"
        if len(key) < 8:
            return "invalid"
        return f"{key[:4]}...{key[-4:]}"
    
    def __repr__(self) -> str:
        """Safe representation without exposing secrets"""
        return f"""ADAMConfig(
    openai_key={self.mask_api_key(self.openai_api_key)},
    anthropic_key={self.mask_api_key(self.anthropic_api_key)},
    xai_key={self.mask_api_key(self.xai_api_key)},
    models={self.get_model_config()},
    storage={self.memory_storage_path}
)"""


# Global config instance
config = ADAMConfig()

# Validate on import
if not config.validate():
    logger.warning("Configuration validation failed - some features may not work")


# Helper function for other modules
def get_config() -> ADAMConfig:
    """Get the global configuration instance"""
    return config