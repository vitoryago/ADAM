"""
ADAM Configuration System
Core configuration for the ADAM AI assistant
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ADAMConfig:
    """Main configuration class for ADAM system"""
    
    def __init__(self):
        # Storage paths
        self.memory_storage_path = os.getenv('MEMORY_STORAGE_PATH', './data/adam_memory')
        self.project_storage_path = os.getenv('PROJECT_STORAGE_PATH', './data/adam_projects')
        self.conversation_storage_path = os.getenv('CONVERSATION_STORAGE_PATH', './data/conversations')
        
        # Ensure directories exist
        for path in [self.memory_storage_path, self.project_storage_path, self.conversation_storage_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Model configuration - check both ADAM format and standard format
        self.embedding_model = (
            os.getenv('ADAM_EMBEDDING_MODEL') or 
            os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
        )
        self.default_simple_model = os.getenv('DEFAULT_SIMPLE_MODEL', 'gpt-4o-mini')
        self.default_complex_model = os.getenv('DEFAULT_COMPLEX_MODEL', 'grok-4')
        self.default_coding_model = os.getenv('DEFAULT_CODING_MODEL', 'gpt-4o')
        
        # ADAM-specific settings
        self.adam_name = os.getenv('ADAM_NAME', 'ADAM')
        self.adam_language = os.getenv('ADAM_LANGUAGE', 'en')
        self.adam_voice_speed = int(os.getenv('ADAM_VOICE_SPEED', '180'))
        self.adam_voice_engine = os.getenv('ADAM_VOICE_ENGINE', 'pyttsx3')
        
        # Cost configuration
        self.daily_cost_limit = float(os.getenv('DAILY_COST_LIMIT', '1.00'))
        self.monthly_cost_limit = float(os.getenv('MONTHLY_COST_LIMIT', '30.00'))
        
        # Memory configuration
        self.memory_confidence_threshold = float(os.getenv('MEMORY_CONFIDENCE_THRESHOLD', '0.7'))
        self.memory_high_confidence_threshold = float(os.getenv('MEMORY_HIGH_CONFIDENCE_THRESHOLD', '0.9'))
        
        # Screen capture configuration
        self.enable_screen_capture = os.getenv('ENABLE_SCREEN_CAPTURE', 'false').lower() == 'true'
        self.screen_monitor_interval = int(os.getenv('SCREEN_MONITOR_INTERVAL', '30'))
        
        # Voice configuration
        self.enable_voice = os.getenv('ENABLE_VOICE', 'false').lower() == 'true'
        self.voice_speed = int(os.getenv('VOICE_SPEED', '150'))
        
        # Logging
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    def get_model_config(self, task_type: str = 'general') -> str:
        """Get appropriate model for task type"""
        model_mapping = {
            'simple': self.default_simple_model,
            'complex': self.default_complex_model,
            'coding': self.default_coding_model,
            'general': self.default_simple_model
        }
        return model_mapping.get(task_type, self.default_simple_model)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'memory_storage_path': self.memory_storage_path,
            'project_storage_path': self.project_storage_path,
            'conversation_storage_path': self.conversation_storage_path,
            'embedding_model': self.embedding_model,
            'default_simple_model': self.default_simple_model,
            'default_complex_model': self.default_complex_model,
            'default_coding_model': self.default_coding_model,
            'daily_cost_limit': self.daily_cost_limit,
            'monthly_cost_limit': self.monthly_cost_limit,
            'memory_confidence_threshold': self.memory_confidence_threshold,
            'memory_high_confidence_threshold': self.memory_high_confidence_threshold,
            'enable_screen_capture': self.enable_screen_capture,
            'screen_monitor_interval': self.screen_monitor_interval,
            'enable_voice': self.enable_voice,
            'voice_speed': self.voice_speed,
            'log_level': self.log_level
        }