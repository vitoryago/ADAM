"""
ADAM Configuration Module

Unified configuration system that loads from environment variables,
.env files, and provides typed dataclass configs for all subsystems.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "sqlite:///./data/adam.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10


@dataclass
class MemoryConfig:
    """Memory system configuration"""
    path: str = "./data/adam_memory"
    search_enabled: bool = True
    search_limit: int = 10
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chromadb_persist_directory: str = "./data/adam_memory/chromadb"
    chromadb_collection_name: str = "adam_memories"
    embedding_model: str = "all-mpnet-base-v2"


@dataclass
class LLMConfig:
    """LLM configuration"""
    default_model_fast: str = "grok-4-fast-non-reasoning"
    default_model_standard: str = "grok-4-fast-non-reasoning"
    default_model_reasoning: str = "grok-4-fast-reasoning"
    default_model_code: str = "grok-4-fast-reasoning"
    use_intelligent_routing: bool = True
    routing_cache_size: int = 1000
    routing_cache_ttl: int = 3600
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class VoiceConfig:
    """Voice and audio configuration"""
    use_openai_tts: bool = True
    voice_speed: int = 180
    voice_engine: str = "openai"
    openai_tts_voice: str = "alloy"
    openai_tts_model: str = "tts-1"
    elevenlabs_voice_id: str = "ZthjuvLPty3kTMaNKVKb"
    elevenlabs_model: str = "eleven_monolingual_v1"
    use_openai_whisper: bool = True
    whisper_model: str = "whisper-1"


@dataclass
class WebConfig:
    """Web interface configuration"""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    api_debug: bool = True
    streamlit_port: int = 8501
    streamlit_host: str = "localhost"
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:8080"
    ])


@dataclass
class SecurityConfig:
    """Security and performance configuration"""
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    cache_ttl: int = 3600
    cache_size: int = 10000
    session_secret_key: str = "change-this-in-production"
    session_timeout: int = 3600


@dataclass
class FeatureFlags:
    """Feature enablement flags"""
    enable_voice: bool = True
    enable_vision: bool = True
    enable_web_search: bool = False
    enable_coworker_mode: bool = False
    enable_team_workspaces: bool = False
    enable_metrics: bool = True


@dataclass
class PathConfig:
    """File and directory paths"""
    knowledge_base_path: str = "./knowledge"
    conversation_dir: str = "./conversations"
    project_dir: str = "./data/adam_projects"
    upload_dir: str = "./data/uploads"
    temp_dir: str = "./data/temp"
    backup_dir: str = "./data/backups"


class UnifiedConfig:
    """
    Unified configuration system for ADAM

    This class loads configuration from:
    1. Environment variables
    2. .env file
    3. Default values

    It provides a single source of truth for all ADAM configuration.
    """

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize unified configuration

        Args:
            env_file: Path to .env file (defaults to .env in project root)
        """
        # Load environment variables
        if env_file is None:
            # Look for .env file in project root
            project_root = Path(__file__).parent.parent.parent
            env_file = project_root / ".env"

        if Path(env_file).exists():
            load_dotenv(env_file, override=True)
            logger.info(f"Loaded configuration from {env_file}")
        else:
            logger.warning(f"Configuration file {env_file} not found, using environment variables and defaults")

        # Initialize configuration sections
        self.core = self._load_core_config()
        self.database = self._load_database_config()
        self.memory = self._load_memory_config()
        self.llm = self._load_llm_config()
        self.voice = self._load_voice_config()
        self.web = self._load_web_config()
        self.security = self._load_security_config()
        self.features = self._load_feature_flags()
        self.paths = self._load_path_config()

        # Create directories
        self._create_directories()

    def _load_core_config(self) -> Dict[str, Any]:
        """Load core ADAM settings"""
        return {
            'name': os.getenv('ADAM_NAME', 'ADAM'),
            'version': os.getenv('ADAM_VERSION', '4.0.0'),
            'language': os.getenv('ADAM_LANGUAGE', 'en'),
            'log_level': os.getenv('ADAM_LOG_LEVEL', 'INFO'),
            'debug': os.getenv('DEBUG', 'false').lower() == 'true'
        }

    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration"""
        return DatabaseConfig(
            url=os.getenv('DATABASE_URL', 'sqlite:///./data/adam.db'),
            echo=os.getenv('LOG_SQL_QUERIES', 'false').lower() == 'true',
            pool_size=int(os.getenv('DB_POOL_SIZE', '5')),
            max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '10'))
        )

    def _load_memory_config(self) -> MemoryConfig:
        """Load memory system configuration"""
        return MemoryConfig(
            path=os.getenv('MEMORY_PATH', './data/adam_memory'),
            search_enabled=os.getenv('MEMORY_SEARCH_ENABLED', 'true').lower() == 'true',
            search_limit=int(os.getenv('MEMORY_SEARCH_LIMIT', '10')),
            chunk_size=int(os.getenv('MEMORY_CHUNK_SIZE', '1000')),
            chunk_overlap=int(os.getenv('MEMORY_CHUNK_OVERLAP', '200')),
            chromadb_persist_directory=os.getenv('CHROMADB_PERSIST_DIRECTORY', './data/adam_memory/chromadb'),
            chromadb_collection_name=os.getenv('CHROMADB_COLLECTION_NAME', 'adam_memories'),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'all-mpnet-base-v2')
        )

    def _load_llm_config(self) -> LLMConfig:
        """Load LLM configuration"""
        return LLMConfig(
            default_model_fast=os.getenv('DEFAULT_MODEL_FAST', 'grok-4-fast-non-reasoning'),
            default_model_standard=os.getenv('DEFAULT_MODEL_STANDARD', 'grok-4-fast-non-reasoning'),
            default_model_reasoning=os.getenv('DEFAULT_MODEL_REASONING', 'grok-4-fast-reasoning'),
            default_model_code=os.getenv('DEFAULT_MODEL_CODE', 'grok-4-fast-reasoning'),
            use_intelligent_routing=os.getenv('USE_INTELLIGENT_ROUTING', 'true').lower() == 'true',
            routing_cache_size=int(os.getenv('ROUTING_CACHE_SIZE', '1000')),
            routing_cache_ttl=int(os.getenv('ROUTING_CACHE_TTL', '3600')),
            max_tokens=int(os.getenv('MAX_TOKENS', '4096')),
            temperature=float(os.getenv('TEMPERATURE', '0.7')),
            top_p=float(os.getenv('TOP_P', '0.9'))
        )

    def _load_voice_config(self) -> VoiceConfig:
        """Load voice configuration"""
        return VoiceConfig(
            use_openai_tts=os.getenv('USE_OPENAI_TTS', 'true').lower() == 'true',
            voice_speed=int(os.getenv('VOICE_SPEED', '180')),
            voice_engine=os.getenv('VOICE_ENGINE', 'openai'),
            openai_tts_voice=os.getenv('OPENAI_TTS_VOICE', 'alloy'),
            openai_tts_model=os.getenv('OPENAI_TTS_MODEL', 'tts-1'),
            elevenlabs_voice_id=os.getenv('ELEVENLABS_VOICE_ID', 'ZthjuvLPty3kTMaNKVKb'),
            elevenlabs_model=os.getenv('ELEVENLABS_MODEL', 'eleven_monolingual_v1'),
            use_openai_whisper=os.getenv('USE_OPENAI_WHISPER', 'true').lower() == 'true',
            whisper_model=os.getenv('WHISPER_MODEL', 'whisper-1')
        )

    def _load_web_config(self) -> WebConfig:
        """Load web configuration"""
        cors_origins = os.getenv('CORS_ORIGINS', '').split(',') if os.getenv('CORS_ORIGINS') else [
            "http://localhost:3000", "http://localhost:5173", "http://localhost:5174", "http://localhost:8080"
        ]

        return WebConfig(
            api_host=os.getenv('API_HOST', '0.0.0.0'),
            api_port=int(os.getenv('API_PORT', '8000')),
            api_reload=os.getenv('API_RELOAD', 'true').lower() == 'true',
            api_debug=os.getenv('API_DEBUG', 'true').lower() == 'true',
            streamlit_port=int(os.getenv('STREAMLIT_PORT', '8501')),
            streamlit_host=os.getenv('STREAMLIT_HOST', 'localhost'),
            cors_origins=[origin.strip() for origin in cors_origins if origin.strip()]
        )

    def _load_security_config(self) -> SecurityConfig:
        """Load security configuration"""
        return SecurityConfig(
            rate_limit_per_minute=int(os.getenv('RATE_LIMIT_PER_MINUTE', '60')),
            rate_limit_per_hour=int(os.getenv('RATE_LIMIT_PER_HOUR', '1000')),
            cache_ttl=int(os.getenv('CACHE_TTL', '3600')),
            cache_size=int(os.getenv('CACHE_SIZE', '10000')),
            session_secret_key=os.getenv('SESSION_SECRET_KEY', 'change-this-in-production'),
            session_timeout=int(os.getenv('SESSION_TIMEOUT', '3600'))
        )

    def _load_feature_flags(self) -> FeatureFlags:
        """Load feature flags"""
        return FeatureFlags(
            enable_voice=os.getenv('ENABLE_VOICE', 'true').lower() == 'true',
            enable_vision=os.getenv('ENABLE_VISION', 'true').lower() == 'true',
            enable_web_search=os.getenv('ENABLE_WEB_SEARCH', 'false').lower() == 'true',
            enable_coworker_mode=os.getenv('ENABLE_COWORKER_MODE', 'false').lower() == 'true',
            enable_team_workspaces=os.getenv('ENABLE_TEAM_WORKSPACES', 'false').lower() == 'true',
            enable_metrics=os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
        )

    def _load_path_config(self) -> PathConfig:
        """Load path configuration"""
        return PathConfig(
            knowledge_base_path=os.getenv('KNOWLEDGE_BASE_PATH', './knowledge'),
            conversation_dir=os.getenv('CONVERSATION_DIR', './conversations'),
            project_dir=os.getenv('PROJECT_DIR', './data/adam_projects'),
            upload_dir=os.getenv('UPLOAD_DIR', './data/uploads'),
            temp_dir=os.getenv('TEMP_DIR', './data/temp'),
            backup_dir=os.getenv('BACKUP_DIR', './data/backups')
        )

    def _create_directories(self):
        """Create necessary directories"""
        dirs_to_create = [
            self.memory.path,
            self.memory.chromadb_persist_directory,
            self.paths.conversation_dir,
            self.paths.project_dir,
            self.paths.upload_dir,
            self.paths.temp_dir,
            self.paths.backup_dir,
            "./data"  # Base data directory
        ]

        for directory in dirs_to_create:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @property
    def api_keys(self) -> Dict[str, Optional[str]]:
        """Get all API keys"""
        return {
            'openai': os.getenv('OPENAI_API_KEY'),
            'xai': os.getenv('XAI_API_KEY'),
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'elevenlabs': os.getenv('ELEVENLABS_API_KEY')
        }

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider"""
        return self.api_keys.get(provider.lower())

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        # Check required API keys
        if not self.get_api_key('openai') and not self.get_api_key('xai'):
            issues.append("At least one of OPENAI_API_KEY or XAI_API_KEY is required")

        # Check database URL
        if not self.database.url:
            issues.append("DATABASE_URL is required")

        # Check secret key in production
        if not self.core['debug'] and self.security.session_secret_key == 'change-this-in-production':
            issues.append("SESSION_SECRET_KEY must be changed in production")

        return issues

    def __str__(self) -> str:
        """String representation of configuration"""
        return f"UnifiedConfig(name={self.core['name']}, version={self.core['version']}, debug={self.core['debug']})"


# Global configuration instance
_config: Optional[UnifiedConfig] = None


def get_config(env_file: Optional[str] = None) -> UnifiedConfig:
    """
    Get the global configuration instance

    Args:
        env_file: Path to .env file (only used on first call)

    Returns:
        UnifiedConfig instance
    """
    global _config
    if _config is None:
        _config = UnifiedConfig(env_file)
    return _config


def reload_config(env_file: Optional[str] = None) -> UnifiedConfig:
    """
    Reload the global configuration

    Args:
        env_file: Path to .env file

    Returns:
        New UnifiedConfig instance
    """
    global _config
    _config = UnifiedConfig(env_file)
    return _config


# Convenience alias
Config = get_config
