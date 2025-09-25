"""
ADAM Configuration Module
"""

from .unified import (
    UnifiedConfig,
    DatabaseConfig,
    MemoryConfig,
    LLMConfig,
    VoiceConfig,
    WebConfig,
    SecurityConfig,
    FeatureFlags,
    PathConfig,
    get_config,
    reload_config
)

# Legacy compatibility (import later to avoid circular import)
try:
    from ..config import Config as LegacyConfig
except ImportError:
    LegacyConfig = None

__all__ = [
    'UnifiedConfig',
    'DatabaseConfig',
    'MemoryConfig',
    'LLMConfig',
    'VoiceConfig',
    'WebConfig',
    'SecurityConfig',
    'FeatureFlags',
    'PathConfig',
    'get_config',
    'reload_config',
    # 'LegacyConfig'  # For backward compatibility - removed due to circular import
]

# Default config instance
Config = get_config