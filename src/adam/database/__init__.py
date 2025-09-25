"""
Unified Database Module for ADAM
Consolidates all database operations and provides migration support
"""

from .models import (
    Base,
    Project,
    Conversation,
    Message,
    Memory,
    MemoryConnection,
    SystemConfiguration,
    UserSession,
    UsageMetrics,
    MessageRole,
    MemoryType
)
from .engine import (
    DatabaseEngine,
    get_engine,
    create_tables,
    get_session
)
from .migrations import (
    MigrationManager,
    run_migrations,
    create_migration
)

__all__ = [
    # Models
    'Base',
    'Project',
    'Conversation',
    'Message',
    'Memory',
    'MemoryConnection',
    'SystemConfiguration',
    'UserSession',
    'UsageMetrics',
    'MessageRole',
    'MemoryType',

    # Engine
    'DatabaseEngine',
    'get_engine',
    'create_tables',
    'get_session',

    # Migrations
    'MigrationManager',
    'run_migrations',
    'create_migration'
]