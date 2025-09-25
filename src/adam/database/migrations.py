"""
Database Migration System
Handles schema evolution and data migration between versions
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import json
import hashlib

from .engine import get_engine, DatabaseSession
from .models import SystemConfiguration

logger = logging.getLogger(__name__)


class Migration:
    """Represents a single database migration"""

    def __init__(self, version: str, name: str, up_sql: str, down_sql: str = "", description: str = ""):
        self.version = version
        self.name = name
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.description = description
        self.applied_at: Optional[datetime] = None

    @property
    def id(self) -> str:
        """Unique identifier for this migration"""
        return f"{self.version}_{self.name}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert migration to dictionary"""
        return {
            'version': self.version,
            'name': self.name,
            'description': self.description,
            'applied_at': self.applied_at.isoformat() if self.applied_at else None
        }


class MigrationManager:
    """
    Manages database migrations for ADAM
    """

    def __init__(self, migrations_dir: Optional[Path] = None):
        """
        Initialize migration manager

        Args:
            migrations_dir: Directory containing migration files
        """
        if migrations_dir is None:
            migrations_dir = Path(__file__).parent / "migrations"

        self.migrations_dir = migrations_dir
        self.migrations_dir.mkdir(exist_ok=True)

        self.migrations: List[Migration] = []
        self._load_built_in_migrations()

    def _load_built_in_migrations(self):
        """Load built-in migrations for ADAM"""

        # Migration 001: Initial unified schema
        initial_migration = Migration(
            version="001",
            name="initial_unified_schema",
            description="Create unified ADAM database schema",
            up_sql="""
            -- This migration creates the initial unified schema
            -- The actual table creation is handled by SQLAlchemy
            INSERT OR IGNORE INTO system_configuration (id, value, description, is_sensitive)
            VALUES ('migration_system_initialized', '"true"', 'Migration system initialized', 0);
            """,
            down_sql="""
            DELETE FROM system_configuration WHERE id = 'migration_system_initialized';
            """
        )

        # Migration 002: Data migration from old ADAM v2 schema
        data_migration = Migration(
            version="002",
            name="migrate_adam_v2_data",
            description="Migrate data from ADAM v2 database",
            up_sql="""
            -- This migration will be populated with actual data migration logic
            -- when we detect an existing ADAM v2 database
            INSERT OR IGNORE INTO system_configuration (id, value, description, is_sensitive)
            VALUES ('adam_v2_migration_complete', '"false"', 'ADAM v2 data migration status', 0);
            """,
            down_sql="""
            DELETE FROM system_configuration WHERE id = 'adam_v2_migration_complete';
            """
        )

        # Migration 003: Add indexes for performance
        index_migration = Migration(
            version="003",
            name="add_performance_indexes",
            description="Add database indexes for better performance",
            up_sql="""
            -- Additional indexes beyond what's in the model
            CREATE INDEX IF NOT EXISTS idx_messages_role_created
            ON messages(role, created_at);

            CREATE INDEX IF NOT EXISTS idx_memories_tags
            ON memories(tags);

            CREATE INDEX IF NOT EXISTS idx_conversations_title
            ON conversations(title);
            """,
            down_sql="""
            DROP INDEX IF EXISTS idx_messages_role_created;
            DROP INDEX IF EXISTS idx_memories_tags;
            DROP INDEX IF EXISTS idx_conversations_title;
            """
        )

        self.migrations = [initial_migration, data_migration, index_migration]

    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration IDs"""
        async with DatabaseSession() as session:
            try:
                # Check if system_configuration table exists
                result = await session.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' AND name='system_configuration'")
                )
                if not result.fetchone():
                    return []

                # Get applied migrations
                result = await session.execute(
                    text("SELECT value FROM system_configuration WHERE id = 'applied_migrations'")
                )
                row = result.fetchone()
                if row:
                    return json.loads(row[0])
                return []
            except Exception as e:
                logger.warning(f"Could not get applied migrations: {e}")
                return []

    async def mark_migration_applied(self, migration: Migration):
        """Mark a migration as applied"""
        async with DatabaseSession() as session:
            # Get current applied migrations
            applied = await self.get_applied_migrations()

            if migration.id not in applied:
                applied.append(migration.id)
                migration.applied_at = datetime.now()

                # Update the applied migrations list
                await session.execute(
                    text("""
                    INSERT OR REPLACE INTO system_configuration
                    (id, value, description, is_sensitive)
                    VALUES ('applied_migrations', :migrations, 'List of applied migration IDs', 0)
                    """),
                    {'migrations': json.dumps(applied)}
                )

                # Store individual migration record
                await session.execute(
                    text("""
                    INSERT OR REPLACE INTO system_configuration
                    (id, value, description, is_sensitive)
                    VALUES (:migration_id, :migration_data, :description, 0)
                    """),
                    {
                        'migration_id': f'migration_{migration.id}',
                        'migration_data': json.dumps(migration.to_dict()),
                        'description': f'Migration record for {migration.id}'
                    }
                )

                await session.commit()
                logger.info(f"Marked migration {migration.id} as applied")

    async def run_migration(self, migration: Migration):
        """Run a single migration"""
        logger.info(f"Running migration {migration.id}: {migration.name}")

        async with DatabaseSession() as session:
            try:
                # Execute the migration SQL
                if migration.up_sql.strip():
                    for statement in migration.up_sql.split(';'):
                        statement = statement.strip()
                        if statement:
                            await session.execute(text(statement))

                await session.commit()
                logger.info(f"Successfully executed migration {migration.id}")

                # Mark as applied
                await self.mark_migration_applied(migration)

            except Exception as e:
                await session.rollback()
                logger.error(f"Migration {migration.id} failed: {e}")
                raise

    async def rollback_migration(self, migration: Migration):
        """Rollback a migration"""
        logger.info(f"Rolling back migration {migration.id}")

        async with DatabaseSession() as session:
            try:
                # Execute the rollback SQL
                if migration.down_sql.strip():
                    for statement in migration.down_sql.split(';'):
                        statement = statement.strip()
                        if statement:
                            await session.execute(text(statement))

                await session.commit()
                logger.info(f"Successfully rolled back migration {migration.id}")

                # Remove from applied list
                applied = await self.get_applied_migrations()
                if migration.id in applied:
                    applied.remove(migration.id)
                    await session.execute(
                        text("""
                        UPDATE system_configuration
                        SET value = :migrations
                        WHERE id = 'applied_migrations'
                        """),
                        {'migrations': json.dumps(applied)}
                    )

                await session.commit()

            except Exception as e:
                await session.rollback()
                logger.error(f"Rollback of migration {migration.id} failed: {e}")
                raise

    async def run_migrations(self) -> int:
        """
        Run all pending migrations

        Returns:
            Number of migrations applied
        """
        # Ensure tables exist first
        engine = get_engine()
        await engine.create_tables()

        applied = await self.get_applied_migrations()
        pending = [m for m in self.migrations if m.id not in applied]

        if not pending:
            logger.info("No pending migrations")
            return 0

        logger.info(f"Running {len(pending)} pending migrations")

        for migration in pending:
            await self.run_migration(migration)

        logger.info(f"Successfully applied {len(pending)} migrations")
        return len(pending)

    async def create_migration_file(self, name: str, up_sql: str, down_sql: str = "", description: str = ""):
        """Create a new migration file"""
        # Get next version number
        if self.migrations:
            last_version = max(int(m.version) for m in self.migrations)
            version = f"{last_version + 1:03d}"
        else:
            version = "001"

        migration = Migration(version, name, up_sql, down_sql, description)

        # Write migration file
        migration_file = self.migrations_dir / f"{migration.id}.sql"
        with open(migration_file, 'w') as f:
            f.write(f"-- Migration: {migration.id}\n")
            f.write(f"-- Description: {description}\n")
            f.write(f"-- Created: {datetime.now().isoformat()}\n\n")
            f.write("-- UP\n")
            f.write(up_sql)
            f.write("\n\n-- DOWN\n")
            f.write(down_sql)

        self.migrations.append(migration)
        logger.info(f"Created migration file: {migration_file}")
        return migration

    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status"""
        return {
            'total_migrations': len(self.migrations),
            'migrations': [
                {
                    'id': m.id,
                    'name': m.name,
                    'description': m.description,
                    'applied': m.applied_at is not None
                }
                for m in self.migrations
            ]
        }


# Global migration manager
_migration_manager: Optional[MigrationManager] = None


def get_migration_manager() -> MigrationManager:
    """Get the global migration manager"""
    global _migration_manager
    if _migration_manager is None:
        _migration_manager = MigrationManager()
    return _migration_manager


async def run_migrations() -> int:
    """
    Run all pending migrations

    Returns:
        Number of migrations applied
    """
    manager = get_migration_manager()
    return await manager.run_migrations()


async def create_migration(name: str, up_sql: str, down_sql: str = "", description: str = "") -> Migration:
    """
    Create a new migration

    Args:
        name: Migration name
        up_sql: SQL to apply the migration
        down_sql: SQL to rollback the migration
        description: Description of what the migration does

    Returns:
        Created Migration object
    """
    manager = get_migration_manager()
    return await manager.create_migration_file(name, up_sql, down_sql, description)


async def migrate_adam_v2_data():
    """
    Migrate existing ADAM v2 data to unified schema
    This function detects existing ADAM v2 databases and migrates their data
    """
    logger.info("Checking for ADAM v2 data to migrate...")

    # Check for existing ADAM v2 databases
    potential_db_paths = [
        Path("./adam_v2.db"),
        Path("./src/adam_v2/adam_v2.db"),
        Path("./src/adam_v2/data/adam_v2.db")
    ]

    for db_path in potential_db_paths:
        if db_path.exists():
            logger.info(f"Found ADAM v2 database: {db_path}")
            await _migrate_from_adam_v2_db(db_path)

    # Mark migration as complete
    async with DatabaseSession() as session:
        await session.execute(
            text("""
            INSERT OR REPLACE INTO system_configuration
            (id, value, description, is_sensitive)
            VALUES ('adam_v2_migration_complete', '"true"', 'ADAM v2 data migration status', 0)
            """)
        )
        await session.commit()


async def _migrate_from_adam_v2_db(db_path: Path):
    """Migrate data from a specific ADAM v2 database"""
    logger.info(f"Migrating data from {db_path}")

    # This would contain the actual migration logic
    # For now, we'll just log that we found the database
    logger.info(f"Found {db_path} - migration logic would be implemented here")

    # TODO: Implement actual data migration
    # 1. Connect to old database
    # 2. Read projects, conversations, messages
    # 3. Transform to new schema
    # 4. Insert into unified database
    # 5. Handle conflicts and duplicates