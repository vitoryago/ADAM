#!/usr/bin/env python3
"""
ADAM Migration Script
Helps transition from the old fragmented architecture to the unified system
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import shutil
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adam.database import run_migrations, create_tables
from adam.config import get_config, reload_config
from adam.database.migrations import migrate_adam_v2_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages the migration from old ADAM to unified ADAM"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.backup_dir = self.project_root / "backup_migration"

    async def run_full_migration(self):
        """Run the complete migration process"""
        logger.info("🚀 Starting ADAM Unified Architecture Migration")
        logger.info("=" * 60)

        try:
            # Step 1: Create backups
            await self._create_backups()

            # Step 2: Setup unified configuration
            await self._setup_unified_config()

            # Step 3: Setup unified database
            await self._setup_unified_database()

            # Step 4: Install package in development mode
            await self._install_package()

            # Step 5: Migrate data from old systems
            await self._migrate_existing_data()

            # Step 6: Create CLI shortcuts
            await self._setup_cli_commands()

            # Step 7: Cleanup old files (optional)
            await self._cleanup_old_files()

            logger.info("✅ Migration completed successfully!")
            logger.info("=" * 60)
            await self._show_next_steps()

        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            logger.error("Check the logs above for details")
            return False

        return True

    async def _create_backups(self):
        """Create backups of important files"""
        logger.info("📦 Creating backups...")

        self.backup_dir.mkdir(exist_ok=True)

        # Backup configuration files
        config_files = [
            ".env",
            "requirements.txt",
            "requirements_web.txt",
            "requirements_coworker.txt",
            "src/adam_v2/.env",
        ]

        for config_file in config_files:
            source = self.project_root / config_file
            if source.exists():
                dest = self.backup_dir / config_file.replace("/", "_")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                logger.info(f"  Backed up: {config_file}")

        # Backup databases
        db_files = [
            "adam_v2.db",
            "src/adam_v2/adam_v2.db",
            "src/adam_v2/data/adam_v2.db",
        ]

        for db_file in db_files:
            source = self.project_root / db_file
            if source.exists():
                dest = self.backup_dir / f"{db_file.replace('/', '_')}.backup"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                logger.info(f"  Backed up database: {db_file}")

        logger.info("✅ Backups created successfully")

    async def _setup_unified_config(self):
        """Setup the unified configuration"""
        logger.info("⚙️  Setting up unified configuration...")

        env_unified = self.project_root / ".env.unified"
        env_file = self.project_root / ".env"

        if env_unified.exists() and not env_file.exists():
            # Copy the unified template to .env
            shutil.copy2(env_unified, env_file)
            logger.info("  Created .env from .env.unified template")
            logger.warning("  ⚠️  Please update .env with your actual API keys!")
        elif env_file.exists():
            logger.info("  Using existing .env file")
        else:
            logger.error("  ❌ No configuration file found. Please create .env from .env.unified")
            raise FileNotFoundError("Configuration file not found")

        # Reload configuration
        reload_config()
        config = get_config()

        # Validate configuration
        issues = config.validate_config()
        if issues:
            logger.warning("  Configuration issues found:")
            for issue in issues:
                logger.warning(f"    - {issue}")

        logger.info("✅ Unified configuration setup complete")

    async def _setup_unified_database(self):
        """Setup the unified database with migrations"""
        logger.info("🗄️  Setting up unified database...")

        try:
            # Create tables
            await create_tables()
            logger.info("  Created database tables")

            # Run migrations
            migrations_applied = await run_migrations()
            logger.info(f"  Applied {migrations_applied} migrations")

        except Exception as e:
            logger.error(f"  Database setup failed: {e}")
            raise

        logger.info("✅ Unified database setup complete")

    async def _install_package(self):
        """Install the ADAM package in development mode"""
        logger.info("📦 Installing ADAM package in development mode...")

        import subprocess

        try:
            # Install in development mode
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("  Package installed successfully")

            # Install consolidated requirements
            if (self.project_root / "requirements-consolidated.txt").exists():
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", "requirements-consolidated.txt"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info("  Dependencies installed from consolidated requirements")

        except subprocess.CalledProcessError as e:
            logger.error(f"  Package installation failed: {e}")
            logger.error(f"  Stdout: {e.stdout}")
            logger.error(f"  Stderr: {e.stderr}")
            raise

        logger.info("✅ Package installation complete")

    async def _migrate_existing_data(self):
        """Migrate data from existing ADAM v2 databases"""
        logger.info("📊 Migrating existing data...")

        try:
            await migrate_adam_v2_data()
            logger.info("  Data migration completed")
        except Exception as e:
            logger.warning(f"  Data migration encountered issues: {e}")
            logger.warning("  You may need to manually import your data")

        logger.info("✅ Data migration complete")

    async def _setup_cli_commands(self):
        """Create CLI command shortcuts"""
        logger.info("🖥️  Setting up CLI commands...")

        try:
            # Test the CLI commands
            import subprocess

            commands_to_test = ["adam-chat", "adam-complete", "adam-server"]
            available_commands = []

            for cmd in commands_to_test:
                try:
                    result = subprocess.run(
                        [cmd, "--help"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        available_commands.append(cmd)
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass

            if available_commands:
                logger.info(f"  Available CLI commands: {', '.join(available_commands)}")
            else:
                logger.warning("  No CLI commands available yet")
                logger.warning("  Run 'pip install -e .' to enable CLI commands")

        except Exception as e:
            logger.warning(f"  CLI setup check failed: {e}")

        logger.info("✅ CLI setup complete")

    async def _cleanup_old_files(self):
        """Cleanup old configuration files (optional)"""
        logger.info("🧹 Cleaning up old files...")

        # Files that can be safely removed after migration
        old_files = [
            "requirements_web.txt",
            "requirements_coworker.txt",
        ]

        for old_file in old_files:
            file_path = self.project_root / old_file
            if file_path.exists():
                response = input(f"Remove old file {old_file}? [y/N]: ")
                if response.lower() == 'y':
                    file_path.unlink()
                    logger.info(f"  Removed: {old_file}")
                else:
                    logger.info(f"  Kept: {old_file}")

        logger.info("✅ Cleanup complete")

    async def _show_next_steps(self):
        """Show next steps to the user"""
        logger.info("🎉 Next Steps:")
        logger.info("=" * 60)
        logger.info("1. Update your .env file with actual API keys")
        logger.info("2. Test the new CLI commands:")
        logger.info("   - adam-chat          # Simple chat interface")
        logger.info("   - adam-complete      # Full interface with memory")
        logger.info("   - adam-server        # Start the FastAPI server")
        logger.info("3. Start the web interface:")
        logger.info("   - streamlit run src/adam/web/app.py")
        logger.info("4. Check the unified configuration:")
        logger.info("   - python -c \"from adam.config import get_config; print(get_config())\"")
        logger.info("5. Test database connectivity:")
        logger.info("   - python -c \"from adam.database import get_engine; import asyncio; asyncio.run(get_engine().health_check())\"")
        logger.info("")
        logger.info("📚 Documentation and help:")
        logger.info("   - Check CLAUDE.md for detailed setup instructions")
        logger.info("   - Backups are stored in: backup_migration/")
        logger.info("")
        logger.info("🐛 If you encounter issues:")
        logger.info("   - Check logs above for specific error messages")
        logger.info("   - Restore from backups if needed")
        logger.info("   - Check API key configuration in .env")


async def main():
    """Main migration entry point"""
    print("ADAM Unified Architecture Migration")
    print("=" * 40)
    print("This script will migrate your ADAM installation to the new unified architecture.")
    print("It will:")
    print("  1. Create backups of your current configuration and data")
    print("  2. Setup the unified configuration system")
    print("  3. Setup the unified database with migrations")
    print("  4. Install the ADAM package in development mode")
    print("  5. Migrate existing data")
    print("  6. Setup CLI commands")
    print("")

    response = input("Do you want to proceed? [y/N]: ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        return

    print("")
    manager = MigrationManager()
    success = await manager.run_full_migration()

    if success:
        print("\n🎉 Migration completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Migration failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())