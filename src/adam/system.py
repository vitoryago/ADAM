"""
ADAM System - Main system class for unified architecture
"""

import asyncio
from typing import Optional

from adam.config import get_config
from adam.database import get_engine
from adam.llm.async_client import AsyncLLMClient


class ADAMSystem:
    """
    Main ADAM system class
    Provides a simple interface to the unified architecture
    """

    def __init__(self, config=None):
        """
        Initialize ADAM system

        Args:
            config: Configuration object (optional)
        """
        self.config = config or get_config()
        self.engine = get_engine()
        self.llm_client: Optional[AsyncLLMClient] = None

    async def initialize(self):
        """Initialize async components"""
        if self.llm_client is None:
            self.llm_client = AsyncLLMClient()
            await self.llm_client.initialize()

    async def complete(self, prompt: str, **kwargs) -> str:
        """
        Simple completion method

        Args:
            prompt: User prompt
            **kwargs: Additional arguments

        Returns:
            Response string
        """
        await self.initialize()

        response = await self.llm_client.complete(prompt, **kwargs)
        if hasattr(response, 'content'):
            return response.content
        return str(response)

    async def close(self):
        """Clean up resources"""
        if self.llm_client:
            await self.llm_client.close()

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# Backward compatibility
class ADAMMemoryAdvanced:
    """Legacy memory system for backward compatibility"""

    def __init__(self, persist_directory=None):
        self.persist_directory = persist_directory

    def search(self, query, k=5):
        """Mock search method"""
        return []

    def add_memory(self, content, metadata=None):
        """Mock add memory method"""
        pass

    def get_statistics(self):
        """Mock statistics method"""
        return {
            'total_memories': 0,
            'collections': 0,
            'last_updated': 'N/A'
        }