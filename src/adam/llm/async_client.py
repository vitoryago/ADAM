"""
Async-First LLM Client for ADAM
Standardized async patterns for all LLM operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, AsyncGenerator, Any
from dataclasses import dataclass
from enum import Enum
import json
import time

from adam.utils import (
    AsyncRetry,
    AsyncTimer,
    AsyncLoggingContext,
    safe_await,
    ensure_coroutine
)

logger = logging.getLogger(__name__)


# Import model-specific SDKs
try:
    from xai_sdk import AsyncClient as XAIAsyncClient
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    logger.warning("xai_sdk not installed. Run: pip install xai-sdk")

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not installed. Run: pip install openai")

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic not installed. Run: pip install anthropic")


@dataclass
class AsyncLLMResponse:
    """Unified async response format"""
    content: str
    model: str
    provider: str
    reasoning_content: Optional[str] = None
    total_tokens: int = 0
    reasoning_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    processing_time: float = 0.0
    raw_response: Optional[Dict] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AsyncLLMProvider(Enum):
    """Available async providers"""
    XAI = "xai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class AsyncLLMClient:
    """
    Async-first unified LLM client
    Implements standardized async patterns throughout
    """

    def __init__(self, config=None):
        """
        Initialize async LLM client

        Args:
            config: LLM configuration object
        """
        from adam.llm.config import LLMConfig
        self.config = config or LLMConfig()
        self.clients = {}
        self._client_locks = {}
        self._initialize_lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """
        Async initialization of clients
        Must be called before using the client
        """
        async with self._initialize_lock:
            if self._initialized:
                return

            async with AsyncLoggingContext(__name__, "client_initialization"):
                await self._initialize_async_clients()
                self._initialized = True

    async def _initialize_async_clients(self):
        """Initialize async API clients for available providers"""

        # Initialize xAI async client
        if XAI_AVAILABLE and self.config.get_api_key("grok"):
            try:
                self.clients[AsyncLLMProvider.XAI] = XAIAsyncClient(
                    api_host="api.x.ai",
                    api_key=self.config.get_api_key("grok"),
                    timeout=3600
                )
                self._client_locks[AsyncLLMProvider.XAI] = asyncio.Semaphore(5)  # Limit concurrent requests
                logger.info("Initialized xAI async client")
            except Exception as e:
                logger.error(f"Failed to initialize xAI client: {e}")

        # Initialize OpenAI async client
        if OPENAI_AVAILABLE and self.config.get_api_key("openai"):
            try:
                self.clients[AsyncLLMProvider.OPENAI] = AsyncOpenAI(
                    api_key=self.config.get_api_key("openai"),
                    timeout=300.0
                )
                self._client_locks[AsyncLLMProvider.OPENAI] = asyncio.Semaphore(10)
                logger.info("Initialized OpenAI async client")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")

        # Initialize Anthropic async client
        if ANTHROPIC_AVAILABLE and self.config.get_api_key("anthropic"):
            try:
                self.clients[AsyncLLMProvider.ANTHROPIC] = AsyncAnthropic(
                    api_key=self.config.get_api_key("anthropic"),
                    timeout=300.0
                )
                self._client_locks[AsyncLLMProvider.ANTHROPIC] = asyncio.Semaphore(5)
                logger.info("Initialized Anthropic async client")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")

    async def _ensure_initialized(self):
        """Ensure client is initialized"""
        if not self._initialized:
            await self.initialize()

    @AsyncRetry(max_attempts=3, base_delay=1.0)
    async def complete(
        self,
        prompt: str,
        model: str = "grok-4-fast-non-reasoning",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[AsyncLLMResponse, AsyncGenerator[str, None]]:
        """
        Get async completion from any configured model

        Args:
            prompt: User prompt
            model: Model name
            system_prompt: Optional system message
            temperature: Response randomness (0-1)
            max_tokens: Maximum response length
            stream: Whether to stream response
            **kwargs: Additional model-specific parameters

        Returns:
            AsyncLLMResponse object or async generator if streaming
        """
        await self._ensure_initialized()

        start_time = time.time()

        async with AsyncLoggingContext(
            __name__,
            "llm_completion",
            model=model,
            prompt_length=len(prompt),
            stream=stream
        ):
            provider = self._get_provider_for_model(model)

            if provider not in self.clients:
                raise ValueError(f"Provider {provider.value} not available or not configured")

            # Use semaphore to limit concurrent requests per provider
            async with self._client_locks[provider]:
                if stream:
                    return self._complete_streaming(provider, model, prompt, system_prompt, temperature, max_tokens, **kwargs)
                else:
                    response = await self._complete_single(provider, model, prompt, system_prompt, temperature, max_tokens, **kwargs)
                    response.processing_time = time.time() - start_time
                    return response

    async def _complete_single(
        self,
        provider: AsyncLLMProvider,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> AsyncLLMResponse:
        """Complete a single (non-streaming) request"""

        if provider == AsyncLLMProvider.XAI:
            return await self._complete_xai(model, prompt, system_prompt, temperature, max_tokens, **kwargs)
        elif provider == AsyncLLMProvider.OPENAI:
            return await self._complete_openai(model, prompt, system_prompt, temperature, max_tokens, **kwargs)
        elif provider == AsyncLLMProvider.ANTHROPIC:
            return await self._complete_anthropic(model, prompt, system_prompt, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _complete_streaming(
        self,
        provider: AsyncLLMProvider,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Complete a streaming request"""

        if provider == AsyncLLMProvider.XAI:
            async for chunk in self._stream_xai(model, prompt, system_prompt, temperature, max_tokens, **kwargs):
                yield chunk
        elif provider == AsyncLLMProvider.OPENAI:
            async for chunk in self._stream_openai(model, prompt, system_prompt, temperature, max_tokens, **kwargs):
                yield chunk
        elif provider == AsyncLLMProvider.ANTHROPIC:
            async for chunk in self._stream_anthropic(model, prompt, system_prompt, temperature, max_tokens, **kwargs):
                yield chunk
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _complete_xai(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> AsyncLLMResponse:
        """Complete request using xAI"""
        client = self.clients[AsyncLLMProvider.XAI]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
                **kwargs
            )

            return AsyncLLMResponse(
                content=response.choices[0].message.content,
                model=model,
                provider="xai",
                total_tokens=response.usage.total_tokens if hasattr(response, 'usage') else 0,
                completion_tokens=response.usage.completion_tokens if hasattr(response, 'usage') else 0,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
            )

        except Exception as e:
            logger.error(f"xAI completion failed: {e}")
            raise

    async def _complete_openai(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> AsyncLLMResponse:
        """Complete request using OpenAI"""
        client = self.clients[AsyncLLMProvider.OPENAI]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            return AsyncLLMResponse(
                content=response.choices[0].message.content,
                model=model,
                provider="openai",
                total_tokens=response.usage.total_tokens,
                completion_tokens=response.usage.completion_tokens,
                raw_response=response.model_dump()
            )

        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            raise

    async def _complete_anthropic(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> AsyncLLMResponse:
        """Complete request using Anthropic"""
        client = self.clients[AsyncLLMProvider.ANTHROPIC]

        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens or 4096,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )

            return AsyncLLMResponse(
                content=response.content[0].text,
                model=model,
                provider="anthropic",
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                completion_tokens=response.usage.output_tokens,
                raw_response=response.model_dump()
            )

        except Exception as e:
            logger.error(f"Anthropic completion failed: {e}")
            raise

    async def _stream_xai(self, model: str, prompt: str, system_prompt: Optional[str],
                          temperature: float, max_tokens: Optional[int], **kwargs) -> AsyncGenerator[str, None]:
        """Stream from xAI"""
        client = self.clients[AsyncLLMProvider.XAI]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            async for chunk in await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
                stream=True,
                **kwargs
            ):
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"xAI streaming failed: {e}")
            raise

    async def _stream_openai(self, model: str, prompt: str, system_prompt: Optional[str],
                            temperature: float, max_tokens: Optional[int], **kwargs) -> AsyncGenerator[str, None]:
        """Stream from OpenAI"""
        client = self.clients[AsyncLLMProvider.OPENAI]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            async for chunk in await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            ):
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise

    async def _stream_anthropic(self, model: str, prompt: str, system_prompt: Optional[str],
                               temperature: float, max_tokens: Optional[int], **kwargs) -> AsyncGenerator[str, None]:
        """Stream from Anthropic"""
        client = self.clients[AsyncLLMProvider.ANTHROPIC]

        try:
            async for chunk in await client.messages.create(
                model=model,
                max_tokens=max_tokens or 4096,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **kwargs
            ):
                if chunk.type == "content_block_delta" and hasattr(chunk.delta, 'text'):
                    yield chunk.delta.text
        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise

    def _get_provider_for_model(self, model: str) -> AsyncLLMProvider:
        """Get provider for a given model"""
        if model.startswith('grok-'):
            return AsyncLLMProvider.XAI
        elif model.startswith(('gpt-', 'o1-', 'o3')):
            return AsyncLLMProvider.OPENAI
        elif model.startswith('claude-'):
            return AsyncLLMProvider.ANTHROPIC
        else:
            # Default to XAI for unknown models (assuming they're Grok variants)
            return AsyncLLMProvider.XAI

    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all configured providers

        Returns:
            Dict mapping provider names to health status
        """
        await self._ensure_initialized()

        results = {}

        for provider in self.clients:
            try:
                # Make a simple test request
                response = await safe_await(
                    self._complete_single(provider, "grok-4-fast-non-reasoning", "test", None, 0.1, 10),
                    default=None,
                    timeout=10.0
                )
                results[provider.value] = response is not None
            except Exception as e:
                logger.warning(f"Health check failed for {provider.value}: {e}")
                results[provider.value] = False

        return results

    async def close(self):
        """Clean up resources"""
        for client in self.clients.values():
            if hasattr(client, 'close'):
                await safe_await(client.close(), default=None)

        self.clients.clear()
        self._client_locks.clear()
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# Global async client instance
_async_client: Optional[AsyncLLMClient] = None


async def get_async_client() -> AsyncLLMClient:
    """
    Get the global async LLM client instance

    Returns:
        AsyncLLMClient instance
    """
    global _async_client
    if _async_client is None:
        _async_client = AsyncLLMClient()
        await _async_client.initialize()
    return _async_client


async def close_global_client():
    """Close the global async client"""
    global _async_client
    if _async_client is not None:
        await _async_client.close()
        _async_client = None