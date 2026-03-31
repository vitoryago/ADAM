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
    from xai_sdk import Client as XAIClient
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
    GEMINI = "gemini"
    LOCAL = "local"


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
        self._local_provider = None

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

        from adam.llm.config import ModelProvider

        # Initialize xAI client (synchronous, will wrap in async)
        if XAI_AVAILABLE and self.config.get_api_key(ModelProvider.GROK):
            try:
                self.clients[AsyncLLMProvider.XAI] = XAIClient(
                    api_host="api.x.ai",
                    api_key=self.config.get_api_key(ModelProvider.GROK)
                )
                self._client_locks[AsyncLLMProvider.XAI] = asyncio.Semaphore(5)  # Limit concurrent requests
                logger.info("Initialized xAI async client")
            except Exception as e:
                logger.error(f"Failed to initialize xAI client: {e}")

        # Initialize OpenAI async client
        if OPENAI_AVAILABLE and self.config.get_api_key(ModelProvider.OPENAI):
            try:
                self.clients[AsyncLLMProvider.OPENAI] = AsyncOpenAI(
                    api_key=self.config.get_api_key(ModelProvider.OPENAI),
                    timeout=300.0
                )
                self._client_locks[AsyncLLMProvider.OPENAI] = asyncio.Semaphore(10)
                logger.info("Initialized OpenAI async client")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")

        # Initialize Anthropic async client
        if ANTHROPIC_AVAILABLE and self.config.get_api_key(ModelProvider.ANTHROPIC):
            try:
                self.clients[AsyncLLMProvider.ANTHROPIC] = AsyncAnthropic(
                    api_key=self.config.get_api_key(ModelProvider.ANTHROPIC),
                    timeout=300.0
                )
                self._client_locks[AsyncLLMProvider.ANTHROPIC] = asyncio.Semaphore(5)
                logger.info("Initialized Anthropic async client")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")

        # Initialize Gemini async client (via OpenAI-compatible endpoint)
        if OPENAI_AVAILABLE and self.config.get_api_key(ModelProvider.GEMINI):
            try:
                import os
                gemini_api_key = self.config.get_api_key(ModelProvider.GEMINI)
                self.clients[AsyncLLMProvider.GEMINI] = AsyncOpenAI(
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    api_key=gemini_api_key,
                    timeout=300.0
                )
                self._client_locks[AsyncLLMProvider.GEMINI] = asyncio.Semaphore(5)
                logger.info("Initialized Gemini async client")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")

        # Initialize local model provider
        import os
        local_enabled = os.getenv("LOCAL_MODEL_ENABLED", "true").lower() == "true"
        if local_enabled and OPENAI_AVAILABLE:
            try:
                from adam.llm.local_provider import LocalModelProvider
                self._local_provider = LocalModelProvider()
                await self._local_provider.discover()
                seen_urls = set()
                for model in self._local_provider.models.values():
                    if model.base_url not in seen_urls:
                        seen_urls.add(model.base_url)
                        self.clients[(AsyncLLMProvider.LOCAL, model.base_url)] = AsyncOpenAI(
                            base_url=model.base_url,
                            api_key="not-needed",
                            timeout=300.0,
                        )
                if self._local_provider.models:
                    self._client_locks[AsyncLLMProvider.LOCAL] = asyncio.Semaphore(3)
                    logger.info("Initialized LOCAL provider with %d model(s)", len(self._local_provider.models))
                    await self._local_provider.start_health_checks()
            except Exception as e:
                logger.error(f"Failed to initialize local provider: {e}")
                self._local_provider = LocalModelProvider(endpoints=[])

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

            # LOCAL clients are stored with tuple keys (LOCAL, base_url)
            if provider == AsyncLLMProvider.LOCAL:
                if not self._local_provider or not self._local_provider.has_model(model):
                    raise ValueError(f"Provider {provider.value} not available or not configured")
            elif provider not in self.clients:
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
        elif provider == AsyncLLMProvider.GEMINI:
            return await self._complete_gemini(model, prompt, system_prompt, temperature, max_tokens, **kwargs)
        elif provider == AsyncLLMProvider.LOCAL:
            return await self._complete_local(model, prompt, system_prompt, temperature, max_tokens, **kwargs)
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
        elif provider == AsyncLLMProvider.GEMINI:
            async for chunk in self._stream_gemini(model, prompt, system_prompt, temperature, max_tokens, **kwargs):
                yield chunk
        elif provider == AsyncLLMProvider.LOCAL:
            async for chunk in self._stream_local(model, prompt, system_prompt, temperature, max_tokens, **kwargs):
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
        from xai_sdk.chat import user, system

        client = self.clients[AsyncLLMProvider.XAI]

        messages = []
        if system_prompt:
            messages.append(system(system_prompt))
        messages.append(user(prompt))

        try:
            # XAI SDK uses synchronous API, wrap in asyncio
            import asyncio

            # Create the chat instance and sample it
            def _create_and_sample():
                chat = client.chat.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens or 4096,
                    **kwargs
                )
                return chat.sample()

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                _create_and_sample
            )

            return AsyncLLMResponse(
                content=response.content,
                model=model,
                provider="xai",
                total_tokens=response.usage.total_tokens if hasattr(response, 'usage') else 0,
                completion_tokens=response.usage.completion_tokens if hasattr(response, 'usage') else 0,
                raw_response=None  # XAI Response doesn't have model_dump
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

    async def _complete_gemini(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> AsyncLLMResponse:
        """Complete request using Gemini (via OpenAI-compatible endpoint)"""
        client = self.clients[AsyncLLMProvider.GEMINI]

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
                provider="gemini",
                total_tokens=response.usage.total_tokens,
                completion_tokens=response.usage.completion_tokens,
                raw_response=response.model_dump()
            )

        except Exception as e:
            logger.error(f"Gemini completion failed: {e}")
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

    async def _stream_gemini(self, model: str, prompt: str, system_prompt: Optional[str],
                             temperature: float, max_tokens: Optional[int], **kwargs) -> AsyncGenerator[str, None]:
        """Stream from Gemini (via OpenAI-compatible endpoint)"""
        client = self.clients[AsyncLLMProvider.GEMINI]

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
            logger.error(f"Gemini streaming failed: {e}")
            raise

    async def _complete_local(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> AsyncLLMResponse:
        """Complete request using a local model (via OpenAI-compatible endpoint)"""
        base_url = self._local_provider.get_base_url(model)
        client = self.clients.get((AsyncLLMProvider.LOCAL, base_url))
        if not client:
            raise ValueError(f"No local client configured for model {model} at {base_url}")

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

            total_tokens = getattr(response.usage, 'total_tokens', 0) if response.usage else 0
            completion_tokens = getattr(response.usage, 'completion_tokens', 0) if response.usage else 0

            return AsyncLLMResponse(
                content=response.choices[0].message.content,
                model=model,
                provider="local",
                total_tokens=total_tokens,
                completion_tokens=completion_tokens,
                cost=0.0,  # Local models are free
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            )

        except Exception as e:
            logger.error(f"Local completion failed for {model}: {e}")
            raise

    async def _stream_local(self, model: str, prompt: str, system_prompt: Optional[str],
                            temperature: float, max_tokens: Optional[int], **kwargs) -> AsyncGenerator[str, None]:
        """Stream from a local model (via OpenAI-compatible endpoint)"""
        base_url = self._local_provider.get_base_url(model)
        client = self.clients.get((AsyncLLMProvider.LOCAL, base_url))
        if not client:
            raise ValueError(f"No local client configured for model {model} at {base_url}")

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
            logger.error(f"Local streaming failed for {model}: {e}")
            raise

    def _get_provider_for_model(self, model: str) -> AsyncLLMProvider:
        """Get provider for a given model"""
        if model.startswith('grok-'):
            return AsyncLLMProvider.XAI
        elif model.startswith(('gpt-', 'o1-', 'o3')):
            return AsyncLLMProvider.OPENAI
        elif model.startswith('claude-'):
            return AsyncLLMProvider.ANTHROPIC
        elif model.startswith('gemini-'):
            return AsyncLLMProvider.GEMINI
        else:
            local = getattr(self, '_local_provider', None)
            if local and local.has_model(model):
                return AsyncLLMProvider.LOCAL
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
        # Stop local provider health checks
        if self._local_provider:
            await safe_await(self._local_provider.stop_health_checks(), default=None)
            self._local_provider = None

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