"""
Async Utilities for ADAM
Provides standardized async patterns and utilities
"""

import asyncio
import logging
import functools
from typing import Any, Callable, Coroutine, Optional, TypeVar, Awaitable, Union
from concurrent.futures import ThreadPoolExecutor
import time
import inspect

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AsyncPatternError(Exception):
    """Raised when async patterns are used incorrectly"""
    pass


def ensure_async(func: Union[Callable[..., T], Callable[..., Awaitable[T]]]) -> Callable[..., Awaitable[T]]:
    """
    Decorator that ensures a function is async-compatible
    If the function is sync, it runs it in a thread pool
    """
    if asyncio.iscoroutinefunction(func):
        return func

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> T:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, functools.partial(func, **kwargs), *args)

    return async_wrapper


def run_sync_in_async(func: Callable[..., T], *args, **kwargs) -> Awaitable[T]:
    """
    Run a synchronous function in an async context using thread pool

    Args:
        func: Synchronous function to run
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Awaitable result
    """
    if asyncio.iscoroutinefunction(func):
        logger.warning(f"Function {func.__name__} is already async, running directly")
        return func(*args, **kwargs)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return loop.run_in_executor(executor, functools.partial(func, **kwargs), *args)


async def run_with_timeout(coro: Coroutine, timeout_seconds: float, default_value: Any = None) -> Any:
    """
    Run a coroutine with a timeout

    Args:
        coro: Coroutine to run
        timeout_seconds: Timeout in seconds
        default_value: Value to return if timeout occurs

    Returns:
        Result of coroutine or default_value if timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"Coroutine timed out after {timeout_seconds}s, returning default value")
        return default_value


class AsyncRetry:
    """Async retry decorator with exponential backoff"""

    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_multiplier: float = 2.0,
                 exceptions: tuple = (Exception,)):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.exceptions = exceptions

    def __call__(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        if not asyncio.iscoroutinefunction(func):
            raise AsyncPatternError(f"AsyncRetry can only be applied to async functions, got {func.__name__}")

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(1, self.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except self.exceptions as e:
                    last_exception = e
                    if attempt == self.max_attempts:
                        break

                    delay = min(self.base_delay * (self.backoff_multiplier ** (attempt - 1)), self.max_delay)
                    logger.warning(f"Attempt {attempt} failed for {func.__name__}, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)

            logger.error(f"All {self.max_attempts} attempts failed for {func.__name__}")
            raise last_exception

        return wrapper


async def gather_with_limit(tasks: list, limit: int = 10) -> list:
    """
    Run multiple async tasks with a concurrency limit

    Args:
        tasks: List of coroutines or awaitables
        limit: Maximum number of concurrent tasks

    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(limit)

    async def run_with_semaphore(task):
        async with semaphore:
            return await task

    limited_tasks = [run_with_semaphore(task) for task in tasks]
    return await asyncio.gather(*limited_tasks)


class AsyncContextManager:
    """Base class for async context managers with proper error handling"""

    async def __aenter__(self):
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup in {self.__class__.__name__}: {e}")
            if exc_type is None:  # Only re-raise if no other exception is being handled
                raise

    async def setup(self):
        """Override this method to implement setup logic"""
        pass

    async def cleanup(self):
        """Override this method to implement cleanup logic"""
        pass


class AsyncTimer:
    """Context manager for timing async operations"""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = 0

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        logger.debug(f"Async {self.name} took {duration:.2f} seconds")


def deprecated_sync(replacement: str):
    """
    Decorator to mark synchronous functions as deprecated
    Encourages migration to async patterns
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.warning(
                f"Function {func.__name__} is deprecated (synchronous). "
                f"Use {replacement} instead for better async compatibility."
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


async def ensure_coroutine(obj: Union[T, Awaitable[T]]) -> T:
    """
    Ensure an object is awaited if it's a coroutine

    Args:
        obj: Object that might be a coroutine

    Returns:
        The awaited result
    """
    if inspect.iscoroutine(obj) or inspect.isawaitable(obj):
        return await obj
    return obj


class AsyncLock:
    """Enhanced async lock with timeout and context info"""

    def __init__(self, name: str = "unnamed_lock", timeout: Optional[float] = None):
        self.name = name
        self.timeout = timeout
        self._lock = asyncio.Lock()
        self._acquired_at = None
        self._acquired_by = None

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire the lock with optional timeout"""
        timeout = timeout or self.timeout

        try:
            if timeout:
                await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
            else:
                await self._lock.acquire()

            self._acquired_at = time.time()
            self._acquired_by = asyncio.current_task()
            logger.debug(f"AsyncLock '{self.name}' acquired by {self._acquired_by}")
            return True

        except asyncio.TimeoutError:
            logger.warning(f"AsyncLock '{self.name}' acquisition timed out after {timeout}s")
            return False

    def release(self):
        """Release the lock"""
        if self._lock.locked():
            duration = time.time() - self._acquired_at if self._acquired_at else 0
            logger.debug(f"AsyncLock '{self.name}' held for {duration:.2f}s")
            self._lock.release()
            self._acquired_at = None
            self._acquired_by = None

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


# Async-first logging context
class AsyncLoggingContext:
    """Async context manager for structured logging"""

    def __init__(self, logger_name: str, operation: str, **context):
        self.logger = logging.getLogger(logger_name)
        self.operation = operation
        self.context = context
        self.start_time = None

    async def __aenter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}", extra=self.context)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation} in {duration:.2f}s",
                extra={**self.context, 'duration': duration}
            )
        else:
            self.logger.error(
                f"Failed {self.operation} after {duration:.2f}s: {exc_val}",
                extra={**self.context, 'duration': duration, 'error': str(exc_val)}
            )


# Convenience functions for common patterns
async def safe_await(awaitable: Awaitable[T], default: T = None, timeout: float = None) -> T:
    """
    Safely await a coroutine with error handling and optional timeout

    Args:
        awaitable: The awaitable to execute
        default: Default value to return on error
        timeout: Optional timeout in seconds

    Returns:
        Result of awaitable or default value
    """
    try:
        if timeout:
            return await asyncio.wait_for(awaitable, timeout=timeout)
        else:
            return await awaitable
    except Exception as e:
        logger.warning(f"safe_await caught exception: {e}, returning default")
        return default


async def async_map(func: Callable[[T], Awaitable[Any]], items: list[T], limit: int = 10) -> list:
    """
    Async version of map with concurrency limit

    Args:
        func: Async function to apply to each item
        items: List of items to process
        limit: Maximum concurrent executions

    Returns:
        List of results
    """
    tasks = [func(item) for item in items]
    return await gather_with_limit(tasks, limit=limit)


# Pattern enforcement
def enforce_async_only(func):
    """
    Decorator that ensures a function is only called from async context
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Check if we're in an async context
            asyncio.current_task()
        except RuntimeError:
            raise AsyncPatternError(
                f"Function {func.__name__} can only be called from async context. "
                f"Use asyncio.run() or call from within an async function."
            )
        return func(*args, **kwargs)
    return wrapper