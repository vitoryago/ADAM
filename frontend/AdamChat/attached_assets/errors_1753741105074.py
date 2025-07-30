#!/usr/bin/env python3
"""
ADAM Error Handling System
Provides custom exceptions, retry logic, and recovery mechanisms
"""

from typing import Optional, Type, Callable, Any
from functools import wraps
import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Categorize errors by severity for appropriate handling"""
    LOW = "low"          # Can be ignored or logged
    MEDIUM = "medium"    # Should be handled but not critical
    HIGH = "high"        # Must be handled, affects functionality
    CRITICAL = "critical" # System-breaking, requires immediate attention


class ADAMError(Exception):
    """Base exception for all ADAM errors"""
    severity = ErrorSeverity.MEDIUM
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        self.message = message
        self.cause = cause
        super().__init__(self.message)
        
    def __str__(self):
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message


# Memory-related errors
class MemoryError(ADAMError):
    """Base class for memory system errors"""
    pass


class StorageError(MemoryError):
    """Failed to store memory"""
    severity = ErrorSeverity.HIGH


class RetrievalError(MemoryError):
    """Failed to retrieve memory"""
    severity = ErrorSeverity.MEDIUM


class CorruptedMemoryError(MemoryError):
    """Memory data is corrupted"""
    severity = ErrorSeverity.CRITICAL


# Network and communication errors
class NetworkError(ADAMError):
    """Network-related errors"""
    severity = ErrorSeverity.HIGH


class LLMError(NetworkError):
    """LLM API errors"""
    pass


class EmbeddingError(NetworkError):
    """Embedding generation errors"""
    pass


# File system errors
class FileSystemError(ADAMError):
    """File system related errors"""
    severity = ErrorSeverity.HIGH


class LoadError(FileSystemError):
    """Failed to load from disk"""
    pass


class SaveError(FileSystemError):
    """Failed to save to disk"""
    pass


# Configuration errors
class ConfigurationError(ADAMError):
    """Configuration related errors"""
    severity = ErrorSeverity.CRITICAL


class MissingAPIKeyError(ConfigurationError):
    """Required API key not found"""
    pass


# Retry decorator with exponential backoff
def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        backoff_factor: Multiplication factor for delay
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback function called on each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


# Error recovery strategies
class RecoveryStrategy:
    """Base class for error recovery strategies"""
    
    def can_recover(self, error: Exception) -> bool:
        """Check if this strategy can handle the error"""
        raise NotImplementedError
    
    def recover(self, error: Exception, context: dict) -> Any:
        """Attempt to recover from the error"""
        raise NotImplementedError


class FallbackJSONRecovery(RecoveryStrategy):
    """Fallback to JSON storage when primary storage fails"""
    
    def __init__(self, fallback_path: str):
        self.fallback_path = fallback_path
        
    def can_recover(self, error: Exception) -> bool:
        return isinstance(error, (StorageError, NetworkError))
    
    def recover(self, error: Exception, context: dict) -> Any:
        """Save to JSON as fallback"""
        import json
        from pathlib import Path
        
        fallback_file = Path(self.fallback_path) / f"fallback_{context.get('memory_id', 'unknown')}.json"
        fallback_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(fallback_file, 'w') as f:
                json.dump(context.get('data', {}), f, indent=2, default=str)
            logger.info(f"Saved to fallback: {fallback_file}")
            return True
        except Exception as e:
            logger.error(f"Fallback recovery failed: {e}")
            return False


class ErrorHandler:
    """Central error handling and recovery coordinator"""
    
    def __init__(self):
        self.recovery_strategies: list[RecoveryStrategy] = []
        self.error_count = {}  # Track errors by type
        
    def add_recovery_strategy(self, strategy: RecoveryStrategy):
        """Add a recovery strategy"""
        self.recovery_strategies.append(strategy)
        
    def handle_error(self, error: Exception, context: Optional[dict] = None) -> Any:
        """
        Handle an error with appropriate recovery strategies
        
        Args:
            error: The exception that occurred
            context: Additional context for recovery
            
        Returns:
            Recovery result or re-raises if unrecoverable
        """
        error_type = type(error).__name__
        self.error_count[error_type] = self.error_count.get(error_type, 0) + 1
        
        # Log based on severity
        if isinstance(error, ADAMError):
            if error.severity == ErrorSeverity.CRITICAL:
                logger.critical(f"Critical error: {error}")
            elif error.severity == ErrorSeverity.HIGH:
                logger.error(f"High severity error: {error}")
            else:
                logger.warning(f"Error: {error}")
        else:
            logger.error(f"Unexpected error: {error}")
        
        # Try recovery strategies
        for strategy in self.recovery_strategies:
            if strategy.can_recover(error):
                try:
                    result = strategy.recover(error, context or {})
                    if result:
                        logger.info(f"Recovered using {strategy.__class__.__name__}")
                        return result
                except Exception as recovery_error:
                    logger.error(f"Recovery strategy failed: {recovery_error}")
        
        # No recovery possible
        raise error
    
    def get_error_stats(self) -> dict:
        """Get error statistics"""
        return {
            "error_counts": self.error_count.copy(),
            "total_errors": sum(self.error_count.values())
        }


# Context manager for error handling
class ErrorContext:
    """Context manager for handling errors in a specific scope"""
    
    def __init__(self, operation_name: str, handler: ErrorHandler, context: dict = None):
        self.operation_name = operation_name
        self.handler = handler
        self.context = context or {}
        
    def __enter__(self):
        logger.debug(f"Starting operation: {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            logger.error(f"Error in {self.operation_name}: {exc_val}")
            try:
                self.handler.handle_error(exc_val, self.context)
                return True  # Suppress the exception
            except Exception:
                return False  # Re-raise
        else:
            logger.debug(f"Completed operation: {self.operation_name}")
            return True