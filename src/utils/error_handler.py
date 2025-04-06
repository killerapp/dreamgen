"""
Centralized error handling for the image generation system.
"""
from typing import Optional, Type, Callable, List, Dict, Any
import functools
import logging
import traceback
import os
import sys
from pathlib import Path
import inspect
import time

# Create logs directory
log_dir = Path('logs')
log_dir.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'error.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ImageGenError(Exception):
    """Base exception class for image generation errors."""
    def __init__(self, message: str, retriable: bool = True, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.retriable = retriable
        self.original_error = original_error
        self.timestamp = time.time()
        self.traceback = traceback.format_exc()

class ModelError(ImageGenError):
    """Errors related to model loading or inference."""
    pass

class PromptError(ImageGenError):
    """Errors related to prompt generation or validation."""
    pass

class ResourceError(ImageGenError):
    """Errors related to system resources (memory, GPU, etc)."""
    pass

class NetworkError(ImageGenError):
    """Errors related to network operations (API calls, model downloads)."""
    pass

class TimeoutError(ImageGenError):
    """Errors related to operations timing out."""
    pass

class FileSystemError(ImageGenError):
    """Errors related to file system operations."""
    pass

# Dictionary mapping exception types to our custom error types
ERROR_MAPPING = {
    "ConnectionError": NetworkError,
    "TimeoutError": TimeoutError,
    "FileNotFoundError": FileSystemError,
    "PermissionError": FileSystemError,
    "OSError": FileSystemError,
    "IOError": FileSystemError,
    "RuntimeError": ModelError,
    "ValueError": ModelError,
    "ImportError": ModelError,
    "ModuleNotFoundError": ModelError,
    "MemoryError": ResourceError,
    "KeyboardInterrupt": ImageGenError
}

# Errors that should not be retried
NON_RETRIABLE_ERRORS = [
    "FileNotFoundError",
    "PermissionError", 
    "ImportError", 
    "ModuleNotFoundError",
    "KeyboardInterrupt",
    "SyntaxError"
]

def classify_error(exception: Exception) -> tuple[Type[ImageGenError], bool]:
    """
    Classify an exception into our error types and determine if it's retriable.
    
    Args:
        exception: The exception to classify
        
    Returns:
        tuple: (error_type, is_retriable)
    """
    error_name = exception.__class__.__name__
    
    # Determine if the error is retriable
    retriable = error_name not in NON_RETRIABLE_ERRORS
    
    # Get the appropriate error type
    error_type = ERROR_MAPPING.get(error_name, ImageGenError)
    
    return error_type, retriable

def handle_errors(error_type: Optional[Type[Exception]] = None,
                 retries: int = 0,
                 cleanup_func: Optional[Callable] = None,
                 retry_delay: float = 1.0):
    """
    Decorator for handling errors in image generation functions.
    
    Args:
        error_type: Specific type of error to catch and re-raise
        retries: Number of retry attempts
        cleanup_func: Function to call for cleanup on error
        retry_delay: Delay in seconds between retry attempts
    """
    def decorator(func):
        is_async = inspect.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                attempts = retries + 1
                last_error = None
                
                for attempt in range(attempts):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        # Get caller information for better logging
                        frame = inspect.currentframe().f_back
                        caller = inspect.getframeinfo(frame) if frame else None
                        caller_info = f" at {caller.filename}:{caller.lineno}" if caller else ""
                        
                        # Log error with traceback for better debugging
                        logger.error(f"Error in {func.__name__}{caller_info}: {str(e)}")
                        if os.environ.get('DEBUG') == '1':
                            logger.error(f"Traceback: {traceback.format_exc()}")
                        
                        # Run cleanup function if provided
                        if cleanup_func:
                            try:
                                result = cleanup_func()
                                # Handle async cleanup functions
                                if inspect.isawaitable(result):
                                    await result
                            except Exception as cleanup_error:
                                logger.error(f"Error during cleanup: {str(cleanup_error)}")
                        
                        # Determine if we should retry based on error type
                        custom_error_type, retriable = classify_error(e)
                        
                        # Skip retry for non-retriable errors
                        if not retriable:
                            logger.warning(f"Not retrying {func.__name__} due to non-retriable error: {type(e).__name__}")
                            break
                            
                        # Retry if we have attempts remaining
                        if attempt < retries:
                            delay = retry_delay * (attempt + 1)  # Exponential backoff
                            logger.info(f"Retrying {func.__name__} in {delay:.1f}s (attempt {attempt + 2}/{attempts})")
                            if delay > 0:
                                import asyncio
                                await asyncio.sleep(delay)
                            continue
                        
                        # We've exhausted retries, raise appropriate error
                        if error_type and isinstance(e, error_type):
                            # If the specific error type is provided and matched, re-raise it directly
                            raise
                        
                        # Package the error in our custom error type
                        if isinstance(e, ImageGenError):
                            # Already our custom type, just re-raise
                            raise
                        else:
                            # Convert to our custom error type
                            raise custom_error_type(
                                f"Failed after {attempts} attempts: {str(e)}",
                                retriable=retriable,
                                original_error=e
                            ) from e
                
                # This should not be reached if attempts > 0, but just in case
                if last_error:
                    raise last_error
                    
            return async_wrapper
        else:
            # Non-async version of the wrapper
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                attempts = retries + 1
                last_error = None
                
                for attempt in range(attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        logger.error(f"Error in {func.__name__}: {str(e)}")
                        if os.environ.get('DEBUG') == '1':
                            logger.error(f"Traceback: {traceback.format_exc()}")
                        
                        if cleanup_func:
                            try:
                                cleanup_func()
                            except Exception as cleanup_error:
                                logger.error(f"Error during cleanup: {str(cleanup_error)}")
                        
                        # Determine if we should retry
                        custom_error_type, retriable = classify_error(e)
                        
                        if not retriable:
                            logger.warning(f"Not retrying {func.__name__} due to non-retriable error: {type(e).__name__}")
                            break
                            
                        if attempt < retries:
                            delay = retry_delay * (attempt + 1)
                            logger.info(f"Retrying {func.__name__} in {delay:.1f}s (attempt {attempt + 2}/{attempts})")
                            if delay > 0:
                                time.sleep(delay)
                            continue
                        
                        if error_type and isinstance(e, error_type):
                            raise
                        
                        if isinstance(e, ImageGenError):
                            raise
                        else:
                            raise custom_error_type(
                                f"Failed after {attempts} attempts: {str(e)}",
                                retriable=retriable,
                                original_error=e
                            ) from e
                
                if last_error:
                    raise last_error
                    
            return sync_wrapper
            
    return decorator

def log_error_details(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log detailed error information to aid debugging.
    
    Args:
        error: The exception to log
        context: Additional context information
    """
    # Get the stack frame where the error occurred
    tb = traceback.extract_tb(sys.exc_info()[2])
    if tb:
        filename, line, func, code = tb[-1]
        error_location = f"{filename}:{line} in {func}"
    else:
        error_location = "unknown location"
    
    # Format the error message
    error_type = type(error).__name__
    error_msg = str(error)
    traceback_str = traceback.format_exc()
    
    # Log basic error info
    logger.error(f"Error: {error_type} at {error_location}: {error_msg}")
    
    # Log context if provided
    if context:
        logger.error(f"Context: {context}")
    
    # Log full traceback in debug mode
    if os.environ.get('DEBUG') == '1':
        logger.error(f"Traceback:\n{traceback_str}")
        
    # If it's our custom error type, log original error too
    if isinstance(error, ImageGenError) and error.original_error:
        logger.error(f"Original error: {type(error.original_error).__name__}: {str(error.original_error)}")
