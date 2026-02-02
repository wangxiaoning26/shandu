"""
Wrapper functions to handle async functions in LangGraph nodes.
"""
import asyncio
from typing import Callable, Any, Awaitable, TypeVar

T = TypeVar('T')

def create_node_wrapper(async_fn: Callable[..., Awaitable[T]]) -> Callable[..., T]:
    """
    Create a wrapper function that properly handles async functions for LangGraph nodes.
    
    Args:
        async_fn: The async function to wrap
        
    Returns:
        A synchronous function that calls the async function
    """
    def wrapped_function(*args, **kwargs):
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in a running event loop, use run_until_complete
                # (This should be rare and only happens in very specific nested scenarios)
                return loop.run_until_complete(async_fn(*args, **kwargs))
            else:
                # Normal case - use asyncio.run()
                return asyncio.run(async_fn(*args, **kwargs))
        except RuntimeError:
            # If we can't get the event loop, create a new one and run the function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(async_fn(*args, **kwargs))
            finally:
                loop.close()
    
    return wrapped_function