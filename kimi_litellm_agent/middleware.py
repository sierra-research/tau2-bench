"""
Middleware for kimi_litellm_agent.

This module re-exports shared middleware for convenience.
"""

from shared_utils.middleware import RootA2AMiddleware

__all__ = ["RootA2AMiddleware"]
