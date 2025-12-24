"""
Shared SSE streaming utilities for tau2_agent.

This module provides utilities for emitting SSE progress events from tau2 agents.
These utilities work with ADK's built-in A2A server to stream progress updates.

Usage:
    from tau2_agent.streaming import (
        EvaluationProgress,
        create_adk_progress_event,
        create_adk_error_event,
        create_adk_result_event,
        TaskState,
    )
"""

from .events import (
    TaskState,
    create_adk_error_event,
    create_adk_progress_event,
    create_adk_result_event,
)
from .progress import EvaluationProgress

__all__ = [
    "EvaluationProgress",
    "TaskState",
    "create_adk_error_event",
    "create_adk_progress_event",
    "create_adk_result_event",
]
