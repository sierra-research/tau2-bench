"""
Event builder functions for ADK streaming events.

This module provides functions to create ADK Event objects with tau2-specific
metadata. ADK's A2aAgentExecutor handles converting these to A2A SSE events.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from google.adk.events.event import Event

    from .progress import EvaluationProgress

# A2A-compliant task state values
TaskState = Literal["submitted", "working", "completed", "failed"]

__all__ = [
    "TaskState",
    "create_adk_error_event",
    "create_adk_progress_event",
    "create_adk_result_event",
]


def create_adk_progress_event(
    invocation_id: str,
    state: TaskState,
    message: str,
    evaluation_id: str | None = None,
    progress: EvaluationProgress | None = None,
    **extra_metadata: object,
) -> Event:
    """Create ADK Event with tau2 progress metadata.

    ADK's A2aAgentExecutor converts this to TaskStatusUpdateEvent for SSE.

    Required metadata: tau2.state, tau2.progress
    Optional metadata: tau2.evaluation_id (included only if provided)

    Args:
        invocation_id: ADK invocation ID for event correlation
        state: Task state (submitted, working, completed, failed)
        message: Human-readable status message
        evaluation_id: Optional evaluation identifier (included in metadata if not None)
        progress: Optional EvaluationProgress for detailed tracking
        **extra_metadata: Additional tau2-namespaced fields merged into metadata

    Returns:
        ADK Event that will be converted to TaskStatusUpdateEvent by ADK
    """
    from google.adk.events.event import Event
    from google.genai.types import Content, Part

    from .metadata import TAU2_EVALUATION_ID, TAU2_PROGRESS, TAU2_STATE

    # Build metadata dict
    metadata: dict[str, object] = {
        TAU2_STATE: state,
    }

    # Add progress metadata
    if progress is not None:
        metadata.update(progress.to_metadata())
    elif state == "submitted":
        metadata[TAU2_PROGRESS] = 0
    elif state == "working":
        raise ValueError(
            "progress is required for 'working' state; "
            "provide an EvaluationProgress instance"
        )
    else:
        # Terminal states (completed, failed): default to 100%
        metadata[TAU2_PROGRESS] = 100

    # Add evaluation ID if provided
    if evaluation_id is not None:
        metadata[TAU2_EVALUATION_ID] = evaluation_id

    # Merge extra metadata
    metadata.update(extra_metadata)

    return Event(
        invocation_id=invocation_id,
        author="tau2_agent",
        content=Content(
            role="model",
            parts=[Part(text=message)],
        ),
        custom_metadata=metadata,
    )


def create_adk_error_event(
    invocation_id: str,
    evaluation_id: str | None,
    error_message: str,
    error_code: str | None = None,
    **extra_metadata: object,
) -> Event:
    """Create ADK Event for error/failure state.

    Args:
        invocation_id: ADK invocation ID
        evaluation_id: Unique evaluation identifier
        error_message: Human-readable error description
        error_code: Optional error code for programmatic handling
        **extra_metadata: Additional tau2-namespaced metadata

    Returns:
        ADK Event with error_code set, converted to failed TaskStatusUpdateEvent
    """
    from google.adk.events.event import Event
    from google.genai.types import Content, Part

    from .metadata import (
        TAU2_ERROR,
        TAU2_ERROR_CODE,
        TAU2_EVALUATION_ID,
        TAU2_STATE,
    )

    # Build metadata dict
    metadata: dict[str, object] = {
        TAU2_STATE: "failed",
        TAU2_ERROR: error_message,
    }

    if error_code is not None:
        metadata[TAU2_ERROR_CODE] = error_code

    if evaluation_id is not None:
        metadata[TAU2_EVALUATION_ID] = evaluation_id

    # Merge extra metadata
    metadata.update(extra_metadata)

    return Event(
        invocation_id=invocation_id,
        author="tau2_agent",
        error_code=error_code,
        error_message=error_message,
        content=Content(
            role="model",
            parts=[Part(text=f"Evaluation failed: {error_message}")],
        ),
        custom_metadata=metadata,
    )


def create_adk_result_event(
    invocation_id: str,
    evaluation_id: str | None,
    results: dict[str, object],
    message: str = "Evaluation complete",
    **extra_metadata: object,
) -> Event:
    """Create ADK Event with evaluation results.

    Args:
        invocation_id: ADK invocation ID
        evaluation_id: Unique evaluation identifier
        results: Evaluation results dict (will be in artifact)
        message: Completion message
        **extra_metadata: Additional tau2-namespaced metadata

    Returns:
        ADK Event with results in content, triggers TaskArtifactUpdateEvent
    """
    import json

    from google.adk.events.event import Event
    from google.genai.types import Content, Part

    from .metadata import TAU2_EVALUATION_ID, TAU2_PROGRESS, TAU2_STATE

    # Build metadata dict
    metadata: dict[str, object] = {
        TAU2_STATE: "completed",
        TAU2_PROGRESS: 100,
    }

    if evaluation_id is not None:
        metadata[TAU2_EVALUATION_ID] = evaluation_id

    # Merge extra metadata
    metadata.update(extra_metadata)

    # Build content with results
    result_text = f"{message}\n\nResults:\n{json.dumps(results, indent=2)}"

    return Event(
        invocation_id=invocation_id,
        author="tau2_agent",
        content=Content(
            role="model",
            parts=[Part(text=result_text)],
        ),
        custom_metadata=metadata,
    )
