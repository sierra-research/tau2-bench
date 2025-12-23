# Data Model: Shared SSE Streaming Utilities

**Feature**: 003-async-evaluation
**Date**: 2025-12-22
**Status**: Complete

## Scope

This document defines the shared data types for SSE streaming utilities. These types are used by both the structured path (004-gym-evaluation) and future NL path enhancements.

## Entities

### 1. EvaluationProgress

Tracks evaluation progress for streaming updates.

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class EvaluationProgress:
    """Track evaluation progress for streaming updates.

    Used by GymOrchestrator and other evaluation runners to calculate
    and emit progress events during evaluation.
    """

    total_tasks: int
    completed_tasks: int = 0
    current_task_id: str | None = None
    current_trial: int = 1
    total_trials: int = 1
    started_at: datetime | None = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def percent(self) -> int:
        """Calculate completion percentage (0-100)."""
        if self.total_tasks == 0:
            return 0
        return int((self.completed_tasks / self.total_tasks) * 100)

    @property
    def elapsed_seconds(self) -> float:
        """Calculate elapsed time since start."""
        if not self.started_at:
            return 0.0
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()

    def to_metadata(self) -> dict[str, any]:
        """Convert to tau2-namespaced metadata dict for event emission."""
        return {
            "tau2.progress": self.percent,
            "tau2.completed_tasks": self.completed_tasks,
            "tau2.total_tasks": self.total_tasks,
            "tau2.current_task_id": self.current_task_id,
            "tau2.current_trial": self.current_trial,
            "tau2.total_trials": self.total_trials,
            "tau2.elapsed_seconds": round(self.elapsed_seconds, 2),
        }

    def increment(self, task_id: str | None = None) -> None:
        """Increment completed count and optionally update current task."""
        self.completed_tasks += 1
        if task_id:
            self.current_task_id = task_id
```

**Validation Rules**:
- `total_tasks`: Must be >= 0
- `completed_tasks`: Must be >= 0 and <= `total_tasks`
- `current_trial`: Must be >= 1 and <= `total_trials`
- `started_at`: Should be set before progress tracking begins

---

### 2. Tau2EventMetadata

Standard metadata fields for tau2 streaming events.

```python
from dataclasses import dataclass
from typing import Literal

TaskState = Literal["submitted", "working", "completed", "failed"]

@dataclass
class Tau2EventMetadata:
    """Standard tau2 metadata fields for A2A/ADK events.

    All fields use 'tau2.' prefix to avoid collisions with
    ADK or other middleware metadata.
    """

    # Required
    state: TaskState

    # Progress (optional, for working state)
    progress: int | None = None
    completed_tasks: int | None = None
    total_tasks: int | None = None
    current_task_id: str | None = None
    elapsed_seconds: float | None = None

    # Evaluation context (optional)
    evaluation_id: str | None = None
    domain: str | None = None
    agent_endpoint: str | None = None

    # Error info (optional, for failed state)
    error: str | None = None
    error_code: str | None = None

    def to_dict(self) -> dict[str, any]:
        """Convert to namespaced dict, excluding None values."""
        result = {"tau2.state": self.state}

        optional_fields = [
            ("tau2.progress", self.progress),
            ("tau2.completed_tasks", self.completed_tasks),
            ("tau2.total_tasks", self.total_tasks),
            ("tau2.current_task_id", self.current_task_id),
            ("tau2.elapsed_seconds", self.elapsed_seconds),
            ("tau2.evaluation_id", self.evaluation_id),
            ("tau2.domain", self.domain),
            ("tau2.agent_endpoint", self.agent_endpoint),
            ("tau2.error", self.error),
            ("tau2.error_code", self.error_code),
        ]

        for key, value in optional_fields:
            if value is not None:
                result[key] = value

        return result
```

---

### 3. Event Builder Functions (Signatures)

These are the primary interface for creating streaming events.

```python
from google.adk.events.event import Event
from a2a.types import TaskStatusUpdateEvent, TaskArtifactUpdateEvent

def create_adk_progress_event(
    invocation_id: str,
    state: TaskState,
    message: str,
    progress: EvaluationProgress | None = None,
    **extra_metadata,
) -> Event:
    """Create ADK Event with tau2 progress metadata.

    Args:
        invocation_id: ADK invocation ID for event correlation
        state: Task state (submitted, working, completed, failed)
        message: Human-readable status message
        progress: Optional EvaluationProgress for detailed tracking
        **extra_metadata: Additional tau2-namespaced metadata

    Returns:
        ADK Event that will be converted to TaskStatusUpdateEvent by ADK
    """
    ...

def create_adk_error_event(
    invocation_id: str,
    error_message: str,
    error_code: str | None = None,
    **extra_metadata,
) -> Event:
    """Create ADK Event for error/failure state.

    Args:
        invocation_id: ADK invocation ID
        error_message: Human-readable error description
        error_code: Optional error code for programmatic handling
        **extra_metadata: Additional tau2-namespaced metadata

    Returns:
        ADK Event with error_code set, converted to failed TaskStatusUpdateEvent
    """
    ...

def create_adk_result_event(
    invocation_id: str,
    results: dict,
    message: str = "Evaluation complete",
    **extra_metadata,
) -> Event:
    """Create ADK Event with evaluation results.

    Args:
        invocation_id: ADK invocation ID
        results: Evaluation results dict (will be in artifact)
        message: Completion message
        **extra_metadata: Additional tau2-namespaced metadata

    Returns:
        ADK Event with results in content, triggers TaskArtifactUpdateEvent
    """
    ...
```

---

## Metadata Namespace

All tau2-specific metadata uses the `tau2.` prefix.

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `tau2.state` | string | Yes | Task state (submitted/working/completed/failed) |
| `tau2.progress` | int | No | Percent complete (0-100) |
| `tau2.total_tasks` | int | No | Total tasks in evaluation |
| `tau2.completed_tasks` | int | No | Tasks completed so far |
| `tau2.current_task_id` | string | No | Currently evaluating task ID |
| `tau2.current_trial` | int | No | Current trial number |
| `tau2.total_trials` | int | No | Total trials per task |
| `tau2.elapsed_seconds` | float | No | Seconds since evaluation started |
| `tau2.evaluation_id` | string | No | Unique evaluation identifier |
| `tau2.domain` | string | No | Evaluation domain (airline, retail, etc.) |
| `tau2.agent_endpoint` | string | No | Agent being evaluated |
| `tau2.error` | string | No | Error message (for failed state) |
| `tau2.error_code` | string | No | Error code (for failed state) |

---

## State Transitions

```
                    ┌─────────────┐
                    │  (request)  │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  submitted  │  ← Initial acknowledgment
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   working   │  ← Progress updates (0% → 100%)
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       ┌───────────┐             ┌──────────┐
       │ completed │             │  failed  │
       └───────────┘             └──────────┘
```

**Transitions**:
- `submitted` → `working`: Evaluation started
- `working` → `working`: Progress update (same state, different percentage)
- `working` → `completed`: All tasks finished successfully
- `working` → `failed`: Unrecoverable error
- `submitted` → `failed`: Error before evaluation started

---

## Integration with A2A Types

The utilities create ADK `Event` objects that ADK's `A2aAgentExecutor` converts to A2A types:

| ADK Event | A2A Event | When |
|-----------|-----------|------|
| `Event` with content | `TaskStatusUpdateEvent` (state=working) | Progress updates |
| `Event` with error_code | `TaskStatusUpdateEvent` (state=failed) | Errors |
| `Event` with results | `TaskArtifactUpdateEvent` | Final results |
| Final `Event` | `TaskStatusUpdateEvent` (final=true) | Completion |

---

## Usage Example

```python
from tau2_agent.streaming import (
    EvaluationProgress,
    create_adk_progress_event,
    create_adk_result_event,
)

async def run_evaluation(self) -> AsyncIterator[Event]:
    progress = EvaluationProgress(total_tasks=5)

    # Emit submitted
    yield create_adk_progress_event(
        invocation_id=self.invocation_id,
        state="submitted",
        message="Starting evaluation",
        evaluation_id=self.evaluation_id,
        domain=self.domain,
    )

    # Emit working with progress
    for task_id in task_ids:
        progress.current_task_id = task_id
        yield create_adk_progress_event(
            invocation_id=self.invocation_id,
            state="working",
            message=f"Evaluating {task_id}",
            progress=progress,
        )

        result = await self._evaluate_task(task_id)
        progress.increment()

    # Emit completed with results
    yield create_adk_result_event(
        invocation_id=self.invocation_id,
        results=aggregated_results,
        message="Evaluation complete",
        evaluation_id=self.evaluation_id,
    )
```

---

## Relationship to Other Specs

| Spec | Shared Types |
|------|--------------|
| 004-gym-evaluation | Uses `EvaluationProgress`, `create_adk_*` functions |
| 006-otel-integration | Shares `tau2.*` namespace for span attributes |
| 002-evaluation-store | Uses `evaluation_id` for correlation |
