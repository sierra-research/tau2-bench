# Quickstart: SSE Streaming Utilities

**Feature**: 003-async-evaluation
**Date**: 2025-12-22

## Overview

The `tau2_agent.streaming` module provides utilities for emitting SSE progress events from tau2 agents. These utilities work with ADK's built-in A2A server to stream progress updates to clients.

## Installation

The utilities are part of the `tau2_agent` package - no additional installation required.

```python
from tau2_agent.streaming import (
    EvaluationProgress,
    create_adk_progress_event,
    create_adk_error_event,
    create_adk_result_event,
)
```

## Basic Usage

### 1. Track Progress with EvaluationProgress

```python
from tau2_agent.streaming import EvaluationProgress

# Create progress tracker
progress = EvaluationProgress(total_tasks=5)

# Update as tasks complete
for task_id in task_ids:
    progress.current_task_id = task_id
    print(f"Progress: {progress.percent}%")

    # ... do work ...

    progress.increment()

print(f"Completed in {progress.elapsed_seconds:.1f}s")
```

### 2. Emit Progress Events in an Agent

```python
from collections.abc import AsyncIterator
from google.adk.agents import BaseAgent
from google.adk.events.event import Event
from tau2_agent.streaming import (
    EvaluationProgress,
    create_adk_progress_event,
    create_adk_result_event,
)

class MyEvaluationAgent(BaseAgent):
    async def _run_async_impl(self, ctx) -> AsyncIterator[Event]:
        progress = EvaluationProgress(total_tasks=5)

        # Emit "submitted" state
        yield create_adk_progress_event(
            invocation_id=ctx.invocation_id,
            state="submitted",
            message="Starting evaluation",
        )

        # Emit "working" state with progress updates
        for task_id in task_ids:
            progress.current_task_id = task_id

            yield create_adk_progress_event(
                invocation_id=ctx.invocation_id,
                state="working",
                message=f"Evaluating task {task_id}",
                progress=progress,
            )

            result = await self._evaluate_task(task_id)
            progress.increment()

        # Emit "completed" state with results
        yield create_adk_result_event(
            invocation_id=ctx.invocation_id,
            evaluation_id=self.evaluation_id,
            results={"success_rate": 0.8, "tasks": results},
            message="Evaluation complete",
        )
```

### 3. Handle Errors

```python
from tau2_agent.streaming import create_adk_error_event

async def _run_async_impl(self, ctx) -> AsyncIterator[Event]:
    try:
        # ... evaluation logic ...
        pass
    except Exception as e:
        yield create_adk_error_event(
            invocation_id=ctx.invocation_id,
            evaluation_id=self.evaluation_id,
            error_message=str(e),
            error_code="EVALUATION_FAILED",
        )
```

## Event Flow

When your agent yields events, ADK handles the rest:

```
Your Agent                      ADK                         Client
    │                            │                            │
    │─── yield Event ───────────►│                            │
    │                            │── convert to A2A ─────────►│
    │                            │   TaskStatusUpdateEvent    │
    │                            │                            │
    │─── yield Event ───────────►│                            │
    │                            │── stream via SSE ─────────►│
    │                            │   {"statusUpdate": {...}}  │
```

## Metadata Reference

All tau2-specific metadata uses the `tau2.` prefix:

| Field | Type | Description |
|-------|------|-------------|
| `tau2.state` | string | submitted/working/completed/failed |
| `tau2.progress` | int | 0-100 percent complete |
| `tau2.completed_tasks` | int | Tasks finished |
| `tau2.total_tasks` | int | Total tasks |
| `tau2.current_task_id` | string | Current task being evaluated |
| `tau2.elapsed_seconds` | float | Time since start |
| `tau2.evaluation_id` | string | Unique evaluation ID |
| `tau2.domain` | string | Evaluation domain |

## Integration with 004-gym-evaluation

The `GymOrchestrator` in 004-gym-evaluation uses these utilities:

```python
# In GymOrchestrator (004-gym-evaluation)
from tau2_agent.streaming import EvaluationProgress, create_adk_progress_event

async def run_evaluation(self) -> AsyncIterator[Event]:
    progress = EvaluationProgress(total_tasks=len(self.task_ids))

    yield create_adk_progress_event(
        invocation_id=self.invocation_id,
        state="submitted",
        message=f"Starting {self.domain} evaluation",
        **{"tau2.domain": self.domain, "tau2.evaluation_id": self.eval_id},
    )

    for task_id in self.task_ids:
        # ... (progress updates as shown above)
        pass
```

## Testing

```python
import pytest
from tau2_agent.streaming import EvaluationProgress, create_adk_progress_event

def test_progress_calculation():
    progress = EvaluationProgress(total_tasks=4)
    assert progress.percent == 0

    progress.increment()
    assert progress.percent == 25

    progress.increment()
    progress.increment()
    assert progress.percent == 75

def test_event_creation():
    event = create_adk_progress_event(
        invocation_id="test-123",
        state="working",
        message="Testing",
        progress=EvaluationProgress(total_tasks=10, completed_tasks=5),
    )

    assert event.invocation_id == "test-123"
    assert event.custom_metadata["tau2.state"] == "working"
    assert event.custom_metadata["tau2.progress"] == 50
```

## Troubleshooting

### Events Not Streaming

1. Ensure your agent yields `Event` objects (not A2A types directly)
2. Verify ADK is running with A2A enabled (`adk api_server --a2a`)
3. Check client connects to `/message/stream` (not `/message/send`)

### Progress Always 0%

1. Call `progress.increment()` after each task completes
2. Pass `progress=progress` to `create_adk_progress_event()`

### Metadata Not Appearing

1. Use `tau2.` prefix for all custom fields
2. Pass extra metadata as keyword arguments:
   ```python
   create_adk_progress_event(
       ...,
       **{"tau2.custom_field": "value"}
   )
   ```

## See Also

- [004-gym-evaluation](../004-gym-evaluation/quickstart.md) - Full evaluation routing
- [002-evaluation-store](../002-evaluation-store/quickstart.md) - Persist evaluation results
- [006-otel-integration](../006-otel-integration/spec.md) - OpenTelemetry tracing
