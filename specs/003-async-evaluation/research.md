# Research: Shared SSE Streaming Utilities

**Feature**: 003-async-evaluation
**Date**: 2025-12-22
**Status**: Complete

## Scope

This research focuses on **shared SSE streaming utilities** that both execution paths (structured and NL) can use. Routing and execution logic is covered in 004-gym-evaluation.

## Research Tasks

### 1. A2A Protocol SSE Event Format

**Question**: What event types does A2A use for SSE streaming?

**Decision**: Use `TaskStatusUpdateEvent` and `TaskArtifactUpdateEvent` from a2a-sdk.

**Rationale**: The A2A protocol specification defines two main event types for streaming:

1. **TaskStatusUpdateEvent** - State changes and progress
   ```json
   {
     "statusUpdate": {
       "taskId": "eval-123",
       "contextId": "ctx-456",
       "status": {
         "state": "working",
         "timestamp": "2025-12-22T10:05:00Z"
       },
       "final": false,
       "metadata": {
         "tau2.progress": 50,
         "tau2.current_task_id": "airline_003"
       }
     }
   }
   ```

2. **TaskArtifactUpdateEvent** - Results and outputs
   ```json
   {
     "artifactUpdate": {
       "taskId": "eval-123",
       "contextId": "ctx-456",
       "artifact": {
         "artifactId": "results",
         "name": "evaluation_results",
         "parts": [{"data": {"success_rate": 0.8}}]
       },
       "lastChunk": true
     }
   }
   ```

**Task Lifecycle States**:
| State | Type | Description |
|-------|------|-------------|
| `submitted` | In-progress | Task acknowledged, not yet processing |
| `working` | In-progress | Active processing |
| `completed` | Terminal | Success |
| `failed` | Terminal | Fatal error |

---

### 2. ADK Event Flow for SSE

**Question**: How do ADK Events become SSE events?

**Decision**: Create ADK `Event` objects with tau2 metadata; ADK's `A2aAgentExecutor` handles conversion.

**Rationale**: ADK provides `A2aAgentExecutor` which:

1. Wraps an ADK agent/runner
2. Iterates over yielded `Event` objects
3. Converts each to A2A events via `convert_event_to_a2a_events()`
4. Enqueues to `EventQueue` for SSE streaming

**Key Insight**: We don't need to build SSE infrastructure - ADK provides it. We just need to:
- Create ADK `Event` objects with appropriate content
- Include tau2-specific metadata in `custom_metadata`

**Event Conversion Flow**:
```
Agent._run_async_impl()
    │
    └─► yields Event(content=..., custom_metadata={"tau2.progress": 50})
            │
            ▼
A2aAgentExecutor._handle_request()
    │
    └─► convert_event_to_a2a_events(event)
            │
            └─► TaskStatusUpdateEvent(
                    status=TaskStatus(state=working, message=...),
                    metadata={"tau2.progress": 50}
                )
                    │
                    ▼
            event_queue.enqueue_event(...)
                    │
                    ▼
            SSE: data: {"statusUpdate": {...}}
```

---

### 3. ADK Event Structure for Progress

**Question**: What should ADK Events contain for progress updates?

**Decision**: Use `Event.content` for message text, `Event.custom_metadata` for tau2-specific fields.

**Rationale**: ADK's `convert_event_to_a2a_events` reads:
- `event.content.parts` → becomes `message.parts` in `TaskStatusUpdateEvent`
- `event.custom_metadata` → preserved in event metadata

**Event Structure**:
```python
from google.adk.events.event import Event
from google.genai.types import Content, Part

def create_adk_progress_event(
    invocation_id: str,
    state: str,
    progress: int,
    message: str,
    **extra_metadata,
) -> Event:
    return Event(
        invocation_id=invocation_id,
        author="tau2_agent",
        content=Content(
            role="model",
            parts=[Part(text=message)],
        ),
        custom_metadata={
            "tau2.state": state,
            "tau2.progress": progress,
            **extra_metadata,
        },
    )
```

---

### 4. Progress Tracking Pattern

**Question**: How should we track and report progress during evaluation?

**Decision**: Simple dataclass with computed properties.

**Rationale**: Keep it simple - no complex state machines or observers.

**Implementation**:
```python
@dataclass
class EvaluationProgress:
    total_tasks: int
    completed_tasks: int = 0
    current_task_id: str | None = None
    started_at: datetime | None = None

    @property
    def percent(self) -> int:
        if self.total_tasks == 0:
            return 0
        return int((self.completed_tasks / self.total_tasks) * 100)

    @property
    def elapsed_seconds(self) -> float:
        if not self.started_at:
            return 0.0
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()

    def to_metadata(self) -> dict:
        return {
            "tau2.progress": self.percent,
            "tau2.completed_tasks": self.completed_tasks,
            "tau2.total_tasks": self.total_tasks,
            "tau2.current_task_id": self.current_task_id,
            "tau2.elapsed_seconds": self.elapsed_seconds,
        }
```

**Alternatives Considered**:
- State machine with transitions - Rejected: Overkill for progress tracking
- Observer pattern - Rejected: Adds complexity, simple yields work fine
- Callback-based - Rejected: Async generators are cleaner in Python

---

### 5. Metadata Namespace

**Question**: How should we namespace tau2-specific metadata?

**Decision**: Use `tau2.` prefix for all tau2-specific metadata keys.

**Rationale**:
- Avoids collisions with ADK or other middleware metadata
- Clear ownership of fields
- Matches 006-otel-integration span attribute naming

**Standard Fields**:
| Key | Type | Description |
|-----|------|-------------|
| `tau2.state` | string | Current task state (submitted/working/completed/failed) |
| `tau2.progress` | int | Percent complete (0-100) |
| `tau2.total_tasks` | int | Total tasks in evaluation |
| `tau2.completed_tasks` | int | Tasks completed so far |
| `tau2.current_task_id` | string | Currently evaluating task ID |
| `tau2.elapsed_seconds` | float | Seconds since evaluation started |
| `tau2.evaluation_id` | string | Unique evaluation identifier |
| `tau2.domain` | string | Evaluation domain (airline, retail, etc.) |

---

### 6. Error Handling

**Question**: How should errors be represented in streaming events?

**Decision**: Use `TaskState.failed` with error message in status.

**Rationale**: A2A protocol has explicit `failed` state. ADK's event converter handles error events.

**Error Event**:
```python
def create_error_event(
    invocation_id: str,
    error_message: str,
    error_code: str | None = None,
) -> Event:
    return Event(
        invocation_id=invocation_id,
        author="tau2_agent",
        error_code=error_code,
        error_message=error_message,
        content=Content(
            role="model",
            parts=[Part(text=f"Evaluation failed: {error_message}")],
        ),
        custom_metadata={
            "tau2.state": "failed",
            "tau2.error": error_message,
        },
    )
```

ADK's `convert_event_to_a2a_events` checks `event.error_code` and creates a `TaskStatusUpdateEvent` with `state=failed`.

---

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| a2a-sdk | >=0.3.12 | A2A protocol types |
| google-adk | >=1.18.0 | ADK Event type |
| pydantic | >=2.0 | Data validation |

---

## Summary

| Area | Decision | Confidence |
|------|----------|------------|
| Event types | Use a2a-sdk types directly | High |
| ADK integration | Create ADK Events, let ADK convert | High |
| Progress tracking | Simple dataclass with properties | High |
| Metadata namespace | `tau2.` prefix | High |
| Error handling | Use Event.error_code, ADK handles | High |

## Relationship to Other Specs

| Spec | Relationship |
|------|--------------|
| 002-evaluation-store | Uses evaluation_id for correlation |
| 004-gym-evaluation | Consumes these utilities for SSE emission |
| 006-otel-integration | Shares tau2. namespace for attributes |
