# Implementation Plan: Shared SSE Streaming Utilities

**Branch**: `003-async-evaluation` | **Date**: 2025-12-22 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-async-evaluation/spec.md`
**Depends On**: `002-evaluation-store` (session persistence)
**Depended On By**: `004-gym-evaluation` (uses streaming utilities)

## Summary

Provide shared SSE streaming utilities for tau2_agent that both the structured path (GymOrchestrator) and natural language path (RunTau2Evaluation) can use. This feature focuses on:

1. **ADK Event builders** - Create ADK `Event` objects with tau2-specific metadata (ADK handles A2A conversion)
2. **Progress tracking** - `EvaluationProgress` dataclass for calculating and formatting progress updates
3. **RunTau2Evaluation integration** - Update existing tool to use streaming utilities and expose trace context

**Note**: Routing logic, GymOrchestrator, and Docker entry point belong in 004-gym-evaluation. This feature provides the shared utilities layer.

## Technical Context

**Language/Version**: Python 3.10+ (per tau2-bench pyproject.toml `requires-python = ">=3.10"`)
**Primary Dependencies**:
- `google-adk>=1.18.0` - ADK `Event` type for agent event emission (ADK handles A2A conversion)
- `a2a-sdk>=0.3.12` - `TaskState` type alias for A2A-compliant state values
- `pydantic>=2.0` - Data validation

**Storage**: N/A (stateless utilities, state managed by 002-evaluation-store)
**Testing**: pytest + pytest-asyncio
**Target Platform**: Linux container (Docker)
**Project Type**: Shared utility module within tau2_agent

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: A2A/ADK/tau2 Compliance ✅

| Requirement | Status | Notes |
|-------------|--------|-------|
| A2A Protocol Compliance | ✅ | Event builders produce A2A-compliant types |
| ADK Integration | ✅ | Helpers create ADK Event objects |
| tau2-bench Extension | ✅ | Utilities in `tau2_agent/` namespace |

### Principle II: Backward Compatibility ✅

| Requirement | Status | Notes |
|-------------|--------|-------|
| Zero Breaking Changes | ✅ | New module, no modifications to existing code |
| Optional Usage | ✅ | Other features can use utilities or not |

### Principle III: Metrics & Observability ✅

| Requirement | Status | Notes |
|-------------|--------|-------|
| Progress Metrics | ✅ | Progress utilities include timestamps, percentages |
| Event Metadata | ✅ | Events include tau2-specific attributes for tracing |

### Principle IV: Testing Philosophy ✅

| Requirement | Status | Notes |
|-------------|--------|-------|
| Unit Tests | ✅ | Test event builders and progress calculations |
| No External Dependencies | ✅ | Tests don't require running servers |

### Principle V: Code Quality Guidelines ✅

| Requirement | Status | Notes |
|-------------|--------|-------|
| Type Hints | ✅ | All public functions fully typed |
| Pure Functions | ✅ | Event builders are stateless |

**Gate Status: PASS**

## Project Structure

### Documentation (this feature)

```text
specs/003-async-evaluation/
├── plan.md              # This file
├── research.md          # SSE streaming patterns research
├── data-model.md        # Shared event type definitions
├── quickstart.md        # How to use the utilities
├── contracts/           # A2A event schemas
└── tasks.md             # Implementation tasks
```

### Source Code (repository root)

```text
tau2_agent/
├── streaming/                    # NEW: Shared streaming utilities
│   ├── __init__.py              # Public exports
│   ├── events.py                # Event builder functions
│   ├── progress.py              # Progress tracking utilities
│   └── metadata.py              # tau2-specific metadata helpers
└── ...

tests/
├── test_streaming/              # NEW: Unit tests
│   ├── test_events.py
│   └── test_progress.py
└── ...
```

**Structure Decision**: New `streaming/` submodule within `tau2_agent/`. This keeps utilities colocated with the agent code while being clearly separated as a shared layer.

## Complexity Tracking

> **No violations identified** - Simple utility module

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Module location | `tau2_agent/streaming/` | Colocated with agent, clearly named |
| Event types | Use a2a-sdk types directly | No custom wrappers needed |
| Progress format | Simple dict with standard fields | Easy to serialize, no complex objects |

## Implementation Approach

### Key Design Principles

1. **Stateless utilities**: All functions take input and return output, no side effects
2. **Type-safe**: Full typing for IDE support and validation
3. **ADK-native**: Create ADK `Event` objects; ADK's `A2aAgentExecutor` handles A2A protocol conversion
4. **tau2.* namespace**: All tau2-specific metadata uses `tau2.` prefix to avoid collisions

### Core Functions

```python
# tau2_agent/streaming/events.py

def create_adk_progress_event(
    invocation_id: str,
    state: TaskState,
    message: str,
    evaluation_id: str,
    progress: EvaluationProgress | None = None,
    **extra_metadata,
) -> Event:
    """Create ADK Event with tau2 progress metadata.

    ADK's A2aAgentExecutor converts this to TaskStatusUpdateEvent for SSE.
    Required metadata: tau2.state, tau2.progress, tau2.evaluation_id
    """
    ...

def create_adk_error_event(
    invocation_id: str,
    evaluation_id: str,
    error_message: str,
    error_code: str | None = None,
    **extra_metadata,
) -> Event:
    """Create ADK Event for error/failure state."""
    ...

def create_adk_result_event(
    invocation_id: str,
    evaluation_id: str,
    results: dict,
    message: str = "Evaluation complete",
    **extra_metadata,
) -> Event:
    """Create ADK Event with evaluation results as artifact."""
    ...
```

### Progress Tracking

```python
# tau2_agent/streaming/progress.py

@dataclass
class EvaluationProgress:
    """Track evaluation progress for streaming updates."""
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
        """Convert to metadata dict for event emission."""
        return {
            "tau2.progress": self.percent,
            "tau2.completed_tasks": self.completed_tasks,
            "tau2.total_tasks": self.total_tasks,
            "tau2.current_task_id": self.current_task_id,
            "tau2.elapsed_seconds": self.elapsed_seconds,
        }
```

## Integration with 004-gym-evaluation

004-gym-evaluation uses these utilities in two places:

### 1. GymOrchestrator (Structured Path)

```python
# In 004's GymOrchestrator
from tau2_agent.streaming import create_adk_progress_event, EvaluationProgress

async def run_evaluation(self) -> AsyncIterator[Event]:
    progress = EvaluationProgress(total_tasks=len(task_ids))

    yield create_adk_progress_event(
        invocation_id=self.invocation_id,
        state="submitted",
        progress=0,
        message=f"Starting evaluation of {progress.total_tasks} tasks",
    )

    for task_id in task_ids:
        progress.current_task_id = task_id
        yield create_adk_progress_event(
            invocation_id=self.invocation_id,
            state="working",
            progress=progress.percent,
            message=f"Evaluating task {task_id}",
            **progress.to_metadata(),
        )

        result = await self._evaluate_task(task_id)
        progress.completed_tasks += 1

    yield create_adk_progress_event(
        invocation_id=self.invocation_id,
        state="completed",
        progress=100,
        message="Evaluation complete",
    )
```

### 2. Tau2RouterAgent Event Flow

```
Tau2RouterAgent._run_async_impl()
    │
    ├─► Structured: GymOrchestrator.run_evaluation()
    │       └─► yields ADK Event (with tau2 metadata)
    │
    └─► NL: LlmAgent.run_async()
            └─► yields ADK Event (standard)

    ▼
ADK A2aAgentExecutor
    │
    └─► convert_event_to_a2a_events()
            └─► TaskStatusUpdateEvent (SSE)
```

## Constitution Check (Post-Design Re-evaluation)

All principles remain satisfied. The refocused scope (shared utilities only) is simpler and cleaner than the original design.

**Post-Design Gate Status: PASS**

---

## Generated Artifacts

| Artifact | Path | Status |
|----------|------|--------|
| Plan | `specs/003-async-evaluation/plan.md` | ✅ Complete |
| Research | `specs/003-async-evaluation/research.md` | ✅ Complete |
| Data Model | `specs/003-async-evaluation/data-model.md` | ✅ Complete |
| Quickstart | `specs/003-async-evaluation/quickstart.md` | ✅ Complete |
| Contracts | `specs/003-async-evaluation/contracts/` | ✅ Complete |

## Next Steps

Run `/speckit.tasks` to generate implementation tasks based on this plan.
