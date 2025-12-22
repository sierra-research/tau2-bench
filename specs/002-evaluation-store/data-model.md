# Data Model: Evaluation Store

**Feature**: 002-evaluation-store
**Date**: 2025-12-22

## Overview

This document defines the data entities, their relationships, validation rules, and state transitions for the evaluation store.

---

## Entities

### 1. Evaluation

The primary entity representing an evaluation run, either in-progress or completed.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `evaluation_id` | `str` | Yes | Unique ID: `eval-{unix_ms}-{random_6_chars}` |
| `trace_id` | `str \| None` | No | W3C Trace Context ID for OTel/Datadog correlation |
| `session_id` | `str \| None` | No | SSE session ID for streaming reconnection |
| `status` | `EvaluationStatus` | Yes | Current state of the evaluation |
| `domain` | `str` | Yes | Domain being evaluated (airline, retail, etc.) |
| `agent_endpoint` | `str \| None` | No | A2A agent endpoint URL |
| `state_history` | `list[StateTransition]` | Yes | Ordered list of state changes |
| `created_at` | `datetime` | Yes | UTC timestamp when evaluation was created |
| `completed_at` | `datetime \| None` | No | UTC timestamp when evaluation completed/failed |
| `request` | `EvaluationRequest` | Yes | Original evaluation request parameters |
| `results` | `EvaluationResults \| None` | No | Final results (only when completed) |
| `error` | `str \| None` | No | Error message (only when failed) |
| `progress` | `Progress \| None` | No | Current progress (only for in-progress) |

### 2. EvaluationStatus (Enum)

```python
class EvaluationStatus(str, Enum):
    SUBMITTED = "submitted"    # Evaluation request received
    WORKING = "working"        # Actively processing tasks
    COMPLETED = "completed"    # Successfully finished
    FAILED = "failed"          # Terminated with error
    ABANDONED = "abandoned"    # No heartbeat for 2+ hours
```

### 3. StateTransition

Records a single state change in the evaluation lifecycle.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `state` | `EvaluationStatus` | Yes | The state entered |
| `at` | `datetime` | Yes | UTC timestamp of transition |
| `progress` | `int \| None` | No | Progress percentage (0-100) when entering working state |

### 4. Progress

Tracks real-time progress for in-progress evaluations.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `current_task` | `int` | Yes | 1-indexed current task number |
| `total_tasks` | `int` | Yes | Total number of tasks |
| `percent` | `int` | Yes | Completion percentage (0-100) |
| `last_heartbeat` | `datetime` | Yes | UTC timestamp of last update |

### 5. EvaluationRequest

Captures the original request parameters.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_llm` | `str \| None` | No | LLM used for user simulation |
| `num_trials` | `int` | Yes | Number of trials per task |
| `num_tasks` | `int` | Yes | Number of tasks to evaluate |

### 6. EvaluationResults

Final evaluation results (only present on completed evaluations).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `success_rate` | `float` | Yes | Overall success rate (0.0-1.0) |
| `total_tasks` | `int` | Yes | Total tasks evaluated |
| `successful` | `int` | Yes | Number of successful tasks |
| `tasks` | `list[TaskResult]` | Yes | Per-task results |

### 7. TaskResult

Individual task result within an evaluation.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_id` | `str` | Yes | Task identifier |
| `success` | `bool` | Yes | Whether task succeeded |
| `reward` | `float` | Yes | Task reward (0.0-1.0) |
| `trajectory` | `list[Message]` | No | Full message trajectory (optional) |

### 8. LogEvent

Structured log event for audit trail.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ts` | `datetime` | Yes | UTC timestamp |
| `level` | `str` | Yes | Log level (info, warning, error) |
| `event` | `str` | Yes | Event type (see Event Types below) |
| `evaluation_id` | `str` | Yes | Associated evaluation ID |
| `trace_id` | `str \| None` | No | OTel trace ID for correlation |
| `session_id` | `str \| None` | No | SSE session ID |
| `*` | `Any` | No | Additional event-specific fields |

---

## Relationships

```
EvaluationRequest ─────┐
                       │
Evaluation ────────────┼── 1:1 relationship
    │                  │
    ├── state_history ─┤── 1:N StateTransition
    │                  │
    ├── progress ──────┤── 1:1 Progress (only in-progress)
    │                  │
    ├── results ───────┤── 1:1 EvaluationResults (only completed)
    │                  │
    └── results.tasks ─┘── 1:N TaskResult

LogEvent ─────────────────── references Evaluation by evaluation_id
```

---

## Validation Rules

### EvaluationID
- Pattern: `^eval-[0-9]{13}-[a-f0-9]{6}$`
- Example: `eval-1732449600000-a1b2c3`

### TraceID
- Pattern: `^[a-f0-9]{32}$` (W3C Trace Context format)
- Example: `4bf92f3577b34da6a3ce929d0e0e4736`

### SessionID
- Pattern: `^sess-[a-z0-9]+$`
- Example: `sess-abc123`

### Timestamps
- Format: ISO 8601 with UTC timezone (Z suffix)
- Example: `2025-12-22T10:00:00Z`

### Progress.percent
- Range: 0-100 inclusive
- Calculated as: `(current_task - 1) / total_tasks * 100`

### Success Rate
- Range: 0.0-1.0 inclusive
- Calculated as: `successful / total_tasks`

### Domain
- Must be one of registered domains (airline, retail, telecom, vacation_rental, mock)

### Status Transitions
- Valid transitions only (see State Transitions below)

---

## State Transitions

```
                    ┌──────────────┐
                    │   SUBMITTED  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
        ┌───────────│   WORKING    │───────────┐
        │           └──────┬───────┘           │
        │                  │                   │
        │ (no heartbeat    │ (all tasks        │ (error during
        │  for 2 hours)    │  complete)        │  processing)
        │                  │                   │
        ▼                  ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   ABANDONED  │   │   COMPLETED  │   │    FAILED    │
└──────────────┘   └──────────────┘   └──────────────┘
     (terminal)         (terminal)         (terminal)
```

### Valid Transitions

| From | To | Trigger |
|------|----|---------|
| SUBMITTED | WORKING | First task starts processing |
| WORKING | COMPLETED | All tasks finish successfully |
| WORKING | FAILED | Unrecoverable error occurs |
| WORKING | ABANDONED | No heartbeat for `TAU2_SESSION_STALE_HOURS` (default: 2) |

### Terminal States
- COMPLETED, FAILED, ABANDONED are terminal - no further transitions allowed
- Evaluations in terminal states are moved from `sessions/` to `evaluations/`

---

## File Storage Layout

### Sessions Directory (Mutable)
```
$TAU2_DATA_DIR/sessions/
├── eval-1732449720000-g7h8i9.json           # Active session
├── eval-1732449720000-g7h8i9.json.tmp       # Atomic write in progress
└── .heartbeat                                # Watchdog marker (optional)
```

### Evaluations Directory (Immutable, Flat)
```
$TAU2_DATA_DIR/evaluations/
├── eval-1732449600000-a1b2c3.json           # Completed
├── eval-1732449660000-d4e5f6.json           # Failed
└── eval-1732535000000-x7y8z9.json           # Completed
```

### Logs Directory
```
$TAU2_DATA_DIR/logs/
├── events.jsonl                              # Current log file
└── archive/
    ├── events.2025-12-20.jsonl.gz           # Compressed old logs
    └── events.2025-12-19.jsonl.gz
```

---

## Event Types

Standard events logged to `events.jsonl`:

| Event | When | Additional Fields |
|-------|------|-------------------|
| `evaluation_created` | Evaluation submitted | `domain`, `agent_endpoint`, `num_tasks` |
| `evaluation_started` | First task begins | `domain` |
| `task_completed` | Single task finishes | `task_num`, `total_tasks`, `success`, `reward` |
| `evaluation_completed` | All tasks done | `success_rate`, `duration_s` |
| `evaluation_failed` | Unrecoverable error | `error_type`, `error_message` |
| `evaluation_abandoned` | Heartbeat timeout | `last_heartbeat`, `stale_hours` |
| `cleanup_started` | Retention cleanup begins | `retention_days` |
| `cleanup_completed` | Retention cleanup ends | `deleted_count`, `duration_s` |

---

## Pydantic Models (Summary)

```python
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import Any

class EvaluationStatus(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"

class StateTransition(BaseModel):
    state: EvaluationStatus
    at: datetime
    progress: int | None = None

class Progress(BaseModel):
    current_task: int = Field(ge=1)
    total_tasks: int = Field(ge=1)
    percent: int = Field(ge=0, le=100)
    last_heartbeat: datetime

class EvaluationRequest(BaseModel):
    user_llm: str | None = None
    num_trials: int = Field(ge=1)
    num_tasks: int = Field(ge=1)

class TaskResult(BaseModel):
    task_id: str
    success: bool
    reward: float = Field(ge=0.0, le=1.0)
    trajectory: list[dict[str, Any]] | None = None

class EvaluationResults(BaseModel):
    success_rate: float = Field(ge=0.0, le=1.0)
    total_tasks: int = Field(ge=1)
    successful: int = Field(ge=0)
    tasks: list[TaskResult]

class Evaluation(BaseModel):
    evaluation_id: str = Field(pattern=r"^eval-\d{13}-[a-f0-9]{6}$")
    trace_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    session_id: str | None = Field(default=None, pattern=r"^sess-[a-z0-9]+$")
    status: EvaluationStatus
    domain: str
    agent_endpoint: str | None = None
    state_history: list[StateTransition]
    created_at: datetime
    completed_at: datetime | None = None
    request: EvaluationRequest
    results: EvaluationResults | None = None
    error: str | None = None
    progress: Progress | None = None

class LogEvent(BaseModel):
    ts: datetime
    level: str
    event: str
    evaluation_id: str
    trace_id: str | None = None
    session_id: str | None = None
```
