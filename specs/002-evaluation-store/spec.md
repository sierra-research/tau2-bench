# Feature Specification: Evaluation Store

**Feature Branch**: `002-evaluation-store`
**Created**: 2025-11-24
**Updated**: 2025-12-22
**Status**: Draft

**Input**: "Persist evaluation results using simple filesystem storage for a platform serving up to 5000 users"

## Problem Statement

tau2_agent needs to persist evaluation results so platforms can retrieve them later. Current implementation returns results synchronously and doesn't store them.

### Additional Context (2025-12-21)

With streaming evaluations (003-async-evaluation) and observability integration (006-otel, 007-datadog):
- Need correlation between stored evaluations and distributed traces
- Need session tracking for SSE reconnection after connection drops
- Need separation of in-progress vs completed evaluations
- Need mandatory retention policies for production deployments

## Design Principles

1. **Simple over clever** - No premature optimization for scale we don't need
2. **Immutable completed records** - Completed evaluations are write-once, never update
3. **Mutable in-progress state** - Active sessions can be updated (progress, heartbeat)
4. **Atomic writes** - temp file + rename ensures no partial reads
5. **Directory as database** - Filesystem is the source of truth
6. **Observability-ready** - Include trace_id for correlation with OTel/Datadog traces
7. **Retention by default** - Automatic cleanup prevents unbounded storage growth
8. **UTC everywhere** - All timestamps in ISO 8601 with Z suffix

## Architecture

```
$DATA_DIR/
├── simulations/                    # EXISTING - CLI batch runs (untouched)
│   └── {timestamp}_{domain}_{agent}.json
├── evaluations/                    # NEW - Completed API evaluations (immutable)
│   ├── eval-1732449600000-a1b2c3.json
│   └── eval-1732449660000-d4e5f6.json
├── sessions/                       # NEW - In-progress evaluations (mutable)
│   ├── eval-1732449720000-g7h8i9.json           # Active evaluation with progress
│   └── eval-1732449720000-g7h8i9.json.tmp       # Atomic write in progress
└── logs/                           # Structured event logs
    ├── events.jsonl               # Rolling JSON lines log
    └── archive/                    # Compressed old logs
```

**Three-directory design**:
- `simulations/` - Existing CLI batch results (unchanged by this feature)
- `sessions/` - Mutable, frequently updated, short-lived (deleted on completion)
- `evaluations/` - Immutable, append-only, retained per policy (API results)

**Note**: `evaluations/` is separate from `simulations/` to avoid modifying existing CLI behavior. Future work may consolidate these.

### ID Generation

Evaluation IDs follow the format `eval-{unix_ms}-{random_6_chars}`:
- `unix_ms`: UTC Unix timestamp in milliseconds
- `random_6_chars`: 6 alphanumeric characters for uniqueness

Example: `eval-1732449600000-x7k9m2`

This format provides:
- **Time-sortability**: Chronological ordering in directory listings
- **Consistency**: Aligns with existing `data/simulations/` timestamp-first naming convention
- **Uniqueness**: Random suffix handles concurrent evaluation creation
- **Human-readability**: Timestamp is easily parseable for debugging

**Collision handling**: Check if file exists before write; raise error on collision (caller retries with new ID). Collision probability is negligible (~10⁻¹⁰ per evaluation) but check prevents silent data loss.

### File Format (Completed Evaluation)

```json
{
  "evaluation_id": "eval-1732449600000-x7k9m2",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "session_id": "sess-xyz789",
  "status": "completed",
  "domain": "airline",
  "agent_endpoint": "http://example.com/agent",
  "state_history": [
    {"state": "submitted", "at": "2025-11-24T10:00:00Z"},
    {"state": "working", "at": "2025-11-24T10:00:01Z", "progress": 0},
    {"state": "working", "at": "2025-11-24T10:07:30Z", "progress": 50},
    {"state": "completed", "at": "2025-11-24T10:15:00Z", "progress": 100}
  ],
  "created_at": "2025-11-24T10:00:00Z",
  "completed_at": "2025-11-24T10:15:00Z",
  "request": {
    "user_llm": "gpt-4o",
    "num_trials": 1,
    "num_tasks": 5
  },
  "results": {
    "success_rate": 0.8,
    "total_tasks": 5,
    "successful": 4,
    "tasks": [...]
  },
  "error": null
}
```

| Field | Purpose | OTel/Datadog Correlation |
|-------|---------|--------------------------|
| `trace_id` | W3C Trace Context ID | Links to OTel spans in Jaeger/Datadog APM |
| `session_id` | SSE session identifier | Enables reconnection to streaming evaluation |
| `state_history` | State machine transitions | Audit trail, debugging failed evaluations |

### File Format (In-Progress Session)

```json
{
  "evaluation_id": "eval-1732449660000-ghi789",
  "trace_id": "8af342b1234c56d7e8f901234a567890",
  "session_id": "sess-abc123",
  "status": "working",
  "domain": "retail",
  "agent_endpoint": "http://example.com/agent",
  "progress": {
    "current_task": 3,
    "total_tasks": 5,
    "percent": 60,
    "last_heartbeat": "2025-11-24T10:12:00Z"
  },
  "state_history": [
    {"state": "submitted", "at": "2025-11-24T10:00:00Z"},
    {"state": "working", "at": "2025-11-24T10:00:01Z", "progress": 0}
  ],
  "created_at": "2025-11-24T10:00:00Z",
  "request": {
    "user_llm": "gpt-4o",
    "num_trials": 1,
    "num_tasks": 5
  }
}
```

### Write Path (Atomic)

```python
from datetime import datetime
from pathlib import Path

def save_session(evaluation_id: str, data: dict) -> None:
    """Save in-progress session (mutable, can be updated)."""
    path = SESSION_DIR / f"{evaluation_id}.json"
    temp_path = SESSION_DIR / f"{evaluation_id}.json.tmp"

    # Update heartbeat
    data["progress"]["last_heartbeat"] = datetime.utcnow().isoformat() + "Z"

    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)

    temp_path.rename(path)


def finalize_evaluation(evaluation_id: str, data: dict) -> None:
    """Move completed evaluation from sessions to evaluations (immutable)."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    path = EVAL_DIR / f"{evaluation_id}.json"
    temp_path = EVAL_DIR / f"{evaluation_id}.json.tmp"

    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)

    temp_path.rename(path)

    # Remove from sessions
    session_path = SESSION_DIR / f"{evaluation_id}.json"
    if session_path.exists():
        session_path.unlink()
```

### Read Path (No Locking Needed)

```python
def get_evaluation(evaluation_id: str) -> dict | None:
    """Read evaluation from either sessions (in-progress) or evaluations (completed)."""
    # Check sessions first (in-progress)
    session_path = SESSION_DIR / f"{evaluation_id}.json"
    if session_path.exists():
        with open(session_path) as f:
            return json.load(f)

    # Check completed evaluations
    eval_path = EVAL_DIR / f"{evaluation_id}.json"
    if eval_path.exists():
        with open(eval_path) as f:
            return json.load(f)

    return None


def get_session_by_trace_id(trace_id: str) -> dict | None:
    """Find session by OTel trace ID (for reconnection)."""
    for path in SESSION_DIR.glob("eval-*.json"):
        with open(path) as f:
            data = json.load(f)
            if data.get("trace_id") == trace_id:
                return data
    return None
```

Why no locking?
- Completed files are immutable after creation
- Atomic rename means file either exists completely or doesn't
- Readers never see partial data
- Session updates are atomic (temp + rename)

### List Path (Simple Scan)

```python
def list_evaluations(
    domain: str | None = None,
    status: str | None = None,
    include_sessions: bool = True,
) -> list[dict]:
    """List evaluations from both sessions and completed store."""
    results = []

    # List in-progress sessions
    if include_sessions:
        for path in SESSION_DIR.glob("eval-*.json"):
            if path.name.endswith('.tmp'):
                continue
            with open(path) as f:
                data = json.load(f)
            if domain and data.get("domain") != domain:
                continue
            if status and data.get("status") != status:
                continue
            results.append(_summary(data))

    # List completed evaluations (flat directory)
    for path in EVAL_DIR.glob("eval-*.json"):
        with open(path) as f:
            data = json.load(f)
        if domain and data.get("domain") != domain:
            continue
        if status and data.get("status") != status:
            continue
        results.append(_summary(data))

    return sorted(results, key=lambda x: x["created_at"], reverse=True)


def _summary(data: dict) -> dict:
    """Extract summary fields for listing."""
    return {
        "evaluation_id": data["evaluation_id"],
        "trace_id": data.get("trace_id"),
        "session_id": data.get("session_id"),
        "status": data["status"],
        "domain": data["domain"],
        "created_at": data["created_at"],
        "progress": data.get("progress", {}).get("percent"),
    }
```

At 5000 files, this takes ~50-100ms. Acceptable for our scale.

## Requirements

### Functional Requirements

- **FR-001**: Save evaluation results as JSON files with atomic writes
- **FR-002**: Retrieve evaluation by ID (checks sessions first, then evaluations)
- **FR-003**: List evaluations with optional domain and status filters
- **FR-004**: Support status values: submitted, working, completed, failed, abandoned
- **FR-005**: Include timestamps for created_at, completed_at, and state_history
- **FR-006**: Store full results payload for completed evaluations

### Observability Integration (P0)

- **FR-006a**: Include `trace_id` field (W3C Trace Context format) for OTel/Datadog correlation
- **FR-006b**: Include `session_id` field for SSE streaming reconnection (see 003-async-evaluation)
- **FR-006c**: Maintain `state_history` array with timestamped state transitions
- **FR-006d**: Support lookup by `trace_id` for reconnection scenarios

### Session Management (P0)

- **FR-006e**: Store in-progress evaluations in `sessions/` directory (mutable)
- **FR-006f**: Store completed evaluations in `evaluations/` directory (immutable, flat)
- **FR-006g**: Update session heartbeat on each progress update
- **FR-006h**: Move session to evaluations atomically on completion (delete session file)

### Retention & Cleanup (P1 - Mandatory)

- **FR-007**: Delete completed evaluations older than retention period (default: 30 days, configurable via `TAU2_RETENTION_DAYS`)
- **FR-007a**: Delete failed evaluations older than 7 days (less retention value)
- **FR-007b**: Mark sessions as `abandoned` if no heartbeat for 2 hours
- **FR-007c**: Delete abandoned sessions after 24 hours
- **FR-008**: Cleanup runs on startup and daily via scheduled task
- **FR-008a**: Cleanup uses file modification time to determine age (simple file-based retention)

### Structured Logging (P1)

- **FR-009**: Emit structured JSON logs to `$DATA_DIR/logs/events.jsonl`
- **FR-009a**: Log events with consistent schema: `{timestamp, level, event, evaluation_id, trace_id, session_id, ...}`
- **FR-009b**: Required events: `evaluation_created`, `evaluation_started`, `task_completed`, `evaluation_completed`, `evaluation_failed`
- **FR-009c**: Include `trace_id` in all log events for OTel correlation
- **FR-009d**: Rotate logs daily, compress logs older than 3 days
- **FR-009e**: Delete logs older than 14 days (configurable via `TAU2_LOG_RETENTION_DAYS`)
- **FR-009f**: Emit events to stdout when `TAU2_LOG_STDOUT=true` (auto-enabled in container environments)

```json
// Example log event (events.jsonl)
{"ts": "2025-12-21T10:00:00Z", "level": "info", "event": "evaluation_started", "evaluation_id": "eval-1732449600000-a1b2c3", "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736", "domain": "airline", "agent_endpoint": "http://example.com/agent"}
{"ts": "2025-12-21T10:05:00Z", "level": "info", "event": "task_completed", "evaluation_id": "eval-1732449600000-a1b2c3", "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736", "task_num": 1, "total_tasks": 5, "success": true, "reward": 0.8}
{"ts": "2025-12-21T10:15:00Z", "level": "info", "event": "evaluation_completed", "evaluation_id": "eval-1732449600000-a1b2c3", "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736", "success_rate": 0.8, "duration_s": 900}
```

**Note**: When OTel (006) or Datadog (007) is enabled, prefer emitting spans over logs for request-level details. These structured logs serve as a fallback and for offline analysis.

### Error Handling (P0)

- **FR-010**: On disk write failure (full disk, permission error, IO error), raise exception immediately
- **FR-010a**: Log error details to stderr (ensures visibility even if log file write fails)
- **FR-010b**: Do not retry writes - let caller decide retry strategy
- **FR-010c**: Clean up temp files on failure (remove `.tmp` files from failed atomic writes)
- **FR-010d**: Check file exists before write; raise `EvaluationIdCollisionError` if ID already exists

## What We're NOT Building

- ❌ Index files - Directory scan is fast enough at our scale
- ❌ File locking for reads - Immutable completed files don't need it
- ❌ Distributed storage - Single node is sufficient
- ❌ Request-level logging - Delegated to OTel spans (006) or Datadog (007)
- ❌ Log aggregation/search - Use OTel backends (Jaeger) or Datadog for that
- ❌ Real-time log streaming - SSE handles progress, logs are for post-hoc analysis

## Success Criteria

- **SC-001**: Can save and retrieve evaluations (both in-progress and completed)
- **SC-002**: List 5000 evaluations in < 500ms
- **SC-003**: No data corruption from concurrent access
- **SC-004**: Works after process restart (sessions persist)
- **SC-005**: Can retrieve evaluation by `trace_id` for OTel correlation
- **SC-006**: Cleanup removes files older than retention period without manual intervention
- **SC-007**: Structured logs contain `trace_id` for correlation with OTel/Datadog traces
- **SC-008**: Stale sessions marked as `abandoned` after 2 hours without heartbeat

## Implementation Estimate

This is ~200 lines of Python. Core modules:

**store.py** (~100 lines):
- `save_session()` - 15 lines
- `finalize_evaluation()` - 20 lines
- `get_evaluation()` - 20 lines
- `get_session_by_trace_id()` - 10 lines
- `list_evaluations()` - 25 lines

**retention.py** (~60 lines):
- `cleanup_expired()` - 25 lines
- `mark_abandoned_sessions()` - 15 lines
- `rotate_logs()` - 20 lines

**events.py** (~40 lines):
- `log_event()` - 15 lines
- `EventLogger` class - 25 lines

## Configuration

```python
# Environment variables with defaults
TAU2_DATA_DIR = os.getenv("TAU2_DATA_DIR", "./data")
TAU2_RETENTION_DAYS = int(os.getenv("TAU2_RETENTION_DAYS", "30"))
TAU2_FAILED_RETENTION_DAYS = int(os.getenv("TAU2_FAILED_RETENTION_DAYS", "7"))
TAU2_LOG_RETENTION_DAYS = int(os.getenv("TAU2_LOG_RETENTION_DAYS", "14"))
TAU2_SESSION_STALE_HOURS = int(os.getenv("TAU2_SESSION_STALE_HOURS", "2"))
TAU2_SESSION_CLEANUP_HOURS = int(os.getenv("TAU2_SESSION_CLEANUP_HOURS", "24"))

# File permissions (all readers run as tau2_agent user/group)
FILE_MODE_DATA = 0o600  # Owner read/write only for evaluation and session files
FILE_MODE_LOGS = 0o640  # Owner read/write, group read for log files (log shippers)

# Logging output configuration
TAU2_LOG_STDOUT = os.getenv("TAU2_LOG_STDOUT", "true" if _in_container() else "false").lower() == "true"
```

## Dependencies

- **006-otel-integration**: Provides `trace_id` via OTel context propagation
- **003-async-evaluation**: Uses `session_id` for SSE reconnection
- **007-datadog-project**: Uses `trace_id` for Datadog APM correlation

## Clarifications

### Session 2025-12-22

- Q: What file permission mode should be used for stored evaluation files? → A: Same user/group only (0600 for data files, 0640 for logs)
- Q: How should the store handle disk write failures? → A: Fail fast with error (raise exception, log to stderr, let caller handle)
- Q: Should structured events be emitted to stdout in addition to the log file? → A: Both stdout and file (configurable via TAU2_LOG_STDOUT, default true in containers)
- Q: What timezone should be used for all timestamps? → A: UTC always (ISO 8601 with Z suffix)
- Q: How should evaluation IDs be generated? → A: Timestamp + random (`eval-{unix_ms}-{random_6_chars}`) for consistency with existing simulation file naming; check-before-write to prevent collision overwrites
- Q: Why separate `evaluations/` from existing `simulations/`? → A: `simulations/` is designed for CLI batch runs (one file per batch, non-atomic writes). `evaluations/` is for concurrent API evaluations (one file per evaluation, atomic writes, session tracking). Keeping them separate avoids modifying working CLI behavior. Future consolidation possible.
- Q: Do we need date-partitioned directories? → A: No. Simple flat directory with file-age-based cleanup is sufficient for our scale. Removed to simplify implementation.

## Open Questions

1. ~~Default retention period - 7 days? 30 days?~~ **Resolved**: 30 days for completed, 7 days for failed
2. ~~Should we store in-progress evaluations here or only completed ones?~~ **Resolved**: Separate `sessions/` for in-progress, `evaluations/` for completed
3. ~~Should we emit events to stdout in addition to file?~~ **Resolved**: Both, configurable via `TAU2_LOG_STDOUT` (default true in containers)
