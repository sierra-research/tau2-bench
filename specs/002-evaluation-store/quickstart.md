# Quickstart: Evaluation Store

**Feature**: 002-evaluation-store

## Installation

No additional dependencies required - uses existing pydantic and loguru from tau2-bench.

## Configuration

Set environment variables (all optional with sensible defaults):

```bash
# Base directory for all storage (default: ./data)
export TAU2_DATA_DIR="./data"

# Retention settings
export TAU2_RETENTION_DAYS="30"       # Completed evaluations
export TAU2_FAILED_RETENTION_DAYS="7" # Failed evaluations
export TAU2_LOG_RETENTION_DAYS="14"   # Log files

# Session settings
export TAU2_SESSION_STALE_HOURS="2"   # Heartbeat timeout
export TAU2_SESSION_CLEANUP_HOURS="24" # Abandoned cleanup

# Logging
export TAU2_LOG_STDOUT="true"         # Also emit to stdout (auto-true in containers)
```

## Basic Usage

### Create and Track an Evaluation

```python
from tau2.store import create_store, create_event_logger

# Initialize
store = create_store()
logger = create_event_logger()

# Create a new evaluation session
evaluation_id = store.create_session(
    domain="airline",
    request={"num_tasks": 5, "num_trials": 1},
    trace_id="4bf92f3577b34da6a3ce929d0e0e4736",  # From OTel
    session_id="sess-abc123",  # For SSE reconnection
    agent_endpoint="http://localhost:8080/a2a",
)

logger.log_event(
    "evaluation_created",
    evaluation_id,
    trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
    domain="airline",
    num_tasks=5,
)

# Update progress as tasks complete
for task_num in range(1, 6):
    store.update_progress(evaluation_id, task_num, total_tasks=5)
    logger.log_event(
        "task_completed",
        evaluation_id,
        task_num=task_num,
        total_tasks=5,
        success=True,
        reward=0.8,
    )

# Complete the evaluation
store.complete_evaluation(
    evaluation_id,
    results={
        "success_rate": 0.8,
        "total_tasks": 5,
        "successful": 4,
        "tasks": [...]
    },
)

logger.log_event(
    "evaluation_completed",
    evaluation_id,
    success_rate=0.8,
    duration_s=900,
)
```

### Retrieve Evaluations

```python
from tau2.store import create_store

store = create_store()

# Get by ID
evaluation = store.get_evaluation("eval-1732449600000-a1b2c3")
if evaluation:
    print(f"Status: {evaluation.status}")
    print(f"Results: {evaluation.results}")

# Find by trace ID (for OTel correlation)
session = store.get_evaluation_by_trace_id("4bf92f3577b34da6a3ce929d0e0e4736")
if session:
    print(f"Found session: {session.evaluation_id}")

# List evaluations with filters
evaluations = store.list_evaluations(
    domain="airline",
    status="completed",
    limit=10,
)
for e in evaluations:
    print(f"{e.evaluation_id}: {e.status} ({e.progress}%)")
```

### Handle Failures

```python
from tau2.store import create_store, EvaluationNotFoundError, InvalidStateError

store = create_store()

try:
    store.fail_evaluation(
        evaluation_id="eval-1732449600000-a1b2c3",
        error="Agent connection timeout",
    )
except EvaluationNotFoundError as e:
    print(f"Not found: {e.evaluation_id}")
except InvalidStateError as e:
    print(f"Invalid state: {e.current_state} (expected: {e.expected_states})")
```

### Run Cleanup

```python
from tau2.store import create_retention_manager

manager = create_retention_manager()

# Mark stale sessions as abandoned
abandoned = manager.mark_abandoned_sessions()
print(f"Marked {len(abandoned)} sessions as abandoned")

# Delete expired evaluations
deleted = manager.cleanup_expired_evaluations()
print(f"Deleted {deleted} expired evaluations")

# Cleanup old abandoned sessions
cleaned = manager.cleanup_abandoned_sessions()
print(f"Cleaned {cleaned} abandoned sessions")

# Rotate and compress logs
rotated = manager.rotate_logs()
print(f"Rotated {rotated} log files")
```

## Integration with tau2-bench

### In Orchestrator (future integration)

```python
from tau2.orchestrator import Orchestrator
from tau2.store import create_store, create_event_logger

store = create_store()
logger = create_event_logger()

class PersistentOrchestrator(Orchestrator):
    """Orchestrator with evaluation persistence."""

    def run_evaluation(self, config):
        # Create session at start
        evaluation_id = store.create_session(
            domain=config.domain,
            request=config.dict(),
            trace_id=get_current_trace_id(),  # From OTel context
        )

        try:
            # Run evaluation with progress tracking
            for task_num, result in enumerate(self._run_tasks(config), 1):
                store.update_progress(evaluation_id, task_num, config.num_tasks)
                logger.log_event(
                    "task_completed",
                    evaluation_id,
                    task_num=task_num,
                    success=result.success,
                )

            # Complete on success
            store.complete_evaluation(evaluation_id, results)
            return results

        except Exception as e:
            # Record failure
            store.fail_evaluation(evaluation_id, str(e))
            raise
```

### Startup Cleanup (in main.py or server startup)

```python
from tau2.store import create_retention_manager

def startup_cleanup():
    """Run cleanup tasks on startup."""
    manager = create_retention_manager()

    # Mark any sessions that died without cleanup
    manager.mark_abandoned_sessions()

    # Remove expired data
    manager.cleanup_expired_evaluations()
    manager.cleanup_abandoned_sessions()
    manager.rotate_logs()
```

## Directory Structure After Use

```
data/
├── simulations/                    # EXISTING - CLI batch runs (untouched)
│   └── {timestamp}_{domain}_{agent}.json
├── evaluations/                    # NEW - Completed API evaluations (flat)
│   ├── eval-1732449600000-a1b2c3.json
│   └── eval-1732449660000-d4e5f6.json
├── sessions/                       # NEW - In-progress evaluations (temporary)
│   └── eval-1732449720000-g7h8i9.json
└── logs/
    ├── events.jsonl               # Current log file
    └── archive/
        └── events.2025-12-20.jsonl.gz
```

## Log Event Examples

View logs with jq:

```bash
# All events for an evaluation
cat data/logs/events.jsonl | jq 'select(.evaluation_id == "eval-123")'

# All failed tasks
cat data/logs/events.jsonl | jq 'select(.event == "task_completed" and .success == false)'

# Events by trace_id (for OTel correlation)
cat data/logs/events.jsonl | jq 'select(.trace_id == "abc123")'
```
