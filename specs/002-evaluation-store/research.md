# Research: Evaluation Store

**Feature**: 002-evaluation-store
**Date**: 2025-12-22
**Status**: Complete

## Overview

This document consolidates research findings for implementing the evaluation store. The spec is well-defined with minimal unknowns, so this research focuses on validating technical approaches.

---

## 1. Atomic Write Pattern (Python/Linux)

### Decision
Use **temp file + atomic rename** pattern with `pathlib.Path.rename()`.

### Rationale
- `os.rename()` and `Path.rename()` are atomic on POSIX systems when source and destination are on the same filesystem
- Prevents partial reads - file either exists completely or doesn't
- Standard pattern used by databases, config management tools, and loggers

### Implementation Pattern
```python
from pathlib import Path
import tempfile
import json

def atomic_write(path: Path, data: dict) -> None:
    """Write data atomically using temp file + rename."""
    # Create temp file in same directory (same filesystem)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())  # Ensure data hits disk before rename

    temp_path.rename(path)  # Atomic on POSIX
```

### Alternatives Considered
1. **Direct write with file locking (fcntl)** - Rejected: More complex, doesn't prevent partial reads on crash
2. **Write-ahead logging (WAL)** - Rejected: Overkill for our scale and use case
3. **Database (SQLite)** - Rejected: Violates "directory as database" design principle

### Key Insight
The `os.fsync()` call is optional but recommended for durability. Without it, data may be in OS buffer cache but not on disk. For our use case, the trade-off is acceptable since we're not a financial system.

---

## 2. JSON Lines Logging Pattern

### Decision
Use **append-only JSON Lines format** with loguru's structured logging.

### Rationale
- Each line is a complete, parseable JSON object
- Append-only enables efficient log rotation without locking
- Compatible with standard log analysis tools (jq, grep, Datadog agents)
- Survives partial writes - corrupted line doesn't affect others

### Implementation Pattern
```python
from loguru import logger
import sys

# Configure loguru for JSON output
logger.configure(
    handlers=[
        {
            "sink": sys.stderr,
            "format": "{message}",
            "serialize": True,  # JSON output
        },
        {
            "sink": "data/logs/events.jsonl",
            "format": "{message}",
            "serialize": True,
            "rotation": "00:00",  # Daily rotation at midnight UTC
            "compression": "gz",
            "retention": "14 days",
        },
    ]
)

# Log structured events
logger.bind(
    event="evaluation_started",
    evaluation_id="eval-123",
    trace_id="abc-456",
).info("Evaluation started")
```

### Alternatives Considered
1. **Custom file appender** - Rejected: loguru handles rotation, compression, retention
2. **Syslog** - Rejected: Adds external dependency, less portable
3. **Single JSON file** - Rejected: Requires rewriting entire file, not crash-safe

### Key Insight
loguru's `serialize=True` outputs JSON Lines format natively. Combined with `rotation` and `compression`, it handles all FR-009 requirements.

---

## 3. Flat Directory Storage

### Decision
Use **flat directory structure** for completed evaluations with file-age-based cleanup.

### Rationale
- Simpler implementation - no directory management needed
- Direct file access by evaluation_id - O(1) lookup
- File modification time provides age for retention cleanup
- Sufficient for our scale (5000 evaluations)

### Implementation Pattern
```python
from datetime import datetime, UTC, timedelta
from pathlib import Path
import os

def get_evaluation_path(evaluation_id: str, base_dir: Path) -> Path:
    """Get path for completed evaluation."""
    eval_dir = base_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir / f"{evaluation_id}.json"

def cleanup_expired(base_dir: Path, retention_days: int) -> int:
    """Remove evaluations older than retention period."""
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    deleted = 0

    eval_dir = base_dir / "evaluations"
    for path in eval_dir.glob("eval-*.json"):
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        if mtime < cutoff:
            path.unlink()
            deleted += 1

    return deleted
```

### Alternatives Considered
1. **Date-partitioned (YYYY/MM/DD)** - Rejected: Added complexity not needed at our scale
2. **Database index** - Rejected: Violates "directory as database" principle

### Key Insight
At 5000 files, a flat directory scan for cleanup takes ~50-100ms - acceptable for a daily cleanup task. The simplicity of flat storage outweighs any marginal performance benefit from date partitioning.

---

## 4. File Permissions

### Decision
Use **0o600 for data files, 0o640 for logs**.

### Rationale
- Data files (evaluations, sessions) contain potentially sensitive task details - owner-only access
- Log files need group read for log shipper processes (Datadog agent, fluentd)
- Follows principle of least privilege
- Consistent with spec FR-010 error handling requirements

### Implementation Pattern
```python
import os
from pathlib import Path

FILE_MODE_DATA = 0o600  # -rw-------
FILE_MODE_LOGS = 0o640  # -rw-r-----

def atomic_write_with_permissions(path: Path, data: dict, mode: int) -> None:
    """Write data atomically with specific permissions."""
    temp_path = path.with_suffix(path.suffix + ".tmp")

    # Set umask temporarily to get exact permissions
    old_umask = os.umask(0o077)
    try:
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        os.chmod(temp_path, mode)
    finally:
        os.umask(old_umask)

    temp_path.rename(path)
```

### Alternatives Considered
1. **World-readable (0o644)** - Rejected: Unnecessary exposure of evaluation data
2. **Group writable (0o660)** - Rejected: Only owner process should write data
3. **ACLs** - Rejected: Overly complex for single-user deployment model

### Key Insight
Docker containers typically run as a single user, so permissions are mainly relevant for shared host deployments and log shipper access.

---

## 5. ID Generation Strategy

### Decision
Use **timestamp-first format** (`eval-{unix_ms}-{random_6_chars}`).

### Rationale
- Time-sortable for chronological ordering in directory listings
- Consistent with existing `data/simulations/` naming convention (timestamp-first)
- Random suffix handles concurrent evaluation creation
- Human-readable timestamp for debugging

### Implementation Pattern
```python
import secrets
import time

def generate_evaluation_id() -> str:
    """Generate unique evaluation ID with timestamp prefix."""
    unix_ms = int(time.time() * 1000)
    random_suffix = secrets.token_hex(3)  # 6 hex chars
    return f"eval-{unix_ms}-{random_suffix}"
```

### Collision Analysis
- 6 hex chars = 16.7 million combinations per millisecond
- At 1000 concurrent evaluations/ms (impossible), collision probability ~0.003%
- Real-world collision probability negligible (~10⁻¹⁰ per evaluation)
- Still check-before-write to prevent silent data loss

### Alternatives Considered
1. **UUID v4** - Rejected: Not time-sortable, harder to debug
2. **UUID v7** - Acceptable alternative, but more complex than needed
3. **Sequential counter** - Rejected: Requires coordination, not distributed-safe

---

## 6. Heartbeat and Stale Session Detection

### Decision
Use **last_heartbeat timestamp in session file** with periodic check.

### Rationale
- Simple file-based approach, no external dependencies
- Session file already exists - just add timestamp field
- Cleanup process checks age of last_heartbeat
- Aligns with spec FR-007b (2-hour stale threshold)

### Implementation Pattern
```python
from datetime import datetime, UTC, timedelta

def update_heartbeat(session_path: Path) -> None:
    """Update session heartbeat timestamp."""
    with open(session_path) as f:
        data = json.load(f)

    data["progress"]["last_heartbeat"] = datetime.now(UTC).isoformat() + "Z"
    atomic_write(session_path, data)

def mark_stale_sessions(sessions_dir: Path, stale_hours: int = 2) -> list[str]:
    """Mark sessions without recent heartbeat as abandoned."""
    cutoff = datetime.now(UTC) - timedelta(hours=stale_hours)
    abandoned = []

    for path in sessions_dir.glob("eval-*.json"):
        with open(path) as f:
            data = json.load(f)

        heartbeat = data.get("progress", {}).get("last_heartbeat")
        if heartbeat:
            heartbeat_time = datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
            if heartbeat_time < cutoff:
                data["status"] = "abandoned"
                atomic_write(path, data)
                abandoned.append(data["evaluation_id"])

    return abandoned
```

### Alternatives Considered
1. **External process/watchdog** - Rejected: Adds deployment complexity
2. **Lock files** - Rejected: Stale locks are harder to clean up
3. **TTL-based cleanup only** - Rejected: Can't distinguish "slow" from "dead"

---

## Summary

All technical approaches are validated and align with the spec. No unresolved unknowns remain. The implementation will use:

1. **Atomic writes**: temp file + rename with optional fsync
2. **Structured logging**: loguru with JSON serialization
3. **Flat directory storage**: Simple evaluations/ directory with file-age cleanup
4. **File permissions**: 0o600 for data, 0o640 for logs
5. **ID generation**: Unix timestamp (ms) + random suffix
6. **Stale detection**: Heartbeat timestamp in session files

Ready for Phase 1: Design artifacts.
