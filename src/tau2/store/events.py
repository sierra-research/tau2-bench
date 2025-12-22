"""
Evaluation Store Event Logger

Structured JSON Lines event logging with file and stdout output.
Supports OTel/Datadog correlation via trace_id and session_id fields.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tau2.store.config import get_data_dir, get_log_stdout
from tau2.store.utils import ensure_directories

# File permission for log files (owner rw, group r)
FILE_MODE_LOGS = 0o640


class EventLogger:
    """Structured event logger for the evaluation store.

    Logs events to a JSON Lines file (events.jsonl) and optionally to stdout.
    All events include timestamp, level, event type, and evaluation_id.
    Optional trace_id and session_id fields enable OTel/Datadog correlation.

    Attributes:
        data_dir: Base directory for logs
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        stdout: bool | None = None,
    ):
        """Initialize the event logger.

        Args:
            data_dir: Base directory for logs (default: $TAU2_DATA_DIR or ./data)
            stdout: Whether to also emit to stdout (default: $TAU2_LOG_STDOUT)
        """
        if data_dir is None:
            data_dir = get_data_dir()
        self._data_dir = Path(data_dir)
        self._dirs = ensure_directories(self._data_dir)

        # Resolve stdout setting
        if stdout is None:
            self._stdout = get_log_stdout()
        else:
            self._stdout = stdout

    @property
    def data_dir(self) -> Path:
        """Get the base data directory."""
        return self._data_dir

    def log_event(
        self,
        event: str,
        evaluation_id: str,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
        level: str = "info",
        **kwargs: Any,
    ) -> None:
        """Log a structured event.

        Events are written to both the log file and optionally stdout.
        Each event is a single JSON object on its own line (JSON Lines format).

        Args:
            event: Event type (evaluation_created, task_completed, etc.)
            evaluation_id: Associated evaluation ID
            trace_id: OTel trace ID for correlation
            session_id: SSE session ID
            level: Log level (info, warning, error)
            **kwargs: Additional event-specific fields
        """
        # Build the event data
        now = datetime.now(timezone.utc)
        ts_str = now.isoformat().replace("+00:00", "Z")

        event_data: dict[str, Any] = {
            "ts": ts_str,
            "level": level,
            "event": event,
            "evaluation_id": evaluation_id,
        }

        # Add optional correlation IDs (only when present to reduce log size)
        if trace_id is not None:
            event_data["trace_id"] = trace_id
        if session_id is not None:
            event_data["session_id"] = session_id

        # Add any extra fields
        event_data.update(kwargs)

        # Serialize to JSON
        json_line = json.dumps(event_data, separators=(",", ":"))

        # Write to file
        self._write_to_file(json_line)

        # Optionally write to stdout
        if self._stdout:
            print(json_line, file=sys.stdout)
            sys.stdout.flush()

    def _write_to_file(self, json_line: str) -> None:
        """Write a JSON line to the events log file.

        Args:
            json_line: The JSON string to write
        """
        logs_dir = self._dirs["logs"]
        events_file = logs_dir / "events.jsonl"

        # Set umask temporarily to get exact permissions
        old_umask = os.umask(0o027)  # Results in 0o640 for files created with 0o666
        try:
            # Check if file exists to set permissions on creation
            file_exists = events_file.exists()

            with open(events_file, "a") as f:
                f.write(json_line + "\n")
                f.flush()
                os.fsync(f.fileno())

            # Set correct permissions if we just created the file
            if not file_exists:
                events_file.chmod(FILE_MODE_LOGS)

        finally:
            os.umask(old_umask)


def create_event_logger(
    data_dir: Path | str | None = None,
    stdout: bool | None = None,
) -> EventLogger:
    """Create an event logger instance.

    Args:
        data_dir: Base directory for logs (default: $TAU2_DATA_DIR or ./data)
        stdout: Whether to also emit to stdout (default: $TAU2_LOG_STDOUT)

    Returns:
        Configured EventLogger instance
    """
    return EventLogger(data_dir=data_dir, stdout=stdout)
