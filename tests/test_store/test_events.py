"""
Tests for the EventLogger in the Evaluation Store.

Tests structured event logging with JSON Lines output, trace_id correlation,
stdout output, and standard event types.
"""

import json
import os
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from tau2.store import LogEvent
from tau2.store.config import get_data_dir
from tau2.store.events import EventLogger, create_event_logger
from tau2.store.utils import ensure_directories


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with required subdirectories."""
    data_dir = tmp_path / "data"
    ensure_directories(data_dir)
    return data_dir


@pytest.fixture
def event_logger(temp_data_dir: Path) -> EventLogger:
    """Create an EventLogger instance for testing."""
    return create_event_logger(data_dir=temp_data_dir, stdout=False)


@pytest.fixture
def sample_evaluation_id() -> str:
    """Return a sample evaluation ID for testing."""
    return "eval-1735300000000-abc123"


@pytest.fixture
def sample_trace_id() -> str:
    """Return a sample W3C trace ID for testing."""
    return "4bf92f3577b34da6a3ce929d0e0e4736"


@pytest.fixture
def sample_session_id() -> str:
    """Return a sample SSE session ID for testing."""
    return "sess-xyz789"


class TestEventLoggerCreation:
    """Tests for EventLogger initialization."""

    def test_create_event_logger_with_defaults(self, temp_data_dir: Path) -> None:
        """Test that create_event_logger works with default settings."""
        logger = create_event_logger(data_dir=temp_data_dir)
        assert logger is not None
        assert isinstance(logger, EventLogger)

    def test_create_event_logger_with_custom_data_dir(
        self, temp_data_dir: Path
    ) -> None:
        """Test that create_event_logger accepts custom data directory."""
        logger = create_event_logger(data_dir=temp_data_dir)
        assert logger.data_dir == temp_data_dir

    def test_create_event_logger_with_stdout_enabled(
        self, temp_data_dir: Path
    ) -> None:
        """Test that create_event_logger can enable stdout output."""
        logger = create_event_logger(data_dir=temp_data_dir, stdout=True)
        assert logger._stdout is True

    def test_create_event_logger_with_stdout_disabled(
        self, temp_data_dir: Path
    ) -> None:
        """Test that create_event_logger can disable stdout output."""
        logger = create_event_logger(data_dir=temp_data_dir, stdout=False)
        assert logger._stdout is False


class TestLogEventToFile:
    """Tests for logging events to the events.jsonl file."""

    def test_log_event_creates_file(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test that logging an event creates the events.jsonl file."""
        event_logger.log_event(
            "evaluation_created",
            sample_evaluation_id,
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        assert events_file.exists()

    def test_log_event_writes_valid_json(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test that logged events are valid JSON."""
        event_logger.log_event(
            "evaluation_created",
            sample_evaluation_id,
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            line = f.readline()
            data = json.loads(line)

        assert data["event"] == "evaluation_created"
        assert data["evaluation_id"] == sample_evaluation_id

    def test_log_event_includes_timestamp(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test that logged events include a timestamp."""
        before = datetime.now(timezone.utc)
        event_logger.log_event(
            "evaluation_created",
            sample_evaluation_id,
        )
        after = datetime.now(timezone.utc)

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())

        ts_str = data["ts"]
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

        assert before <= ts <= after

    def test_log_event_appends_to_file(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test that multiple events are appended to the same file."""
        event_logger.log_event("evaluation_created", sample_evaluation_id)
        event_logger.log_event("evaluation_started", sample_evaluation_id)
        event_logger.log_event("task_completed", sample_evaluation_id, task_num=1)

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            lines = f.readlines()

        assert len(lines) == 3

        events = [json.loads(line) for line in lines]
        assert events[0]["event"] == "evaluation_created"
        assert events[1]["event"] == "evaluation_started"
        assert events[2]["event"] == "task_completed"

    def test_log_event_includes_level(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test that logged events include the log level."""
        event_logger.log_event(
            "evaluation_failed",
            sample_evaluation_id,
            level="error",
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())

        assert data["level"] == "error"

    def test_log_event_default_level_is_info(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test that the default log level is 'info'."""
        event_logger.log_event(
            "evaluation_created",
            sample_evaluation_id,
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())

        assert data["level"] == "info"

    def test_log_event_with_extra_fields(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test that additional keyword arguments are included in the event."""
        event_logger.log_event(
            "task_completed",
            sample_evaluation_id,
            task_num=5,
            total_tasks=10,
            success=True,
            reward=0.8,
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())

        assert data["task_num"] == 5
        assert data["total_tasks"] == 10
        assert data["success"] is True
        assert data["reward"] == 0.8


class TestLogEventWithTraceId:
    """Tests for logging events with OTel trace_id correlation."""

    def test_log_event_with_trace_id(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
        sample_trace_id: str,
    ) -> None:
        """Test that trace_id is included in the logged event."""
        event_logger.log_event(
            "evaluation_created",
            sample_evaluation_id,
            trace_id=sample_trace_id,
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())

        assert data["trace_id"] == sample_trace_id

    def test_log_event_without_trace_id(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test that trace_id can be omitted."""
        event_logger.log_event(
            "evaluation_created",
            sample_evaluation_id,
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())

        assert data.get("trace_id") is None

    def test_log_event_with_session_id(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
        sample_session_id: str,
    ) -> None:
        """Test that session_id is included in the logged event."""
        event_logger.log_event(
            "evaluation_created",
            sample_evaluation_id,
            session_id=sample_session_id,
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())

        assert data["session_id"] == sample_session_id

    def test_log_event_with_all_correlation_ids(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
        sample_trace_id: str,
        sample_session_id: str,
    ) -> None:
        """Test that all correlation IDs are included together."""
        event_logger.log_event(
            "evaluation_created",
            sample_evaluation_id,
            trace_id=sample_trace_id,
            session_id=sample_session_id,
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())

        assert data["evaluation_id"] == sample_evaluation_id
        assert data["trace_id"] == sample_trace_id
        assert data["session_id"] == sample_session_id


class TestLogEventToStdout:
    """Tests for logging events to stdout."""

    def test_log_event_to_stdout_when_enabled(
        self,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test that events are written to stdout when enabled."""
        logger = create_event_logger(data_dir=temp_data_dir, stdout=True)

        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            logger.log_event(
                "evaluation_created",
                sample_evaluation_id,
            )

        output = captured_output.getvalue()
        assert output.strip()  # Should have output

        # Verify it's valid JSON
        data = json.loads(output.strip())
        assert data["event"] == "evaluation_created"
        assert data["evaluation_id"] == sample_evaluation_id

    def test_log_event_not_to_stdout_when_disabled(
        self,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test that events are not written to stdout when disabled."""
        logger = create_event_logger(data_dir=temp_data_dir, stdout=False)

        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            logger.log_event(
                "evaluation_created",
                sample_evaluation_id,
            )

        output = captured_output.getvalue()
        assert output == ""  # Should be empty

    def test_log_event_to_both_file_and_stdout(
        self,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test that events are written to both file and stdout."""
        logger = create_event_logger(data_dir=temp_data_dir, stdout=True)

        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            logger.log_event(
                "evaluation_created",
                sample_evaluation_id,
                domain="airline",
            )

        # Check stdout
        stdout_data = json.loads(captured_output.getvalue().strip())
        assert stdout_data["event"] == "evaluation_created"

        # Check file
        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            file_data = json.loads(f.readline())

        assert file_data["event"] == "evaluation_created"
        assert file_data["domain"] == "airline"

    def test_stdout_from_env_variable(self, temp_data_dir: Path) -> None:
        """Test that TAU2_LOG_STDOUT env variable controls stdout output."""
        with patch.dict(os.environ, {"TAU2_LOG_STDOUT": "true"}):
            logger = create_event_logger(data_dir=temp_data_dir)
            assert logger._stdout is True

        with patch.dict(os.environ, {"TAU2_LOG_STDOUT": "false"}):
            logger = create_event_logger(data_dir=temp_data_dir)
            assert logger._stdout is False


class TestStandardEventTypes:
    """Tests for standard event types as defined in data-model.md."""

    def test_evaluation_created_event(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test logging evaluation_created event with expected fields."""
        event_logger.log_event(
            "evaluation_created",
            sample_evaluation_id,
            domain="airline",
            agent_endpoint="http://localhost:8080/a2a",
            num_tasks=10,
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())

        assert data["event"] == "evaluation_created"
        assert data["domain"] == "airline"
        assert data["agent_endpoint"] == "http://localhost:8080/a2a"
        assert data["num_tasks"] == 10

    def test_evaluation_started_event(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test logging evaluation_started event."""
        event_logger.log_event(
            "evaluation_started",
            sample_evaluation_id,
            domain="retail",
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())

        assert data["event"] == "evaluation_started"
        assert data["domain"] == "retail"

    def test_task_completed_event(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test logging task_completed event with task details."""
        event_logger.log_event(
            "task_completed",
            sample_evaluation_id,
            task_num=3,
            total_tasks=10,
            success=True,
            reward=0.95,
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())

        assert data["event"] == "task_completed"
        assert data["task_num"] == 3
        assert data["total_tasks"] == 10
        assert data["success"] is True
        assert data["reward"] == 0.95

    def test_evaluation_completed_event(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test logging evaluation_completed event with results."""
        event_logger.log_event(
            "evaluation_completed",
            sample_evaluation_id,
            success_rate=0.85,
            duration_s=3600,
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())

        assert data["event"] == "evaluation_completed"
        assert data["success_rate"] == 0.85
        assert data["duration_s"] == 3600

    def test_evaluation_failed_event(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test logging evaluation_failed event with error details."""
        event_logger.log_event(
            "evaluation_failed",
            sample_evaluation_id,
            level="error",
            error_type="ConnectionError",
            error_message="Agent connection timeout",
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())

        assert data["event"] == "evaluation_failed"
        assert data["level"] == "error"
        assert data["error_type"] == "ConnectionError"
        assert data["error_message"] == "Agent connection timeout"

    def test_evaluation_abandoned_event(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test logging evaluation_abandoned event."""
        event_logger.log_event(
            "evaluation_abandoned",
            sample_evaluation_id,
            level="warning",
            last_heartbeat="2025-12-22T10:00:00Z",
            stale_hours=2,
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())

        assert data["event"] == "evaluation_abandoned"
        assert data["level"] == "warning"
        assert data["last_heartbeat"] == "2025-12-22T10:00:00Z"
        assert data["stale_hours"] == 2

    def test_cleanup_started_event(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test logging cleanup_started event."""
        event_logger.log_event(
            "cleanup_started",
            sample_evaluation_id,
            retention_days=30,
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())

        assert data["event"] == "cleanup_started"
        assert data["retention_days"] == 30

    def test_cleanup_completed_event(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test logging cleanup_completed event."""
        event_logger.log_event(
            "cleanup_completed",
            sample_evaluation_id,
            deleted_count=5,
            duration_s=1.5,
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())

        assert data["event"] == "cleanup_completed"
        assert data["deleted_count"] == 5
        assert data["duration_s"] == 1.5


class TestLogEventModel:
    """Tests for LogEvent model integration."""

    def test_log_event_can_be_parsed_as_model(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
        sample_trace_id: str,
    ) -> None:
        """Test that logged events can be parsed back as LogEvent models."""
        event_logger.log_event(
            "evaluation_created",
            sample_evaluation_id,
            trace_id=sample_trace_id,
            domain="airline",
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())

        log_event = LogEvent(**data)
        assert log_event.event == "evaluation_created"
        assert log_event.evaluation_id == sample_evaluation_id
        assert log_event.trace_id == sample_trace_id


class TestFilePermissions:
    """Tests for log file permissions."""

    def test_log_file_has_correct_permissions(
        self,
        event_logger: EventLogger,
        temp_data_dir: Path,
        sample_evaluation_id: str,
    ) -> None:
        """Test that log files have 0o640 permissions (group readable)."""
        event_logger.log_event(
            "evaluation_created",
            sample_evaluation_id,
        )

        events_file = temp_data_dir / "logs" / "events.jsonl"
        mode = events_file.stat().st_mode & 0o777

        # Log files should be 0o640 (owner rw, group r, others none)
        assert mode == 0o640
