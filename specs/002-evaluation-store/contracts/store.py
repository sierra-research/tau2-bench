"""
Evaluation Store Interface Contract

This file defines the public API for the evaluation store module.
Implementation goes in src/tau2/store/store.py
"""

from abc import ABC, abstractmethod
from pathlib import Path

from .models import Evaluation, EvaluationSummary


class EvaluationStoreProtocol(ABC):
    """Protocol for evaluation storage operations."""

    @abstractmethod
    def create_session(
        self,
        domain: str,
        request: dict,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
        agent_endpoint: str | None = None,
    ) -> str:
        """Create a new in-progress evaluation session.

        Args:
            domain: Evaluation domain (airline, retail, etc.)
            request: Original evaluation request parameters
            trace_id: W3C Trace Context ID for OTel correlation
            session_id: SSE session ID for reconnection
            agent_endpoint: A2A agent endpoint URL

        Returns:
            Generated evaluation_id

        Raises:
            EvaluationIdCollisionError: If generated ID already exists (should retry)
            IOError: If unable to write to filesystem
        """
        ...

    @abstractmethod
    def update_progress(
        self,
        evaluation_id: str,
        current_task: int,
        total_tasks: int,
    ) -> None:
        """Update progress of an in-progress evaluation.

        Updates the progress fields and refreshes the heartbeat timestamp.
        Transitions status from SUBMITTED to WORKING on first update.

        Args:
            evaluation_id: Evaluation to update
            current_task: 1-indexed current task number
            total_tasks: Total number of tasks

        Raises:
            EvaluationNotFoundError: If evaluation_id not in sessions
            InvalidStateError: If evaluation is not in SUBMITTED or WORKING state
            IOError: If unable to write to filesystem
        """
        ...

    @abstractmethod
    def complete_evaluation(
        self,
        evaluation_id: str,
        results: dict,
    ) -> None:
        """Complete an evaluation and move to immutable storage.

        Moves evaluation from sessions/ to evaluations/.

        Args:
            evaluation_id: Evaluation to complete
            results: Final evaluation results

        Raises:
            EvaluationNotFoundError: If evaluation_id not in sessions
            InvalidStateError: If evaluation is already terminal
            IOError: If unable to write to filesystem
        """
        ...

    @abstractmethod
    def fail_evaluation(
        self,
        evaluation_id: str,
        error: str,
    ) -> None:
        """Mark an evaluation as failed and move to immutable storage.

        Moves evaluation from sessions/ to evaluations/.

        Args:
            evaluation_id: Evaluation to fail
            error: Error message describing the failure

        Raises:
            EvaluationNotFoundError: If evaluation_id not in sessions
            InvalidStateError: If evaluation is already terminal
            IOError: If unable to write to filesystem
        """
        ...

    @abstractmethod
    def get_evaluation(self, evaluation_id: str) -> Evaluation | None:
        """Retrieve evaluation by ID.

        Checks sessions first (in-progress), then evaluations (completed).

        Args:
            evaluation_id: Evaluation to retrieve

        Returns:
            Evaluation if found, None otherwise
        """
        ...

    @abstractmethod
    def get_evaluation_by_trace_id(self, trace_id: str) -> Evaluation | None:
        """Find session by OTel trace ID.

        Only searches in-progress sessions, not completed evaluations.

        Args:
            trace_id: W3C Trace Context ID

        Returns:
            Evaluation if found in sessions, None otherwise
        """
        ...

    @abstractmethod
    def list_evaluations(
        self,
        *,
        domain: str | None = None,
        status: str | None = None,
        include_sessions: bool = True,
        limit: int = 100,
    ) -> list[EvaluationSummary]:
        """List evaluations with optional filters.

        Args:
            domain: Filter by domain name
            status: Filter by evaluation status
            include_sessions: Whether to include in-progress sessions
            limit: Maximum number of results

        Returns:
            List of evaluation summaries, sorted by created_at descending
        """
        ...


class RetentionManagerProtocol(ABC):
    """Protocol for retention and cleanup operations."""

    @abstractmethod
    def cleanup_expired_evaluations(self) -> int:
        """Remove completed evaluations older than retention period.

        Uses TAU2_RETENTION_DAYS for completed evaluations (default: 30).
        Uses TAU2_FAILED_RETENTION_DAYS for failed evaluations (default: 7).

        Returns:
            Number of evaluations deleted
        """
        ...

    @abstractmethod
    def mark_abandoned_sessions(self) -> list[str]:
        """Mark stale sessions as abandoned.

        Sessions without heartbeat for TAU2_SESSION_STALE_HOURS (default: 2)
        are marked as abandoned.

        Returns:
            List of evaluation_ids that were marked abandoned
        """
        ...

    @abstractmethod
    def cleanup_abandoned_sessions(self) -> int:
        """Remove abandoned sessions older than cleanup threshold.

        Sessions abandoned for longer than TAU2_SESSION_CLEANUP_HOURS (default: 24)
        are deleted.

        Returns:
            Number of sessions deleted
        """
        ...

    @abstractmethod
    def rotate_logs(self) -> int:
        """Compress old logs and delete expired ones.

        Compresses logs older than 3 days.
        Deletes logs older than TAU2_LOG_RETENTION_DAYS (default: 14).

        Returns:
            Number of log files processed
        """
        ...


class EventLoggerProtocol(ABC):
    """Protocol for structured event logging."""

    @abstractmethod
    def log_event(
        self,
        event: str,
        evaluation_id: str,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
        level: str = "info",
        **kwargs,
    ) -> None:
        """Log a structured event.

        Events are written to both the log file and optionally stdout.

        Args:
            event: Event type (evaluation_created, task_completed, etc.)
            evaluation_id: Associated evaluation ID
            trace_id: OTel trace ID for correlation
            session_id: SSE session ID
            level: Log level (info, warning, error)
            **kwargs: Additional event-specific fields
        """
        ...


# Factory function signature
def create_store(data_dir: Path | str | None = None) -> EvaluationStoreProtocol:
    """Create an evaluation store instance.

    Args:
        data_dir: Base directory for storage (default: $TAU2_DATA_DIR or ./data)

    Returns:
        Configured EvaluationStore instance
    """
    ...


def create_retention_manager(
    data_dir: Path | str | None = None,
) -> RetentionManagerProtocol:
    """Create a retention manager instance.

    Args:
        data_dir: Base directory for storage (default: $TAU2_DATA_DIR or ./data)

    Returns:
        Configured RetentionManager instance
    """
    ...


def create_event_logger(
    data_dir: Path | str | None = None,
    stdout: bool | None = None,
) -> EventLoggerProtocol:
    """Create an event logger instance.

    Args:
        data_dir: Base directory for logs (default: $TAU2_DATA_DIR or ./data)
        stdout: Whether to also emit to stdout (default: $TAU2_LOG_STDOUT)

    Returns:
        Configured EventLogger instance
    """
    ...
