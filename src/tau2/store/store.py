"""
Evaluation Store - Core Storage Operations

Filesystem-based evaluation storage with atomic writes, session tracking,
and two-directory design (sessions for in-progress, evaluations for completed).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from tau2.store.config import get_data_dir
from tau2.store.exceptions import (
    EvaluationIdCollisionError,
    EvaluationNotFoundError,
    InvalidStateError,
)
from tau2.store.models import (
    Evaluation,
    EvaluationRequest,
    EvaluationResults,
    EvaluationStatus,
    EvaluationSummary,
    Progress,
    StateTransition,
)
from tau2.store.utils import (
    atomic_write,
    ensure_directories,
    generate_evaluation_id,
    get_evaluation_path,
    get_session_path,
)


class EvaluationStore:
    """Filesystem-based evaluation storage.

    Uses a two-directory design:
    - sessions/: Mutable in-progress evaluations
    - evaluations/: Immutable completed evaluations

    All writes are atomic using temp file + rename pattern.
    """

    def __init__(self, data_dir: Path | str | None = None):
        """Initialize the evaluation store.

        Args:
            data_dir: Base directory for storage (default: $TAU2_DATA_DIR or ./data)
        """
        if data_dir is None:
            data_dir = get_data_dir()
        self._data_dir = Path(data_dir)
        self._dirs = ensure_directories(self._data_dir)

    @property
    def data_dir(self) -> Path:
        """Get the base data directory."""
        return self._data_dir

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
            EvaluationIdCollisionError: If generated ID already exists
            IOError: If unable to write to filesystem
        """
        evaluation_id = generate_evaluation_id()
        session_path = get_session_path(evaluation_id, self._data_dir)
        eval_path = get_evaluation_path(evaluation_id, self._data_dir)

        # Check for collision (extremely unlikely but possible)
        if session_path.exists() or eval_path.exists():
            raise EvaluationIdCollisionError(evaluation_id)

        now = datetime.now(timezone.utc)

        # Parse request into EvaluationRequest model
        eval_request = EvaluationRequest(**request)

        # Create initial evaluation with SUBMITTED status
        evaluation = Evaluation(
            evaluation_id=evaluation_id,
            trace_id=trace_id,
            session_id=session_id,
            status=EvaluationStatus.SUBMITTED,
            domain=domain,
            agent_endpoint=agent_endpoint,
            state_history=[StateTransition(state=EvaluationStatus.SUBMITTED, at=now)],
            created_at=now,
            request=eval_request,
        )

        # Write atomically to sessions directory
        atomic_write(session_path, evaluation.model_dump(mode="json"))

        return evaluation_id

    def update_progress(
        self,
        evaluation_id: str,
        current_task: int,
        total_tasks: int,
    ) -> None:
        """Update progress of an in-progress evaluation.

        Updates the progress fields and refreshes the heartbeat timestamp.
        Transitions status to WORKING on first update.

        Args:
            evaluation_id: Evaluation to update
            current_task: 1-indexed current task number
            total_tasks: Total number of tasks

        Raises:
            EvaluationNotFoundError: If evaluation_id not in sessions
            InvalidStateError: If evaluation is in terminal state
            IOError: If unable to write to filesystem
        """
        session_path = get_session_path(evaluation_id, self._data_dir)

        if not session_path.exists():
            raise EvaluationNotFoundError(evaluation_id)

        # Load current evaluation
        with open(session_path) as f:
            data = json.load(f)

        evaluation = Evaluation.model_validate(data)

        # Check for terminal states
        terminal_states = {
            EvaluationStatus.COMPLETED,
            EvaluationStatus.FAILED,
            EvaluationStatus.ABANDONED,
        }
        if evaluation.status in terminal_states:
            raise InvalidStateError(
                evaluation_id,
                evaluation.status.value,
                [EvaluationStatus.SUBMITTED.value, EvaluationStatus.WORKING.value],
            )

        now = datetime.now(timezone.utc)

        # Calculate progress percentage: (current_task - 1) / total_tasks * 100
        percent = int((current_task - 1) / total_tasks * 100)

        # Update progress
        evaluation.progress = Progress(
            current_task=current_task,
            total_tasks=total_tasks,
            percent=percent,
            last_heartbeat=now,
        )

        # Transition to WORKING if currently SUBMITTED
        if evaluation.status == EvaluationStatus.SUBMITTED:
            evaluation.status = EvaluationStatus.WORKING
            evaluation.state_history.append(
                StateTransition(
                    state=EvaluationStatus.WORKING,
                    at=now,
                    progress=percent,
                )
            )

        # Write atomically
        atomic_write(session_path, evaluation.model_dump(mode="json"))

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
        session_path = get_session_path(evaluation_id, self._data_dir)
        eval_path = get_evaluation_path(evaluation_id, self._data_dir)

        if not session_path.exists():
            raise EvaluationNotFoundError(evaluation_id)

        # Load current evaluation
        with open(session_path) as f:
            data = json.load(f)

        evaluation = Evaluation.model_validate(data)

        # Check for terminal states
        terminal_states = {
            EvaluationStatus.COMPLETED,
            EvaluationStatus.FAILED,
            EvaluationStatus.ABANDONED,
        }
        if evaluation.status in terminal_states:
            raise InvalidStateError(
                evaluation_id,
                evaluation.status.value,
                [EvaluationStatus.SUBMITTED.value, EvaluationStatus.WORKING.value],
            )

        now = datetime.now(timezone.utc)

        # Parse results
        eval_results = EvaluationResults.model_validate(results)

        # Update evaluation
        evaluation.status = EvaluationStatus.COMPLETED
        evaluation.completed_at = now
        evaluation.results = eval_results
        evaluation.progress = None  # Clear progress for completed evaluations
        evaluation.state_history.append(
            StateTransition(state=EvaluationStatus.COMPLETED, at=now)
        )

        # Write to evaluations directory atomically
        atomic_write(eval_path, evaluation.model_dump(mode="json"))

        # Remove from sessions directory
        session_path.unlink()

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
        session_path = get_session_path(evaluation_id, self._data_dir)
        eval_path = get_evaluation_path(evaluation_id, self._data_dir)

        if not session_path.exists():
            raise EvaluationNotFoundError(evaluation_id)

        # Load current evaluation
        with open(session_path) as f:
            data = json.load(f)

        evaluation = Evaluation.model_validate(data)

        # Check for terminal states
        terminal_states = {
            EvaluationStatus.COMPLETED,
            EvaluationStatus.FAILED,
            EvaluationStatus.ABANDONED,
        }
        if evaluation.status in terminal_states:
            raise InvalidStateError(
                evaluation_id,
                evaluation.status.value,
                [EvaluationStatus.SUBMITTED.value, EvaluationStatus.WORKING.value],
            )

        now = datetime.now(timezone.utc)

        # Update evaluation
        evaluation.status = EvaluationStatus.FAILED
        evaluation.completed_at = now
        evaluation.error = error
        evaluation.progress = None  # Clear progress for failed evaluations
        evaluation.state_history.append(
            StateTransition(state=EvaluationStatus.FAILED, at=now)
        )

        # Write to evaluations directory atomically
        atomic_write(eval_path, evaluation.model_dump(mode="json"))

        # Remove from sessions directory
        session_path.unlink()

    def get_evaluation(self, evaluation_id: str) -> Evaluation | None:
        """Retrieve evaluation by ID.

        Checks sessions first (in-progress), then evaluations (completed).

        Args:
            evaluation_id: Evaluation to retrieve

        Returns:
            Evaluation if found, None otherwise
        """
        # Check sessions first (in-progress)
        session_path = get_session_path(evaluation_id, self._data_dir)
        if session_path.exists():
            with open(session_path) as f:
                data = json.load(f)
            return Evaluation.model_validate(data)

        # Check evaluations (completed)
        eval_path = get_evaluation_path(evaluation_id, self._data_dir)
        if eval_path.exists():
            with open(eval_path) as f:
                data = json.load(f)
            return Evaluation.model_validate(data)

        return None

    def get_evaluation_by_trace_id(self, trace_id: str) -> Evaluation | None:
        """Find session by OTel trace ID.

        Only searches in-progress sessions, not completed evaluations.

        Args:
            trace_id: W3C Trace Context ID

        Returns:
            Evaluation if found in sessions, None otherwise
        """
        sessions_dir = self._dirs["sessions"]

        for session_file in sessions_dir.glob("*.json"):
            with open(session_file) as f:
                data = json.load(f)

            if data.get("trace_id") == trace_id:
                return Evaluation.model_validate(data)

        return None

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
        summaries: list[EvaluationSummary] = []

        # Collect from sessions if requested
        if include_sessions:
            sessions_dir = self._dirs["sessions"]
            for session_file in sessions_dir.glob("*.json"):
                with open(session_file) as f:
                    data = json.load(f)

                # Apply filters
                if domain is not None and data.get("domain") != domain:
                    continue
                if status is not None and data.get("status") != status:
                    continue

                # Create summary
                progress_data = data.get("progress")
                summaries.append(
                    EvaluationSummary(
                        evaluation_id=data["evaluation_id"],
                        trace_id=data.get("trace_id"),
                        session_id=data.get("session_id"),
                        status=EvaluationStatus(data["status"]),
                        domain=data["domain"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        progress=progress_data.get("percent")
                        if progress_data
                        else None,
                    )
                )

        # Collect from evaluations
        evaluations_dir = self._dirs["evaluations"]
        for eval_file in evaluations_dir.glob("*.json"):
            with open(eval_file) as f:
                data = json.load(f)

            # Apply filters
            if domain is not None and data.get("domain") != domain:
                continue
            if status is not None and data.get("status") != status:
                continue

            # Create summary
            summaries.append(
                EvaluationSummary(
                    evaluation_id=data["evaluation_id"],
                    trace_id=data.get("trace_id"),
                    session_id=data.get("session_id"),
                    status=EvaluationStatus(data["status"]),
                    domain=data["domain"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    progress=None,  # Completed evaluations don't have progress
                )
            )

        # Sort by created_at descending (newest first)
        summaries.sort(key=lambda s: s.created_at, reverse=True)

        # Apply limit
        return summaries[:limit]


def create_store(data_dir: Path | str | None = None) -> EvaluationStore:
    """Create an evaluation store instance.

    Args:
        data_dir: Base directory for storage (default: $TAU2_DATA_DIR or ./data)

    Returns:
        Configured EvaluationStore instance
    """
    return EvaluationStore(data_dir)
