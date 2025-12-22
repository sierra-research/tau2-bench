"""
Tests for Evaluation Store - User Story 3: Retention & Cleanup

Tests cover cleanup of expired evaluations, stale session detection,
abandoned session cleanup, and file age-based cleanup.
"""

import json
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from tau2.store import (
    EvaluationStatus,
    create_store,
)
from tau2.store.config import FILE_MODE_DATA
from tau2.store.utils import atomic_write


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set environment variable for the store
        old_env = os.environ.get("TAU2_DATA_DIR")
        os.environ["TAU2_DATA_DIR"] = tmpdir
        yield Path(tmpdir)
        # Restore original environment
        if old_env is not None:
            os.environ["TAU2_DATA_DIR"] = old_env
        else:
            os.environ.pop("TAU2_DATA_DIR", None)


@pytest.fixture
def store(temp_data_dir):
    """Create a store instance with temporary directory."""
    return create_store(temp_data_dir)


@pytest.fixture
def retention_manager(temp_data_dir):
    """Create a retention manager instance with temporary directory."""
    from tau2.store import create_retention_manager

    return create_retention_manager(temp_data_dir)


@pytest.fixture
def sample_request():
    """Sample evaluation request parameters."""
    return {"num_tasks": 5, "num_trials": 1, "user_llm": "gpt-4"}


def create_old_evaluation_file(
    evaluations_dir: Path,
    evaluation_id: str,
    status: str = "completed",
    days_old: int = 35,
) -> Path:
    """Create an evaluation file with a specific file modification time.

    Args:
        evaluations_dir: Directory to create the file in
        evaluation_id: Evaluation ID for the file
        status: Evaluation status (completed, failed, abandoned)
        days_old: How many days old the file should be

    Returns:
        Path to the created file
    """
    evaluations_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    created_at = now - timedelta(days=days_old)

    evaluation_data = {
        "evaluation_id": evaluation_id,
        "trace_id": None,
        "session_id": None,
        "status": status,
        "domain": "airline",
        "agent_endpoint": None,
        "state_history": [
            {"state": "submitted", "at": created_at.isoformat()},
            {"state": status, "at": created_at.isoformat()},
        ],
        "created_at": created_at.isoformat(),
        "completed_at": created_at.isoformat() if status != "abandoned" else None,
        "request": {"num_tasks": 5, "num_trials": 1},
        "results": {
            "success_rate": 0.8,
            "total_tasks": 5,
            "successful": 4,
            "tasks": [],
        }
        if status == "completed"
        else None,
        "error": "Test error" if status == "failed" else None,
        "progress": None,
    }

    file_path = evaluations_dir / f"{evaluation_id}.json"
    atomic_write(file_path, evaluation_data)

    # Set file modification time to simulate age
    old_timestamp = (now - timedelta(days=days_old)).timestamp()
    os.utime(file_path, (old_timestamp, old_timestamp))

    return file_path


def create_old_session_file(
    sessions_dir: Path,
    evaluation_id: str,
    hours_since_heartbeat: int = 3,
    status: str = "working",
) -> Path:
    """Create a session file with a stale heartbeat.

    Args:
        sessions_dir: Directory to create the file in
        evaluation_id: Evaluation ID for the file
        hours_since_heartbeat: How many hours since last heartbeat
        status: Evaluation status

    Returns:
        Path to the created file
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    created_at = now - timedelta(hours=hours_since_heartbeat + 1)
    last_heartbeat = now - timedelta(hours=hours_since_heartbeat)

    session_data = {
        "evaluation_id": evaluation_id,
        "trace_id": None,
        "session_id": None,
        "status": status,
        "domain": "airline",
        "agent_endpoint": None,
        "state_history": [
            {"state": "submitted", "at": created_at.isoformat()},
            {"state": "working", "at": created_at.isoformat()},
        ],
        "created_at": created_at.isoformat(),
        "completed_at": None,
        "request": {"num_tasks": 5, "num_trials": 1},
        "results": None,
        "error": None,
        "progress": {
            "current_task": 2,
            "total_tasks": 5,
            "percent": 20,
            "last_heartbeat": last_heartbeat.isoformat(),
        },
    }

    file_path = sessions_dir / f"{evaluation_id}.json"
    atomic_write(file_path, session_data)

    return file_path


# =============================================================================
# T036: Test fixtures are defined above
# =============================================================================


class TestCleanupExpiredEvaluations:
    """Tests for cleanup_expired_evaluations method (T037)."""

    def test_cleanup_removes_old_completed_evaluations(
        self, temp_data_dir, retention_manager
    ):
        """cleanup_expired_evaluations should remove evaluations older than retention period."""
        evaluations_dir = temp_data_dir / "evaluations"

        # Create old evaluation (35 days old, default retention is 30)
        create_old_evaluation_file(
            evaluations_dir,
            "eval-1000000000000-aaaaaa",
            status="completed",
            days_old=35,
        )

        # Create recent evaluation (5 days old)
        create_old_evaluation_file(
            evaluations_dir,
            "eval-1000000000001-bbbbbb",
            status="completed",
            days_old=5,
        )

        deleted = retention_manager.cleanup_expired_evaluations()

        assert deleted == 1
        assert not (evaluations_dir / "eval-1000000000000-aaaaaa.json").exists()
        assert (evaluations_dir / "eval-1000000000001-bbbbbb.json").exists()

    def test_cleanup_respects_retention_days_env_var(self, temp_data_dir):
        """cleanup_expired_evaluations should respect TAU2_RETENTION_DAYS."""
        from tau2.store import create_retention_manager

        evaluations_dir = temp_data_dir / "evaluations"

        # Create evaluation 10 days old
        create_old_evaluation_file(
            evaluations_dir,
            "eval-1000000000000-aaaaaa",
            status="completed",
            days_old=10,
        )

        # Set retention to 5 days
        old_env = os.environ.get("TAU2_RETENTION_DAYS")
        os.environ["TAU2_RETENTION_DAYS"] = "5"
        try:
            manager = create_retention_manager(temp_data_dir)
            deleted = manager.cleanup_expired_evaluations()

            assert deleted == 1
        finally:
            if old_env is not None:
                os.environ["TAU2_RETENTION_DAYS"] = old_env
            else:
                os.environ.pop("TAU2_RETENTION_DAYS", None)

    def test_cleanup_returns_zero_when_nothing_to_delete(
        self, temp_data_dir, retention_manager
    ):
        """cleanup_expired_evaluations should return 0 when no evaluations are expired."""
        evaluations_dir = temp_data_dir / "evaluations"

        # Create recent evaluation
        create_old_evaluation_file(
            evaluations_dir,
            "eval-1000000000000-aaaaaa",
            status="completed",
            days_old=5,
        )

        deleted = retention_manager.cleanup_expired_evaluations()

        assert deleted == 0

    def test_cleanup_handles_empty_directory(self, temp_data_dir, retention_manager):
        """cleanup_expired_evaluations should handle empty evaluations directory."""
        # Ensure directory exists but is empty
        evaluations_dir = temp_data_dir / "evaluations"
        evaluations_dir.mkdir(parents=True, exist_ok=True)

        deleted = retention_manager.cleanup_expired_evaluations()

        assert deleted == 0


class TestCleanupFailedEvaluationsShorterRetention:
    """Tests for failed evaluations having shorter retention (T038)."""

    def test_failed_evaluations_use_shorter_retention(
        self, temp_data_dir, retention_manager
    ):
        """Failed evaluations should use TAU2_FAILED_RETENTION_DAYS (default 7)."""
        evaluations_dir = temp_data_dir / "evaluations"

        # Create failed evaluation 10 days old (should be deleted with 7-day retention)
        create_old_evaluation_file(
            evaluations_dir,
            "eval-1000000000000-aaaaaa",
            status="failed",
            days_old=10,
        )

        # Create completed evaluation 10 days old (should NOT be deleted with 30-day retention)
        create_old_evaluation_file(
            evaluations_dir,
            "eval-1000000000001-bbbbbb",
            status="completed",
            days_old=10,
        )

        deleted = retention_manager.cleanup_expired_evaluations()

        assert deleted == 1
        assert not (evaluations_dir / "eval-1000000000000-aaaaaa.json").exists()
        assert (evaluations_dir / "eval-1000000000001-bbbbbb.json").exists()

    def test_recent_failed_evaluations_not_deleted(
        self, temp_data_dir, retention_manager
    ):
        """Failed evaluations within retention period should not be deleted."""
        evaluations_dir = temp_data_dir / "evaluations"

        # Create failed evaluation 3 days old (within 7-day retention)
        create_old_evaluation_file(
            evaluations_dir,
            "eval-1000000000000-aaaaaa",
            status="failed",
            days_old=3,
        )

        deleted = retention_manager.cleanup_expired_evaluations()

        assert deleted == 0
        assert (evaluations_dir / "eval-1000000000000-aaaaaa.json").exists()

    def test_failed_retention_days_env_var(self, temp_data_dir):
        """cleanup_expired_evaluations should respect TAU2_FAILED_RETENTION_DAYS."""
        from tau2.store import create_retention_manager

        evaluations_dir = temp_data_dir / "evaluations"

        # Create failed evaluation 3 days old
        create_old_evaluation_file(
            evaluations_dir,
            "eval-1000000000000-aaaaaa",
            status="failed",
            days_old=3,
        )

        # Set failed retention to 2 days
        old_env = os.environ.get("TAU2_FAILED_RETENTION_DAYS")
        os.environ["TAU2_FAILED_RETENTION_DAYS"] = "2"
        try:
            manager = create_retention_manager(temp_data_dir)
            deleted = manager.cleanup_expired_evaluations()

            assert deleted == 1
        finally:
            if old_env is not None:
                os.environ["TAU2_FAILED_RETENTION_DAYS"] = old_env
            else:
                os.environ.pop("TAU2_FAILED_RETENTION_DAYS", None)


class TestMarkAbandonedSessions:
    """Tests for mark_abandoned_sessions method (T039)."""

    def test_mark_stale_sessions_as_abandoned(self, temp_data_dir, retention_manager):
        """mark_abandoned_sessions should mark stale sessions as abandoned."""
        sessions_dir = temp_data_dir / "sessions"

        # Create stale session (3 hours since heartbeat, default threshold is 2)
        create_old_session_file(
            sessions_dir,
            "eval-1000000000000-aaaaaa",
            hours_since_heartbeat=3,
        )

        abandoned = retention_manager.mark_abandoned_sessions()

        assert len(abandoned) == 1
        assert "eval-1000000000000-aaaaaa" in abandoned

        # Verify the session file was updated
        with open(sessions_dir / "eval-1000000000000-aaaaaa.json") as f:
            data = json.load(f)
        assert data["status"] == "abandoned"

    def test_active_sessions_not_marked(self, temp_data_dir, retention_manager):
        """mark_abandoned_sessions should not mark active sessions."""
        sessions_dir = temp_data_dir / "sessions"

        # Create active session (0.5 hours since heartbeat)
        create_old_session_file(
            sessions_dir,
            "eval-1000000000000-aaaaaa",
            hours_since_heartbeat=0,
        )

        abandoned = retention_manager.mark_abandoned_sessions()

        assert len(abandoned) == 0

        # Verify the session status is unchanged
        with open(sessions_dir / "eval-1000000000000-aaaaaa.json") as f:
            data = json.load(f)
        assert data["status"] == "working"

    def test_respects_stale_hours_env_var(self, temp_data_dir):
        """mark_abandoned_sessions should respect TAU2_SESSION_STALE_HOURS."""
        from tau2.store import create_retention_manager

        sessions_dir = temp_data_dir / "sessions"

        # Create session 1 hour since heartbeat
        create_old_session_file(
            sessions_dir,
            "eval-1000000000000-aaaaaa",
            hours_since_heartbeat=1,
        )

        # Set stale threshold to 0.5 hours
        old_env = os.environ.get("TAU2_SESSION_STALE_HOURS")
        os.environ["TAU2_SESSION_STALE_HOURS"] = "0"  # 0 means immediately stale
        try:
            manager = create_retention_manager(temp_data_dir)
            abandoned = manager.mark_abandoned_sessions()

            assert len(abandoned) == 1
        finally:
            if old_env is not None:
                os.environ["TAU2_SESSION_STALE_HOURS"] = old_env
            else:
                os.environ.pop("TAU2_SESSION_STALE_HOURS", None)

    def test_mark_abandoned_updates_state_history(self, temp_data_dir, retention_manager):
        """mark_abandoned_sessions should add ABANDONED to state_history."""
        sessions_dir = temp_data_dir / "sessions"

        create_old_session_file(
            sessions_dir,
            "eval-1000000000000-aaaaaa",
            hours_since_heartbeat=3,
        )

        retention_manager.mark_abandoned_sessions()

        with open(sessions_dir / "eval-1000000000000-aaaaaa.json") as f:
            data = json.load(f)

        # Should have SUBMITTED, WORKING, and ABANDONED in history
        assert len(data["state_history"]) == 3
        assert data["state_history"][-1]["state"] == "abandoned"

    def test_already_abandoned_sessions_not_remarked(
        self, temp_data_dir, retention_manager
    ):
        """mark_abandoned_sessions should skip already abandoned sessions."""
        sessions_dir = temp_data_dir / "sessions"

        # Create already abandoned session
        create_old_session_file(
            sessions_dir,
            "eval-1000000000000-aaaaaa",
            hours_since_heartbeat=3,
            status="abandoned",
        )

        abandoned = retention_manager.mark_abandoned_sessions()

        # Should not be in the returned list (already abandoned)
        assert len(abandoned) == 0

    def test_handles_empty_sessions_directory(self, temp_data_dir, retention_manager):
        """mark_abandoned_sessions should handle empty sessions directory."""
        sessions_dir = temp_data_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        abandoned = retention_manager.mark_abandoned_sessions()

        assert len(abandoned) == 0


class TestCleanupAbandonedSessions:
    """Tests for cleanup_abandoned_sessions method (T040)."""

    def test_cleanup_old_abandoned_sessions(self, temp_data_dir, retention_manager):
        """cleanup_abandoned_sessions should delete abandoned sessions past cleanup threshold."""
        sessions_dir = temp_data_dir / "sessions"

        # Create old abandoned session (set file mtime to 25 hours ago, threshold is 24)
        file_path = create_old_session_file(
            sessions_dir,
            "eval-1000000000000-aaaaaa",
            hours_since_heartbeat=25,
            status="abandoned",
        )
        # Set file modification time to simulate age
        old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=25)).timestamp()
        os.utime(file_path, (old_timestamp, old_timestamp))

        deleted = retention_manager.cleanup_abandoned_sessions()

        assert deleted == 1
        assert not file_path.exists()

    def test_recent_abandoned_sessions_not_deleted(
        self, temp_data_dir, retention_manager
    ):
        """cleanup_abandoned_sessions should not delete recent abandoned sessions."""
        sessions_dir = temp_data_dir / "sessions"

        # Create recently abandoned session (file created now)
        create_old_session_file(
            sessions_dir,
            "eval-1000000000000-aaaaaa",
            hours_since_heartbeat=3,
            status="abandoned",
        )

        deleted = retention_manager.cleanup_abandoned_sessions()

        assert deleted == 0
        assert (sessions_dir / "eval-1000000000000-aaaaaa.json").exists()

    def test_active_sessions_not_deleted(self, temp_data_dir, retention_manager):
        """cleanup_abandoned_sessions should only delete abandoned status sessions."""
        sessions_dir = temp_data_dir / "sessions"

        # Create working session with old file time
        file_path = create_old_session_file(
            sessions_dir,
            "eval-1000000000000-aaaaaa",
            hours_since_heartbeat=0,
            status="working",
        )
        old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=25)).timestamp()
        os.utime(file_path, (old_timestamp, old_timestamp))

        deleted = retention_manager.cleanup_abandoned_sessions()

        assert deleted == 0
        assert file_path.exists()

    def test_respects_cleanup_hours_env_var(self, temp_data_dir):
        """cleanup_abandoned_sessions should respect TAU2_SESSION_CLEANUP_HOURS."""
        from tau2.store import create_retention_manager

        sessions_dir = temp_data_dir / "sessions"

        # Create abandoned session with file 2 hours old
        file_path = create_old_session_file(
            sessions_dir,
            "eval-1000000000000-aaaaaa",
            hours_since_heartbeat=3,
            status="abandoned",
        )
        old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=2)).timestamp()
        os.utime(file_path, (old_timestamp, old_timestamp))

        # Set cleanup threshold to 1 hour
        old_env = os.environ.get("TAU2_SESSION_CLEANUP_HOURS")
        os.environ["TAU2_SESSION_CLEANUP_HOURS"] = "1"
        try:
            manager = create_retention_manager(temp_data_dir)
            deleted = manager.cleanup_abandoned_sessions()

            assert deleted == 1
        finally:
            if old_env is not None:
                os.environ["TAU2_SESSION_CLEANUP_HOURS"] = old_env
            else:
                os.environ.pop("TAU2_SESSION_CLEANUP_HOURS", None)


class TestFileAgeCleanup:
    """Tests for file age-based cleanup logic (T041)."""

    def test_file_mtime_determines_age(self, temp_data_dir, retention_manager):
        """Cleanup should use file modification time to determine age."""
        evaluations_dir = temp_data_dir / "evaluations"

        # Create evaluation with recent created_at but old file mtime
        file_path = create_old_evaluation_file(
            evaluations_dir,
            "eval-1000000000000-aaaaaa",
            status="completed",
            days_old=0,  # Recent created_at in the data
        )
        # But make the file 35 days old
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=35)).timestamp()
        os.utime(file_path, (old_timestamp, old_timestamp))

        deleted = retention_manager.cleanup_expired_evaluations()

        assert deleted == 1

    def test_new_file_with_old_created_at_not_deleted(
        self, temp_data_dir, retention_manager
    ):
        """Files with recent mtime should not be deleted even if created_at is old."""
        evaluations_dir = temp_data_dir / "evaluations"

        # Create evaluation - file will be recent but created_at is old
        create_old_evaluation_file(
            evaluations_dir,
            "eval-1000000000000-aaaaaa",
            status="completed",
            days_old=35,  # Old created_at
        )
        # But DON'T modify the file time - it will be current

        # Reset file mtime to now
        file_path = evaluations_dir / "eval-1000000000000-aaaaaa.json"
        now_timestamp = datetime.now(timezone.utc).timestamp()
        os.utime(file_path, (now_timestamp, now_timestamp))

        deleted = retention_manager.cleanup_expired_evaluations()

        assert deleted == 0

    def test_mixed_file_ages(self, temp_data_dir, retention_manager):
        """Cleanup should correctly identify and delete only old files."""
        evaluations_dir = temp_data_dir / "evaluations"

        # Create multiple evaluations with different file ages
        old_file = create_old_evaluation_file(
            evaluations_dir,
            "eval-1000000000000-aaaaaa",
            status="completed",
            days_old=35,
        )

        recent_file = create_old_evaluation_file(
            evaluations_dir,
            "eval-1000000000001-bbbbbb",
            status="completed",
            days_old=5,
        )

        borderline_file = create_old_evaluation_file(
            evaluations_dir,
            "eval-1000000000002-cccccc",
            status="completed",
            days_old=29,  # Just under the boundary (30 days retention)
        )

        deleted = retention_manager.cleanup_expired_evaluations()

        assert deleted == 1  # Only the 35-day-old file
        assert not old_file.exists()
        assert recent_file.exists()
        assert borderline_file.exists()  # Under boundary, should NOT be deleted

    def test_session_cleanup_uses_file_age(self, temp_data_dir, retention_manager):
        """Session cleanup should use file modification time."""
        sessions_dir = temp_data_dir / "sessions"

        # Create abandoned session with recent heartbeat but old file
        file_path = create_old_session_file(
            sessions_dir,
            "eval-1000000000000-aaaaaa",
            hours_since_heartbeat=1,  # Recent heartbeat in data
            status="abandoned",
        )
        # Make file 25 hours old
        old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=25)).timestamp()
        os.utime(file_path, (old_timestamp, old_timestamp))

        deleted = retention_manager.cleanup_abandoned_sessions()

        assert deleted == 1


class TestRetentionManagerIntegration:
    """Integration tests for RetentionManager with real store operations."""

    def test_full_cleanup_workflow(self, temp_data_dir, store, retention_manager):
        """Test complete cleanup workflow with store and retention manager."""
        evaluations_dir = temp_data_dir / "evaluations"
        sessions_dir = temp_data_dir / "sessions"
        sample_request = {"num_tasks": 5, "num_trials": 1}

        # 1. Create and complete an evaluation (recent)
        eval_id_1 = store.create_session(domain="airline", request=sample_request)
        store.update_progress(eval_id_1, current_task=1, total_tasks=5)
        store.complete_evaluation(
            eval_id_1,
            results={
                "success_rate": 1.0,
                "total_tasks": 5,
                "successful": 5,
                "tasks": [],
            },
        )

        # 2. Create an old completed evaluation
        create_old_evaluation_file(
            evaluations_dir,
            "eval-1000000000000-old111",
            status="completed",
            days_old=35,
        )

        # 3. Create a stale session
        create_old_session_file(
            sessions_dir,
            "eval-1000000000001-stale1",
            hours_since_heartbeat=3,
            status="working",
        )

        # Run cleanup workflow
        abandoned = retention_manager.mark_abandoned_sessions()
        deleted_evals = retention_manager.cleanup_expired_evaluations()

        # Verify results
        assert len(abandoned) == 1
        assert "eval-1000000000001-stale1" in abandoned
        assert deleted_evals == 1

        # Recent evaluation should still exist
        assert store.get_evaluation(eval_id_1) is not None

    def test_cleanup_preserves_active_sessions(
        self, temp_data_dir, store, retention_manager
    ):
        """Cleanup should preserve active sessions."""
        sample_request = {"num_tasks": 5, "num_trials": 1}

        # Create active session
        eval_id = store.create_session(domain="airline", request=sample_request)
        store.update_progress(eval_id, current_task=2, total_tasks=5)

        # Run abandonment check
        abandoned = retention_manager.mark_abandoned_sessions()

        assert len(abandoned) == 0

        # Session should still be retrievable and in working state
        evaluation = store.get_evaluation(eval_id)
        assert evaluation is not None
        assert evaluation.status == EvaluationStatus.WORKING
