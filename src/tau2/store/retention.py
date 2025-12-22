"""
Evaluation Store Retention and Cleanup

Handles cleanup of expired evaluations, stale session detection,
abandoned session cleanup, and log rotation.
"""

import fcntl
import gzip
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tau2.store.config import (
    get_data_dir,
    get_failed_retention_days,
    get_log_retention_days,
    get_retention_days,
    get_session_cleanup_hours,
    get_session_stale_hours,
)
from tau2.store.models import EvaluationStatus
from tau2.store.utils import atomic_write, ensure_directories


class RetentionManager:
    """Manages retention and cleanup for evaluation storage.

    Handles:
    - Cleanup of expired completed/failed evaluations (file-age based)
    - Detection and marking of stale sessions as abandoned
    - Cleanup of old abandoned sessions
    - Log rotation and compression
    """

    def __init__(self, data_dir: Path | str | None = None):
        """Initialize the retention manager.

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

    def cleanup_expired_evaluations(self) -> int:
        """Remove completed/failed evaluations older than retention period.

        Uses TAU2_RETENTION_DAYS for completed evaluations (default: 30).
        Uses TAU2_FAILED_RETENTION_DAYS for failed evaluations (default: 7).

        Retention is determined by file modification time, not the created_at
        field in the evaluation data.

        Returns:
            Number of evaluations deleted
        """
        evaluations_dir = self._dirs["evaluations"]
        retention_days = get_retention_days()
        failed_retention_days = get_failed_retention_days()

        now = datetime.now(timezone.utc)
        completed_cutoff = now - timedelta(days=retention_days)
        failed_cutoff = now - timedelta(days=failed_retention_days)

        deleted = 0

        for eval_file in evaluations_dir.glob("*.json"):
            try:
                # Use file modification time for age
                file_mtime = datetime.fromtimestamp(
                    eval_file.stat().st_mtime, tz=timezone.utc
                )

                # Read the file to check status
                with open(eval_file) as f:
                    data = json.load(f)

                status = data.get("status", "")

                # Determine cutoff based on status
                if status == EvaluationStatus.FAILED.value:
                    cutoff = failed_cutoff
                else:
                    cutoff = completed_cutoff

                # Delete if file is older than cutoff
                if file_mtime < cutoff:
                    eval_file.unlink()
                    deleted += 1

            except (OSError, json.JSONDecodeError):
                # Skip files that can't be read
                continue

        return deleted

    def mark_abandoned_sessions(self) -> list[str]:
        """Mark stale sessions as abandoned.

        Sessions without heartbeat for TAU2_SESSION_STALE_HOURS (default: 2)
        are marked as abandoned.

        For sessions with progress, uses the last_heartbeat timestamp.
        For sessions without progress (e.g., SUBMITTED state), uses file mtime
        as fallback to prevent orphaned sessions.

        Already abandoned sessions are skipped.

        Returns:
            List of evaluation_ids that were marked abandoned
        """
        sessions_dir = self._dirs["sessions"]
        stale_hours = get_session_stale_hours()

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=stale_hours)

        abandoned = []

        for session_file in sessions_dir.glob("*.json"):
            try:
                # Use file locking to prevent race conditions during read-modify-write
                with open(session_file, "r+") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        data = json.load(f)

                        # Skip if already abandoned
                        if data.get("status") == EvaluationStatus.ABANDONED.value:
                            continue

                        # Determine last activity time
                        # Priority: heartbeat > file mtime
                        last_activity: datetime | None = None

                        progress = data.get("progress")
                        if progress is not None:
                            heartbeat_str = progress.get("last_heartbeat")
                            if heartbeat_str is not None:
                                last_activity = datetime.fromisoformat(
                                    heartbeat_str.replace("Z", "+00:00")
                                )

                        # Fallback to file mtime for sessions without progress
                        if last_activity is None:
                            last_activity = datetime.fromtimestamp(
                                session_file.stat().st_mtime, tz=timezone.utc
                            )

                        # Mark as abandoned if stale
                        if last_activity < cutoff:
                            data["status"] = EvaluationStatus.ABANDONED.value

                            # Add to state history
                            state_history = data.get("state_history", [])
                            state_history.append(
                                {
                                    "state": EvaluationStatus.ABANDONED.value,
                                    "at": now.isoformat().replace("+00:00", "Z"),
                                }
                            )
                            data["state_history"] = state_history

                            # Write updated data while still holding lock
                            atomic_write(session_file, data)

                            abandoned.append(data["evaluation_id"])
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            except (OSError, json.JSONDecodeError, KeyError):
                # Skip files that can't be read or parsed
                continue

        return abandoned

    def cleanup_abandoned_sessions(self) -> int:
        """Remove abandoned sessions older than cleanup threshold.

        Sessions abandoned for longer than TAU2_SESSION_CLEANUP_HOURS (default: 24)
        are deleted.

        Uses file modification time to determine age.

        Returns:
            Number of sessions deleted
        """
        sessions_dir = self._dirs["sessions"]
        cleanup_hours = get_session_cleanup_hours()

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=cleanup_hours)

        deleted = 0

        for session_file in sessions_dir.glob("*.json"):
            try:
                # Check file modification time
                file_mtime = datetime.fromtimestamp(
                    session_file.stat().st_mtime, tz=timezone.utc
                )

                # Only delete if file is old enough
                if file_mtime >= cutoff:
                    continue

                # Read to check status
                with open(session_file) as f:
                    data = json.load(f)

                # Only delete abandoned sessions
                if data.get("status") == EvaluationStatus.ABANDONED.value:
                    session_file.unlink()
                    deleted += 1

            except (OSError, json.JSONDecodeError):
                # Skip files that can't be read
                continue

        return deleted

    def rotate_logs(self) -> int:
        """Compress old logs and delete expired ones.

        Compresses logs older than 3 days.
        Deletes logs older than TAU2_LOG_RETENTION_DAYS (default: 14).

        Returns:
            Number of log files processed
        """
        logs_dir = self._dirs["logs"]
        retention_days = get_log_retention_days()

        now = datetime.now(timezone.utc)
        compress_cutoff = now - timedelta(days=3)
        delete_cutoff = now - timedelta(days=retention_days)

        processed = 0

        # Create archive directory if needed
        archive_dir = logs_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        # Process log files
        for log_file in logs_dir.glob("*.jsonl"):
            if log_file.name == "events.jsonl":
                # Don't process current log file
                continue

            try:
                file_mtime = datetime.fromtimestamp(
                    log_file.stat().st_mtime, tz=timezone.utc
                )

                # Delete if past retention
                if file_mtime < delete_cutoff:
                    log_file.unlink()
                    processed += 1
                    continue

                # Compress if old enough but not yet compressed
                if file_mtime < compress_cutoff:
                    archive_path = archive_dir / f"{log_file.name}.gz"
                    with (
                        open(log_file, "rb") as f_in,
                        gzip.open(archive_path, "wb") as f_out,
                    ):
                        shutil.copyfileobj(f_in, f_out)
                    log_file.unlink()
                    processed += 1

            except OSError:
                continue

        # Also check archive for expired compressed logs
        for gz_file in archive_dir.glob("*.gz"):
            try:
                file_mtime = datetime.fromtimestamp(
                    gz_file.stat().st_mtime, tz=timezone.utc
                )
                if file_mtime < delete_cutoff:
                    gz_file.unlink()
                    processed += 1
            except OSError:
                continue

        return processed


def create_retention_manager(
    data_dir: Path | str | None = None,
) -> RetentionManager:
    """Create a retention manager instance.

    Args:
        data_dir: Base directory for storage (default: $TAU2_DATA_DIR or ./data)

    Returns:
        Configured RetentionManager instance
    """
    return RetentionManager(data_dir)
