"""
Evaluation Store Configuration

Environment-based configuration for the evaluation store module.
All settings have sensible defaults for development.
"""

import os
from pathlib import Path


def get_data_dir() -> Path:
    """Get the base data directory for storage.

    Returns:
        Path to data directory (default: ./data)
    """
    return Path(os.environ.get("TAU2_DATA_DIR", "./data"))


def get_retention_days() -> int:
    """Get retention period for completed evaluations.

    Returns:
        Number of days to retain completed evaluations (default: 30)
    """
    return int(os.environ.get("TAU2_RETENTION_DAYS", "30"))


def get_failed_retention_days() -> int:
    """Get retention period for failed evaluations.

    Failed evaluations use shorter retention since they're typically
    for debugging and don't need long-term storage.

    Returns:
        Number of days to retain failed evaluations (default: 7)
    """
    return int(os.environ.get("TAU2_FAILED_RETENTION_DAYS", "7"))


def get_session_stale_hours() -> int:
    """Get threshold for marking sessions as stale/abandoned.

    Sessions without heartbeat updates for this duration are
    marked as abandoned.

    Returns:
        Hours before a session is considered stale (default: 2)
    """
    return int(os.environ.get("TAU2_SESSION_STALE_HOURS", "2"))


def get_session_cleanup_hours() -> int:
    """Get threshold for cleaning up abandoned sessions.

    Abandoned sessions older than this are permanently deleted.

    Returns:
        Hours before abandoned sessions are deleted (default: 24)
    """
    return int(os.environ.get("TAU2_SESSION_CLEANUP_HOURS", "24"))


def get_log_retention_days() -> int:
    """Get retention period for log files.

    Returns:
        Number of days to retain log files (default: 14)
    """
    return int(os.environ.get("TAU2_LOG_RETENTION_DAYS", "14"))


def get_log_stdout() -> bool:
    """Check if log events should also be written to stdout.

    Returns:
        True if events should be emitted to stdout
    """
    return os.environ.get("TAU2_LOG_STDOUT", "").lower() in ("1", "true", "yes")


# File permission modes
FILE_MODE_DATA = 0o600  # -rw------- (owner read/write only)
FILE_MODE_LOGS = 0o640  # -rw-r----- (owner read/write, group read)
