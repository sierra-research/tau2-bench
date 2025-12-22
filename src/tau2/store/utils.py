"""
Evaluation Store Utilities

Core utility functions for atomic writes, ID generation, and directory management.
"""

import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

from tau2.store.config import FILE_MODE_DATA, get_data_dir


def atomic_write(path: Path, data: dict[str, Any], mode: int = FILE_MODE_DATA) -> None:
    """Write data atomically using temp file + rename pattern.

    This ensures that readers never see partial writes. The file either
    exists completely or doesn't exist at all.

    Args:
        path: Target file path
        data: Dictionary to serialize as JSON
        mode: File permission mode (default: 0o600)

    Raises:
        IOError: If unable to write to filesystem
    """
    # Create temp file in same directory (same filesystem for atomic rename)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    # Set restrictive umask temporarily
    old_umask = os.umask(0o077)
    try:
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())  # Ensure data hits disk before rename

        temp_path.chmod(mode)
        temp_path.rename(path)  # Atomic on POSIX
    finally:
        os.umask(old_umask)


def generate_evaluation_id() -> str:
    """Generate unique evaluation ID with timestamp prefix.

    Format: eval-{unix_ms}-{random_6_chars}

    The timestamp prefix makes IDs chronologically sortable.
    The random suffix handles concurrent evaluation creation.

    Returns:
        Unique evaluation ID string

    Example:
        >>> generate_evaluation_id()
        'eval-1732449600000-a1b2c3'
    """
    unix_ms = int(time.time() * 1000)
    random_suffix = secrets.token_hex(3)  # 6 hex chars
    return f"eval-{unix_ms}-{random_suffix}"


def ensure_directories(data_dir: Path | None = None) -> dict[str, Path]:
    """Ensure all required directories exist.

    Creates the following directory structure:
        data_dir/
        ├── evaluations/   # Completed evaluations (immutable)
        ├── sessions/      # In-progress sessions (mutable)
        └── logs/          # Structured event logs

    Args:
        data_dir: Base data directory (default: from TAU2_DATA_DIR env var)

    Returns:
        Dictionary with paths to each directory:
        - 'base': Base data directory
        - 'evaluations': Completed evaluations directory
        - 'sessions': In-progress sessions directory
        - 'logs': Log files directory

    Raises:
        IOError: If unable to create directories
    """
    if data_dir is None:
        data_dir = get_data_dir()

    data_dir = Path(data_dir)

    dirs = {
        "base": data_dir,
        "evaluations": data_dir / "evaluations",
        "sessions": data_dir / "sessions",
        "logs": data_dir / "logs",
    }

    for _name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def get_session_path(evaluation_id: str, data_dir: Path | None = None) -> Path:
    """Get path for in-progress session file.

    Args:
        evaluation_id: Evaluation ID
        data_dir: Base data directory (default: from TAU2_DATA_DIR env var)

    Returns:
        Path to session JSON file
    """
    if data_dir is None:
        data_dir = get_data_dir()

    return Path(data_dir) / "sessions" / f"{evaluation_id}.json"


def get_evaluation_path(evaluation_id: str, data_dir: Path | None = None) -> Path:
    """Get path for completed evaluation file.

    Args:
        evaluation_id: Evaluation ID
        data_dir: Base data directory (default: from TAU2_DATA_DIR env var)

    Returns:
        Path to evaluation JSON file
    """
    if data_dir is None:
        data_dir = get_data_dir()

    return Path(data_dir) / "evaluations" / f"{evaluation_id}.json"
