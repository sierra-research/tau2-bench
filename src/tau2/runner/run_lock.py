# Copyright Sierra
"""Exclusive claims for result directories written by benchmark runs."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

RUN_LOCK_FILENAME = ".tau2-run.lock"


@contextmanager
def claim_run_directories(run_dirs: list[Path]) -> Iterator[None]:
    """Prevent independent controllers from writing the same run directory.

    Claims are acquired in sorted order so two multi-run controllers cannot
    deadlock while requesting overlapping directory sets. Lock files remain as
    harmless provenance; the kernel releases every claim when its process exits.
    """
    handles = []
    try:
        for run_dir in sorted({Path(path).resolve() for path in run_dirs}):
            run_dir.mkdir(parents=True, exist_ok=True)
            lock_path = run_dir / RUN_LOCK_FILENAME
            handle = lock_path.open("a+")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.close()
                owner = "unknown"
                try:
                    owner = lock_path.read_text().strip() or owner
                except OSError:
                    pass
                raise RuntimeError(
                    f"Run directory {run_dir} is already being written "
                    f"(lock owner {owner})."
                ) from error
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            finally:
                handle.close()
