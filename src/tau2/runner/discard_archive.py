"""Append-safe archive for simulations discarded by the hallucination gate.

Each worker process appends to its own JSONL file
(``results_user_hallucination.<pid>.jsonl``) under
``<save_dir>/hallucination_discarded/``, so concurrent worker processes never
share a file and cannot interleave writes. Within a process, a lock serializes
worker threads. Every record is a single line: an ``info`` header when the
file is created, then one ``discarded_simulation`` record (task + simulation)
per discard, so each file is self-contained provenance.

The archive is diagnostics only. Nothing in the run pipeline reads it back,
and no failure here may fail a simulation: append errors are logged, the
offending file is rotated aside to ``*.corrupt``, and the append is retried on
a fresh file — if that also fails, the record is dropped with an error log.
"""

import json
import os
import threading
from pathlib import Path
from typing import Literal, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

from tau2.data_model.simulation import Info, SimulationRun
from tau2.data_model.tasks import Task

DISCARDED_DIR_NAME = "hallucination_discarded"

# Serializes appends from worker threads within one process; distinct
# processes write distinct per-pid files and need no cross-process lock.
_append_lock = threading.Lock()


class DiscardArchiveHeader(BaseModel):
    """First line of each per-worker archive file: run-level provenance."""

    record: Literal["info"] = "info"
    info: Info = Field(description="Run-level info of the batch that discarded.")


class DiscardedSimulationRecord(BaseModel):
    """One archived discard: the task and the simulation that was thrown away."""

    record: Literal["discarded_simulation"] = "discarded_simulation"
    task: Task = Field(description="The task whose simulation was discarded.")
    simulation: SimulationRun = Field(description="The discarded simulation run.")


def _archive_path(save_dir: Path) -> Path:
    return (
        save_dir
        / DISCARDED_DIR_NAME
        / f"results_user_hallucination.{os.getpid()}.jsonl"
    )


def _rotate_corrupt(path: Path) -> None:
    corrupt = path.with_name(path.name + ".corrupt")
    n = 1
    while corrupt.exists():
        corrupt = path.with_name(f"{path.name}.corrupt{n}")
        n += 1
    path.rename(corrupt)
    logger.error(f"Rotated unusable discard archive {path} to {corrupt}")


def _append(
    path: Path, header: DiscardArchiveHeader, record: DiscardedSimulationRecord
) -> None:
    payload = record.model_dump_json() + "\n"
    if not path.exists() or path.stat().st_size == 0:
        payload = header.model_dump_json() + "\n" + payload
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(payload)


def archive_discarded_simulation(
    save_dir: Union[str, Path],
    info: Info,
    task: Task,
    simulation: SimulationRun,
) -> Optional[Path]:
    """Append one discarded simulation to this worker's archive file.

    Never raises: on any failure the error is logged, an unusable target is
    rotated to ``*.corrupt`` and the append retried once, and ``None`` is
    returned if the record could not be written.
    """
    try:
        path = _archive_path(Path(save_dir))
        path.parent.mkdir(parents=True, exist_ok=True)
        header = DiscardArchiveHeader(info=info)
        record = DiscardedSimulationRecord(task=task, simulation=simulation)
        with _append_lock:
            try:
                _append(path, header, record)
            except OSError as e:
                logger.error(
                    f"Discard archive {path} is unusable ({e}); rotating and retrying"
                )
                _rotate_corrupt(path)
                _append(path, header, record)
        return path
    except Exception as e:
        logger.error(
            f"Failed to archive discarded simulation for task {task.id}; "
            f"continuing without archiving: {e}"
        )
        return None


def load_discarded_simulations(
    save_dir: Union[str, Path],
) -> list[DiscardedSimulationRecord]:
    """Read every per-worker archive under save_dir, for diagnostics.

    Unparseable lines are logged and skipped — a partially corrupt archive
    never raises.
    """
    records: list[DiscardedSimulationRecord] = []
    directory = Path(save_dir) / DISCARDED_DIR_NAME
    if not directory.exists():
        return records
    for path in sorted(directory.glob("results_user_hallucination.*.jsonl")):
        with open(path, "r", encoding="utf-8") as fp:
            for lineno, line in enumerate(fp, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    if payload.get("record") == "discarded_simulation":
                        records.append(
                            DiscardedSimulationRecord.model_validate(payload)
                        )
                    elif payload.get("record") != "info":
                        raise ValueError(
                            f"unknown record type {payload.get('record')!r}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Skipping unparseable discard-archive line {path}:{lineno}: {e}"
                    )
    return records
