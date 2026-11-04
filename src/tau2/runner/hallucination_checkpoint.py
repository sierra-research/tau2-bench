"""Recover a reviewed voice simulation after result checkpoint or delivery fails."""

import hashlib
import json
import os
import uuid
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from tau2.data_model.simulation import RunConfig, SimulationRun


class HallucinationReviewCheckpoint(BaseModel):
    """A reviewed scored simulation awaiting a successful result checkpoint."""

    task_id: str
    trial: int
    seed: int
    config_hash: str
    simulation: SimulationRun


def review_checkpoint_path(
    save_dir: Path | None, task_id: str, trial: int, seed: int
) -> Path | None:
    """Give each task/trial/seed in one output directory a stable file name."""
    if save_dir is None:
        return None
    key = f"{task_id}\0{trial}\0{seed}".encode()
    return (
        save_dir
        / "hallucination_review_checkpoints"
        / (f"{hashlib.sha256(key).hexdigest()}.json")
    )


def review_config_hash(config: RunConfig) -> str:
    """Prevent a recovered simulation from crossing run configurations."""
    encoded = json.dumps(
        config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_review_checkpoint(
    path: Path | None,
    *,
    task_id: str,
    trial: int,
    seed: int,
    config_hash: str,
) -> HallucinationReviewCheckpoint | None:
    """Load only the checkpoint for this exact queued unit and configuration.

    A checkpoint for any other unit or configuration is discarded.
    """
    if path is None or not path.exists():
        return None
    checkpoint = HallucinationReviewCheckpoint.model_validate_json(path.read_text())
    if (
        checkpoint.task_id != task_id
        or checkpoint.trial != trial
        or checkpoint.seed != seed
        or checkpoint.config_hash != config_hash
        or checkpoint.simulation.task_id != task_id
        or checkpoint.simulation.trial != trial
    ):
        # A leftover file from another configuration or unit is stale, not
        # recoverable: drop it and run the unit afresh.
        logger.warning(f"Discarding stale hallucination review checkpoint {path}")
        path.unlink(missing_ok=True)
        return None
    return checkpoint


def save_review_checkpoint(
    path: Path, checkpoint: HallucinationReviewCheckpoint
) -> None:
    """Replace the checkpoint atomically so a worker crash cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(checkpoint.model_dump_json(indent=2) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
