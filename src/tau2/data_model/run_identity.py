# Copyright Sierra
"""Content-derived identity for a results run.

A filesystem path LOCATES a run; it does not IDENTIFY one. Paths move when the
tree is reorganised, and a deleted run regenerated under the same
``--save-to`` lands at exactly the same path with entirely different content.
Anything that records "which run did I read" (packet provenance, samplers)
therefore records a :class:`RunIdentity` alongside the path, and resolves by
the identity.

The identity is derived from the run's METADATA only — the results-level
timestamp plus the set of simulation ids. Both survive a move (they live
inside ``results.json``) and both change when the run is re-executed: the
runner stamps a fresh timestamp on a fresh run and every simulation gets a new
uuid. Post-hoc judging (``tau2 judges``) rewrites simulation FILES but keeps
their ids, so re-judging a run does not change its identity — which is the
intent: the identity names the run, not the annotations layered on it.

Cost is bounded by ``results.json``: a dir-format run (every voice run) is
identified from its ``simulation_index`` without opening a single simulation
file, let alone the gigabytes of audio beside them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Optional

from pydantic import BaseModel, Field

RESULTS_METADATA_NAME = "results.json"
SIMULATIONS_DIR_NAME = "simulations"

# Long enough that an accidental collision across a repo's worth of runs is not
# a thing; short enough to read in a manifest diff.
RUN_ID_LENGTH = 16


class RunIdentity(BaseModel):
    """Which run a results path held, independent of where it sat.

    ``run_id`` is the comparison key; ``timestamp`` and ``num_simulations``
    are carried for legibility (a mismatch report can say what changed).
    """

    run_id: Annotated[
        str,
        Field(
            description="sha256 of (results timestamp, sorted simulation ids), "
            f"truncated to {RUN_ID_LENGTH} hex chars."
        ),
    ]
    timestamp: Annotated[
        Optional[str],
        Field(description="The run's results-level timestamp, if it recorded one."),
    ] = None
    num_simulations: Annotated[
        int, Field(description="Simulations the run held when it was identified.")
    ] = 0


def _metadata_path(path: Path) -> Path:
    """The ``results.json`` for a run given its dir OR that file itself."""
    path = Path(path)
    return path / RESULTS_METADATA_NAME if path.is_dir() else path


def _ids_of(entries: object, field: str, meta_path: Path) -> list[str]:
    """Pull ``id`` off every entry, or say loudly which run is malformed.

    Raised as ``ValueError`` on purpose: callers that sweep a whole run tree
    (``tau2 annotate source-audit``) catch it per run, so one hand-edited or
    truncated ``results.json`` is reported and skipped instead of aborting
    the sweep before any manifest is checked.
    """
    if not isinstance(entries, list):
        raise ValueError(f"{meta_path}: '{field}' is not a list")
    ids = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError(f"{meta_path}: '{field}' entry without an id: {entry!r}")
        ids.append(entry["id"])
    return ids


def _simulation_ids(meta: dict, meta_path: Path) -> list[str]:
    """Simulation ids, from the cheapest source the run's format offers.

    Dir format carries ``simulation_index`` in the metadata; monolithic runs
    carry the simulations inline. A dir-format run written without an index
    falls back to the simulations/ file names — which ARE the ids, so still no
    simulation file is opened.
    """
    index = meta.get("simulation_index")
    if index:
        return _ids_of(index, "simulation_index", meta_path)
    inline = meta.get("simulations")
    if inline:
        return _ids_of(inline, "simulations", meta_path)
    sims_dir = meta_path.parent / SIMULATIONS_DIR_NAME
    if sims_dir.is_dir():
        return [f.stem for f in sims_dir.glob("*.json")]
    return []


def compute_run_identity(path: Path) -> RunIdentity:
    """Identify the run at ``path`` (a run dir or its ``results.json``).

    Raises ``FileNotFoundError`` if no ``results.json`` is there — a caller
    recording provenance must not be able to record an identity for a run it
    did not actually read — and ``ValueError`` (which ``JSONDecodeError``
    subclasses) if the metadata that IS there cannot be trusted.
    """
    meta_path = _metadata_path(path)
    if not meta_path.is_file():
        raise FileNotFoundError(f"no {RESULTS_METADATA_NAME} at {path}")
    meta = json.loads(meta_path.read_text())
    if not isinstance(meta, dict):
        raise ValueError(f"{meta_path}: results metadata is not an object")
    timestamp = meta.get("timestamp")
    if timestamp is not None and not isinstance(timestamp, str):
        raise ValueError(f"{meta_path}: 'timestamp' is not a string")
    sim_ids = _simulation_ids(meta, meta_path)

    digest = hashlib.sha256()
    digest.update((timestamp or "").encode())
    for sim_id in sorted(sim_ids):
        digest.update(b"\0")
        digest.update(sim_id.encode())
    return RunIdentity(
        run_id=digest.hexdigest()[:RUN_ID_LENGTH],
        timestamp=timestamp,
        num_simulations=len(sim_ids),
    )
