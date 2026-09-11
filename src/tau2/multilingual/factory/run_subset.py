# Copyright Sierra
"""Seeded subset of an existing run results directory (``tau2 factory subset-run``).

Judge waves (delivery re-judging, calibration packets) sometimes need to run
on a REDUCED corpus — e.g. 30 of a run's 200 calls. This verb draws that
subset deterministically and writes a complete, self-consistent dir-format
run directory next to the source, so every downstream consumer (judges,
``tau2 annotate`` packet builders, ``Results.load``) reads it exactly like a
real run:

- ``results.json`` — the source metadata verbatim (``timestamp``/``info``/
  ``tasks`` untouched) with the ``simulation_index`` pruned to the drawn sims
  and a typed ``subset_provenance`` block stamped INSIDE ``info``;
- ``simulations/<sim-id>.json`` — the drawn sim files, copied;
- ``artifacts/task_<task_id>/sim_<sim-id>/`` — ONLY the drawn sims' artifact
  subtrees (the runner keys artifacts per-sim, not per-task — see
  ``runner/batch.py`` — so trials of the same task never share a subtree and
  the copy is exact).

The draw rule is named and versioned (``run-subset-draw-v1``): sort the
simulation index by ``(task_id, trial, id)``, then ``random.Random(seed)
.sample(sorted_index, count)``. Same source + seed + count → the same subset,
always.

The source directory is never modified (copy, never move). Failures are loud:
a count over the index size, a missing sim file, a missing artifact subtree,
or an already-existing output directory each abort before anything is
written.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from tau2.data_model.simulation import SimulationIndexEntry
from tau2.multilingual.factory.author_form import FactoryDraftError
from tau2.utils.utils import get_commit_hash

RUN_SUBSET_DRAW_RULE = "run-subset-draw-v1"

#: Key under which the provenance block is stamped into the output
#: results.json ``info`` object.
SUBSET_PROVENANCE_KEY = "subset_provenance"


class RunSubsetProvenance(BaseModel):
    """Where a subset run directory came from — stamped into its
    ``results.json`` ``info.subset_provenance``."""

    model_config = ConfigDict(extra="forbid")

    draw_rule: Annotated[
        Literal["run-subset-draw-v1"],
        Field(
            description="Named draw rule: sort the simulation index by "
            "(task_id, trial, id), then random.Random(seed)"
            ".sample(sorted_index, count)."
        ),
    ] = RUN_SUBSET_DRAW_RULE
    source: Annotated[str, Field(description="Resolved source results directory.")]
    seed: Annotated[int, Field(description="Draw seed.")]
    count: Annotated[int, Field(gt=0, description="Sims drawn.")]
    source_index_size: Annotated[
        int, Field(gt=0, description="Simulation-index size of the source run.")
    ]
    git_sha: Annotated[str, Field(description="Repo HEAD at subset time.")]
    created_at: Annotated[
        str, Field(description="Subset wall-clock time (UTC ISO 8601).")
    ]


class RunSubsetOutcome(BaseModel):
    """What ``tau2 factory subset-run`` wrote."""

    model_config = ConfigDict(extra="forbid")

    out_dir: Annotated[Path, Field(description="The subset run directory.")]
    drawn: Annotated[
        list[SimulationIndexEntry],
        Field(description="The drawn index entries, in output-index order."),
    ]
    artifact_dirs_copied: Annotated[
        int, Field(ge=0, description="Per-sim artifact subtrees copied.")
    ]
    provenance: Annotated[
        RunSubsetProvenance,
        Field(description="The provenance stamped into the output info."),
    ]


def _sim_artifact_dir(run_dir: Path, entry: SimulationIndexEntry) -> Path:
    """The runner's per-sim artifact subtree (``runner/batch.py`` layout)."""
    return run_dir / "artifacts" / f"task_{entry.task_id}" / f"sim_{entry.id}"


def _load_source_metadata(source: Path) -> tuple[dict, list[SimulationIndexEntry]]:
    """Read the source results.json and its typed simulation index.

    The metadata dict is a short-lived pass-through payload: ``timestamp``/
    ``info``/``tasks`` must land in the output byte-faithfully (an Info
    round-trip through the model would silently drop fields the current
    schema does not know), so only ``simulation_index`` is validated into
    typed entries here.
    """
    results_json = source / "results.json"
    if not results_json.is_file():
        raise FactoryDraftError(f"source has no results.json: {results_json}")
    meta = json.loads(results_json.read_text())
    if not isinstance(meta, dict):
        raise FactoryDraftError(f"{results_json} is not a JSON object")
    raw_index = meta.get("simulation_index")
    if not raw_index:
        raise FactoryDraftError(
            f"{results_json} has no simulation_index — subset-run only "
            "operates on dir-format runs with a populated index"
        )
    if not isinstance(meta.get("info"), dict):
        raise FactoryDraftError(
            f"{results_json} 'info' is not a JSON object — nowhere to stamp "
            "the subset provenance"
        )
    index = [SimulationIndexEntry.model_validate(entry) for entry in raw_index]
    return meta, index


def draw_run_subset(
    index: list[SimulationIndexEntry], seed: int, count: int
) -> set[str]:
    """The ``run-subset-draw-v1`` rule: drawn sim ids from an index.

    Sort by ``(task_id, trial, id)`` — task_id stringified so int- and
    str-keyed runs sort the same way — then a single
    ``random.Random(seed).sample`` over the sorted list.
    """
    import random

    if count > len(index):
        raise FactoryDraftError(
            f"--count {count} exceeds the source index size ({len(index)} sims)"
        )
    ordered = sorted(index, key=lambda e: (str(e.task_id), e.trial, e.id))
    return {entry.id for entry in random.Random(seed).sample(ordered, count)}


def build_run_subset(
    source: Path, out: Path, *, seed: int, count: int
) -> RunSubsetOutcome:
    """Draw and write a seeded subset of one dir-format run directory.

    Deterministic (same source + seed + count → the same subset), copy-only
    (the source is never modified), and loud on every failure mode: count
    over the index, missing sim files, missing artifact subtrees, sim files
    whose recorded id contradicts their filename, or a pre-existing output
    directory.
    """
    source = Path(source).resolve()
    out = Path(out).resolve()
    if not source.is_dir():
        raise FactoryDraftError(f"source run directory not found: {source}")
    if out.exists():
        raise FactoryDraftError(
            f"output directory already exists: {out} — refusing to overwrite "
            "(delete it or pick a fresh --out)"
        )
    if out == source or source in out.parents:
        raise FactoryDraftError(
            f"--out {out} is inside the source run directory {source}"
        )
    if count <= 0:
        raise FactoryDraftError(f"--count must be positive, got {count}")

    meta, index = _load_source_metadata(source)
    drawn_ids = draw_run_subset(index, seed, count)
    # Output index preserves the SOURCE index order, filtered to the draw —
    # the draw rule owns which sims are in, not how the file orders them.
    drawn = [entry for entry in index if entry.id in drawn_ids]

    # Verify the full copy plan BEFORE writing anything.
    missing_sims = [
        entry.id
        for entry in drawn
        if not (source / "simulations" / f"{entry.id}.json").is_file()
    ]
    if missing_sims:
        raise FactoryDraftError(
            f"{len(missing_sims)} drawn sim file(s) missing under "
            f"{source / 'simulations'}: {sorted(missing_sims)}"
        )
    missing_artifacts = [
        entry.id for entry in drawn if not _sim_artifact_dir(source, entry).is_dir()
    ]
    if missing_artifacts:
        raise FactoryDraftError(
            f"{len(missing_artifacts)} drawn sim(s) have no artifact subtree "
            f"under {source / 'artifacts'}: {sorted(missing_artifacts)} — "
            "the subset would be unusable for audio judging"
        )
    for entry in drawn:
        sim_path = source / "simulations" / f"{entry.id}.json"
        recorded = json.loads(sim_path.read_text()).get("id")
        if recorded != entry.id:
            raise FactoryDraftError(
                f"sim file {sim_path} records id {recorded!r} — the source "
                "run is corrupt"
            )

    provenance = RunSubsetProvenance(
        source=str(source),
        seed=seed,
        count=count,
        source_index_size=len(index),
        git_sha=get_commit_hash(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    out.mkdir(parents=True)
    (out / "simulations").mkdir()
    for entry in drawn:
        shutil.copy2(
            source / "simulations" / f"{entry.id}.json",
            out / "simulations" / f"{entry.id}.json",
        )
        target = _sim_artifact_dir(out, entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(_sim_artifact_dir(source, entry), target)

    out_meta = dict(meta)
    out_meta["info"] = {
        **meta["info"],
        SUBSET_PROVENANCE_KEY: provenance.model_dump(mode="json"),
    }
    out_meta["simulation_index"] = [entry.model_dump(mode="json") for entry in drawn]
    (out / "results.json").write_text(json.dumps(out_meta, indent=2))

    return RunSubsetOutcome(
        out_dir=out,
        drawn=drawn,
        artifact_dirs_copied=len(drawn),
        provenance=provenance,
    )
