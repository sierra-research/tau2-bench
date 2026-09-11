# Copyright Sierra
"""Fixed per-domain task subsets — the frozen 50 a benchmark run scores on.

Public surface:

- :func:`resolve_subset` — what a run's ``--task-subset`` value means.
- :func:`apply_subset` — filter a loaded task set to a resolved subset.
- :func:`build_subset` — the seeded stratified draw behind ``tau2 tasks subset``.
"""

from tau2.task_subsets.sampler import (
    DOMAIN_STRATIFIER,
    STRATIFIERS,
    allocate,
    build_subset,
    stratifier_for,
)
from tau2.task_subsets.schema import (
    StratumRecord,
    SubsetFrame,
    SubsetStrategy,
    TaskSubset,
    frame_digest,
)
from tau2.task_subsets.store import (
    SUBSET_ALL,
    SUBSET_AUTO,
    apply_subset,
    canonical_subset_name,
    check_frame,
    list_subsets,
    load_subset,
    resolve_subset,
    save_subset,
    subset_covering_ids,
    subset_path,
    task_subset_dir,
)

__all__ = [
    "DOMAIN_STRATIFIER",
    "STRATIFIERS",
    "SUBSET_ALL",
    "SUBSET_AUTO",
    "StratumRecord",
    "SubsetFrame",
    "SubsetStrategy",
    "TaskSubset",
    "allocate",
    "apply_subset",
    "build_subset",
    "canonical_subset_name",
    "check_frame",
    "frame_digest",
    "list_subsets",
    "load_subset",
    "resolve_subset",
    "save_subset",
    "stratifier_for",
    "subset_covering_ids",
    "subset_path",
    "task_subset_dir",
]
