"""Safe artifact reads, canonical-frame verification, and immutable publication."""

import hashlib
import os
import re
import tempfile
from pathlib import Path

from tau2.task_subsets.sampler import build_subset
from tau2.task_subsets.schema import (
    CanonicalFrame,
    Domain,
    DrawRequest,
    TaskSubset,
    Verification,
    frame_digest,
)

# Immutable released artifacts, not configurable sampling defaults. New frames
# must receive new names; editing an existing published ID set is never a redraw.
SHIPPED_SUBSET_SHA256 = {
    "airline_50": "055b5d4f8a0a09d297226ee1fe6f2fb01115ac0953e705388fd3dc34b46998db",
    "retail_50": "dc7490f023d1bf9c345edf9c37a8e658298ac3a5be884e012bfb271b3678f2be",
    "telecom_50": "c6376fe23ee7f8c5d94d38b2ba9864c638cfda25f4d4efd8a8dfb2ecdd9f9c03",
}


def task_subset_dir() -> Path:
    """Read the current configured data root, including isolated installations."""
    import tau2.utils

    return tau2.utils.DATA_DIR / "tau2" / "task_subsets"


def subset_path(name: str) -> Path:
    """Resolve a safe artifact name without allowing traversal or symlink escape."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("invalid subset name")
    root = task_subset_dir().resolve()
    path = root / f"{name}.json"
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("subset path must not be a symlink or escape its directory")
    return path


def list_subsets() -> list[str]:
    """List names; loading separately validates each declared artifact."""
    return sorted(path.stem for path in task_subset_dir().glob("*.json"))


def load_subset(name: str) -> TaskSubset:
    """Read the exact named artifact; missing or modified official data fails."""
    path = subset_path(name)
    payload = path.read_bytes()
    if name in SHIPPED_SUBSET_SHA256:
        if hashlib.sha256(payload).hexdigest() != SHIPPED_SUBSET_SHA256[name]:
            raise ValueError(f"frozen subset {name} changed; publish a new name")
    subset = TaskSubset.model_validate_json(payload)
    if subset.name != name:
        raise ValueError("subset filename and declared name disagree")
    return subset


def load_frame(domain: Domain, task_split: str | None = "base") -> CanonicalFrame:
    """Use the ordinary registered canonical loader and its explicit split."""
    # Validate before looking up any registry name.
    from pydantic import TypeAdapter

    from tau2.registry import registry

    domain = TypeAdapter(Domain).validate_python(domain)
    tasks = registry.get_tasks_loader(domain)(task_split_name=task_split)
    return CanonicalFrame(domain=domain, task_split=task_split, tasks=tasks)


def verify_subset(
    subset: TaskSubset, *, frame: CanonicalFrame | None = None
) -> Verification:
    """Reject drift, wrong order, changed strata or a nonreproducible draw."""
    subset = TaskSubset.model_validate(subset.model_dump())
    frame = frame or load_frame(subset.domain, subset.frame.task_split)
    if (
        frame.domain != subset.domain
        or frame.task_split != subset.frame.task_split
        or len(frame.tasks) != subset.frame.size
        or frame_digest(frame.task_ids) != subset.frame.digest
    ):
        raise ValueError(f"canonical frame drift for {subset.name}")
    expected = build_subset(
        DrawRequest(
            name=subset.name, domain=subset.domain, size=subset.size, seed=subset.seed
        ),
        frame,
        created=subset.created,
        git_commit=subset.git_commit,
    )
    if subset.strata != expected.strata or subset.stratifier != expected.stratifier:
        raise ValueError(f"source strata changed for {subset.name}")
    if subset.task_ids != expected.task_ids:
        raise ValueError(
            f"selected order or deterministic draw changed for {subset.name}"
        )
    return Verification(
        name=subset.name,
        frame_size=len(frame.tasks),
        selected_size=subset.size,
        frame_digest=subset.frame.digest,
    )


def verify_installed_bindings(subset: TaskSubset) -> list[str]:
    """Check released membership through explicit runtime metadata, never suffixes.

    Exploratory new draws are verified against their canonical frame, not against
    installed packs intentionally materialized only for the official frozen50.
    """
    if subset.name not in SHIPPED_SUBSET_SHA256:
        return []
    from tau2.multilingual.runtime import list_languages, load_language, load_pack

    languages = []
    for language in list_languages():
        pack = load_pack(language)
        if subset.domain not in pack.domains:
            continue
        loaded = load_language(language, subset.domain, mode="text")
        canonical = {entry.canonical_task_id for entry in loaded.metadata.values()}
        if not set(subset.task_ids) <= canonical:
            raise ValueError(f"{pack.language}/{subset.domain} misses frozen task IDs")
        languages.append(pack.language)
    return languages


def save_subset(subset: TaskSubset) -> Path:
    """Atomically create a new name; identical reruns reuse original provenance."""
    subset = TaskSubset.model_validate(subset.model_dump())
    if subset.name in SHIPPED_SUBSET_SHA256:
        raise ValueError("official subset names are reserved; choose a new name")
    verify_subset(subset)
    path = subset_path(subset.name)
    comparable = subset.model_dump(exclude={"created", "git_commit"})

    def unchanged() -> bool:
        existing = load_subset(subset.name)
        if existing.model_dump(exclude={"created", "git_commit"}) != comparable:
            raise FileExistsError(f"{path} is immutable; choose a new name")
        return True

    if path.exists():
        unchanged()
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = subset.model_dump_json(indent=2) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            unchanged()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
