# Copyright Sierra
"""``tau2 factory seed-tasks`` — materialize a domain's multilingual seed set.

Domains with a GENERATED task pool (telecom: 2285 tasks) don't run the whole
pool; the multilingual pipeline runs a reviewed split of it. This verb
deterministically materializes that split as the domain's multilingual SOURCE
task set (the profile's ``source_tasks_filename``, e.g.
``tasks_multilingual.json``) plus a provenance manifest, and then emits the
per-language arm files through the shared emitter in
``tau2.multilingual.factory.task_arms`` — task ids suffixed ``_<lang>``, prose
kept byte-identical to the English source, because english-prompt mode is how
every multilingual run works.

The complement is ``tau2 factory arm-tasks``, which emits arms for domains
whose source set is hand-CURATED (nothing to materialize); the profile's
``curated_source`` decides which verb owns a domain's arm files.

Selection order matches the domain's registry loader (``get_tasks(split)``):
the pool file's order filtered by split membership, so the seed set and a
plain ``tau2 run --domain <domain>`` see the same tasks in the same order.

The selected tasks are then CALLER-DIVERSIFIED (``caller_diversity``): each
task is rewritten onto one of the domain's reviewed ``caller_pool.yaml``
identities with a deterministic 50/50 phone/name-DOB auth split, and the
English arm gets a caller-gender voice-routing sidecar. The pool file and
the domain DB are never modified — renames ship per task via
``initialization_data`` patches.

Everything here is deterministic and idempotent: re-running writes nothing
when content is unchanged (the manifest keeps its original provenance sha).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field

from tau2.multilingual.domain_profiles import (
    REGISTRY_TASKS_FILENAME,
    get_domain_profile,
)
from tau2.multilingual.factory.author_form import FactoryDraftError
from tau2.multilingual.factory.caller_diversity import (
    AuthMode,
    caller_pool_path,
    check_seed_customer_patch_integrity,
    diversify_seed_tasks,
    load_caller_pool,
    task_auth_mode,
)
from tau2.multilingual.factory.task_arms import (
    check_arm_emission_allowed,
    emit_language_arms,
    write_if_changed,
)
from tau2.multilingual.invariants import caller_set_user_info
from tau2.multilingual.localize_lib import data_dir, load_domain_db

# The generated pool + split definitions every seedable domain ships.
POOL_FILENAME = REGISTRY_TASKS_FILENAME
SPLITS_FILENAME = "split_tasks.json"


class SeedTasksOutcome(BaseModel):
    """What ``tau2 factory seed-tasks`` produced (or found already in place)."""

    model_config = ConfigDict(extra="forbid")

    domain: Annotated[str, Field(description="The seeded domain")]
    split: Annotated[str, Field(description="The materialized split name")]
    task_count: Annotated[int, Field(description="Tasks in the seed set")]
    source_path: Annotated[
        Path, Field(description="The materialized multilingual source task file")
    ]
    manifest_path: Annotated[Path, Field(description="The provenance manifest")]
    emitted: Annotated[
        dict[str, Path],
        Field(description="Per-language arm files ({lang: path})"),
    ]
    written: Annotated[
        list[Path],
        Field(
            description="Files actually (re)written this run — empty when "
            "everything was already up to date"
        ),
    ]


def _git_sha() -> str:
    """HEAD sha for the provenance manifest, '-dirty'-suffixed when the tree
    has uncommitted changes (best-effort, advisory only — ``pool_sha256`` is
    the manifest's verifiable provenance: a branch rebase rewrites commit
    shas, so a bare HEAD sha can become unreachable in every clone).
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=data_dir(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if not sha:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=data_dir(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_seed_tasks(
    domain: str,
    split: str = "base",
    langs: Optional[list[str]] = None,
) -> SeedTasksOutcome:
    """Materialize the seed split + per-language arm files (see module doc).

    Raises:
        FactoryDraftError: Unknown/unseedable domain, unknown split, split ids
            missing from the pool, or an emitted set failing the entity
            invariants.
    """
    try:
        profile = get_domain_profile(domain)
    except KeyError as exc:
        raise FactoryDraftError(str(exc)) from exc
    if profile.curated_source:
        raise FactoryDraftError(
            f"Domain '{domain}' ships a curated {POOL_FILENAME} — there is "
            "nothing to materialize, so its arm files come from `tau2 factory "
            f"arm-tasks --domain {domain}`. seed-tasks is for domains whose "
            "profile declares a separate multilingual source file."
        )
    langs = list(langs or ["en"])
    check_arm_emission_allowed(profile, langs)

    domain_dir = data_dir() / "tau2" / "domains" / domain
    pool_path = domain_dir / POOL_FILENAME
    splits_path = domain_dir / SPLITS_FILENAME
    for path in (pool_path, splits_path):
        if not path.exists():
            raise FactoryDraftError(f"Domain '{domain}' has no {path.name}")
    pool = json.loads(pool_path.read_text())
    splits = json.loads(splits_path.read_text())
    if split not in splits:
        raise FactoryDraftError(
            f"Unknown split '{split}' for domain '{domain}' — "
            f"{splits_path.name} defines {sorted(splits)}"
        )
    wanted = set(splits[split])
    tasks = [task for task in pool if task["id"] in wanted]
    missing = sorted(wanted - {task["id"] for task in tasks})
    if missing:
        raise FactoryDraftError(
            f"Split '{split}' names {len(missing)} task id(s) absent from "
            f"{pool_path.name}, e.g. {missing[0]!r}"
        )

    # Caller diversification: the generated pool holds the caller constant
    # (a nuisance parameter to generation, experimental material to us), so
    # the seed split rewrites each task onto one of the reviewed
    # caller_pool.yaml identities with a 50/50 phone/name-DOB auth split.
    # Seedable domains must ship a pool (load fails loud without one).
    caller_pool = load_caller_pool(domain)
    db = load_domain_db(domain)
    tasks = diversify_seed_tasks(tasks, caller_pool, db)
    seed_problems = check_seed_customer_patch_integrity(tasks, db, caller_pool)
    if seed_problems:
        raise FactoryDraftError(
            "Diversified seed set fails patch integrity: "
            + "; ".join(seed_problems[:5])
        )

    written: list[Path] = []
    source_path = domain_dir / profile.source_tasks_filename
    source_text = json.dumps(tasks, ensure_ascii=False, indent=2) + "\n"
    write_if_changed(source_path, source_text, written)

    # The manifest keeps its original provenance when the content is
    # unchanged — the sha records what PRODUCED the artifact, not the last
    # time someone re-ran the verb.
    manifest_path = source_path.with_suffix(".manifest.json")
    if source_path in written or not manifest_path.exists():
        manifest = {
            "domain": domain,
            "split": split,
            "source_pool": POOL_FILENAME,
            # The verifiable provenance: the derivation input's content hash
            # (commit shas are advisory — a rebase strands them).
            "pool_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
            "caller_pool_sha256": hashlib.sha256(
                caller_pool_path(domain).read_bytes()
            ).hexdigest(),
            "git_sha": _git_sha(),
            "task_count": len(tasks),
            "task_ids": [task["id"] for task in tasks],
            # Derived FROM the artifact (single implementations in
            # caller_diversity), recorded here as reviewable provenance.
            "caller_assignment": {
                task["id"]: caller_set_user_info(task)["name"] for task in tasks
            },
            "name_auth_task_ids": [
                task["id"]
                for task in tasks
                if task_auth_mode(task) is AuthMode.NAME_DOB
            ],
        }
        write_if_changed(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            written,
        )

    # The diversified seed set IS the arms' English source: arm and source
    # carry the same customer patch, so the shared emitter's invariant gate
    # (entities, eval criteria, customer-patch integrity) degenerates to a
    # no-op rename and only the id suffix differs.
    emitted = emit_language_arms(domain, tasks, langs, db, profile, written)

    return SeedTasksOutcome(
        domain=domain,
        split=split,
        task_count=len(tasks),
        source_path=source_path,
        manifest_path=manifest_path,
        emitted=emitted,
        written=written,
    )
