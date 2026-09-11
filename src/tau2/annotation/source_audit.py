"""Audit (and repair) the run sources recorded in packet manifest provenance.

Every packet manifest records, per build invocation, the results it read
(``provenance.builds[].results``) as a path PLUS the identity of the run that
path held (see ``tau2.data_model.run_identity``). The path locates; the
identity identifies. Resolution is identity-first, so this module can tell
apart the ways a recorded source stops being true:

- **RESOLVED** — the recorded path still holds the recorded run.
- **RELOCATED** — that run now lives elsewhere. Found by identity anywhere
  under the simulations root, so a reorganised tree repairs exactly; the
  repair keeps the recorded path's shape (run dir vs ``results.json``).
- **REPLACED** — the path still exists, but what sits there is NOT the run
  that was read: the run was deleted and re-executed to the same
  ``--save-to``. A path-matching audit cannot see this at all; it is the
  reason identity is recorded.
- **AMBIGUOUS** — the recorded run's identity is present in more than one
  place (a copy, not a move). Reported, never guessed at.
- **MISSING** — that run is nowhere under the simulations root any more.

Repairs are themselves provenance-bearing: each write appends a
``source_audit`` record carrying the timestamp, git sha, every path it
rewrote, and every loss it observed, so a repaired manifest still says what it
read at build time.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from tau2.annotation.artifacts import provenance_stamp, write_json_artifact
from tau2.annotation.provenance import RecordedRun, build_records
from tau2.data_model.run_identity import RESULTS_METADATA_NAME, compute_run_identity

DEFAULT_ANNOTATIONS_ROOT = Path("data/annotations")
DEFAULT_SIMULATIONS_ROOT = Path("data/simulations")


class SourceStatus(str, Enum):
    """What became of one recorded source run."""

    RESOLVED = "resolved"
    RELOCATED = "relocated"
    REPLACED = "replaced"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"


class SourceCheck(BaseModel):
    """One recorded results source, and what became of it."""

    manifest: Annotated[str, Field(description="Manifest the source was read from.")]
    build_index: Annotated[
        int, Field(description="Index into provenance.builds for this source.")
    ]
    recorded_path: Annotated[
        str, Field(description="The path as written at build time.")
    ]
    recorded_run_id: Annotated[
        str, Field(description="Identity of the run that was read at build time.")
    ]
    status: Annotated[SourceStatus, Field(description="Resolution outcome.")]
    current_path: Annotated[
        Optional[str],
        Field(description="Where the recorded run lives now (RELOCATED only)."),
    ] = None
    found_run_id: Annotated[
        Optional[str],
        Field(
            description="Identity of the run occupying the recorded path "
            "(REPLACED only; None when nothing readable is there)."
        ),
    ] = None
    candidates: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Every location holding the recorded run (AMBIGUOUS only).",
        ),
    ]


class SourceRepair(BaseModel):
    """One recorded path rewritten because its run moved."""

    run_id: Annotated[str, Field(description="Identity of the run that moved.")]
    previous_path: Annotated[str, Field(description="Path recorded at build time.")]
    current_path: Annotated[str, Field(description="Path written by the audit.")]


class SourceLoss(BaseModel):
    """A recorded source that no longer resolves to the run that was read.

    Written into the manifest verbatim and compared verbatim: a second audit
    pass re-records a loss only when it differs from every loss already
    recorded, so an unchanged tree writes nothing.
    """

    status: Annotated[
        SourceStatus, Field(description="REPLACED, AMBIGUOUS, or MISSING.")
    ]
    path: Annotated[str, Field(description="The recorded path (never rewritten).")]
    run_id: Annotated[str, Field(description="Identity recorded at build time.")]
    found_run_id: Annotated[
        Optional[str],
        Field(description="Identity now at that path (REPLACED only)."),
    ] = None
    candidates: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Locations holding the recorded run (AMBIGUOUS only).",
        ),
    ]


class SourceAuditRecord(BaseModel):
    """One audit pass's delta, appended to ``provenance.source_audit``."""

    created_at: Annotated[str, Field(description="Audit wall-clock time (UTC).")]
    git_sha: Annotated[str, Field(description="Repo HEAD at audit time.")]
    repaired: Annotated[
        list[SourceRepair], Field(description="Paths rewritten by this pass.")
    ]
    losses: Annotated[
        list[SourceLoss], Field(description="Losses first observed by this pass.")
    ]


class ManifestProblem(BaseModel):
    """A manifest the audit could not read. Loud, local, and not fatal."""

    manifest: Annotated[str, Field(description="Manifest that failed to parse.")]
    problem: Annotated[str, Field(description="Why it could not be audited.")]


class SourceAuditReport(BaseModel):
    """Provenance-bearing record of one audit pass."""

    created_at: Annotated[str, Field(description="Audit wall-clock time (UTC).")]
    git_sha: Annotated[str, Field(description="Repo HEAD at audit time.")]
    annotations_root: Annotated[str, Field(description="Manifest tree scanned.")]
    simulations_root: Annotated[str, Field(description="Run tree searched.")]
    applied: Annotated[
        bool, Field(description="Whether relocations were written back.")
    ]
    checks: Annotated[list[SourceCheck], Field(description="One per recorded source.")]
    unreadable: Annotated[
        list[ManifestProblem],
        Field(
            default_factory=list,
            description="Manifests skipped because their provenance did not "
            "validate (rebuild them).",
        ),
    ]

    def count(self, status: SourceStatus) -> int:
        return sum(1 for c in self.checks if c.status is status)

    @property
    def manifests_repaired(self) -> set[str]:
        return {c.manifest for c in self.checks if c.status is SourceStatus.RELOCATED}


def _reshape(run_dir: Path, recorded: Path) -> Path:
    """``run_dir`` written in the shape the recorded path used."""
    if recorded.name == RESULTS_METADATA_NAME:
        return run_dir / RESULTS_METADATA_NAME
    return run_dir


def _index_runs(simulations_root: Path) -> dict[str, list[Path]]:
    """Map every run's identity to the run dir(s) holding it.

    Only ``results.json`` is read per run — no simulation files, no audio.
    Duplicate identities are kept so the caller can refuse to guess.
    """
    index: dict[str, list[Path]] = {}
    for results_json in sorted(simulations_root.rglob(RESULTS_METADATA_NAME)):
        run_dir = results_json.parent
        try:
            identity = compute_run_identity(run_dir)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(f"{run_dir}: cannot identify run ({exc}) — not indexable")
            continue
        index.setdefault(identity.run_id, []).append(run_dir)
    return index


def _identity_at(recorded: Path) -> Optional[str]:
    """The id of whatever run occupies ``recorded`` now, if any."""
    try:
        return compute_run_identity(recorded).run_id
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _resolve(
    recorded_run: RecordedRun, index: dict[str, list[Path]]
) -> tuple[SourceStatus, Optional[Path], Optional[str], list[Path]]:
    """Identity-first resolution: (status, relocated_to, found_id, candidates)."""
    recorded = Path(recorded_run.path)
    run_id = recorded_run.identity.run_id

    # Ask the recorded path itself FIRST, not the index. A source can sit
    # outside --simulations-root (an archive volume, another tree) and be
    # perfectly healthy; going through the index would find no candidate and
    # slander it as replaced. This also makes RESOLVED shape- and
    # relative-vs-absolute-agnostic: it compares runs, not path strings.
    found_here = _identity_at(recorded)
    if found_here == run_id:
        return SourceStatus.RESOLVED, None, run_id, []

    candidates = index.get(run_id, [])
    if len(candidates) == 1:
        return SourceStatus.RELOCATED, candidates[0], None, []
    if len(candidates) > 1:
        # The same run in two places is a copy, not a move: repairing would
        # pick one blind, so a human decides which is the real source.
        logger.warning(
            f"{recorded}: run {run_id} found in {len(candidates)} places "
            f"({', '.join(str(c) for c in candidates)}) — not repairing"
        )
        return SourceStatus.AMBIGUOUS, None, None, list(candidates)
    if recorded.exists():
        # The path is live but holds something else: the run was deleted and
        # re-executed to the same --save-to since the packet was built.
        logger.warning(
            f"{recorded}: recorded run {run_id} was replaced by "
            f"{found_here or 'an unreadable run'} — the packet's source is gone"
        )
        return SourceStatus.REPLACED, None, found_here, []
    return SourceStatus.MISSING, None, None, []


def _prior_losses(provenance: dict) -> set[str]:
    """Losses already recorded, as their serialized form (the dedupe key)."""
    prior = provenance.get("source_audit") or []
    return {
        loss.model_dump_json()
        for record in prior
        for loss in SourceAuditRecord.model_validate(record).losses
    }


def audit_sources(
    annotations_root: Path = DEFAULT_ANNOTATIONS_ROOT,
    simulations_root: Path = DEFAULT_SIMULATIONS_ROOT,
    *,
    apply: bool = False,
) -> SourceAuditReport:
    """Check every manifest's recorded source runs; optionally repair moves.

    With ``apply``, RELOCATED paths are rewritten in place and the manifest
    gains a ``source_audit`` record. Every other non-resolving status leaves
    the recorded path exactly as written — the manifest keeps the path it
    actually read, and the audit record names the loss beside it.

    Applying is idempotent: records are deltas (a repair makes the path
    resolve; a loss is only re-recorded if it differs from every loss already
    recorded), so a second pass over an unchanged tree writes nothing.
    """
    annotations_root = Path(annotations_root)
    simulations_root = Path(simulations_root)
    index = _index_runs(simulations_root)
    checks: list[SourceCheck] = []
    unreadable: list[ManifestProblem] = []

    for manifest_path in sorted(annotations_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        provenance = manifest.get("provenance") or {}
        try:
            builds = build_records(provenance)
            prior_losses = _prior_losses(provenance)
        except ValidationError as exc:
            # One unrebuilt manifest must not abort the sweep; it is reported
            # (and counted) rather than skipped silently or guessed at.
            logger.error(f"{manifest_path}: provenance does not validate — {exc}")
            unreadable.append(
                ManifestProblem(manifest=str(manifest_path), problem=str(exc))
            )
            continue
        if not builds:
            continue

        repairs: list[SourceRepair] = []
        losses: list[SourceLoss] = []

        for build_index, build in enumerate(builds):
            for position, recorded_run in enumerate(build.results):
                status, current, found, candidates = _resolve(recorded_run, index)
                checks.append(
                    SourceCheck(
                        manifest=str(manifest_path),
                        build_index=build_index,
                        recorded_path=recorded_run.path,
                        recorded_run_id=recorded_run.identity.run_id,
                        status=status,
                        current_path=str(_reshape(current, Path(recorded_run.path)))
                        if current is not None
                        else None,
                        found_run_id=found if status is SourceStatus.REPLACED else None,
                        candidates=[str(c) for c in candidates],
                    )
                )
                if status is SourceStatus.RELOCATED and current is not None:
                    repaired = str(_reshape(current, Path(recorded_run.path)))
                    repairs.append(
                        SourceRepair(
                            run_id=recorded_run.identity.run_id,
                            previous_path=recorded_run.path,
                            current_path=repaired,
                        )
                    )
                    build.results[position] = recorded_run.model_copy(
                        update={"path": repaired}
                    )
                elif status is not SourceStatus.RESOLVED:
                    losses.append(
                        SourceLoss(
                            status=status,
                            path=recorded_run.path,
                            run_id=recorded_run.identity.run_id,
                            found_run_id=found,
                            candidates=[str(c) for c in candidates],
                        )
                    )

        if not apply:
            continue

        new_losses = [
            loss for loss in losses if loss.model_dump_json() not in prior_losses
        ]
        if not (repairs or new_losses):
            continue

        record = SourceAuditRecord(
            **provenance_stamp(), repaired=repairs, losses=new_losses
        )
        # The validated records are the source of truth for what gets written
        # back — repairs were applied to them, not to the raw JSON.
        manifest["provenance"]["builds"] = [b.model_dump(mode="json") for b in builds]
        manifest["provenance"].setdefault("source_audit", []).append(
            record.model_dump(mode="json")
        )
        write_json_artifact(manifest_path, manifest)
        logger.info(
            f"{manifest_path}: {len(repairs)} repaired, {len(new_losses)} "
            f"newly-observed losses"
        )

    return SourceAuditReport(
        **provenance_stamp(),
        annotations_root=str(annotations_root),
        simulations_root=str(simulations_root),
        applied=apply,
        checks=checks,
        unreadable=unreadable,
    )


def format_report(report: SourceAuditReport) -> str:
    """Human-readable summary; one line per non-resolved source."""
    lines = [
        f"{len(report.checks)} recorded source runs across "
        f"{len({c.manifest for c in report.checks})} manifests",
        f"  resolved  {report.count(SourceStatus.RESOLVED)}",
        f"  relocated {report.count(SourceStatus.RELOCATED)}"
        f"{' (rewritten)' if report.applied else ' (run with --apply to fix)'}",
        f"  replaced  {report.count(SourceStatus.REPLACED)}",
        f"  ambiguous {report.count(SourceStatus.AMBIGUOUS)}",
        f"  missing   {report.count(SourceStatus.MISSING)}",
    ]
    for check in report.checks:
        if check.status is SourceStatus.RELOCATED:
            lines.append(f"  RELOCATED {check.recorded_path} -> {check.current_path}")
        elif check.status is SourceStatus.REPLACED:
            lines.append(
                f"  REPLACED  {check.recorded_path} (read run "
                f"{check.recorded_run_id}, now {check.found_run_id or 'unreadable'})"
            )
        elif check.status is SourceStatus.AMBIGUOUS:
            lines.append(
                f"  AMBIGUOUS {check.recorded_path} -> {', '.join(check.candidates)}"
            )
        elif check.status is SourceStatus.MISSING:
            lines.append(f"  MISSING   {check.recorded_path}")
    for problem in report.unreadable:
        lines.append(f"  UNREADABLE {problem.manifest} — rebuild the packet")
    return "\n".join(lines)
