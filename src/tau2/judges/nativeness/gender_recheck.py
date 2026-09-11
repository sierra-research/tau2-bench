# Copyright Sierra
"""Deterministic gender-agreement recheck over stored runs.

``tau2 judges recheck-gender`` re-derives ONLY the ``gender_agreement`` factor
on frozen results with the current deterministic checker
(``GENDER_AGREEMENT_CHECKER_VERSION``), so a checker fix — e.g. v2's
pinned-greeting exclusion — can be applied to already-judged cells without
paying for a full LLM re-judge.

Scope is deliberately narrow: only sims whose stored gender check is a
deterministic precheck FAIL (evidence prefixed ``gender-``) are candidates.
Verdicts the LLM judge produced (free-text evidence, PASS/NO_OPPORTUNITY
outcomes) are never touched — the deterministic checker cannot speak for them.
A candidate whose v2 precheck no longer fails is recorded NO_OPPORTUNITY (the
precheck's only non-FAIL outcome); a genuine model-turn violation stays FAIL
with current-version evidence. The per-sim checker context is rebuilt through
the same harness functions benchmark scoring uses
(``checker_context_for_simulation`` / ``resolve_participants``), and the
NativenessInfo aggregates are recomputed with the harness's own aggregation
rule (``aggregate_nativeness_checks``).

Before any sim is patched, a provenance-stamped snapshot of the prior verdicts
is written at the root (``gender_recheck_snapshot.json``; an existing snapshot
is never overwritten — later passes that still change sims write numbered
siblings). Re-running the verb is idempotent: a second pass finds nothing to
change and writes nothing.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

from loguru import logger
from pydantic import BaseModel, Field

from tau2.data_model.simulation import JudgeOutcome, Results, SimulationRun
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import get_simulation_language_info
from tau2.judges.attach import resolve_participants
from tau2.judges.nativeness.checkers import (
    CHECKER_REGISTRY,
    GENDER_AGREEMENT_CHECKER_VERSION,
)
from tau2.judges.nativeness.factors import judge_factors_for
from tau2.judges.nativeness.harness import (
    aggregate_nativeness_checks,
    checker_context_for_simulation,
    precheck_factor_check,
)

GENDER_FACTOR_ID = "gender_agreement"

# Deterministic gender evidence is always prefixed "gender-<version>: ..." —
# free-text LLM reasoning never is. This is how a stored verdict is attributed
# to the precheck rather than the LLM judge.
_DETERMINISTIC_EVIDENCE_PREFIX = "gender-"

SNAPSHOT_BASENAME = "gender_recheck_snapshot"


class GenderRecheckPriorVerdict(BaseModel):
    """One patched sim's stored gender verdict, preserved for recovery."""

    cell: Annotated[
        str, Field(description="Cell directory, relative to the recheck root.")
    ]
    sim_id: Annotated[str, Field(description="Simulation id (sim file stem).")]
    task_id: Annotated[str, Field(description="Task the sim ran.")]
    language: Annotated[str, Field(description="Language the factor was scored for.")]
    prior_outcome: Annotated[
        JudgeOutcome, Field(description="Stored gender_agreement outcome.")
    ]
    prior_evidence: Annotated[
        Optional[str], Field(default=None, description="Stored evidence text.")
    ]
    prior_violation_count: Annotated[
        int, Field(default=0, description="Stored violation count.")
    ]
    prior_observed_severity: Annotated[
        int, Field(default=0, description="Stored observed severity.")
    ]
    prior_score: Annotated[
        Optional[float],
        Field(default=None, description="Stored per-sim nativeness score."),
    ]
    new_outcome: Annotated[
        JudgeOutcome, Field(description="Recomputed gender_agreement outcome.")
    ]
    new_evidence: Annotated[
        Optional[str], Field(default=None, description="Recomputed evidence text.")
    ]


class GenderRecheckSnapshot(BaseModel):
    """Provenance-stamped record of the verdicts a recheck pass replaced."""

    created_at: Annotated[
        str, Field(description="UTC ISO timestamp the snapshot was written.")
    ]
    tool: Annotated[
        str,
        Field(
            default="tau2 judges recheck-gender",
            description="Verb that produced this snapshot.",
        ),
    ]
    root: Annotated[str, Field(description="Recheck root the snapshot covers.")]
    checker_version_after: Annotated[
        str,
        Field(description="GENDER_AGREEMENT_CHECKER_VERSION the recheck applied."),
    ]
    checker_versions_before: Annotated[
        list[str],
        Field(
            description="Distinct prior deterministic checker versions observed "
            "in the replaced verdicts (from their evidence prefixes)."
        ),
    ]
    sims: Annotated[
        list[GenderRecheckPriorVerdict],
        Field(description="Prior verdict of every patched sim."),
    ]


class GenderRecheckRootReport(BaseModel):
    """Per-root recheck outcome, printed as the verb's summary."""

    root: Annotated[str, Field(description="Recheck root directory.")]
    cells: Annotated[int, Field(ge=0, description="Dir-format cells scanned.")]
    sims_scanned: Annotated[
        int, Field(ge=0, description="Sim files with a stored gender check.")
    ]
    candidates: Annotated[
        int,
        Field(ge=0, description="Sims whose stored verdict was a precheck FAIL."),
    ]
    fails_before: Annotated[
        int, Field(ge=0, description="gender_agreement FAILs before the recheck.")
    ]
    fails_after: Annotated[
        int, Field(ge=0, description="gender_agreement FAILs after the recheck.")
    ]
    patched: Annotated[int, Field(ge=0, description="Sim files rewritten.")]
    skipped: Annotated[
        int,
        Field(
            ge=0,
            description="Candidates skipped (missing task, unresolvable "
            "language, or no hybrid gender factor for the language) — logged, "
            "never failing the pass.",
        ),
    ]
    snapshot_path: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Snapshot written for this pass; None when nothing "
            "changed (idempotent re-run).",
        ),
    ]


class _SimPatch(BaseModel):
    """One pending sim rewrite plus its snapshot entry."""

    sim_path: str
    prior: GenderRecheckPriorVerdict
    patched_sim: SimulationRun


def _checker_version_of(evidence: Optional[str]) -> str:
    """The checker version stamped in deterministic evidence (else 'unknown')."""
    text = evidence or ""
    if not text.startswith(_DETERMINISTIC_EVIDENCE_PREFIX):
        return "unknown"
    head = text.split(":", 1)[0]
    return head.removeprefix(_DETERMINISTIC_EVIDENCE_PREFIX) or "unknown"


def _next_snapshot_path(root: Path) -> Path:
    """The first unused snapshot filename — the original is never overwritten."""
    candidate = root / f"{SNAPSHOT_BASENAME}.json"
    counter = 2
    while candidate.exists():
        candidate = root / f"{SNAPSHOT_BASENAME}_{counter}.json"
        counter += 1
    return candidate


def _recheck_sim(
    sim: SimulationRun,
    task: Task,
    *,
    agent_provider: Optional[str],
    domain: Optional[str],
) -> Optional[SimulationRun]:
    """Recompute the gender factor on one candidate sim.

    Returns the patched sim (factor check replaced, aggregates recomputed), or
    None when the recomputed verdict is byte-identical to the stored one.
    Raises ValueError for structural problems the caller should count as
    skips (no resolvable language, no hybrid gender factor).
    """
    info = sim.nativeness_info
    assert info is not None  # caller guarantees a stored gender check
    language, _script = get_simulation_language_info(sim)
    if language is None:
        language = info.language
    if not language:
        raise ValueError("no resolvable language")

    factor = next(
        (
            f
            for f in judge_factors_for(language)
            if f.id == GENDER_FACTOR_ID and f.enabled and f.precheck_type is not None
        ),
        None,
    )
    if factor is None:
        raise ValueError(
            f"language {language!r} defines no enabled hybrid {GENDER_FACTOR_ID} factor"
        )
    checker = CHECKER_REGISTRY.get(factor.precheck_type)
    if checker is None:
        raise ValueError(f"unknown precheck {factor.precheck_type!r}")

    participants = resolve_participants(
        sim,
        task,
        language=language,
        agent_provider=agent_provider,
        domain=domain,
    )
    context = checker_context_for_simulation(
        sim,
        task,
        language,
        agent_gender=participants.agent_gender,
        caller_gender=participants.caller_gender,
        caller_name=participants.caller_name,
    )
    outcome, evidence = checker(context)
    new_check = precheck_factor_check(factor, outcome, evidence)

    stored = next(c for c in info.factor_checks if c.id == GENDER_FACTOR_ID)
    if new_check == stored:
        return None
    patched = sim.model_copy(deep=True)
    patched_info = patched.nativeness_info
    assert patched_info is not None
    patched_info.factor_checks = [
        new_check if check.id == GENDER_FACTOR_ID else check
        for check in patched_info.factor_checks
    ]
    aggregates = aggregate_nativeness_checks(patched_info.factor_checks)
    for field, value in aggregates.model_dump().items():
        setattr(patched_info, field, value)
    return patched


def recheck_gender_root(root: Path) -> GenderRecheckRootReport:
    """Recheck every dir-format cell under ``root`` (``<cell>/simulations/*.json``).

    Reads each sim once, recomputes the deterministic gender verdict for every
    precheck-FAIL candidate, snapshots the prior verdicts (once, before any
    write), patches the sim files in place, and returns the per-root report.
    """
    root = Path(root)
    cell_dirs = sorted({p.parent.parent for p in root.rglob("simulations/*.json")})
    if not cell_dirs:
        raise FileNotFoundError(
            f"no dir-format cells (<cell>/simulations/*.json) found under {root}"
        )

    report = GenderRecheckRootReport(
        root=str(root),
        cells=len(cell_dirs),
        sims_scanned=0,
        candidates=0,
        fails_before=0,
        fails_after=0,
        patched=0,
        skipped=0,
    )
    patches: list[_SimPatch] = []

    for cell in cell_dirs:
        meta = Results.load_metadata(cell)
        tasks_by_id = {t.id: t for t in meta.tasks}
        audio_cfg = meta.info.audio_native_config
        agent_provider = audio_cfg.provider if audio_cfg is not None else None
        domain = meta.info.environment_info.domain_name

        for sim_path in sorted((cell / "simulations").glob("*.json")):
            sim = SimulationRun.model_validate_json(sim_path.read_text())
            info = sim.nativeness_info
            if info is None:
                continue
            stored = next(
                (c for c in info.factor_checks if c.id == GENDER_FACTOR_ID), None
            )
            if stored is None:
                continue
            report.sims_scanned += 1
            fails_now = stored.outcome is JudgeOutcome.FAIL
            report.fails_before += fails_now
            if not (
                fails_now
                and (stored.evidence or "").startswith(_DETERMINISTIC_EVIDENCE_PREFIX)
            ):
                # Not a deterministic precheck FAIL: LLM verdicts and non-FAIL
                # outcomes are out of scope for a deterministic recheck.
                report.fails_after += fails_now
                continue
            report.candidates += 1
            task = tasks_by_id.get(sim.task_id)
            if task is None:
                logger.warning(
                    f"recheck-gender: no task for sim {sim.id} "
                    f"(task_id={sim.task_id}) in {cell}; skipping"
                )
                report.skipped += 1
                report.fails_after += 1  # left as stored (FAIL)
                continue
            try:
                patched = _recheck_sim(
                    sim, task, agent_provider=agent_provider, domain=domain
                )
            except ValueError as exc:
                logger.warning(f"recheck-gender: sim {sim.id} in {cell}: {exc}")
                report.skipped += 1
                report.fails_after += 1  # left as stored (FAIL)
                continue
            if patched is None:
                report.fails_after += 1  # unchanged FAIL (current-version verdict)
                continue
            patched_info = patched.nativeness_info
            assert patched_info is not None
            new_check = next(
                c for c in patched_info.factor_checks if c.id == GENDER_FACTOR_ID
            )
            report.fails_after += new_check.outcome is JudgeOutcome.FAIL
            patches.append(
                _SimPatch(
                    sim_path=str(sim_path),
                    prior=GenderRecheckPriorVerdict(
                        cell=str(cell.relative_to(root)) if cell != root else ".",
                        sim_id=sim.id,
                        task_id=sim.task_id,
                        language=patched_info.language or "",
                        prior_outcome=stored.outcome,
                        prior_evidence=stored.evidence,
                        prior_violation_count=stored.violation_count,
                        prior_observed_severity=stored.observed_severity,
                        prior_score=info.score,
                        new_outcome=new_check.outcome,
                        new_evidence=new_check.evidence,
                    ),
                    patched_sim=patched,
                )
            )

    if not patches:
        return report

    snapshot = GenderRecheckSnapshot(
        created_at=datetime.now(timezone.utc).isoformat(),
        root=str(root),
        checker_version_after=GENDER_AGREEMENT_CHECKER_VERSION,
        checker_versions_before=sorted(
            {_checker_version_of(patch.prior.prior_evidence) for patch in patches}
        ),
        sims=[patch.prior for patch in patches],
    )
    snapshot_path = _next_snapshot_path(root)
    snapshot_path.write_text(snapshot.model_dump_json(indent=2))
    report.snapshot_path = str(snapshot_path)

    for patch in patches:
        Path(patch.sim_path).write_text(patch.patched_sim.model_dump_json(indent=2))
        report.patched += 1
    return report
