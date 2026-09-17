# Copyright Sierra
"""Deterministic post-hoc scoring correction for the tau-Elicitation paper.

The frozen calls are immutable evidence.  This module verifies the reviewer
archive's authoritative result manifest and emits a separate correction ledger
for calls that were incorrectly scored because the historical ``NAME`` fold did
not equate the spoken medication unit ``milligram(s)`` with ``mg``.

Only an accepted ``submit_fields`` write to the task's expected record can be
corrected.  Failed/repeated tool attempts, exploratory calls outside the result
manifest, and failures that remain unequal under the current fold are retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tau2.domains.intake.folds import FoldKind, fold_value

SCORING_CORRECTION_VERSION = "tau-elicit-scoring-correction-v1"
DEFAULT_ARTIFACT = Path("analysis_inputs/intake_scoring_correction_2026-09-16.json")
_ACCEPTED_SUBMISSION_PREFIX = "Fields submitted to record"
_MILLIGRAM = re.compile(r"\bmilligrams?\b", re.IGNORECASE)
_NON_DECOMPOSING = str.maketrans(
    {
        "ı": "i",
        "ø": "o",
        "ł": "l",
        "đ": "d",
        "ð": "d",
        "þ": "th",
        "æ": "ae",
        "œ": "oe",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ManifestCell(BaseModel):
    """Authoritative scope and provenance for one archived result cell."""

    model_config = ConfigDict(extra="ignore")

    results_path: str
    cohort: str
    system: str
    condition: str
    rows: Annotated[int, Field(gt=0)]
    tasks: Annotated[int, Field(gt=0)]
    trials: list[int]
    passes: Annotated[int, Field(ge=0)]
    pass_rate: Annotated[float, Field(ge=0, le=1)]
    transcript_file: str
    transcript_sha256: str

    @field_validator("trials", mode="before")
    @classmethod
    def _parse_trials(cls, value: object) -> object:
        if isinstance(value, str):
            return json.loads(value)
        return value


class PromptSnapshotRef(BaseModel):
    """Content-addressed task snapshot referenced by one result cell."""

    model_config = ConfigDict(extra="ignore")

    archive_path: str
    sha256: str


class PromptCell(BaseModel):
    """Prompt-manifest entry needed to recover historical task contracts."""

    model_config = ConfigDict(extra="ignore")

    cell: str
    task_snapshots: list[PromptSnapshotRef]


class PromptManifest(BaseModel):
    """Validated projection of the reviewer prompt manifest."""

    model_config = ConfigDict(extra="ignore")

    cells: list[PromptCell]


class CompactTurn(BaseModel):
    """Compact tool event fields needed to reconstruct accepted writes."""

    model_config = ConfigDict(extra="ignore")

    order: int
    role: str
    source: str
    tool_name: Optional[str] = None
    tool_arguments: Optional[dict[str, Any]] = None
    tool_result: Optional[str] = None


class CompactCall(BaseModel):
    """Validated compact call fields used by the correction detector."""

    model_config = ConfigDict(extra="ignore")

    cell: str
    cohort: str
    simulation_id: str
    task_id: str
    trial: int
    reward: float
    turns: list[CompactTurn]


class TaskContract(BaseModel):
    """Historical expected write and fold rules for one task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    record_id: str
    expected_fields: dict[str, str]
    fold_kinds: dict[str, FoldKind]


class FieldCorrection(BaseModel):
    """One field whose equality changes under the corrected fold."""

    model_config = ConfigDict(extra="forbid")

    field_name: Literal["current_medication"]
    fold_kind: Literal["name"]
    submitted_value: str
    expected_value: str
    legacy_submitted_fold: str
    legacy_expected_fold: str
    corrected_fold: str


class CorrectedCall(BaseModel):
    """One immutable archived call with a corrected derived reward."""

    model_config = ConfigDict(extra="forbid")

    results_path: str
    transcript_file: str
    cohort: str
    system: str
    condition: str
    simulation_id: str
    task_id: str
    trial: int
    original_reward: Literal[0.0]
    corrected_reward: Literal[1.0]
    accepted_submission_order: int
    reason: Literal["medication_unit_alias"] = "medication_unit_alias"
    fields: list[FieldCorrection]


class CellCorrectionSummary(BaseModel):
    """Original and corrected counts for one authoritative manifest cell."""

    model_config = ConfigDict(extra="forbid")

    results_path: str
    transcript_file: str
    transcript_sha256: str
    cohort: str
    system: str
    condition: str
    rows: int
    tasks: int
    trials: list[int]
    original_passes: int
    corrections: int
    corrected_passes: int
    original_pass_rate: float
    corrected_pass_rate: float


class CorrectionChecks(BaseModel):
    """Scope checks completed before any reward was corrected."""

    model_config = ConfigDict(extra="forbid")

    manifest_cells: int
    transcript_rows: int
    unique_simulation_ids: int
    transcript_hashes_verified: int
    cell_row_counts_verified: int
    cell_task_counts_verified: int
    cell_trial_sets_verified: int
    cell_original_pass_counts_verified: int


class ScoringCorrectionArtifact(BaseModel):
    """Provenance-bearing correction layer over immutable frozen outcomes."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCORING_CORRECTION_VERSION
    correction_rule: str
    scope: str
    original_transcripts_unchanged: Literal[True] = True
    results_manifest_path: str
    results_manifest_sha256: str
    prompt_manifest_path: str
    prompt_manifest_sha256: str
    checks: CorrectionChecks
    corrected_calls: int
    corrected_calls_by_cohort: dict[str, int]
    cells: list[CellCorrectionSummary]
    calls: list[CorrectedCall]


def _legacy_fold_name(value: str) -> str:
    """Exact historical NAME fold, before medication-unit normalization."""
    decomposed = unicodedata.normalize("NFD", value.strip())
    folded = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    ).casefold()
    folded = folded.translate(_NON_DECOMPOSING)
    folded = folded.replace("'", "").replace("’", "")
    folded = folded.replace("&", " and ").replace("%", " percent ")
    folded = re.sub(r"[-,/()]", " ", folded)
    folded = folded.replace(".", "")
    folded = re.sub(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def _fold(kind: FoldKind, value: str, *, corrected: bool) -> str:
    if kind is FoldKind.NAME and not corrected:
        return _legacy_fold_name(value)
    return fold_value(kind, value)


def _read_manifest(path: Path) -> list[ManifestCell]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [ManifestCell.model_validate(row) for row in csv.DictReader(handle)]


def _task_contracts(
    release_root: Path,
    prompt_cell: PromptCell,
) -> dict[str, TaskContract]:
    contracts: dict[str, TaskContract] = {}
    for reference in prompt_cell.task_snapshots:
        path = release_root / reference.archive_path
        if _sha256(path) != reference.sha256:
            raise ValueError(f"Task snapshot hash mismatch: {reference.archive_path}")
        document = json.loads(path.read_text())
        tasks = document.get("tasks") if isinstance(document, dict) else document
        if not isinstance(tasks, list):
            raise ValueError(
                f"Task snapshot has no task list: {reference.archive_path}"
            )
        for task in tasks:
            task_id = str(task["id"])
            actions = task["initial_state"]["initialization_actions"]
            seed_records = [row for row in actions if row["func_name"] == "seed_record"]
            if len(seed_records) != 1:
                raise ValueError(f"Expected one seed_record action for {task_id}")
            record = seed_records[0]["arguments"]["record"]
            folds = {
                str(field["name"]): FoldKind(str(field["fold"]))
                for field in record["fields"]
            }
            expected_fields: dict[str, str] = {}
            expected_record_ids: set[str] = set()
            for action in task["evaluation_criteria"].get("actions") or []:
                if action.get("name") != "submit_fields":
                    continue
                arguments = action.get("arguments") or {}
                expected_record_ids.add(str(arguments.get("record_id")))
                expected_fields.update(
                    {
                        str(name): str(value)
                        for name, value in (arguments.get("fields") or {}).items()
                    }
                )
            if not expected_fields or len(expected_record_ids) != 1:
                raise ValueError(f"Unsupported submit_fields contract for {task_id}")
            missing_folds = set(expected_fields) - set(folds)
            if missing_folds:
                raise ValueError(
                    f"Missing folds for {task_id}: {sorted(missing_folds)}"
                )
            contract = TaskContract(
                task_id=task_id,
                record_id=next(iter(expected_record_ids)),
                expected_fields=expected_fields,
                fold_kinds={name: folds[name] for name in expected_fields},
            )
            previous = contracts.get(task_id)
            if previous is not None and previous != contract:
                raise ValueError(f"Historical task contract drift for {task_id}")
            contracts[task_id] = contract
    return contracts


def _accepted_submission(
    call: CompactCall,
    contract: TaskContract,
) -> tuple[dict[str, str], int] | None:
    accepted: dict[str, str] = {}
    last_order = -1
    for turn in sorted(call.turns, key=lambda row: row.order):
        if (
            turn.role != "tool"
            or turn.source != "agent_tool_event"
            or turn.tool_name != "submit_fields"
            or not str(turn.tool_result or "").startswith(_ACCEPTED_SUBMISSION_PREFIX)
        ):
            continue
        arguments = turn.tool_arguments or {}
        if str(arguments.get("record_id")) != contract.record_id:
            continue
        for name, value in (arguments.get("fields") or {}).items():
            if isinstance(value, str):
                accepted[str(name)] = value
        last_order = turn.order
    if last_order < 0:
        return None
    return accepted, last_order


def _detect_correction(
    call: CompactCall,
    cell: ManifestCell,
    contract: TaskContract,
) -> CorrectedCall | None:
    if call.reward != 0.0:
        return None
    submission = _accepted_submission(call, contract)
    if submission is None:
        return None
    submitted, order = submission
    if not set(contract.expected_fields) <= set(submitted):
        return None

    legacy_equal: dict[str, bool] = {}
    corrected_equal: dict[str, bool] = {}
    for name, expected in contract.expected_fields.items():
        kind = contract.fold_kinds[name]
        legacy_equal[name] = _fold(kind, submitted[name], corrected=False) == _fold(
            kind, expected, corrected=False
        )
        corrected_equal[name] = _fold(kind, submitted[name], corrected=True) == _fold(
            kind, expected, corrected=True
        )
    if all(legacy_equal.values()) or not all(corrected_equal.values()):
        return None

    changed = [name for name in contract.expected_fields if not legacy_equal[name]]
    if changed != ["current_medication"]:
        raise ValueError(
            f"Unexpected fold-sensitive fields for {call.simulation_id}: {changed}"
        )
    name = changed[0]
    kind = contract.fold_kinds[name]
    raw_pair = f"{submitted[name]} {contract.expected_fields[name]}"
    if kind is not FoldKind.NAME or _MILLIGRAM.search(raw_pair) is None:
        raise ValueError(
            f"Correction is not a medication-unit alias: {call.simulation_id}"
        )
    corrected_fold = _fold(kind, submitted[name], corrected=True)
    return CorrectedCall(
        results_path=cell.results_path,
        transcript_file=cell.transcript_file,
        cohort=cell.cohort,
        system=cell.system,
        condition=cell.condition,
        simulation_id=call.simulation_id,
        task_id=call.task_id,
        trial=call.trial,
        original_reward=0.0,
        corrected_reward=1.0,
        accepted_submission_order=order,
        fields=[
            FieldCorrection(
                field_name="current_medication",
                fold_kind="name",
                submitted_value=submitted[name],
                expected_value=contract.expected_fields[name],
                legacy_submitted_fold=_fold(kind, submitted[name], corrected=False),
                legacy_expected_fold=_fold(
                    kind, contract.expected_fields[name], corrected=False
                ),
                corrected_fold=corrected_fold,
            )
        ],
    )


def build_scoring_correction(release_root: Path) -> ScoringCorrectionArtifact:
    """Verify the full manifest and derive the immutable reward-correction layer."""
    release_root = release_root.resolve()
    manifest_path = release_root / "results" / "manifest.csv"
    prompt_manifest_path = release_root / "prompts" / "manifest.json"
    cells = _read_manifest(manifest_path)
    if len({cell.results_path for cell in cells}) != len(cells):
        raise ValueError("Duplicate result cells in authoritative manifest")
    prompt_manifest = PromptManifest.model_validate_json(
        prompt_manifest_path.read_text()
    )
    prompt_by_cell = {cell.cell: cell for cell in prompt_manifest.cells}
    if set(prompt_by_cell) != {cell.results_path for cell in cells}:
        raise ValueError("Prompt-manifest and result-manifest cell rosters differ")

    all_simulation_ids: set[str] = set()
    corrections: list[CorrectedCall] = []
    summaries: list[CellCorrectionSummary] = []
    total_rows = 0
    for cell in sorted(cells, key=lambda row: row.results_path):
        transcript_path = release_root / cell.transcript_file
        if _sha256(transcript_path) != cell.transcript_sha256:
            raise ValueError(f"Transcript hash mismatch: {cell.transcript_file}")
        calls = [
            CompactCall.model_validate_json(line)
            for line in transcript_path.read_text().splitlines()
            if line
        ]
        if len(calls) != cell.rows:
            raise ValueError(
                f"Row count mismatch for {cell.results_path}: {len(calls)} != {cell.rows}"
            )
        task_ids = {call.task_id for call in calls}
        if len(task_ids) != cell.tasks:
            raise ValueError(f"Task count mismatch for {cell.results_path}")
        if sorted({call.trial for call in calls}) != cell.trials:
            raise ValueError(f"Trial set mismatch for {cell.results_path}")
        if len({call.simulation_id for call in calls}) != len(calls):
            raise ValueError(f"Duplicate simulation id within {cell.results_path}")
        overlap = all_simulation_ids & {call.simulation_id for call in calls}
        if overlap:
            raise ValueError(
                f"Simulation ids repeated across cells: {sorted(overlap)[:3]}"
            )
        all_simulation_ids.update(call.simulation_id for call in calls)
        original_passes = sum(call.reward == 1.0 for call in calls)
        if original_passes != cell.passes:
            raise ValueError(f"Pass count mismatch for {cell.results_path}")
        if abs(original_passes / len(calls) - cell.pass_rate) > 1e-12:
            raise ValueError(f"Pass rate mismatch for {cell.results_path}")

        contracts = _task_contracts(release_root, prompt_by_cell[cell.results_path])
        missing_contracts = task_ids - set(contracts)
        if missing_contracts:
            raise ValueError(
                f"Missing task contracts for {cell.results_path}: "
                f"{sorted(missing_contracts)[:3]}"
            )
        cell_corrections = [
            correction
            for call in calls
            if (correction := _detect_correction(call, cell, contracts[call.task_id]))
            is not None
        ]
        corrections.extend(cell_corrections)
        corrected_passes = original_passes + len(cell_corrections)
        summaries.append(
            CellCorrectionSummary(
                results_path=cell.results_path,
                transcript_file=cell.transcript_file,
                transcript_sha256=cell.transcript_sha256,
                cohort=cell.cohort,
                system=cell.system,
                condition=cell.condition,
                rows=cell.rows,
                tasks=cell.tasks,
                trials=cell.trials,
                original_passes=original_passes,
                corrections=len(cell_corrections),
                corrected_passes=corrected_passes,
                original_pass_rate=original_passes / cell.rows,
                corrected_pass_rate=corrected_passes / cell.rows,
            )
        )
        total_rows += len(calls)

    corrections.sort(
        key=lambda row: (
            row.results_path,
            row.task_id,
            row.trial,
            row.simulation_id,
        )
    )
    return ScoringCorrectionArtifact(
        correction_rule=(
            "Re-evaluate immutable archived outcomes after adding the whole-word "
            "NAME-fold alias milligram/milligrams -> mg. Correct only reward-zero "
            "calls whose accepted submit_fields write targets the expected record, "
            "whose complete expected field set is unequal under the historical "
            "fold, and whose complete field set is equal under the corrected fold."
        ),
        scope=(
            "Every call in the 29 cells listed by results/manifest.csv; no "
            "exploratory or superseded result root is eligible."
        ),
        results_manifest_path="results/manifest.csv",
        results_manifest_sha256=_sha256(manifest_path),
        prompt_manifest_path="prompts/manifest.json",
        prompt_manifest_sha256=_sha256(prompt_manifest_path),
        checks=CorrectionChecks(
            manifest_cells=len(cells),
            transcript_rows=total_rows,
            unique_simulation_ids=len(all_simulation_ids),
            transcript_hashes_verified=len(cells),
            cell_row_counts_verified=len(cells),
            cell_task_counts_verified=len(cells),
            cell_trial_sets_verified=len(cells),
            cell_original_pass_counts_verified=len(cells),
        ),
        corrected_calls=len(corrections),
        corrected_calls_by_cohort=dict(
            sorted(Counter(row.cohort for row in corrections).items())
        ),
        cells=summaries,
        calls=corrections,
    )


def artifact_path(release_root: Path) -> Path:
    """Return the canonical correction-artifact path for a reviewer archive."""
    return release_root / DEFAULT_ARTIFACT


def write_scoring_correction(
    release_root: Path,
    *,
    output: Optional[Path] = None,
) -> ScoringCorrectionArtifact:
    """Build and write the canonical deterministic correction artifact."""
    artifact = build_scoring_correction(release_root)
    target = output or artifact_path(release_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(artifact.model_dump_json(indent=2) + "\n")
    return artifact


def load_correction_map(release_root: Path) -> dict[str, float]:
    """Load corrected rewards keyed by immutable simulation id."""
    document = ScoringCorrectionArtifact.model_validate_json(
        artifact_path(release_root).read_text()
    )
    return {row.simulation_id: row.corrected_reward for row in document.calls}


def corrected_reward(row: dict[str, Any], corrections: dict[str, float]) -> float:
    """Return a call's corrected reward without mutating its archived record."""
    return float(corrections.get(str(row["simulation_id"]), row["reward"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("papers/tau-intake/v1/reproduction"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    artifact = build_scoring_correction(args.root)
    rendered = artifact.model_dump_json(indent=2) + "\n"
    target = args.output or artifact_path(args.root)
    if args.check:
        if not target.exists() or target.read_text() != rendered:
            print(f"FAIL: scoring correction differs from {target}")
            sys.exit(1)
        print(
            f"PASS: {artifact.corrected_calls} corrected calls across "
            f"{artifact.checks.manifest_cells} manifest cells"
        )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered)
    print(f"Wrote {artifact.corrected_calls} corrections to {target}")


if __name__ == "__main__":
    main()
