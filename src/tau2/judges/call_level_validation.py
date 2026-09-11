# Copyright Sierra
"""Typed, offline verification for frozen call-level judge validations.

The package exposes the paper-facing ``tool_use`` measure as one final decision per
call while retaining its runtime factor ids as provenance. Portuguese register and
regional measures and Mandarin modal particles are also call-level because their
human review units are whole calls. This verifier consumes only the frozen package;
it never calls a model or depends on the construction inputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tau2.judges.validation_archive import (
    ARCHIVE_LANGUAGES,
    BinaryLabel,
    MeasureId,
    SourceSet,
    StoredPredictionPromptContract,
)

CALL_LEVEL_SCHEMA_VERSION = "tau-multi-call-level-validation-v1"
HISTORICAL_PROMPT_REVISION = "6813fceab54fe6c7c44733be9935e6d38aaca848"
TOOL_USE_SUBTYPES = (
    "incorrect_tool_parameters",
    "auth_arg_mismatch",
    "agent_caused_tool_error",
)
SCORING_LABELS = {BinaryLabel.PASS, BinaryLabel.VIOLATION}
SOURCE_ORDER = {SourceSet.PRECISION: 0, SourceSet.RECALL: 1}


class CallMeasure(str, Enum):
    """Paper-facing measures validated at the whole-call level."""

    TOOL_USE = "tool_use"
    REGISTER_FORMALITY = "register_formality"
    REGIONAL_CONSISTENCY = "regional_consistency"
    MODAL_PARTICLES = "modal_particles"


PACKAGE_CONTRACT: dict[CallMeasure, tuple[tuple[str, ...], tuple[str, ...]]] = {
    CallMeasure.TOOL_USE: (ARCHIVE_LANGUAGES, TOOL_USE_SUBTYPES),
    CallMeasure.REGISTER_FORMALITY: (("pt",), ("register_formality",)),
    CallMeasure.REGIONAL_CONSISTENCY: (("pt",), ("regional_consistency",)),
    CallMeasure.MODAL_PARTICLES: (("zh",), ("modal_particles",)),
}

# Frozen confusion matrices make accidental cohort or label drift fail loudly.
EXPECTED_CONFUSION: dict[
    tuple[CallMeasure, str], tuple[int, int, int, int, int, int]
] = {
    (CallMeasure.TOOL_USE, "es"): (38, 6, 2, 0, 0, 0),
    (CallMeasure.TOOL_USE, "pt"): (25, 15, 0, 2, 0, 0),
    (CallMeasure.TOOL_USE, "hi"): (30, 9, 5, 1, 0, 0),
    (CallMeasure.TOOL_USE, "ko"): (41, 4, 1, 0, 1, 0),
    (CallMeasure.TOOL_USE, "zh"): (42, 4, 0, 1, 0, 0),
    (CallMeasure.REGISTER_FORMALITY, "pt"): (15, 18, 1, 0, 0, 0),
    (CallMeasure.REGIONAL_CONSISTENCY, "pt"): (6, 18, 0, 1, 0, 0),
    (CallMeasure.MODAL_PARTICLES, "zh"): (23, 9, 0, 3, 0, 0),
}

# Canonical JSON digests pin the complete stored-prediction contract: wrapper
# text, criteria, judge settings, source revision, and source-file hashes. This
# keeps a rewritten prompt plus freshly regenerated manifests from validating.
EXPECTED_STORED_PROMPT_SHA256: dict[CallMeasure, str] = {
    CallMeasure.TOOL_USE: (
        "900a3df85a11e9cd2c334773f9c8254232cd3a1c4feb9a969242c0fa89cba1ef"
    ),
    CallMeasure.REGISTER_FORMALITY: (
        "76454fa0595b2d076f2d8a4cf37a5496bf8720024cdc74163c9669020f522fe4"
    ),
    CallMeasure.REGIONAL_CONSISTENCY: (
        "244c587c2163e0739fa833800b3d885a0f9cba45a616f39ebbf9bd2d1454725f"
    ),
    CallMeasure.MODAL_PARTICLES: (
        "00f237406d24c58b0470c69a2a5874d8afc948948f96cf4136eca70763a14dbe"
    ),
}


def _blank_to_none(value):
    if isinstance(value, str) and value.strip().upper() in {"", "NA", "N/A"}:
        return None
    return value


def _json_list(value) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, tuple):
        value = list(value)
    return value


def _source_sort_key(source: SourceSet) -> int:
    if source not in SOURCE_ORDER:
        raise ValueError(f"unsupported call-level source set: {source.value}")
    return SOURCE_ORDER[source]


def _expected_row_id(measure: CallMeasure, language: str, simulation_id: str) -> str:
    payload = (
        f"{CALL_LEVEL_SCHEMA_VERSION}\0{measure.value}\0{language}\0{simulation_id}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


class CallValidationRow(BaseModel):
    """One final human label aligned with one stored call-level judge verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_id: Annotated[
        str, Field(pattern=r"^[0-9a-f]{20}$", description="Stable call-row id.")
    ]
    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    measure_id: Annotated[
        CallMeasure, Field(description="Paper-facing call-level measure.")
    ]
    evaluation_level: Literal["call"]
    simulation_id: Annotated[str, Field(min_length=1, description="Simulation id.")]
    task_id: Annotated[str, Field(min_length=1, description="Benchmark task id.")]
    source_sets: Annotated[
        list[SourceSet], Field(min_length=1, description="Contributing packets.")
    ]
    source_row_ids: Annotated[
        list[str],
        Field(min_length=1, description="Contributing normalized annotation rows."),
    ]
    source_factor_ids: Annotated[
        list[str],
        Field(min_length=1, description="Runtime factors combined into the measure."),
    ]
    human_label: Annotated[BinaryLabel, Field(description="Final human label.")]
    human_note: Annotated[
        Optional[str], Field(description="Final human explanation, when supplied.")
    ] = None
    judge_label: Annotated[BinaryLabel, Field(description="Stored judge label.")]
    judge_reason: Annotated[
        Optional[str], Field(description="Stored judge explanation, when supplied.")
    ] = None
    agreement: Annotated[
        Optional[bool], Field(description="Agreement for two scorable labels.")
    ] = None
    judge_models: Annotated[
        list[str], Field(min_length=1, description="Recorded judge models.")
    ]
    prompt_versions: Annotated[
        list[str], Field(description="Recorded prompt versions, when applicable.")
    ]
    rubric_versions: Annotated[
        list[str], Field(min_length=1, description="Recorded rubric versions.")
    ]

    _optional_blanks = field_validator(
        "human_note",
        "judge_reason",
        "agreement",
        mode="before",
    )(_blank_to_none)
    _parse_lists = field_validator(
        "source_sets",
        "source_row_ids",
        "source_factor_ids",
        "judge_models",
        "prompt_versions",
        "rubric_versions",
        mode="before",
    )(_json_list)

    @field_validator("agreement", mode="before")
    @classmethod
    def _parse_agreement(cls, value):
        value = _blank_to_none(value)
        if value is None or isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        return value

    @model_validator(mode="after")
    def _row_is_coherent(self) -> "CallValidationRow":
        languages, source_factors = PACKAGE_CONTRACT[self.measure_id]
        if self.language not in languages:
            raise ValueError(
                f"{self.measure_id.value} is not validated for {self.language}"
            )
        if self.row_id != _expected_row_id(
            self.measure_id, self.language, self.simulation_id
        ):
            raise ValueError("row_id does not match the stable call key")
        expected_agreement = (
            self.human_label == self.judge_label
            if self.human_label in SCORING_LABELS and self.judge_label in SCORING_LABELS
            else None
        )
        if self.agreement != expected_agreement:
            raise ValueError("agreement does not match the final labels")
        if self.source_sets != sorted(set(self.source_sets), key=_source_sort_key):
            raise ValueError("source_sets must be unique and canonically ordered")
        for field_name in (
            "source_row_ids",
            "source_factor_ids",
            "judge_models",
            "prompt_versions",
            "rubric_versions",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
        observed_factors = set(self.source_factor_ids)
        expected_factors = set(source_factors)
        if self.measure_id is CallMeasure.TOOL_USE:
            if not observed_factors <= expected_factors:
                raise ValueError("tool_use row contains an unknown runtime factor")
            expected_order = [
                factor for factor in TOOL_USE_SUBTYPES if factor in observed_factors
            ]
            if self.source_factor_ids != expected_order:
                raise ValueError("tool_use factors must use canonical subtype order")
        elif observed_factors != expected_factors:
            raise ValueError("factor-specific row has the wrong runtime factor")
        return self


class CallMetric(BaseModel):
    """Confusion counts and agreement statistics for one language and measure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str
    measure_id: CallMeasure
    evaluation_level: Literal["call"]
    tp: Annotated[int, Field(ge=0)]
    tn: Annotated[int, Field(ge=0)]
    fp: Annotated[int, Field(ge=0)]
    fn: Annotated[int, Field(ge=0)]
    no_opportunity: Annotated[int, Field(ge=0)]
    errors: Annotated[int, Field(ge=0)]
    n: Annotated[int, Field(ge=0)]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    kappa: Optional[float]

    @model_validator(mode="after")
    def _count_is_coherent(self) -> "CallMetric":
        if self.n != self.tp + self.tn + self.fp + self.fn:
            raise ValueError("metric n does not match the confusion counts")
        return self


class CallPromptArtifact(BaseModel):
    """Exact stored-prediction prompt associated with one validation package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-call-prompt-v2"]
    measure_id: CallMeasure
    languages: list[str]
    stored_prediction_prompt: StoredPredictionPromptContract


class PackageFile(BaseModel):
    """One package member protected by SHA-256."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Annotated[str, Field(min_length=1, description="Relative artifact path.")]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256.")]
    bytes: Annotated[int, Field(ge=0, description="Byte size.")]
    rows: Annotated[Optional[int], Field(default=None, ge=0)]

    @model_validator(mode="after")
    def _path_is_relative(self) -> "PackageFile":
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("manifest paths must stay within the package")
        return self


class CallPackageManifest(BaseModel):
    """Provenance and coverage for one call-level validation measure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-call-package-v1"]
    measure_id: CallMeasure
    evaluation_level: Literal["call"]
    languages: list[str]
    source_factor_ids: list[str]
    rows: Annotated[int, Field(ge=0)]
    scorable_rows: Annotated[int, Field(ge=0)]
    files: list[PackageFile]


class CallLevelManifest(BaseModel):
    """Digest inventory for all retained call-level validation packages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-call-level-root-v1"]
    packages: list[CallMeasure]
    files: list[PackageFile]


class CallLevelValidationReport(BaseModel):
    """Offline package-verification outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: Path
    packages: Annotated[int, Field(ge=0)]
    rows: Annotated[int, Field(ge=0)]
    metric_rows: Annotated[int, Field(ge=0)]
    files_verified: Annotated[int, Field(ge=0)]
    metrics: dict[str, CallMetric]


def sha256_file(path: Path) -> str:
    """Return a file's byte-level SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_header = list(model.model_fields)
        if reader.fieldnames != expected_header:
            raise ValueError(
                f"{path}: CSV header differs from the typed contract: "
                f"{reader.fieldnames} != {expected_header}"
            )
        return [model.model_validate(row) for row in reader]


def read_call_validation_rows(path: Path) -> list[CallValidationRow]:
    """Read one canonical call-level validation CSV through its typed contract."""
    return [
        CallValidationRow.model_validate(row)
        for row in _read_csv(path, CallValidationRow)
    ]


def _read_metrics(path: Path) -> list[CallMetric]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: metrics must be a JSON list")
    return [CallMetric.model_validate(row) for row in raw]


def compute_call_metrics(rows: list[CallValidationRow]) -> list[CallMetric]:
    """Recompute P/R/F1 and Cohen's kappa from final call-level rows."""
    groups: dict[tuple[str, CallMeasure], list[CallValidationRow]] = defaultdict(list)
    for row in rows:
        groups[(row.language, row.measure_id)].append(row)
    metrics: list[CallMetric] = []
    for (language, measure), group in sorted(
        groups.items(),
        key=lambda item: (ARCHIVE_LANGUAGES.index(item[0][0]), item[0][1].value),
    ):
        tp = sum(
            row.human_label is BinaryLabel.VIOLATION
            and row.judge_label is BinaryLabel.VIOLATION
            for row in group
        )
        tn = sum(
            row.human_label is BinaryLabel.PASS and row.judge_label is BinaryLabel.PASS
            for row in group
        )
        fp = sum(
            row.human_label is BinaryLabel.PASS
            and row.judge_label is BinaryLabel.VIOLATION
            for row in group
        )
        fn = sum(
            row.human_label is BinaryLabel.VIOLATION
            and row.judge_label is BinaryLabel.PASS
            for row in group
        )
        no_opportunity = sum(
            BinaryLabel.NO_OPPORTUNITY in {row.human_label, row.judge_label}
            for row in group
        )
        errors = sum(
            BinaryLabel.ERROR in {row.human_label, row.judge_label} for row in group
        )
        n = tp + tn + fp + fn
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        kappa = None
        if n:
            observed = (tp + tn) / n
            human_positive = (tp + fn) / n
            judge_positive = (tp + fp) / n
            expected = human_positive * judge_positive + (1 - human_positive) * (
                1 - judge_positive
            )
            if expected != 1:
                kappa = (observed - expected) / (1 - expected)
        metrics.append(
            CallMetric(
                language=language,
                measure_id=measure,
                evaluation_level="call",
                tp=tp,
                tn=tn,
                fp=fp,
                fn=fn,
                no_opportunity=no_opportunity,
                errors=errors,
                n=n,
                precision=precision,
                recall=recall,
                f1=f1,
                kappa=kappa,
            )
        )
    return metrics


def _metric_payload(metrics: list[CallMetric]) -> list[dict]:
    return [metric.model_dump(mode="json") for metric in metrics]


def _verify_file(root: Path, record: PackageFile) -> Path:
    path = (root / record.path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"artifact path escapes package: {record.path}")
    if not path.is_file():
        raise ValueError(f"missing validation artifact: {record.path}")
    if path.stat().st_size != record.bytes or sha256_file(path) != record.sha256:
        raise ValueError(f"validation artifact digest mismatch: {record.path}")
    return path


def _verify_prompt(
    prompt: CallPromptArtifact,
    measure: CallMeasure,
    languages: tuple[str, ...],
    source_factors: tuple[str, ...],
    rows: list[CallValidationRow],
) -> None:
    if prompt.measure_id is not measure or prompt.languages != list(languages):
        raise ValueError(f"prompt coverage mismatch for {measure.value}")
    contract = prompt.stored_prediction_prompt
    contract_payload = json.dumps(
        contract.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if (
        hashlib.sha256(contract_payload).hexdigest()
        != EXPECTED_STORED_PROMPT_SHA256[measure]
    ):
        raise ValueError(f"stored prompt content differs for {measure.value}")
    if contract.source_revision != HISTORICAL_PROMPT_REVISION:
        raise ValueError(f"stored prompt revision differs for {measure.value}")
    if contract.languages != list(languages):
        raise ValueError(f"stored prompt languages differ for {measure.value}")
    if contract.measure_ids != [MeasureId(measure.value)]:
        raise ValueError(f"stored prompt measure differs for {measure.value}")
    observed_prompts = {
        version for row in rows for version in row.prompt_versions if version
    }
    observed_rubrics = {
        version for row in rows for version in row.rubric_versions if version
    }
    observed_models = {
        model
        for row in rows
        for combined in row.judge_models
        for model in combined.split(" | ")
        if model != "deterministic"
    }
    if observed_prompts != {contract.prompt_version}:
        raise ValueError(f"stored prompt version differs for {measure.value}")
    if observed_rubrics != {contract.rubric_version}:
        raise ValueError(f"stored rubric version differs for {measure.value}")
    if observed_models != set(contract.judge_models):
        raise ValueError(f"stored judge model differs for {measure.value}")
    for row in rows:
        row_models = {
            model
            for combined in row.judge_models
            for model in combined.split(" | ")
            if model != "deterministic"
        }
        if row_models:
            if row_models != set(contract.judge_models):
                raise ValueError(
                    f"row judge model differs for {measure.value}: {row.row_id}"
                )
            if row.prompt_versions != [contract.prompt_version]:
                raise ValueError(
                    f"row prompt version differs for {measure.value}: {row.row_id}"
                )
        else:
            valid_deterministic_no_opportunity = (
                measure is CallMeasure.TOOL_USE
                and row.judge_models == ["deterministic"]
                and row.human_label is BinaryLabel.NO_OPPORTUNITY
                and row.judge_label is BinaryLabel.NO_OPPORTUNITY
                and row.source_factor_ids == list(TOOL_USE_SUBTYPES)
            )
            if not valid_deterministic_no_opportunity:
                raise ValueError(
                    f"unexpected deterministic-only row for {measure.value}: "
                    f"{row.row_id}"
                )
            if row.prompt_versions:
                raise ValueError(
                    f"deterministic row has a prompt version for {measure.value}: "
                    f"{row.row_id}"
                )
        if row.rubric_versions != [contract.rubric_version]:
            raise ValueError(
                f"row rubric version differs for {measure.value}: {row.row_id}"
            )
    if measure is CallMeasure.TOOL_USE:
        observed = [factor.criterion_id for factor in contract.quality_criteria]
        if (
            contract.prompt_family != "quality"
            or set(observed) != set(source_factors)
            or contract.nativeness_criteria
        ):
            raise ValueError("tool_use prompt factors differ from the contract")
    else:
        criteria = contract.nativeness_criteria
        observed = [
            factor.factor_id for factors in criteria.values() for factor in factors
        ]
        if (
            contract.prompt_family != "nativeness"
            or observed != list(source_factors)
            or contract.quality_criteria
        ):
            raise ValueError(f"prompt factors differ for {measure.value}")


def verify_call_level_validation_package(
    root: Path,
) -> CallLevelValidationReport:
    """Hash, parse, align, and recompute the frozen package without model calls."""
    root = root.resolve()
    manifest = CallLevelManifest.model_validate_json(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    expected_measures = list(CallMeasure)
    if manifest.packages != expected_measures:
        raise ValueError("call-level package order differs from the contract")
    expected_paths = {"README.md"} | {
        f"{measure.value}/{name}"
        for measure in CallMeasure
        for name in ("manifest.json", "metrics.json", "prompt.json", "validation.csv")
    }
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "manifest.json"
    }
    recorded_paths = {record.path for record in manifest.files}
    if actual_paths != expected_paths or recorded_paths != expected_paths:
        raise ValueError("call-level package file inventory differs from the contract")
    if len(manifest.files) != len(recorded_paths):
        raise ValueError("call-level root manifest contains duplicate paths")
    for record in manifest.files:
        _verify_file(root, record)

    all_rows: list[CallValidationRow] = []
    metrics_by_key: dict[str, CallMetric] = {}
    seen_row_ids: set[str] = set()
    for measure in CallMeasure:
        languages, source_factors = PACKAGE_CONTRACT[measure]
        package_root = root / measure.value
        package_manifest = CallPackageManifest.model_validate_json(
            (package_root / "manifest.json").read_text(encoding="utf-8")
        )
        expected_manifest_values = (
            package_manifest.measure_id is measure
            and package_manifest.languages == list(languages)
            and package_manifest.source_factor_ids == list(source_factors)
        )
        if not expected_manifest_values:
            raise ValueError(f"manifest contract drifted for {measure.value}")
        package_files = {record.path: record for record in package_manifest.files}
        if set(package_files) != {"validation.csv", "metrics.json", "prompt.json"}:
            raise ValueError(f"package inventory differs for {measure.value}")
        if len(package_manifest.files) != len(package_files):
            raise ValueError(f"duplicate package paths for {measure.value}")
        for record in package_manifest.files:
            _verify_file(package_root, record)

        rows = read_call_validation_rows(package_root / "validation.csv")
        keys = [(row.language, row.simulation_id) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate call rows for {measure.value}")
        for row in rows:
            if row.measure_id is not measure:
                raise ValueError(f"wrong measure row in {measure.value}")
            if row.row_id in seen_row_ids:
                raise ValueError(f"duplicate package row_id: {row.row_id}")
            seen_row_ids.add(row.row_id)
        if package_manifest.rows != len(rows):
            raise ValueError(f"row count differs for {measure.value}")
        scorable = sum(
            row.human_label in SCORING_LABELS and row.judge_label in SCORING_LABELS
            for row in rows
        )
        if package_manifest.scorable_rows != scorable:
            raise ValueError(f"scorable-row count differs for {measure.value}")
        validation_record = package_files["validation.csv"]
        if validation_record.rows != len(rows):
            raise ValueError(f"validation file row count differs for {measure.value}")

        stored_metrics = _read_metrics(package_root / "metrics.json")
        recomputed_metrics = compute_call_metrics(rows)
        if _metric_payload(stored_metrics) != _metric_payload(recomputed_metrics):
            raise ValueError(f"stored metrics do not recompute for {measure.value}")
        if package_files["metrics.json"].rows != len(stored_metrics):
            raise ValueError(f"metric file row count differs for {measure.value}")
        observed_confusion = {
            (metric.measure_id, metric.language): (
                metric.tp,
                metric.tn,
                metric.fp,
                metric.fn,
                metric.no_opportunity,
                metric.errors,
            )
            for metric in stored_metrics
        }
        expected_confusion = {
            key: value for key, value in EXPECTED_CONFUSION.items() if key[0] is measure
        }
        if observed_confusion != expected_confusion:
            raise ValueError(f"confusion matrix differs for {measure.value}")
        for metric in stored_metrics:
            metrics_by_key[f"{metric.language}:{metric.measure_id.value}"] = metric

        prompt = CallPromptArtifact.model_validate_json(
            (package_root / "prompt.json").read_text(encoding="utf-8")
        )
        _verify_prompt(prompt, measure, languages, source_factors, rows)
        if package_files["prompt.json"].rows is not None:
            raise ValueError(f"prompt row metadata must be empty for {measure.value}")
        all_rows.extend(rows)

    return CallLevelValidationReport(
        root=root,
        packages=len(CallMeasure),
        rows=len(all_rows),
        metric_rows=len(metrics_by_key),
        files_verified=len(manifest.files),
        metrics=metrics_by_key,
    )


__all__ = [
    "CALL_LEVEL_SCHEMA_VERSION",
    "EXPECTED_CONFUSION",
    "PACKAGE_CONTRACT",
    "TOOL_USE_SUBTYPES",
    "CallLevelValidationReport",
    "CallMeasure",
    "CallMetric",
    "CallPackageManifest",
    "CallPromptArtifact",
    "CallValidationRow",
    "compute_call_metrics",
    "read_call_validation_rows",
    "verify_call_level_validation_package",
]
