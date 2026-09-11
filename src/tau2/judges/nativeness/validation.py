# Copyright Sierra
"""Typed τ-Multilingual combined-naturalness validation artifacts.

The checked-in reviewer package is deliberately boring: one selected prompt,
one held-out final-label dataset, and one row-aligned judge result for each
language. This module is the sole reader and verifier for that contract. It
recomputes every count and metric instead of trusting serialized summaries.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.config import (
    DEFAULT_TAU_MULTI_NATURALNESS_JUDGE,
    DEFAULT_TAU_MULTI_NATURALNESS_JUDGE_ARGS,
)
from tau2.data_model.simulation import JudgeOutcome, NativenessFactorCheck
from tau2.judges.nativeness.factors import NativenessFactorConfig, judge_factors_for
from tau2.judges.nativeness.judge import NATIVENESS_JUDGE_PROMPT_VERSION

VALIDATION_FACTOR_ID = "natural_word_choice"
VALIDATION_FACTOR_NAME = "combined_utterance_naturalness"
VALIDATION_LANGUAGES = ("es", "hi", "ko", "pt", "zh")
COMBINED_NATURALNESS_RUBRIC_VERSION = "nativeness-rubric-v20"


class ValidationSplit(str, Enum):
    """Closed validation split names."""

    DEV = "dev"
    TEST = "test"


class ValidationRole(str, Enum):
    """Semantic role of a frozen split."""

    CALIBRATION = "calibration"
    HELD_OUT = "held_out_validation"


class ValidationLabel(str, Enum):
    """Final human label for one utterance."""

    PASS = "pass"
    VIOLATION = "violation"


class FileReference(BaseModel):
    """Relative artifact path and byte digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Annotated[str, Field(description="Path relative to the containing file.")]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256.")]


class ValidationJudgeConfig(BaseModel):
    """Effective judge settings recorded on one validation result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Annotated[str, Field(description="LLM judge model id.")]
    model_args: Annotated[dict, Field(description="Arguments passed to the model.")]
    prompt_version: Annotated[str, Field(description="Judge wrapper prompt version.")]
    rubric_version: Annotated[str, Field(description="Nativeness rubric version.")]
    max_concurrency: Annotated[
        int, Field(ge=1, description="Maximum in-flight utterance judge calls.")
    ]


class ValidationPrompt(BaseModel):
    """The one selected language rubric used by a validation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-utterance-naturalness-prompt-v1"]
    language: Annotated[str, Field(description="ISO language code.")]
    factor_id: Literal["natural_word_choice"]
    factor_name: Literal["combined_utterance_naturalness"]
    prompt_version: Annotated[str, Field(description="Judge wrapper prompt version.")]
    rubric_version: Annotated[str, Field(description="Nativeness rubric version.")]
    question: Annotated[str, Field(description="Language-specific binary question.")]
    opportunity: Annotated[
        str, Field(description="When the utterance offers a scoring opportunity.")
    ]
    native_does: Annotated[str, Field(description="Passing behavior and examples.")]
    ai_likely_does: Annotated[
        str, Field(description="Violating behavior and examples.")
    ]


class ValidationDatasetRow(BaseModel):
    """One final human label at the stable utterance key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_index: Annotated[int, Field(ge=0, description="Zero-based row position.")]
    simulation_id: Annotated[str, Field(description="Source simulation id.")]
    task_id: Annotated[str, Field(description="Source task id.")]
    canonical_task_id: Annotated[
        Optional[str],
        Field(description="Language-independent task identity, when recorded."),
    ] = None
    agent_turn_index: Annotated[
        int, Field(ge=0, description="Zero-based delivered agent-turn index.")
    ]
    agent_turn_id: Annotated[
        Optional[str],
        Field(description="Stable human-readable delivered-turn id, when recorded."),
    ] = None
    domain: Annotated[str, Field(description="Benchmark domain.")]
    experiment: Annotated[str, Field(description="Source experiment cell.")]
    provider: Annotated[str, Field(description="Agent provider family.")]
    preceding_customer_text: Annotated[
        Optional[str], Field(description="Immediately preceding customer speech.")
    ] = None
    agent_text: Annotated[str, Field(min_length=1, description="Agent utterance.")]
    source_interrupted: Annotated[
        bool, Field(description="Whether the caller interrupted this utterance.")
    ]
    label: Annotated[ValidationLabel, Field(description="Final human label.")]
    source_factors: Annotated[
        list[str], Field(description="Original annotation factors pooled into label.")
    ]


class ValidationDataset(BaseModel):
    """One language/split label corpus and its deterministic summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-utterance-naturalness-dataset-v1"]
    language: Annotated[str, Field(description="ISO language code.")]
    split: Annotated[ValidationSplit, Field(description="Dev or test split.")]
    role: Annotated[ValidationRole, Field(description="Calibration or held-out role.")]
    evaluation_level: Literal["utterance"]
    grouping_unit: Literal["simulation_id+agent_turn_index"]
    factor_id: Literal["natural_word_choice"]
    factor_name: Literal["combined_utterance_naturalness"]
    included_factors: Annotated[
        list[str], Field(description="Source human factors pooled into this label.")
    ]
    n_rows: Annotated[int, Field(ge=0, description="Number of utterance rows.")]
    n_positive: Annotated[int, Field(ge=0, description="Human violation count.")]
    n_negative: Annotated[int, Field(ge=0, description="Human pass count.")]
    n_calls: Annotated[int, Field(ge=0, description="Unique source simulations.")]
    n_task_groups: Annotated[int, Field(ge=0, description="Unique source tasks.")]
    domain_counts: Annotated[
        dict[str, int], Field(description="Rows by benchmark domain.")
    ]
    provider_counts: Annotated[
        dict[str, int], Field(description="Rows by provider family.")
    ]
    positive_source_factor_counts: Annotated[
        dict[str, int], Field(description="Positive labels by original factor.")
    ]
    rows: Annotated[list[ValidationDatasetRow], Field(description="Final labels.")]

    @model_validator(mode="after")
    def _summary_matches_rows(self) -> "ValidationDataset":
        keys = [(row.simulation_id, row.agent_turn_index) for row in self.rows]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate (simulation_id, agent_turn_index) dataset key")
        if [row.row_index for row in self.rows] != list(range(len(self.rows))):
            raise ValueError("dataset row_index values must be contiguous and ordered")
        positives = sum(row.label is ValidationLabel.VIOLATION for row in self.rows)
        expected = {
            "n_rows": len(self.rows),
            "n_positive": positives,
            "n_negative": len(self.rows) - positives,
            "n_calls": len({row.simulation_id for row in self.rows}),
            "n_task_groups": len(
                {row.canonical_task_id or row.task_id for row in self.rows}
            ),
        }
        observed = {name: getattr(self, name) for name in expected}
        if observed != expected:
            raise ValueError(f"dataset summary mismatch: {observed=} {expected=}")
        if self.domain_counts != dict(
            sorted(Counter(r.domain for r in self.rows).items())
        ):
            raise ValueError("dataset domain_counts do not match rows")
        if self.provider_counts != dict(
            sorted(Counter(r.provider for r in self.rows).items())
        ):
            raise ValueError("dataset provider_counts do not match rows")
        source_counts = Counter(
            factor
            for row in self.rows
            if row.label is ValidationLabel.VIOLATION
            for factor in row.source_factors
        )
        if self.positive_source_factor_counts != dict(sorted(source_counts.items())):
            raise ValueError("positive_source_factor_counts do not match rows")
        included = set(self.included_factors)
        unknown = sorted(
            {
                factor
                for row in self.rows
                for factor in row.source_factors
                if factor not in included
            }
        )
        if unknown:
            raise ValueError(
                f"row source_factors are not included by the dataset: {unknown}"
            )
        return self


class BinaryMetrics(BaseModel):
    """Binary confusion matrix and derived agreement metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tp: Annotated[int, Field(ge=0, description="True positives.")]
    fn: Annotated[int, Field(ge=0, description="False negatives.")]
    fp: Annotated[int, Field(ge=0, description="False positives.")]
    tn: Annotated[int, Field(ge=0, description="True negatives.")]
    n_errors: Annotated[int, Field(ge=0, description="Judge error rows.")]
    precision: Annotated[Optional[float], Field(description="TP / (TP + FP).")]
    recall: Annotated[Optional[float], Field(description="TP / (TP + FN).")]
    f1: Annotated[Optional[float], Field(description="Harmonic mean of P and R.")]
    kappa: Annotated[
        Optional[float], Field(description="Binary Cohen's kappa agreement.")
    ]


class ValidationResultRow(BaseModel):
    """One row-aligned model prediction and its complete factor verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_index: Annotated[int, Field(ge=0, description="Dataset row position.")]
    simulation_id: Annotated[str, Field(description="Source simulation id.")]
    agent_turn_index: Annotated[
        int, Field(ge=0, description="Source delivered agent-turn index.")
    ]
    label_positive: Annotated[bool, Field(description="Human label is violation.")]
    judge_positive: Annotated[bool, Field(description="Judge outcome is FAIL.")]
    judge_outcome: Annotated[JudgeOutcome, Field(description="Judge outcome.")]
    judge_check: Annotated[
        NativenessFactorCheck, Field(description="Complete production factor check.")
    ]

    @model_validator(mode="after")
    def _outcomes_agree(self) -> "ValidationResultRow":
        if self.judge_outcome is not self.judge_check.outcome:
            raise ValueError("judge_outcome differs from judge_check.outcome")
        if self.judge_positive != (self.judge_outcome is JudgeOutcome.FAIL):
            raise ValueError("judge_positive differs from judge outcome")
        if self.judge_check.id != VALIDATION_FACTOR_ID:
            raise ValueError("validation result contains the wrong factor")
        return self


class ValidationResults(BaseModel):
    """Predictions aligned one-for-one with a validation dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-utterance-naturalness-results-v1"]
    language: Annotated[str, Field(description="ISO language code.")]
    split: Annotated[ValidationSplit, Field(description="Dev or test split.")]
    role: Annotated[ValidationRole, Field(description="Calibration or held-out role.")]
    evaluation_level: Literal["utterance"]
    factor_id: Literal["natural_word_choice"]
    factor_name: Literal["combined_utterance_naturalness"]
    dataset: Annotated[FileReference, Field(description="Input label artifact.")]
    prompt: Annotated[FileReference, Field(description="Selected prompt artifact.")]
    judge: Annotated[ValidationJudgeConfig, Field(description="Effective settings.")]
    n_rows: Annotated[int, Field(ge=0, description="Number of result rows.")]
    metrics: Annotated[
        BinaryMetrics, Field(description="Recomputed agreement metrics.")
    ]
    rows: Annotated[list[ValidationResultRow], Field(description="Row predictions.")]


class AcceptancePolicy(BaseModel):
    """Frozen gate used for each language/split report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_precision: Annotated[float, Field(ge=0, le=1, description="P floor.")]
    minimum_recall: Annotated[float, Field(ge=0, le=1, description="R floor.")]
    minimum_human_positives: Annotated[
        int, Field(ge=1, description="Minimum positive-label evidence.")
    ]
    minimum_kappa_when_both_classes_have_at_least_5: Annotated[
        float, Field(ge=-1, le=1, description="Kappa floor when both classes suffice.")
    ]


FIXED_VALIDATION_ACCEPTANCE_POLICY = AcceptancePolicy(
    minimum_precision=0.75,
    minimum_recall=0.75,
    minimum_human_positives=5,
    minimum_kappa_when_both_classes_have_at_least_5=0.60,
)


class ManifestJudgeConfig(BaseModel):
    """Model settings common to every validation split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Annotated[str, Field(description="LLM judge model id.")]
    model_args: Annotated[dict, Field(description="Arguments passed to the model.")]


class ManifestDatasetSummary(FileReference):
    """Manifest summary for one label dataset."""

    n_rows: Annotated[int, Field(ge=0, description="Dataset row count.")]
    n_positive: Annotated[int, Field(ge=0, description="Violation labels.")]
    n_negative: Annotated[int, Field(ge=0, description="Pass labels.")]


class ManifestResultSummary(FileReference):
    """Manifest summary for one result artifact."""

    max_concurrency: Annotated[int, Field(ge=1, description="Execution ceiling.")]
    metrics: Annotated[BinaryMetrics, Field(description="Agreement metrics.")]
    passes_acceptance_policy: Annotated[
        bool, Field(description="Whether frozen acceptance gates pass.")
    ]


class ManifestPromptSummary(FileReference):
    """Manifest summary for one selected language prompt."""

    rubric_version: Annotated[str, Field(description="Nativeness rubric version.")]
    prompt_version: Annotated[str, Field(description="Judge prompt version.")]


class ManifestSplitSummary(BaseModel):
    """Manifest references for one language/split pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Annotated[ValidationRole, Field(description="Semantic split role.")]
    dataset: Annotated[ManifestDatasetSummary, Field(description="Label summary.")]
    results: Annotated[ManifestResultSummary, Field(description="Result summary.")]


class ManifestLanguageSummary(BaseModel):
    """Selected prompt and shipped split summaries for a language."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: Annotated[ManifestPromptSummary, Field(description="Selected prompt.")]
    splits: Annotated[
        dict[ValidationSplit, ManifestSplitSummary],
        Field(description="Exactly the reviewer-facing split summaries."),
    ]


class ValidationManifest(BaseModel):
    """Root inventory and summaries for the complete validation package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-validation-manifest-v2"]
    benchmark: Literal["tau-multi"]
    validation: Literal["utterance_naturalness"]
    evaluation_level: Literal["utterance"]
    factor_id: Literal["natural_word_choice"]
    factor_name: Literal["combined_utterance_naturalness"]
    split_roles: Annotated[
        dict[ValidationSplit, ValidationRole], Field(description="Frozen split roles.")
    ]
    judge: Annotated[ManifestJudgeConfig, Field(description="Common judge settings.")]
    acceptance_policy: Annotated[
        AcceptancePolicy, Field(description="Frozen per-split acceptance gates.")
    ]
    files: Annotated[dict[str, str], Field(description="Artifact SHA-256 inventory.")]
    languages: Annotated[
        dict[str, ManifestLanguageSummary], Field(description="Per-language summaries.")
    ]


class ValidationPackageReport(BaseModel):
    """Offline validation outcome suitable for CLI JSON output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: Annotated[str, Field(description="Resolved package root.")]
    files_verified: Annotated[int, Field(ge=0, description="Hashed artifacts.")]
    rows_verified: Annotated[int, Field(ge=0, description="Aligned result rows.")]
    metrics: Annotated[
        dict[str, dict[str, BinaryMetrics]],
        Field(description="Recomputed metrics by language then split."),
    ]


def sha256_file(path: Path) -> str:
    """Return the byte-level SHA-256 for ``path``."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_binary_metrics(rows: list[ValidationResultRow]) -> BinaryMetrics:
    """Recompute the confusion matrix, P/R/F1, and binary Cohen's kappa."""
    usable = [row for row in rows if row.judge_outcome is not JudgeOutcome.ERROR]
    tp = sum(row.label_positive and row.judge_positive for row in usable)
    fn = sum(row.label_positive and not row.judge_positive for row in usable)
    fp = sum(not row.label_positive and row.judge_positive for row in usable)
    tn = sum(not row.label_positive and not row.judge_positive for row in usable)
    errors = len(rows) - len(usable)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    n = tp + fn + fp + tn
    if not n:
        kappa = None
    else:
        observed = (tp + tn) / n
        expected = (((tp + fn) * (tp + fp)) + ((fp + tn) * (fn + tn))) / (n * n)
        kappa = (observed - expected) / (1 - expected) if expected != 1 else None
    return BinaryMetrics(
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        n_errors=errors,
        precision=precision,
        recall=recall,
        f1=f1,
        kappa=kappa,
    )


def passes_acceptance_policy(metrics: BinaryMetrics, policy: AcceptancePolicy) -> bool:
    """Apply the frozen evidence and performance gates to one split."""
    positives = metrics.tp + metrics.fn
    negatives = metrics.fp + metrics.tn
    if metrics.n_errors:
        return False
    if positives < policy.minimum_human_positives:
        return False
    if metrics.precision is None or metrics.precision < policy.minimum_precision:
        return False
    if metrics.recall is None or metrics.recall < policy.minimum_recall:
        return False
    if positives >= 5 and negatives >= 5:
        return bool(
            metrics.kappa is not None
            and metrics.kappa >= policy.minimum_kappa_when_both_classes_have_at_least_5
        )
    return True


def _resolve_inside(parent: Path, relative: str) -> Path:
    path = (parent / relative).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError(f"artifact path escapes package: {relative!r}")
    return path


def _resolve_file_reference(
    package_root: Path, containing_file: Path, reference: FileReference
) -> Path:
    """Resolve one file-local reference while keeping it inside the package."""
    path = (containing_file.parent / reference.path).resolve()
    if not path.is_relative_to(package_root.resolve()):
        raise ValueError(f"artifact reference escapes package: {reference.path!r}")
    return path


def _assert_metrics_equal(observed: BinaryMetrics, expected: BinaryMetrics) -> None:
    for field in ("tp", "fn", "fp", "tn", "n_errors"):
        if getattr(observed, field) != getattr(expected, field):
            raise ValueError(f"metric {field} does not reproduce")
    for field in ("precision", "recall", "f1", "kappa"):
        left, right = getattr(observed, field), getattr(expected, field)
        if left is None or right is None:
            if left is not right:
                raise ValueError(f"metric {field} does not reproduce")
        elif abs(left - right) > 1e-12:
            raise ValueError(f"metric {field} does not reproduce: {left} != {right}")


def runtime_combined_factor(language: str) -> NativenessFactorConfig:
    """Return the sole frozen combined factor, rejecting any wiring drift."""
    factors = [
        factor
        for factor in judge_factors_for(language)
        if factor.id == VALIDATION_FACTOR_ID and factor.enabled
    ]
    if len(factors) != 1 or factors[0].params is None:
        raise ValueError(f"{language}: expected one enabled {VALIDATION_FACTOR_ID}")
    factor = factors[0]
    if (
        factor.type != "judge"
        or factor.evaluation_level != "utterance"
        or factor.aggregation != "any"
        or factor.precheck_type is not None
    ):
        raise ValueError(f"{language}: combined naturalness factor wiring drifted")
    return factor


def runtime_validation_prompt(language: str) -> ValidationPrompt:
    """Project the production factor into the checked-in prompt contract."""
    factor = runtime_combined_factor(language)
    assert factor.params is not None
    rubric = factor.params
    return ValidationPrompt(
        schema_version="tau-multi-utterance-naturalness-prompt-v1",
        language=language,
        factor_id=VALIDATION_FACTOR_ID,
        factor_name=VALIDATION_FACTOR_NAME,
        prompt_version=NATIVENESS_JUDGE_PROMPT_VERSION,
        rubric_version=COMBINED_NATURALNESS_RUBRIC_VERSION,
        question=rubric.language_question or rubric.question,
        opportunity=rubric.opportunity,
        native_does=rubric.language_allowed,
        ai_likely_does=rubric.language_violation,
    )


def _format_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_validation_readme(manifest: ValidationManifest) -> str:
    """Render the human-facing metrics table from the typed manifest."""
    reasoning_effort = manifest.judge.model_args.get(
        "reasoning_effort", "provider default"
    )
    lines = [
        "# Utterance naturalness validation",
        "",
        "Held-out annotation datasets, frozen judge prompts, per-utterance predictions, and aggregate validation metrics for the tau-multi paper.",
        "",
        "- `test` is the final held-out validation split for the frozen paper judge; only final test rows are included.",
        "- All rows are agent utterances. A violation means the utterance contains at least one issue covered by that language's combined rubric.",
        f"- `prompt.json` is the exact rubric used with `{manifest.judge.model}` at `{reasoning_effort}` reasoning effort. The effective concurrency is recorded per result.",
        "- `dataset.json` contains final annotations only. `results.json` contains the frozen predictions and recomputed metrics.",
        "- `manifest.json` records the acceptance policy and SHA-256 hashes.",
        "",
        "## Metrics",
        "",
        "| Language | Split | Rows (+/−) | Precision | Recall | F1 | κ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for language in VALIDATION_LANGUAGES:
        for split in (ValidationSplit.TEST,):
            summary = manifest.languages[language].splits[split]
            dataset = summary.dataset
            metrics = summary.results.metrics
            lines.append(
                f"| {language} | {split.value} | {dataset.n_rows} "
                f"({dataset.n_positive}/{dataset.n_negative}) | "
                f"{_format_metric(metrics.precision)} | "
                f"{_format_metric(metrics.recall)} | "
                f"{_format_metric(metrics.f1)} | "
                f"{_format_metric(metrics.kappa)} |"
            )
    lines.extend(
        [
            "",
            "The stable runtime factor id is `natural_word_choice`; it names the combined utterance-naturalness target. Every language rubric combines natural word choice and translationese. Hindi and Portuguese also cover verb and core-grammar errors; Spanish additionally covers those errors, Madrid regional consistency, and written diacritics.",
            "",
            "## Recompute metrics",
            "",
            "The confusion matrix is computed directly from each result row's `label_positive` and `judge_positive` fields. Precision, recall, F1, and Cohen's κ in `results.json` and `manifest.json` are derived from those counts.",
        ]
    )
    return "\n".join(lines) + "\n"


def validate_validation_package(root: Path) -> ValidationPackageReport:
    """Hash, parse, align, and recompute a frozen validation package offline."""
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = ValidationManifest.model_validate_json(manifest_path.read_text())
    if tuple(sorted(manifest.languages)) != tuple(sorted(VALIDATION_LANGUAGES)):
        raise ValueError("validation manifest must contain exactly es/hi/ko/pt/zh")
    if manifest.split_roles != {
        ValidationSplit.TEST: ValidationRole.HELD_OUT,
    }:
        raise ValueError("validation split roles drifted")
    if manifest.judge.model != DEFAULT_TAU_MULTI_NATURALNESS_JUDGE or (
        manifest.judge.model_args != DEFAULT_TAU_MULTI_NATURALNESS_JUDGE_ARGS
    ):
        raise ValueError("validation manifest judge settings drifted")
    if manifest.acceptance_policy != FIXED_VALIDATION_ACCEPTANCE_POLICY:
        raise ValueError("validation acceptance policy drifted")

    expected_files = {
        f"{language}/{name}"
        for language in VALIDATION_LANGUAGES
        for name in (
            "prompt.json",
            "test/dataset.json",
            "test/results.json",
        )
    }
    if set(manifest.files) != expected_files:
        raise ValueError("validation manifest file inventory is not exact")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".work" not in path.relative_to(root).parts
    }
    expected_disk_files = expected_files | {"README.md", "manifest.json"}
    if actual_files != expected_disk_files:
        raise ValueError(
            "validation package disk inventory is not exact: "
            f"missing={sorted(expected_disk_files - actual_files)} "
            f"unexpected={sorted(actual_files - expected_disk_files)}"
        )
    for relative, digest in manifest.files.items():
        path = _resolve_inside(root, relative)
        if sha256_file(path) != digest:
            raise ValueError(f"validation artifact hash mismatch: {relative}")

    reports: dict[str, dict[str, BinaryMetrics]] = {}
    total_rows = 0
    for language in VALIDATION_LANGUAGES:
        language_summary = manifest.languages[language]
        prompt_path = _resolve_inside(root, language_summary.prompt.path)
        expected_prompt_path = (root / language / "prompt.json").resolve()
        if prompt_path != expected_prompt_path:
            raise ValueError(f"{language}: manifest prompt path is not canonical")
        prompt = ValidationPrompt.model_validate_json(prompt_path.read_text())
        if prompt != runtime_validation_prompt(language):
            raise ValueError(f"{language}: packaged prompt differs from runtime rubric")
        if language_summary.prompt.sha256 != sha256_file(prompt_path):
            raise ValueError(f"{language}: prompt summary hash mismatch")
        if (
            manifest.files[language_summary.prompt.path]
            != language_summary.prompt.sha256
        ):
            raise ValueError(f"{language}: prompt summary differs from inventory")
        if (
            language_summary.prompt.prompt_version != prompt.prompt_version
            or language_summary.prompt.rubric_version != prompt.rubric_version
        ):
            raise ValueError(f"{language}: prompt summary versions mismatch")

        reports[language] = {}
        if set(language_summary.splits) != {ValidationSplit.TEST}:
            raise ValueError(f"{language}: manifest needs exactly the held-out test")
        for split in (ValidationSplit.TEST,):
            summary = language_summary.splits[split]
            dataset_path = _resolve_inside(root, summary.dataset.path)
            results_path = _resolve_inside(root, summary.results.path)
            expected_split_root = (root / language / split.value).resolve()
            if dataset_path != expected_split_root / "dataset.json":
                raise ValueError(
                    f"{language}/{split.value}: manifest dataset path is not canonical"
                )
            if results_path != expected_split_root / "results.json":
                raise ValueError(
                    f"{language}/{split.value}: manifest results path is not canonical"
                )
            dataset = ValidationDataset.model_validate_json(dataset_path.read_text())
            results = ValidationResults.model_validate_json(results_path.read_text())
            if dataset.language != language or results.language != language:
                raise ValueError(f"{language}/{split.value}: language mismatch")
            if dataset.split is not split or results.split is not split:
                raise ValueError(f"{language}/{split.value}: split mismatch")
            if dataset.role is not summary.role or results.role is not summary.role:
                raise ValueError(f"{language}/{split.value}: role mismatch")
            if summary.role is not manifest.split_roles[split]:
                raise ValueError(f"{language}/{split.value}: split role drifted")
            if results.n_rows != len(results.rows) or results.n_rows != dataset.n_rows:
                raise ValueError(f"{language}/{split.value}: row-count mismatch")
            if results.dataset.sha256 != sha256_file(dataset_path):
                raise ValueError(f"{language}/{split.value}: dataset hash mismatch")
            if results.prompt.sha256 != sha256_file(prompt_path):
                raise ValueError(f"{language}/{split.value}: prompt hash mismatch")
            if summary.dataset.sha256 != sha256_file(dataset_path):
                raise ValueError(
                    f"{language}/{split.value}: dataset summary hash mismatch"
                )
            if summary.results.sha256 != sha256_file(results_path):
                raise ValueError(
                    f"{language}/{split.value}: result summary hash mismatch"
                )
            if manifest.files[summary.dataset.path] != summary.dataset.sha256:
                raise ValueError(
                    f"{language}/{split.value}: dataset summary differs from inventory"
                )
            if manifest.files[summary.results.path] != summary.results.sha256:
                raise ValueError(
                    f"{language}/{split.value}: result summary differs from inventory"
                )
            if (
                _resolve_file_reference(root, results_path, results.dataset)
                != dataset_path
            ):
                raise ValueError(
                    f"{language}/{split.value}: results dataset path mismatch"
                )
            if (
                _resolve_file_reference(root, results_path, results.prompt)
                != prompt_path
            ):
                raise ValueError(
                    f"{language}/{split.value}: results prompt path mismatch"
                )
            if results.judge.model != DEFAULT_TAU_MULTI_NATURALNESS_JUDGE or (
                results.judge.model_args != DEFAULT_TAU_MULTI_NATURALNESS_JUDGE_ARGS
            ):
                raise ValueError(f"{language}/{split.value}: judge settings drifted")
            if (
                results.judge.prompt_version != prompt.prompt_version
                or results.judge.rubric_version != prompt.rubric_version
            ):
                raise ValueError(f"{language}/{split.value}: judge versions mismatch")
            if summary.results.max_concurrency != results.judge.max_concurrency:
                raise ValueError(
                    f"{language}/{split.value}: concurrency summary drifted"
                )

            for label, prediction in zip(dataset.rows, results.rows, strict=True):
                key = (label.simulation_id, label.agent_turn_index)
                if key != (prediction.simulation_id, prediction.agent_turn_index):
                    raise ValueError(
                        f"{language}/{split.value}: result alignment drift"
                    )
                if label.row_index != prediction.row_index:
                    raise ValueError(f"{language}/{split.value}: row index drift")
                if prediction.label_positive != (
                    label.label is ValidationLabel.VIOLATION
                ):
                    raise ValueError(f"{language}/{split.value}: label drift")
                if prediction.judge_check.evaluation_level != "utterance":
                    raise ValueError(f"{language}/{split.value}: non-utterance check")
                units = prediction.judge_check.unit_results
                if prediction.judge_outcome is not JudgeOutcome.ERROR and (
                    len(units) != 1 or units[0].unit_index != label.agent_turn_index
                ):
                    raise ValueError(f"{language}/{split.value}: unit result drift")
            metrics = compute_binary_metrics(results.rows)
            _assert_metrics_equal(results.metrics, metrics)
            _assert_metrics_equal(summary.results.metrics, metrics)
            if summary.results.passes_acceptance_policy != passes_acceptance_policy(
                metrics, FIXED_VALIDATION_ACCEPTANCE_POLICY
            ):
                raise ValueError(f"{language}/{split.value}: acceptance gate drift")
            if (
                summary.dataset.n_rows != dataset.n_rows
                or summary.dataset.n_positive != dataset.n_positive
                or summary.dataset.n_negative != dataset.n_negative
            ):
                raise ValueError(f"{language}/{split.value}: dataset summary drift")
            reports[language][split.value] = metrics
            total_rows += dataset.n_rows

    readme_path = root / "README.md"
    if not readme_path.is_file() or readme_path.read_text() != render_validation_readme(
        manifest
    ):
        raise ValueError("validation README is not generated from the manifest")
    return ValidationPackageReport(
        root=str(root),
        files_verified=len(manifest.files),
        rows_verified=total_rows,
        metrics=reports,
    )
