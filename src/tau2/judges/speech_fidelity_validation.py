# Copyright Sierra
"""Typed, offline verification for frozen Speech-fidelity validation evidence.

The paper validation unit is one delivered agent utterance.  The archived model
prediction is the presence of any fidelity-axis finding produced by the v5
delivery judge *after* applying the paper's deterministic final-second filter.
This module deliberately validates that historical contract rather than the
current runtime delivery prompt, which may continue to evolve.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SPEECH_FIDELITY_LANGUAGES = ("es", "hi", "ko", "pt", "zh")
SPEECH_FIDELITY_MEASURE_ID = "speech_fidelity"
FROZEN_DELIVERY_JUDGE_MODEL = "gemini/gemini-3.1-pro-preview"
FROZEN_DELIVERY_JUDGE_ARGS = {
    "temperature": 0.0,
    "max_tokens": 8192,
    "timeout": 120,
}
FROZEN_DELIVERY_PROMPT_VERSION = "v5"
FROZEN_DELIVERY_PROMPT_REVISION = "e1ca72ea1e5b399ffcf3e88f62a746e2ad19591d"
FROZEN_DELIVERY_PROMPT_SHA256 = (
    "6c9f075c6be0e43443b829d53312a4863087ac7b08d9805356d5923d7c82546c"
)
FROZEN_FIDELITY_FILTER_VERSION = "v1"
FROZEN_FIDELITY_FINAL_WINDOW_SECONDS = 1.0
FROZEN_COHORT_POLICY = (
    "One fixed curated utterance cohort per language. Direct human labels are "
    "aligned to their delivered utterance, while reviewed no-finding calls supply "
    "projected negative utterances. All predictions apply final-second filter v1 "
    "uniformly to stored finding spans and clip durations."
)

# The exact fixed cohorts. Keeping the confusion
# matrices in code makes accidental cohort or label drift fail loudly.
EXPECTED_CONFUSION: dict[str, tuple[int, int, int, int]] = {
    # language: (TP, TN, FP, FN)
    "es": (23, 32, 1, 4),
    "hi": (19, 38, 1, 2),
    "ko": (22, 27, 4, 7),
    "pt": (20, 34, 0, 6),
    "zh": (18, 26, 6, 2),
}

EXPECTED_STORED_REFERENCE_ALIGNMENT: dict[str, dict[str, int]] = {
    "es": {"exact": 27, "stored_prefix": 33},
    "hi": {"exact": 38, "stored_prefix": 22},
    "ko": {"exact": 53, "stored_prefix": 7},
    "pt": {"exact": 23, "stored_prefix": 37},
    "zh": {"exact": 46, "stored_prefix": 6},
}

_TIME_TOKEN_V1 = re.compile(r"\d+:\d+(?:\.\d+)?|\d+(?:\.\d+)?")


def _time_range_bounds_seconds_v1(
    value: Optional[str],
) -> Optional[tuple[float, float]]:
    """Parse the loose timestamp formats used by frozen final-window filter v1."""
    points: list[float] = []
    for token in _TIME_TOKEN_V1.findall(value or ""):
        if ":" in token:
            minutes, seconds = token.split(":", 1)
            points.append(60 * float(minutes) + float(seconds))
        else:
            points.append(float(token))
    if not points:
        return None
    return points[0], points[-1]


def _is_in_final_utterance_window_v1(
    time_range: Optional[str],
    clip_duration_seconds: float,
    *,
    window_seconds: float = FROZEN_FIDELITY_FINAL_WINDOW_SECONDS,
) -> bool:
    """Apply the exact deterministic final-window rule used by this package."""
    bounds = _time_range_bounds_seconds_v1(time_range)
    if bounds is None:
        return False
    start, end = bounds
    return (
        end >= max(0.0, clip_duration_seconds - window_seconds)
        and start <= clip_duration_seconds + window_seconds
    )


class ValidationLabel(str, Enum):
    """Final human or judge binary label for one utterance."""

    PASS = "pass"
    VIOLATION = "violation"


class ValidationSourceSet(str, Enum):
    """Neutral public designation for the frozen human cohort."""

    FIXED_CURATED = "fixed_curated"


class ValidationLabelOrigin(str, Enum):
    """How an utterance-level human label was obtained."""

    DIRECT_UTTERANCE = "direct_utterance"
    PROJECTED_CALL_PASS = "projected_call_pass"


class StoredReferenceAlignment(str, Enum):
    """Relation between the human reference and stored judge metadata preview."""

    EXACT = "exact"
    STORED_PREFIX = "stored_prefix"


def _normalize_transcript(value: str) -> str:
    """Normalize only Unicode whitespace for cross-artifact alignment."""
    return " ".join(value.split())


def classify_stored_reference_alignment(
    reference_transcript: str, stored_reference_preview: str
) -> StoredReferenceAlignment:
    """Classify the stored v5 reference preview against the human reference."""
    human = _normalize_transcript(reference_transcript)
    preview = _normalize_transcript(stored_reference_preview)
    if human == preview:
        return StoredReferenceAlignment.EXACT
    if human.startswith(preview) and len(preview) < len(human):
        return StoredReferenceAlignment.STORED_PREFIX
    raise ValueError(
        "stored judge reference preview must equal or prefix the human reference "
        "after whitespace normalization"
    )


class FileReference(BaseModel):
    """Path relative to the referring artifact and its byte digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Annotated[str, Field(description="Relative artifact path.")]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256.")]


class AcceptancePolicy(BaseModel):
    """Frozen judge-validity gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_precision: Annotated[float, Field(ge=0, le=1, description="P floor.")]
    minimum_recall: Annotated[float, Field(ge=0, le=1, description="R floor.")]
    minimum_human_positives: Annotated[
        int, Field(ge=1, description="Minimum positive human labels.")
    ]
    minimum_kappa_when_both_classes_have_at_least_5: Annotated[
        float, Field(ge=-1, le=1, description="Conditional Cohen's kappa floor.")
    ]


SPEECH_FIDELITY_ACCEPTANCE_POLICY = AcceptancePolicy(
    minimum_precision=0.75,
    minimum_recall=0.75,
    minimum_human_positives=5,
    minimum_kappa_when_both_classes_have_at_least_5=0.60,
)


class FrozenRubricFactor(BaseModel):
    """One language-specific rubric included in the historical v5 prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    factor_id: Annotated[str, Field(description="Runtime delivery factor id.")]
    severity: Annotated[int, Field(ge=1, le=3, description="Prompted severity.")]
    listen_for: Annotated[str, Field(description="Exact language-pack instruction.")]


class FrozenLanguageRubric(BaseModel):
    """Language-specific prompt material used alongside the common v5 prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    display_name: Annotated[str, Field(description="Prompted language name.")]
    source: Literal["pack"]
    factors: Annotated[
        list[FrozenRubricFactor], Field(description="Factors rendered into the prompt.")
    ]


class SpeechFidelityPrompt(BaseModel):
    """Exact historical v5 delivery-judge prompt and language rubric material."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-speech-fidelity-prompt-v1"]
    prompt_version: Literal["v5"]
    source_revision: Literal["e1ca72ea1e5b399ffcf3e88f62a746e2ad19591d"]
    model: Annotated[str, Field(description="Judge model id.")]
    model_args: Annotated[dict, Field(description="Effective judge arguments.")]
    system_prompt: Annotated[str, Field(description="Exact system prompt.")]
    user_prompt_template: Annotated[str, Field(description="Exact user template.")]
    output_shape: Annotated[str, Field(description="Exact requested JSON contract.")]
    language_rubrics: Annotated[
        dict[str, FrozenLanguageRubric],
        Field(description="Exact pack rubric content used by language."),
    ]

    @model_validator(mode="after")
    def _historical_configuration_is_exact(self) -> "SpeechFidelityPrompt":
        if self.model != FROZEN_DELIVERY_JUDGE_MODEL:
            raise ValueError("frozen Speech-fidelity judge model drifted")
        if self.model_args != FROZEN_DELIVERY_JUDGE_ARGS:
            raise ValueError("frozen Speech-fidelity judge arguments drifted")
        if set(self.language_rubrics) != set(SPEECH_FIDELITY_LANGUAGES):
            raise ValueError("prompt needs exactly the five paper languages")
        return self


class SpeechFidelityDatasetRow(BaseModel):
    """One final human label at a stable delivered-utterance key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_index: Annotated[int, Field(ge=0, description="Zero-based package row.")]
    row_id: Annotated[
        str, Field(pattern=r"^[0-9a-f]{20}$", description="Stable utterance row id.")
    ]
    simulation_id: Annotated[str, Field(description="Source simulation id.")]
    task_id: Annotated[str, Field(description="Localized task id.")]
    label_origin: Annotated[
        ValidationLabelOrigin, Field(description="Utterance-label derivation.")
    ]
    clip_id: Annotated[
        Optional[str], Field(description="Human packet clip id, when available.")
    ] = None
    source_utterance_idx: Annotated[
        int, Field(ge=0, description="Delivery result utterance index.")
    ]
    agent_turn_index: Annotated[
        int, Field(ge=0, description="Zero-based audible agent-turn index.")
    ]
    agent_turn_id: Annotated[
        Optional[str], Field(description="Human-readable turn id, when recorded.")
    ] = None
    domain: Annotated[str, Field(description="Benchmark domain.")]
    provider: Annotated[str, Field(description="Voice-agent experiment provider.")]
    locale: Annotated[str, Field(description="Prompted locale.")]
    preceding_customer_text: Annotated[
        Optional[str], Field(description="Immediately preceding customer text.")
    ] = None
    reference_transcript: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Human-aligned transcript after removing unsynthesized control "
                "tokens from the delivered reference."
            ),
        ),
    ]
    source_interrupted: Annotated[
        bool, Field(description="v5 was_interrupted prompt value.")
    ]
    human_label: Annotated[ValidationLabel, Field(description="Final human label.")]

    @model_validator(mode="after")
    def _label_origin_is_coherent(self) -> "SpeechFidelityDatasetRow":
        if "###STOP###" in self.reference_transcript:
            raise ValueError(
                "reference transcript contains an unsynthesized control token"
            )
        if self.label_origin is ValidationLabelOrigin.PROJECTED_CALL_PASS:
            if self.human_label is not ValidationLabel.PASS:
                raise ValueError("a projected call-pass label must be pass")
        return self


class SpeechFidelityDataset(BaseModel):
    """One language's fixed utterance-level human-validation cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-speech-fidelity-dataset-v3"]
    language: Annotated[str, Field(description="ISO language code.")]
    split: Literal["test"]
    role: Literal["fixed_curated_validation"]
    source_set: ValidationSourceSet
    evaluation_level: Literal["utterance"]
    grouping_unit: Literal["language+simulation_id+agent_turn_index"]
    measure_id: Literal["speech_fidelity"]
    runtime_factor_ids: Annotated[
        list[str], Field(description="Runtime ids retained only as provenance.")
    ]
    n_rows: Annotated[int, Field(ge=0, description="Row count.")]
    n_positive: Annotated[int, Field(ge=0, description="Human violations.")]
    n_negative: Annotated[int, Field(ge=0, description="Human passes.")]
    n_calls: Annotated[int, Field(ge=0, description="Unique simulations.")]
    domain_counts: Annotated[dict[str, int], Field(description="Rows by domain.")]
    provider_counts: Annotated[dict[str, int], Field(description="Rows by provider.")]
    label_origin_counts: Annotated[
        dict[str, int], Field(description="Rows by label derivation.")
    ]
    rows: Annotated[list[SpeechFidelityDatasetRow], Field(description="Final labels.")]

    @model_validator(mode="after")
    def _summary_matches_rows(self) -> "SpeechFidelityDataset":
        if self.language not in SPEECH_FIDELITY_LANGUAGES:
            raise ValueError(f"unsupported Speech-fidelity language: {self.language}")
        if self.runtime_factor_ids != ["fidelity"]:
            raise ValueError("speech_fidelity must derive from fidelity-axis findings")
        if [row.row_index for row in self.rows] != list(range(len(self.rows))):
            raise ValueError("dataset row_index values must be contiguous")
        keys = [
            (self.language, row.simulation_id, row.agent_turn_index)
            for row in self.rows
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate delivered-utterance dataset key")
        if any(not row.row_id for row in self.rows):
            raise ValueError("dataset row_id cannot be empty")
        positives = sum(
            row.human_label is ValidationLabel.VIOLATION for row in self.rows
        )
        expected_scalars = {
            "n_rows": len(self.rows),
            "n_positive": positives,
            "n_negative": len(self.rows) - positives,
            "n_calls": len({row.simulation_id for row in self.rows}),
        }
        if {name: getattr(self, name) for name in expected_scalars} != expected_scalars:
            raise ValueError("dataset scalar summaries do not match rows")
        expected_counters = {
            "domain_counts": Counter(row.domain for row in self.rows),
            "provider_counts": Counter(row.provider for row in self.rows),
            "label_origin_counts": Counter(row.label_origin.value for row in self.rows),
        }
        for field_name, counter in expected_counters.items():
            if getattr(self, field_name) != dict(sorted(counter.items())):
                raise ValueError(f"{field_name} does not match dataset rows")
        return self


class SpeechFidelityFinding(BaseModel):
    """One stored v5 fidelity-axis finding and deterministic filter decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    finding_index: Annotated[int, Field(ge=0, description="Original finding index.")]
    axis: Literal["fidelity"]
    category: Annotated[str, Field(description="Judge finding category.")]
    time_range: Annotated[
        Optional[str], Field(description="Approximate span emitted by the judge.")
    ] = None
    issue: Annotated[
        Optional[str], Field(description="Judge explanation, when emitted.")
    ] = None
    severity: Annotated[int, Field(ge=1, le=3, description="Finding severity.")]
    confidence: Annotated[
        Optional[float], Field(ge=0, le=1, description="Finding confidence.")
    ] = None
    excluded_by_final_second_filter: Annotated[
        bool, Field(description="Whether deterministic filter v1 removes the finding.")
    ]


class BinaryMetrics(BaseModel):
    """Binary confusion counts and agreement metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tp: Annotated[int, Field(ge=0, description="True positives.")]
    tn: Annotated[int, Field(ge=0, description="True negatives.")]
    fp: Annotated[int, Field(ge=0, description="False positives.")]
    fn: Annotated[int, Field(ge=0, description="False negatives.")]
    precision: Annotated[Optional[float], Field(description="TP / (TP + FP).")]
    recall: Annotated[Optional[float], Field(description="TP / (TP + FN).")]
    f1: Annotated[Optional[float], Field(description="Harmonic mean of P and R.")]
    kappa: Annotated[Optional[float], Field(description="Binary Cohen's kappa.")]

    @property
    def n(self) -> int:
        """Number of scored observations."""
        return self.tp + self.tn + self.fp + self.fn


class SpeechFidelityResultRow(BaseModel):
    """One row-aligned v5 prediction before and after filter v1."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_index: Annotated[int, Field(ge=0, description="Dataset row position.")]
    row_id: Annotated[str, Field(description="Stable dataset row id.")]
    simulation_id: Annotated[str, Field(description="Source simulation id.")]
    source_utterance_idx: Annotated[
        int, Field(ge=0, description="Delivery result utterance index.")
    ]
    agent_turn_index: Annotated[
        int, Field(ge=0, description="Audible agent-turn index.")
    ]
    label_positive: Annotated[
        bool, Field(description="Final human label is violation.")
    ]
    judge_raw_positive: Annotated[
        bool, Field(description="v5 emitted at least one fidelity finding.")
    ]
    judge_positive: Annotated[
        bool,
        Field(description="At least one fidelity finding remains after filter v1."),
    ]
    clip_duration_seconds: Annotated[
        float, Field(gt=0, description="Exact judged audio-clip duration.")
    ]
    stored_reference_preview: Annotated[
        str,
        Field(
            min_length=1,
            max_length=200,
            description="Exact reference preview retained in the stored v5 result.",
        ),
    ]
    stored_reference_alignment: StoredReferenceAlignment
    judge_summary: Annotated[
        Optional[str], Field(description="Stored v5 overall explanation, when emitted.")
    ] = None
    findings: Annotated[
        list[SpeechFidelityFinding], Field(description="All fidelity findings.")
    ]

    @model_validator(mode="after")
    def _prediction_matches_findings(self) -> "SpeechFidelityResultRow":
        if [finding.finding_index for finding in self.findings] != list(
            range(len(self.findings))
        ):
            raise ValueError("finding indexes must be contiguous")
        if self.judge_raw_positive != bool(self.findings):
            raise ValueError("raw prediction does not match fidelity findings")
        retained = [
            finding
            for finding in self.findings
            if not finding.excluded_by_final_second_filter
        ]
        if self.judge_positive != bool(retained):
            raise ValueError("filtered prediction does not match retained findings")
        for finding in self.findings:
            expected = _is_in_final_utterance_window_v1(
                finding.time_range,
                self.clip_duration_seconds,
                window_seconds=FROZEN_FIDELITY_FINAL_WINDOW_SECONDS,
            )
            if finding.excluded_by_final_second_filter != expected:
                raise ValueError(
                    "stored final-second decision differs from deterministic filter v1"
                )
        return self


class JudgeConfig(BaseModel):
    """Historical v5 judge and deterministic post-filter settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Annotated[str, Field(description="Judge model id.")]
    model_args: Annotated[dict, Field(description="Effective model arguments.")]
    prompt_version: Literal["v5"]
    axis: Literal["fidelity"]
    threshold: Literal["any_retained_finding"]
    filter_version: Literal["v1"]
    final_window_seconds: Literal[1.0]

    @model_validator(mode="after")
    def _settings_are_frozen(self) -> "JudgeConfig":
        if self.model != FROZEN_DELIVERY_JUDGE_MODEL:
            raise ValueError("result judge model drifted")
        if self.model_args != FROZEN_DELIVERY_JUDGE_ARGS:
            raise ValueError("result judge arguments drifted")
        return self


class SpeechFidelityResults(BaseModel):
    """Stored v5 predictions aligned to one language dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-speech-fidelity-results-v2"]
    language: Annotated[str, Field(description="ISO language code.")]
    split: Literal["test"]
    role: Literal["fixed_curated_validation"]
    evaluation_level: Literal["utterance"]
    measure_id: Literal["speech_fidelity"]
    runtime_factor_ids: Annotated[
        list[str], Field(description="Runtime ids retained only as provenance.")
    ]
    dataset: Annotated[FileReference, Field(description="Aligned human dataset.")]
    prompt: Annotated[FileReference, Field(description="Exact v5 prompt artifact.")]
    judge: Annotated[JudgeConfig, Field(description="Judge and filter settings.")]
    n_rows: Annotated[int, Field(ge=0, description="Result row count.")]
    stored_reference_alignment_counts: Annotated[
        dict[str, int], Field(description="Rows by stored-reference alignment class.")
    ]
    metrics: Annotated[BinaryMetrics, Field(description="Stored aggregate metrics.")]
    rows: Annotated[list[SpeechFidelityResultRow], Field(description="Predictions.")]

    @model_validator(mode="after")
    def _summary_matches_rows(self) -> "SpeechFidelityResults":
        expected = dict(
            sorted(
                Counter(
                    row.stored_reference_alignment.value for row in self.rows
                ).items()
            )
        )
        if self.stored_reference_alignment_counts != expected:
            raise ValueError(
                "stored_reference_alignment_counts does not match result rows"
            )
        return self


class DatasetSummary(FileReference):
    """Manifest dataset reference and class counts."""

    n_rows: Annotated[int, Field(ge=0, description="Dataset row count.")]
    n_positive: Annotated[int, Field(ge=0, description="Human violations.")]
    n_negative: Annotated[int, Field(ge=0, description="Human passes.")]


class ResultSummary(FileReference):
    """Manifest result reference and recomputed metrics."""

    metrics: Annotated[BinaryMetrics, Field(description="Agreement metrics.")]
    stored_reference_alignment_counts: Annotated[
        dict[str, int], Field(description="Rows by stored-reference alignment class.")
    ]
    passes_acceptance_policy: Annotated[
        bool, Field(description="Whether the language cohort passes the frozen gate.")
    ]


class LanguageSummary(BaseModel):
    """Files and summaries for one language's fixed test cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: Annotated[DatasetSummary, Field(description="Human label artifact.")]
    results: Annotated[ResultSummary, Field(description="Judge result artifact.")]


class SpeechFidelityManifest(BaseModel):
    """Inventory and frozen summaries for the complete validation package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-speech-fidelity-manifest-v3"]
    benchmark: Literal["tau-multi"]
    validation: Literal["speech_fidelity"]
    evaluation_level: Literal["utterance"]
    measure_id: Literal["speech_fidelity"]
    split: Literal["test"]
    role: Literal["fixed_curated_validation"]
    cohort_policy: Annotated[str, Field(description="How fixed cohorts were formed.")]
    judge: Annotated[JudgeConfig, Field(description="Historical judge settings.")]
    acceptance_policy: Annotated[
        AcceptancePolicy, Field(description="Frozen validity gate.")
    ]
    prompt: Annotated[FileReference, Field(description="Exact v5 prompt artifact.")]
    files: Annotated[dict[str, str], Field(description="SHA-256 file inventory.")]
    languages: Annotated[
        dict[str, LanguageSummary], Field(description="Per-language summaries.")
    ]


class SpeechFidelityPackageReport(BaseModel):
    """Offline verification outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: Annotated[str, Field(description="Resolved package root.")]
    files_verified: Annotated[int, Field(ge=0, description="Hashed artifacts.")]
    rows_verified: Annotated[int, Field(ge=0, description="Aligned utterances.")]
    metrics: Annotated[
        dict[str, BinaryMetrics], Field(description="Recomputed metrics by language.")
    ]


def sha256_file(path: Path) -> str:
    """Return a file's byte-level SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_binary_metrics(rows: list[SpeechFidelityResultRow]) -> BinaryMetrics:
    """Recompute the confusion matrix, P/R/F1, and binary Cohen's kappa."""
    tp = sum(row.label_positive and row.judge_positive for row in rows)
    tn = sum(not row.label_positive and not row.judge_positive for row in rows)
    fp = sum(not row.label_positive and row.judge_positive for row in rows)
    fn = sum(row.label_positive and not row.judge_positive for row in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    n = tp + tn + fp + fn
    if not n:
        kappa = None
    else:
        observed = (tp + tn) / n
        expected = (((tp + fn) * (tp + fp)) + ((fp + tn) * (fn + tn))) / (n * n)
        kappa = (observed - expected) / (1 - expected) if expected != 1 else None
    return BinaryMetrics(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        kappa=kappa,
    )


def passes_acceptance_policy(metrics: BinaryMetrics, policy: AcceptancePolicy) -> bool:
    """Apply the paper's frozen judge-validity gate."""
    positives = metrics.tp + metrics.fn
    negatives = metrics.tn + metrics.fp
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


def _resolve_reference(
    package_root: Path, referring_file: Path, reference: FileReference
) -> Path:
    target = (referring_file.parent / reference.path).resolve()
    if not target.is_relative_to(package_root.resolve()):
        raise ValueError(f"artifact reference escapes package: {reference.path!r}")
    if sha256_file(target) != reference.sha256:
        raise ValueError(f"artifact reference hash mismatch: {reference.path}")
    return target


def verify_speech_fidelity_validation_package(
    root: Path,
) -> SpeechFidelityPackageReport:
    """Hash, parse, align, and recompute the frozen package without source runs."""
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = SpeechFidelityManifest.model_validate_json(manifest_path.read_text())
    if manifest.judge != JudgeConfig(
        model=FROZEN_DELIVERY_JUDGE_MODEL,
        model_args=FROZEN_DELIVERY_JUDGE_ARGS,
        prompt_version=FROZEN_DELIVERY_PROMPT_VERSION,
        axis="fidelity",
        threshold="any_retained_finding",
        filter_version=FROZEN_FIDELITY_FILTER_VERSION,
        final_window_seconds=FROZEN_FIDELITY_FINAL_WINDOW_SECONDS,
    ):
        raise ValueError("manifest judge/filter configuration drifted")
    if manifest.acceptance_policy != SPEECH_FIDELITY_ACCEPTANCE_POLICY:
        raise ValueError("manifest acceptance policy drifted")
    if manifest.cohort_policy != FROZEN_COHORT_POLICY:
        raise ValueError("manifest cohort policy drifted")
    if set(manifest.languages) != set(SPEECH_FIDELITY_LANGUAGES):
        raise ValueError("manifest must contain exactly es/hi/ko/pt/zh")

    expected_files = {"README.md", "prompt.json"} | {
        f"{language}/test/{name}"
        for language in SPEECH_FIDELITY_LANGUAGES
        for name in ("dataset.json", "results.json")
    }
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(manifest.files) != expected_files or actual_files != expected_files:
        raise ValueError("manifest file inventory is not exact")
    for relative, expected_digest in manifest.files.items():
        path = _resolve_inside(root, relative)
        if sha256_file(path) != expected_digest:
            raise ValueError(f"validation artifact hash mismatch: {relative}")

    prompt_path = _resolve_inside(root, manifest.prompt.path)
    if prompt_path != root / "prompt.json":
        raise ValueError("prompt path is not canonical")
    if manifest.prompt.sha256 != sha256_file(prompt_path):
        raise ValueError("manifest prompt hash mismatch")
    if manifest.prompt.sha256 != FROZEN_DELIVERY_PROMPT_SHA256:
        raise ValueError("exact historical v5 prompt artifact drifted")
    if manifest.files[manifest.prompt.path] != manifest.prompt.sha256:
        raise ValueError("prompt summary differs from file inventory")
    prompt = SpeechFidelityPrompt.model_validate_json(prompt_path.read_text())
    if (
        prompt.prompt_version != manifest.judge.prompt_version
        or prompt.model != manifest.judge.model
        or prompt.model_args != manifest.judge.model_args
    ):
        raise ValueError("prompt and manifest judge provenance differ")

    reports: dict[str, BinaryMetrics] = {}
    total_rows = 0
    seen_row_ids: set[str] = set()
    for language in SPEECH_FIDELITY_LANGUAGES:
        summary = manifest.languages[language]
        dataset_path = _resolve_inside(root, summary.dataset.path)
        results_path = _resolve_inside(root, summary.results.path)
        expected_root = root / language / "test"
        if dataset_path != expected_root / "dataset.json":
            raise ValueError(f"{language}: dataset path is not canonical")
        if results_path != expected_root / "results.json":
            raise ValueError(f"{language}: results path is not canonical")
        dataset = SpeechFidelityDataset.model_validate_json(dataset_path.read_text())
        results = SpeechFidelityResults.model_validate_json(results_path.read_text())
        if dataset.language != language or results.language != language:
            raise ValueError(f"{language}: language mismatch")
        if dataset.source_set is not ValidationSourceSet.FIXED_CURATED:
            raise ValueError(f"{language}: dataset is not the fixed curated cohort")
        if dataset.n_rows != results.n_rows or results.n_rows != len(results.rows):
            raise ValueError(f"{language}: row-count mismatch")
        if summary.dataset.sha256 != sha256_file(dataset_path):
            raise ValueError(f"{language}: dataset summary hash mismatch")
        if summary.results.sha256 != sha256_file(results_path):
            raise ValueError(f"{language}: results summary hash mismatch")
        if manifest.files[summary.dataset.path] != summary.dataset.sha256:
            raise ValueError(f"{language}: dataset summary differs from inventory")
        if manifest.files[summary.results.path] != summary.results.sha256:
            raise ValueError(f"{language}: result summary differs from inventory")
        if _resolve_reference(root, results_path, results.dataset) != dataset_path:
            raise ValueError(f"{language}: result dataset reference differs")
        if _resolve_reference(root, results_path, results.prompt) != prompt_path:
            raise ValueError(f"{language}: result prompt reference differs")
        if results.judge != manifest.judge:
            raise ValueError(f"{language}: result judge configuration drifted")

        if len(dataset.rows) != len(results.rows):
            raise ValueError(f"{language}: row alignment length mismatch")
        for dataset_row, result_row in zip(dataset.rows, results.rows, strict=True):
            expected_stored_alignment = classify_stored_reference_alignment(
                dataset_row.reference_transcript,
                result_row.stored_reference_preview,
            )
            aligned = (
                dataset_row.row_index == result_row.row_index
                and dataset_row.row_id == result_row.row_id
                and dataset_row.simulation_id == result_row.simulation_id
                and dataset_row.source_utterance_idx == result_row.source_utterance_idx
                and dataset_row.agent_turn_index == result_row.agent_turn_index
                and (dataset_row.human_label is ValidationLabel.VIOLATION)
                == result_row.label_positive
                and result_row.stored_reference_alignment is expected_stored_alignment
            )
            if not aligned:
                raise ValueError(f"{language}: dataset/result row alignment mismatch")
            if dataset_row.row_id in seen_row_ids:
                raise ValueError(f"duplicate package row_id: {dataset_row.row_id}")
            seen_row_ids.add(dataset_row.row_id)

        metrics = compute_binary_metrics(results.rows)
        if metrics != results.metrics or metrics != summary.results.metrics:
            raise ValueError(f"{language}: stored metrics do not recompute")
        expected = EXPECTED_CONFUSION[language]
        if (metrics.tp, metrics.tn, metrics.fp, metrics.fn) != expected:
            raise ValueError(
                f"{language}: confusion matrix differs from the fixed cohort"
            )
        if summary.dataset.n_rows != dataset.n_rows:
            raise ValueError(f"{language}: manifest dataset row count differs")
        if summary.dataset.n_positive != dataset.n_positive:
            raise ValueError(f"{language}: manifest positive count differs")
        if summary.dataset.n_negative != dataset.n_negative:
            raise ValueError(f"{language}: manifest negative count differs")
        if (
            results.stored_reference_alignment_counts
            != summary.results.stored_reference_alignment_counts
            or results.stored_reference_alignment_counts
            != EXPECTED_STORED_REFERENCE_ALIGNMENT[language]
        ):
            raise ValueError(f"{language}: stored-reference alignment counts differ")
        passed = passes_acceptance_policy(metrics, manifest.acceptance_policy)
        if passed != summary.results.passes_acceptance_policy or not passed:
            raise ValueError(f"{language}: acceptance-policy result differs")
        reports[language] = metrics
        total_rows += len(results.rows)

    readme = (root / "README.md").read_text(encoding="utf-8")
    if readme != render_speech_fidelity_readme(manifest):
        raise ValueError("README does not match the typed package summaries")

    return SpeechFidelityPackageReport(
        root=str(root),
        files_verified=len(manifest.files),
        rows_verified=total_rows,
        metrics=reports,
    )


def _format_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_speech_fidelity_readme(manifest: SpeechFidelityManifest) -> str:
    """Render the reviewer-facing README directly from the typed manifest."""
    lines = [
        "# Utterance-level Speech fidelity validation",
        "",
        "Final human labels, stored v5 delivery-judge findings, deterministic "
        "filter decisions, and aggregate validation metrics for tau-multi's fixed "
        "human-validated cohort.",
        "",
        "- The unit is one delivered agent utterance, keyed by language, simulation, and agent-turn index.",
        "- Every row belongs to the single neutral `fixed_curated` source set.",
        "- `dataset.json` preserves the human-aligned `reference_transcript` after removing unsynthesized control tokens.",
        "- Delivery judge v5 received that full cleaned reference but persisted at most its first 200 characters in `stored_reference_preview`; the preview is metadata, not evidence that the audio was truncated.",
        "- `stored_reference_alignment` is verifier-enforced after whitespace normalization: `exact` or `stored_prefix`. Prefix status is not inferred from the interruption flag.",
        "- `direct_utterance` labels were assigned to the displayed utterance; `projected_call_pass` negatives inherit a call-level no-finding label.",
        "- `results.json` preserves every fidelity finding and marks findings removed by the final-second filter.",
        "- A judge violation means at least one fidelity finding remains after excluding findings whose approximate span reaches the final 1.0 seconds of the agent clip.",
        "- These validation metrics test the underlying detector at any retained severity; the paper's downstream Experience and standalone Speech-fidelity summaries separately require severity 2 or higher.",
        "- `prompt.json` preserves the exact delivery-judge v5 prompt, model settings, and language-pack rubric material.",
        "- The public measure is `speech_fidelity`; runtime factor ids are provenance, not independently gated measures.",
        "",
        "## Metrics",
        "",
        "| Language | Rows (+/−) | Stored preview exact/prefix | TP | TN | FP | FN | Precision | Recall | F1 | κ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for language in SPEECH_FIDELITY_LANGUAGES:
        summary = manifest.languages[language]
        metrics = summary.results.metrics
        alignments = summary.results.stored_reference_alignment_counts
        lines.append(
            f"| {language} | {summary.dataset.n_rows} "
            f"({summary.dataset.n_positive}/{summary.dataset.n_negative}) | "
            f"{alignments['exact']}/{alignments['stored_prefix']} | "
            f"{metrics.tp} | {metrics.tn} | {metrics.fp} | {metrics.fn} | "
            f"{_format_metric(metrics.precision)} | "
            f"{_format_metric(metrics.recall)} | "
            f"{_format_metric(metrics.f1)} | "
            f"{_format_metric(metrics.kappa)} |"
        )
    lines.extend(
        [
            "",
            "## Offline verification",
            "",
            "`verify_speech_fidelity_validation_package()` validates every JSON contract and SHA-256 reference, checks row alignment, reapplies filter v1 to each stored finding, recomputes every confusion matrix and metric, and requires the exact fixed language results above.",
        ]
    )
    return "\n".join(lines) + "\n"
