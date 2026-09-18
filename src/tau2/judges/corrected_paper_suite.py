# Copyright Sierra
"""Safe replay adapter for the frozen generic τ-Multilingual judge suite.

The paper's generic nativeness, interaction-quality, and audio-delivery
verdicts were produced by the suite at commit :data:`FROZEN_GENERIC_SUITE_COMMIT`.
That commit predates fields in today's :class:`~tau2.data_model.simulation.SimulationRun`;
letting it save current results in place can therefore silently discard data.

This module makes the boundary explicit:

* lock exactly the ten corrected Korean/Mandarin retail trial-0 cells;
* copy their mutable JSON and stereo audio into an isolated legacy workspace;
* run only the verified frozen suite against those copies;
* rebuild a current-schema output from the immutable source JSON, importing only
  ``nativeness_info``, ``quality_info``, and ``delivery_info``; and
* emit the paper's deterministic one-second final-window exclusion sidecar.

Preparing, checking, merging, and building the sidecar make no external calls.
The frozen suite invocation is a separate, explicit, resumable operation.

The paper replay ultimately selected fewer factors than that original all-in-one
suite. ``compose_cross_workspace_judgments`` is the strict promotion path for
those split runs: exact 7+3 quality, delivery, and Mandarin modal-particle
outputs are read from explicit roots and only their owned top-level field is
imported into a fresh current-schema corpus. The v16 natural-word-choice replay
remains a separate utterance sidecar.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.config import (
    DEFAULT_DELIVERY_FIDELITY_END_EXCLUSION_SECONDS,
    DELIVERY_FIDELITY_FINDING_FILTER_VERSION,
)
from tau2.data_model.simulation import (
    DeliveryInfo,
    JudgeOutcome,
    NativenessInfo,
    QualityInfo,
    Results,
    SimulationRun,
    TerminationReason,
)
from tau2.evaluator.evaluator import get_simulation_language_info
from tau2.judges.delivery.disk_audio import find_both_wav
from tau2.judges.delivery.postprocess import (
    is_in_final_utterance_window,
    time_range_bounds_seconds,
)
from tau2.metrics.interaction_quality import agent_utterance_tick_spans

FROZEN_GENERIC_SUITE_COMMIT = "6813fceab54fe6c7c44733be9935e6d38aaca848"
GENERIC_COHORT_SCHEMA_VERSION = "tau-multi-corrected-generic-judges-v1"
GENERIC_MERGE_SCHEMA_VERSION = "tau-multi-corrected-generic-merge-v1"
COMPOSE_SCHEMA_VERSION = "tau-multi-corrected-cross-workspace-compose-v1"
MANIFEST_FILENAME = "manifest.json"
EXECUTION_RECEIPT_FILENAME = "execution.json"
MERGE_REPORT_FILENAME = "merge_report.json"
COMPOSE_REPORT_FILENAME = "compose_report.json"
LEGACY_DIRNAME = "legacy"
MERGED_DIRNAME = "merged"
JUDGE_FIELDS = ("nativeness_info", "quality_info", "delivery_info")
ROUNDTRIP_PROBE_FIELD = "_corrected_paper_frozen_roundtrip_probe"
JUDGE_EVIDENCE_FIELDS = (
    "id",
    "task_id",
    "trial",
    "messages",
    "ticks",
    "termination_reason",
    "speech_environment",
    "agent_provider",
    "agent_voice",
    "effect_timeline",
    "policy",
)

NATIVENESS_MODEL = "gpt-5.5"
NATIVENESS_ARGS = {"reasoning_effort": "medium"}
NATIVENESS_PROMPT_VERSION = "v15"
NATIVENESS_RUBRIC_VERSION = "nativeness-rubric-v19"

QUALITY_MODEL = "gpt-5.5"
QUALITY_PROMPT_VERSION = "quality-judge-v7"
QUALITY_RUBRIC_VERSION = "quality-rubric-v4"
QUALITY_METRICS_VERSION = "quality-metrics-v5"
QUALITY_FACTOR_ARGS: dict[str, dict[str, str]] = {
    "unnecessary_repetition": {"reasoning_effort": "none"},
    "agent_caused_tool_error": {"reasoning_effort": "high"},
    "auth_arg_mismatch": {"reasoning_effort": "high"},
    "incorrect_tool_parameters": {"reasoning_effort": "high"},
    "unnecessary_tool_call": {"reasoning_effort": "xhigh"},
}
QUALITY_PRECLASSIFIED_OUTCOMES: dict[str, frozenset[JudgeOutcome]] = {
    "unnecessary_repetition": frozenset({JudgeOutcome.NO_OPPORTUNITY}),
    "agent_caused_tool_error": frozenset({JudgeOutcome.NO_OPPORTUNITY}),
    "auth_arg_mismatch": frozenset({JudgeOutcome.NO_OPPORTUNITY, JudgeOutcome.PASS}),
    "incorrect_tool_parameters": frozenset({JudgeOutcome.NO_OPPORTUNITY}),
    "unnecessary_tool_call": frozenset({JudgeOutcome.NO_OPPORTUNITY}),
}
QUALITY_DETERMINISTIC_FACTORS = frozenset(
    {
        "responsiveness",
        "yielding",
        "inappropriate_interruption",
        "backchannel_selectivity",
        "vocal_tic_selectivity",
        "non_directed_selectivity",
        "monologue",
    }
)
PAPER_QUALITY_FACTOR_ARGS: dict[str, dict[str, str]] = {
    "agent_caused_tool_error": {"reasoning_effort": "high"},
    "auth_arg_mismatch": {"reasoning_effort": "high"},
    "incorrect_tool_parameters": {"reasoning_effort": "high"},
}
PAPER_QUALITY_FACTOR_IDS = QUALITY_DETERMINISTIC_FACTORS | frozenset(
    PAPER_QUALITY_FACTOR_ARGS
)
EXCLUDED_QUALITY_FACTOR_IDS = frozenset(
    {"unnecessary_repetition", "unnecessary_tool_call"}
)
PAPER_MODAL_FACTOR_IDS = frozenset(
    {
        "backchannel_frequency",
        "email_symbol_verbalization",
        "register_formality",
        "modal_particles",
    }
)
PAPER_MODAL_DETERMINISTIC_FACTOR_IDS = PAPER_MODAL_FACTOR_IDS - {"modal_particles"}
PAPER_SYSTEM_LABELS = {
    "openai_minimal": "OpenAI minimal",
    "openai_xhigh": "OpenAI xhigh",
    "gemini_minimal": "Gemini minimal",
    "gemini_high": "Gemini high",
    "xai_provider_default": "xAI",
}

DELIVERY_MODEL = "gemini/gemini-3.1-pro-preview"
DELIVERY_ARGS = {"temperature": 0.0, "max_tokens": 8192, "timeout": 120}
DELIVERY_PROMPT_VERSION = "v5"
DELIVERY_SAMPLE_RATE = 1.0
DELIVERY_MAX_SEGMENTS = 100

CANONICAL_FINAL_WINDOW_CSV_SHA256 = (
    "94575b3f52af53c7471cb3fa044c5cf05fef694541a7e798294c274c26ace591"
)
CANONICAL_FINAL_WINDOW_MANIFEST_SHA256 = (
    "f372056904e20990a0038a038501482484ee6bd10535a7f25c1a01d4cdd7179b"
)
CANONICAL_FINAL_WINDOW_ROWS = 7_436


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def canonical_results_files_sha256(sources: list[dict[str, object]]) -> str:
    """Hash ordered canonical result headers as ``path<TAB>sha256`` lines."""
    payload = "\n".join(f"{row['path']}\t{row['sha256']}" for row in sources)
    return hashlib.sha256(payload.encode()).hexdigest()


def canonical_cohort_fingerprint_sha256(sources: list[dict[str, object]]) -> str:
    """Hash ordered canonical result headers including trial-zero counts."""
    payload = "\n".join(
        f"{row['path']}\t{row['sha256']}\t{row['trial_0_calls']}" for row in sources
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 for one provenance-bearing file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def nonjudge_sha256(payload: dict) -> str:
    """Hash a simulation after removing the only three importable fields."""
    stripped = dict(payload)
    for field in JUDGE_FIELDS:
        stripped.pop(field, None)
    return _sha256_value(stripped)


def _results_nonindex_sha256(payload: dict) -> str:
    stripped = dict(payload)
    stripped.pop("simulation_index", None)
    return _sha256_value(stripped)


def judge_evidence_sha256(payload: dict) -> str:
    """Hash only evidence consumed by all three frozen judge families.

    Every field in this projection exists at the frozen commit.  It therefore
    survives a legacy round-trip even when unrelated fields added later do not.
    """
    return _sha256_value({field: payload.get(field) for field in JUDGE_EVIDENCE_FIELDS})


class FrozenGenericJudgeContract(BaseModel):
    """Exact models, prompts, rubrics, metrics, and coverage of the paper suite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    commit: Literal["6813fceab54fe6c7c44733be9935e6d38aaca848"] = (
        FROZEN_GENERIC_SUITE_COMMIT
    )
    nativeness_model: Literal["gpt-5.5"] = NATIVENESS_MODEL
    nativeness_args: dict[str, str] = Field(
        default_factory=lambda: dict(NATIVENESS_ARGS)
    )
    nativeness_prompt_version: Literal["v15"] = NATIVENESS_PROMPT_VERSION
    nativeness_rubric_version: Literal["nativeness-rubric-v19"] = (
        NATIVENESS_RUBRIC_VERSION
    )
    quality_model: Literal["gpt-5.5"] = QUALITY_MODEL
    quality_factor_args: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {
            factor: dict(args) for factor, args in QUALITY_FACTOR_ARGS.items()
        }
    )
    quality_prompt_version: Literal["quality-judge-v7"] = QUALITY_PROMPT_VERSION
    quality_rubric_version: Literal["quality-rubric-v4"] = QUALITY_RUBRIC_VERSION
    quality_metrics_version: Literal["quality-metrics-v5"] = QUALITY_METRICS_VERSION
    delivery_model: Literal["gemini/gemini-3.1-pro-preview"] = DELIVERY_MODEL
    delivery_args: dict[str, float | int] = Field(
        default_factory=lambda: dict(DELIVERY_ARGS)
    )
    delivery_prompt_version: Literal["v5"] = DELIVERY_PROMPT_VERSION
    delivery_sample_rate: Literal[1.0] = DELIVERY_SAMPLE_RATE
    delivery_max_segments: Literal[100] = DELIVERY_MAX_SEGMENTS


class GenericJudgeCell(BaseModel):
    """One language/system results root in the corrected trial-0 cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: Annotated[str, Field(min_length=2, description="ISO language code.")]
    system_slug: Annotated[str, Field(min_length=1, description="Paper system slug.")]
    source_relative_path: Annotated[
        str,
        Field(min_length=1, description="Cell path below the supplied source root."),
    ]


class GenericJudgeCohortContract(BaseModel):
    """Closed cell inventory and trial selection for one adapter invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trial: Annotated[int, Field(ge=0, description="Only this trial is materialized.")]
    calls_per_cell: Annotated[
        int, Field(ge=1, description="Required unique task calls per cell.")
    ]
    untouched_trial: Annotated[
        int,
        Field(ge=0, description="Only permitted source trial outside the cohort."),
    ]
    untouched_calls: Annotated[
        int,
        Field(ge=0, description="Required source calls left outside the cohort."),
    ]
    cells: Annotated[
        tuple[GenericJudgeCell, ...],
        Field(min_length=1, description="Complete language/system cell inventory."),
    ]

    @model_validator(mode="after")
    def _unique_cells_and_paths(self) -> "GenericJudgeCohortContract":
        if self.trial == self.untouched_trial:
            raise ValueError("selected and untouched trials must differ")
        keys = [(cell.language, cell.system_slug) for cell in self.cells]
        paths = [cell.source_relative_path for cell in self.cells]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate language/system cell in cohort contract")
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate source path in cohort contract")
        for value in paths:
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe source-relative cell path: {value}")
        return self

    @property
    def expected_calls(self) -> int:
        return len(self.cells) * self.calls_per_cell


CANONICAL_GENERIC_COHORT = GenericJudgeCohortContract(
    trial=0,
    calls_per_cell=50,
    untouched_trial=1,
    untouched_calls=400,
    cells=(
        GenericJudgeCell(
            language="ko",
            system_slug="openai_minimal",
            source_relative_path=(
                "retail_name_roles_v1_korean_retail/ko_retail_openai_minimal"
            ),
        ),
        GenericJudgeCell(
            language="ko",
            system_slug="openai_xhigh",
            source_relative_path=(
                "retail_name_roles_v1_korean_retail/ko_retail_openai_xhigh"
            ),
        ),
        GenericJudgeCell(
            language="ko",
            system_slug="gemini_minimal",
            source_relative_path=(
                "retail_name_roles_v1_korean_retail/ko_retail_gemini_minimal"
            ),
        ),
        GenericJudgeCell(
            language="ko",
            system_slug="gemini_high",
            source_relative_path=(
                "retail_name_roles_v1_korean_retail/ko_retail_gemini_high"
            ),
        ),
        GenericJudgeCell(
            language="ko",
            system_slug="xai_provider_default",
            source_relative_path=(
                "retail_name_roles_xai_v1_korean_retail/ko_retail_xai_provider_default"
            ),
        ),
        GenericJudgeCell(
            language="zh",
            system_slug="openai_minimal",
            source_relative_path=(
                "retail_name_roles_v1_mandarin_retail/zh_retail_openai_minimal"
            ),
        ),
        GenericJudgeCell(
            language="zh",
            system_slug="openai_xhigh",
            source_relative_path=(
                "retail_name_roles_v1_mandarin_retail/zh_retail_openai_xhigh"
            ),
        ),
        GenericJudgeCell(
            language="zh",
            system_slug="gemini_minimal",
            source_relative_path=(
                "retail_name_roles_v1_mandarin_retail/zh_retail_gemini_minimal"
            ),
        ),
        GenericJudgeCell(
            language="zh",
            system_slug="gemini_high",
            source_relative_path=(
                "retail_name_roles_v1_mandarin_retail/zh_retail_gemini_high"
            ),
        ),
        GenericJudgeCell(
            language="zh",
            system_slug="xai_provider_default",
            source_relative_path=(
                "retail_name_roles_xai_v1_mandarin_retail/"
                "zh_retail_xai_provider_default"
            ),
        ),
    ),
)


class LockedJudgeCall(BaseModel):
    """Immutable source identity for one selected trial-0 call and its audio."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    simulation_id: Annotated[str, Field(min_length=1, description="Simulation id.")]
    task_id: Annotated[str, Field(min_length=1, description="Task id.")]
    trial: Annotated[int, Field(ge=0, description="Selected trial index.")]
    simulation_path: Annotated[
        str, Field(description="Simulation path relative to its results cell.")
    ]
    simulation_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Source JSON hash.")
    ]
    nonjudge_sha256: Annotated[
        str,
        Field(pattern=r"^[0-9a-f]{64}$", description="Source non-judge payload hash."),
    ]
    judge_evidence_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Frozen-schema-stable judge evidence hash.",
        ),
    ]
    audio_path: Annotated[str, Field(description="WAV path relative to its cell.")]
    audio_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Stereo WAV hash.")
    ]
    audio_channels: Annotated[int, Field(ge=1, description="WAV channel count.")]
    audio_frames: Annotated[int, Field(gt=0, description="WAV frame count.")]
    audio_sample_rate: Annotated[int, Field(gt=0, description="WAV sample rate.")]


class LockedJudgeCell(BaseModel):
    """Hashed source metadata and selected calls for one results cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str
    system_slug: str
    source_relative_path: str
    results_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Source results hash.")
    ]
    calls: Annotated[list[LockedJudgeCall], Field(description="Selected calls.")]


class GenericPreparationCounts(BaseModel):
    """Hard-gated counts from an API-free cohort preflight."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cells: Annotated[int, Field(ge=0, description="Selected results roots.")]
    calls: Annotated[int, Field(ge=0, description="Unique selected calls.")]
    stereo_audio_files: Annotated[int, Field(ge=0, description="Validated WAVs.")]
    infrastructure_errors: Annotated[int, Field(ge=0, description="Infra failures.")]
    duplicates: Annotated[int, Field(ge=0, description="Duplicate task/trial pairs.")]
    unselected_calls: Annotated[int, Field(ge=0, description="Calls left in source.")]


class GenericJudgeManifest(BaseModel):
    """Typed, complete lock for the isolated frozen-suite workspace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-corrected-generic-judges-v1"]
    source_root: Annotated[str, Field(description="Absolute immutable source root.")]
    workspace: Annotated[str, Field(description="Absolute isolated workspace.")]
    cohort: GenericJudgeCohortContract
    judges: FrozenGenericJudgeContract
    cells: Annotated[list[LockedJudgeCell], Field(description="Hashed cell inventory.")]
    counts: GenericPreparationCounts
    cohort_fingerprint_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Cohort lock fingerprint.")
    ]


class FrozenWorktreeIdentity(BaseModel):
    """Verified source-code identity used to execute the legacy suite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    commit: Literal["6813fceab54fe6c7c44733be9935e6d38aaca848"]
    clean: Literal[True]


class SuiteExecutionPlan(BaseModel):
    """Auditable subprocess invocation; constructing it performs no API calls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frozen_worktree: str
    frozen_commit: Literal["6813fceab54fe6c7c44733be9935e6d38aaca848"]
    pythonpath: str
    command: list[str]
    calls: int
    cells: int


class FrozenSchemaRoundTripReport(BaseModel):
    """API-free proof that every staged call survives frozen serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frozen_worktree: str
    frozen_commit: Literal["6813fceab54fe6c7c44733be9935e6d38aaca848"]
    commands: list[list[str]]
    cells: int
    calls: int
    judge_evidence_mismatches: Literal[0]
    source_hash_mismatches: Literal[0]
    external_api_calls: Literal[False]


class SuitePreflightReport(BaseModel):
    """Complete no-API preflight emitted before the paid frozen suite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan: SuiteExecutionPlan
    frozen_schema_roundtrip: FrozenSchemaRoundTripReport


class SuiteExecutionReceipt(BaseModel):
    """Successful frozen-suite invocation recorded by the current adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-corrected-generic-judges-v1"]
    finished_at: str
    plan: SuiteExecutionPlan
    frozen_schema_roundtrip: FrozenSchemaRoundTripReport
    returncode: Literal[0]


class GenericMergeReport(BaseModel):
    """Proof that only judge siblings changed in the current-schema output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-corrected-generic-merge-v1"]
    created_at: str
    output_root: str
    cells: int
    calls: int
    imported_fields: tuple[str, str, str]
    source_hash_mismatches: int
    nonjudge_hash_mismatches: int
    refreshed_indexes: int


class ComposeInputIdentity(BaseModel):
    """One independently judged workspace used by a field-wise promotion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["quality", "delivery", "modal_particles"]
    workspace: str
    manifest_path: str
    manifest_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Manifest byte hash.")
    ]
    declared_workspace: str
    declared_workspace_matches: bool
    cohort_lock_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Manifest hash after removing only its workspace path.",
        ),
    ]
    imported_field: Literal["quality_info", "delivery_info", "nativeness_info"]
    imported_calls: Annotated[int, Field(ge=0)]


class CrossWorkspaceComposeReport(BaseModel):
    """Proof for an atomic, field-wise current-schema judge promotion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-corrected-cross-workspace-compose-v1"]
    created_at: str
    output_root: str
    merged_root: str
    source_root: str
    authoritative_manifest_path: str
    authoritative_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    cohort_fingerprint_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    cohort_lock_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    inputs: tuple[ComposeInputIdentity, ComposeInputIdentity, ComposeInputIdentity]
    cells: Annotated[int, Field(ge=0)]
    calls: Annotated[int, Field(ge=0)]
    imported_quality_calls: Annotated[int, Field(ge=0)]
    imported_delivery_calls: Annotated[int, Field(ge=0)]
    delivery_error_utterances: Annotated[
        int,
        Field(
            ge=0,
            description="Imported delivery ERROR rows retained as unscored.",
        ),
    ]
    imported_modal_nativeness_calls: Annotated[int, Field(ge=0)]
    preserved_source_nativeness_calls: Annotated[int, Field(ge=0)]
    quality_factor_ids: tuple[str, ...]
    excluded_quality_factor_ids: tuple[str, ...]
    modal_factor_ids: tuple[str, ...]
    merged_results_fingerprint_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    merged_simulations_fingerprint_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$")
    ]
    source_hash_mismatches: Literal[0]
    nonjudge_hash_mismatches: Literal[0]
    audio_hash_mismatches: Literal[0]
    refreshed_indexes: Annotated[int, Field(ge=0)]
    unselected_calls_not_copied: Annotated[int, Field(ge=0)]
    natural_word_choice_storage: Literal["separate_sidecar"]


class FinalWindowExclusionRow(BaseModel):
    """One raw v5 fidelity finding excluded by the paper's deterministic rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sim_id: str
    language: str
    domain: str
    system: str
    utterance_idx: int
    finding_index: int
    severity: int
    time_range: str
    clip_duration_seconds: float
    span_start_seconds: float
    span_end_seconds: float


class GeneratedFinalWindowExclusionManifest(BaseModel):
    """Generation provenance for a final-window exclusion sidecar."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact: str
    filter_version: Literal["v1"]
    window_seconds: Literal[1.0]
    cohort: str
    excluded_findings: int
    excluded_severity_2plus_findings: int
    by_language: dict[str, int]
    by_system: dict[str, int]
    generation_results_files_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    replacement_cohort_fingerprint_sha256: str
    replacement_calls: int
    canonical_sidecar_sha256: str | None = None


class FinalWindowExclusionManifest(GeneratedFinalWindowExclusionManifest):
    """Final-window sidecar bound to the complete canonical paper cohort."""

    final_window_csv_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    canonical_results_files_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    canonical_cohort_fingerprint_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$")
    ]
    canonical_results_files: Annotated[int, Field(ge=1)]
    canonical_trial0_calls: Annotated[int, Field(ge=1)]


def _safe_cell_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"cell path escapes source root: {relative}")
    return path


def _uses_canonical_contract(contract: GenericJudgeCohortContract) -> bool:
    return contract == CANONICAL_GENERIC_COHORT


def _validate_production_metadata(metadata: Results, cell: GenericJudgeCell) -> None:
    info = metadata.info
    if info.environment_info.domain_name != "retail":
        raise ValueError(
            f"corrected judge cell is not retail: {cell.source_relative_path}"
        )
    subset = info.task_subset
    if subset is None or (
        subset.name != "retail_50" or subset.size != 50 or subset.tasks_scored != 50
    ):
        raise ValueError(
            f"corrected judge cell is not the frozen retail_50 frame: "
            f"{cell.source_relative_path}"
        )
    expected = {
        "openai_minimal": ("openai", "gpt-realtime-2", "minimal"),
        "openai_xhigh": ("openai", "gpt-realtime-2", "xhigh"),
        "gemini_minimal": (
            "gemini",
            "gemini-3.1-flash-live-preview",
            "minimal",
        ),
        "gemini_high": ("gemini", "gemini-3.1-flash-live-preview", "high"),
        "xai_provider_default": (
            "xai",
            "grok-voice-think-fast-1.0",
            "provider_default",
        ),
    }
    audio = info.audio_native_config
    observed = (
        audio.provider if audio else None,
        audio.model if audio else None,
        audio.reasoning_effort if audio else None,
    )
    if observed != expected[cell.system_slug]:
        raise ValueError(
            f"corrected judge arm drifted for {cell.system_slug}: {observed}"
        )


def _wav_identity(path: Path) -> tuple[int, int, int]:
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            frames = audio.getnframes()
            rate = audio.getframerate()
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"invalid both.wav: {path}") from exc
    if channels != 2:
        raise ValueError(f"both.wav is not stereo: {path}")
    if frames <= 0 or rate <= 0:
        raise ValueError(f"both.wav is empty or malformed: {path}")
    return channels, frames, rate


def _plan_cell(
    source_root: Path,
    cell: GenericJudgeCell,
    contract: GenericJudgeCohortContract,
) -> tuple[LockedJudgeCell, int]:
    cell_root = _safe_cell_path(source_root, cell.source_relative_path)
    results_path = cell_root / "results.json"
    if not results_path.is_file():
        raise ValueError(f"missing corrected results root: {results_path}")
    metadata = Results.load_metadata(cell_root)
    if _uses_canonical_contract(contract):
        _validate_production_metadata(metadata, cell)
    index = metadata.simulation_index
    if index is None:
        raise ValueError(f"results root has no simulation index: {results_path}")
    unexpected_trials = [
        entry
        for entry in index
        if entry.trial not in {contract.trial, contract.untouched_trial}
    ]
    if unexpected_trials:
        raise ValueError(
            f"cell contains {len(unexpected_trials)} calls outside selected trial "
            f"{contract.trial} and untouched trial {contract.untouched_trial}: "
            f"{results_path}"
        )
    infra = [
        entry
        for entry in index
        if entry.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR.value
    ]
    if infra:
        raise ValueError(
            f"source cell contains {len(infra)} infrastructure errors: {results_path}"
        )
    selected = [entry for entry in index if entry.trial == contract.trial]
    pairs = [(str(entry.task_id), entry.trial) for entry in selected]
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"duplicate task/trial pair in selected cell: {results_path}")
    if len(selected) != contract.calls_per_cell:
        raise ValueError(
            f"selected cell has {len(selected)} usable trial-{contract.trial} calls; "
            f"expected {contract.calls_per_cell}: {results_path}"
        )
    known_tasks = {str(task.id) for task in metadata.tasks}
    calls: list[LockedJudgeCall] = []
    for entry in sorted(selected, key=lambda item: (str(item.task_id), item.id)):
        sim_path = cell_root / "simulations" / f"{entry.id}.json"
        raw = sim_path.read_bytes()
        payload = json.loads(raw)
        simulation = SimulationRun.model_validate(payload)
        if (
            simulation.id != entry.id
            or simulation.trial != contract.trial
            or str(simulation.task_id) not in known_tasks
        ):
            raise ValueError(f"simulation/index identity mismatch: {sim_path}")
        if simulation.termination_reason is TerminationReason.INFRASTRUCTURE_ERROR:
            raise ValueError(f"selected infrastructure-error simulation: {sim_path}")
        language, _script = get_simulation_language_info(simulation)
        if language != cell.language:
            raise ValueError(
                f"selected simulation language {language!r} != {cell.language!r}: "
                f"{sim_path}"
            )
        if not simulation.ticks:
            raise ValueError(
                f"selected voice simulation has no tick timeline: {sim_path}"
            )
        wav_path = find_both_wav(cell_root, simulation)
        if wav_path is None:
            raise ValueError(f"selected simulation has no both.wav: {sim_path}")
        channels, frames, sample_rate = _wav_identity(wav_path)
        calls.append(
            LockedJudgeCall(
                simulation_id=simulation.id,
                task_id=str(simulation.task_id),
                trial=simulation.trial,
                simulation_path=str(sim_path.relative_to(cell_root)),
                simulation_sha256=hashlib.sha256(raw).hexdigest(),
                nonjudge_sha256=nonjudge_sha256(payload),
                judge_evidence_sha256=judge_evidence_sha256(payload),
                audio_path=str(wav_path.relative_to(cell_root)),
                audio_sha256=sha256_file(wav_path),
                audio_channels=channels,
                audio_frames=frames,
                audio_sample_rate=sample_rate,
            )
        )
    untouched = [entry for entry in index if entry.trial == contract.untouched_trial]
    untouched_pairs = [(str(entry.task_id), entry.trial) for entry in untouched]
    if len(untouched_pairs) != len(set(untouched_pairs)):
        raise ValueError(f"duplicate untouched task/trial pair: {results_path}")
    return (
        LockedJudgeCell(
            language=cell.language,
            system_slug=cell.system_slug,
            source_relative_path=cell.source_relative_path,
            results_sha256=sha256_file(results_path),
            calls=calls,
        ),
        len(untouched),
    )


def plan_frozen_judge_workspace(
    *,
    source_root: Path,
    workspace: Path,
    contract: GenericJudgeCohortContract = CANONICAL_GENERIC_COHORT,
) -> GenericJudgeManifest:
    """Scan and hash the exact cohort without writing or invoking a judge."""
    source_root = source_root.expanduser().resolve()
    workspace = workspace.expanduser().resolve()
    if workspace == source_root or workspace.is_relative_to(source_root):
        raise ValueError("judge workspace must be outside the immutable source root")
    cells: list[LockedJudgeCell] = []
    unselected = 0
    seen_simulations: set[str] = set()
    for cell in contract.cells:
        locked, cell_unselected = _plan_cell(source_root, cell, contract)
        unselected += cell_unselected
        for call in locked.calls:
            if call.simulation_id in seen_simulations:
                raise ValueError(
                    f"duplicate simulation id across cells: {call.simulation_id}"
                )
            seen_simulations.add(call.simulation_id)
        cells.append(locked)
    if len(seen_simulations) != contract.expected_calls:
        raise ValueError("selected unique-call count does not match cohort contract")
    if unselected != contract.untouched_calls:
        raise ValueError(
            f"source has {unselected} untouched trial-{contract.untouched_trial} "
            f"calls; expected {contract.untouched_calls}"
        )
    counts = GenericPreparationCounts(
        cells=len(cells),
        calls=len(seen_simulations),
        stereo_audio_files=len(seen_simulations),
        infrastructure_errors=0,
        duplicates=0,
        unselected_calls=unselected,
    )
    fingerprint = _sha256_value(
        {
            "cohort": contract.model_dump(mode="json"),
            "judges": FrozenGenericJudgeContract().model_dump(mode="json"),
            "cells": [cell.model_dump(mode="json") for cell in cells],
        }
    )
    return GenericJudgeManifest(
        schema_version=GENERIC_COHORT_SCHEMA_VERSION,
        source_root=str(source_root),
        workspace=str(workspace),
        cohort=contract,
        judges=FrozenGenericJudgeContract(),
        cells=cells,
        counts=counts,
        cohort_fingerprint_sha256=fingerprint,
    )


def _write_json(path: Path, value: BaseModel | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _stage_cell(source_root: Path, temporary: Path, cell: LockedJudgeCell) -> None:
    source_cell = source_root / cell.source_relative_path
    staged_cell = temporary / LEGACY_DIRNAME / cell.source_relative_path
    selected_ids = {call.simulation_id for call in cell.calls}
    metadata = json.loads((source_cell / "results.json").read_text())
    metadata["simulation_index"] = [
        {
            **entry,
            "quality": None,
            "nativeness": None,
            "fidelity": None,
            "intonation": None,
        }
        for entry in metadata.get("simulation_index", [])
        if entry.get("id") in selected_ids
    ]
    _write_json(staged_cell / "results.json", metadata)
    for call in cell.calls:
        source_sim = source_cell / call.simulation_path
        payload = json.loads(source_sim.read_text())
        for field in JUDGE_FIELDS:
            payload[field] = None
        _write_json(staged_cell / call.simulation_path, payload)
        source_audio = source_cell / call.audio_path
        staged_audio = staged_cell / call.audio_path
        staged_audio.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_audio, staged_audio)


def _verify_sources(manifest: GenericJudgeManifest) -> None:
    source_root = Path(manifest.source_root)
    for cell in manifest.cells:
        cell_root = source_root / cell.source_relative_path
        if sha256_file(cell_root / "results.json") != cell.results_sha256:
            raise ValueError(
                f"source results hash drifted: {cell.source_relative_path}"
            )
        for call in cell.calls:
            sim_path = cell_root / call.simulation_path
            payload = json.loads(sim_path.read_text())
            if sha256_file(sim_path) != call.simulation_sha256:
                raise ValueError(
                    f"source simulation hash drifted: {call.simulation_id}"
                )
            if nonjudge_sha256(payload) != call.nonjudge_sha256:
                raise ValueError(f"source non-judge hash drifted: {call.simulation_id}")
            audio_path = cell_root / call.audio_path
            if sha256_file(audio_path) != call.audio_sha256:
                raise ValueError(f"source audio hash drifted: {call.simulation_id}")


def _verify_legacy_materialization(manifest: GenericJudgeManifest) -> None:
    """Verify every field before any frozen-schema process has saved a sim."""
    legacy_root = Path(manifest.workspace) / LEGACY_DIRNAME
    for cell in manifest.cells:
        cell_root = legacy_root / cell.source_relative_path
        metadata = json.loads((cell_root / "results.json").read_text())
        observed_ids = {entry["id"] for entry in metadata.get("simulation_index", [])}
        expected_ids = {call.simulation_id for call in cell.calls}
        if observed_ids != expected_ids:
            raise ValueError(f"legacy cell index drifted: {cell.source_relative_path}")
        observed_files = {
            path.stem for path in (cell_root / "simulations").glob("*.json")
        }
        if observed_files != expected_ids:
            raise ValueError(
                f"legacy simulation inventory drifted: {cell.source_relative_path}"
            )
        for call in cell.calls:
            payload = json.loads((cell_root / call.simulation_path).read_text())
            if nonjudge_sha256(payload) != call.nonjudge_sha256:
                raise ValueError(
                    f"legacy non-judge payload drifted: {call.simulation_id}"
                )
            if sha256_file(cell_root / call.audio_path) != call.audio_sha256:
                raise ValueError(f"legacy audio drifted: {call.simulation_id}")


def _verify_legacy_identity_audio(manifest: GenericJudgeManifest) -> None:
    """Verify the inventory, call identity, and audio that judges consume."""
    legacy_root = Path(manifest.workspace) / LEGACY_DIRNAME
    for cell in manifest.cells:
        cell_root = legacy_root / cell.source_relative_path
        metadata = json.loads((cell_root / "results.json").read_text())
        expected_ids = {call.simulation_id for call in cell.calls}
        observed_ids = {entry["id"] for entry in metadata.get("simulation_index", [])}
        observed_files = {
            path.stem for path in (cell_root / "simulations").glob("*.json")
        }
        if observed_ids != expected_ids or observed_files != expected_ids:
            raise ValueError(
                f"runnable legacy inventory drifted: {cell.source_relative_path}"
            )
        for call in cell.calls:
            payload = json.loads((cell_root / call.simulation_path).read_text())
            if (
                payload.get("id") != call.simulation_id
                or str(payload.get("task_id")) != call.task_id
                or payload.get("trial") != call.trial
            ):
                raise ValueError(f"legacy call identity drifted: {call.simulation_id}")
            if sha256_file(cell_root / call.audio_path) != call.audio_sha256:
                raise ValueError(f"runnable legacy audio drifted: {call.simulation_id}")


def _verify_runnable_legacy(manifest: GenericJudgeManifest) -> None:
    """Accept pristine calls or exact partial outputs from an interrupted suite.

    The frozen quality extractor deterministically annotates tool-result
    ``turn_idx`` values while judging, so a paid partial pass is not byte-equal
    to the pristine evidence.  Pristine calls must still reproduce the lock;
    touched calls must instead carry valid frozen provenance on every judge
    sibling present.  This preserves safe gap-fill resumability without ever
    trusting legacy non-judge fields for the current-schema merge.
    """
    _verify_legacy_identity_audio(manifest)
    legacy_root = Path(manifest.workspace) / LEGACY_DIRNAME
    for cell in manifest.cells:
        cell_root = legacy_root / cell.source_relative_path
        for call in cell.calls:
            payload = json.loads((cell_root / call.simulation_path).read_text())
            present = [
                field for field in JUDGE_FIELDS if payload.get(field) is not None
            ]
            if not present:
                if judge_evidence_sha256(payload) != call.judge_evidence_sha256:
                    raise ValueError(
                        f"pristine judge evidence drifted: {call.simulation_id}"
                    )
                continue
            if payload.get("nativeness_info") is not None:
                _validate_nativeness_provenance(
                    payload["nativeness_info"], call.simulation_id
                )
            if payload.get("quality_info") is not None:
                _validate_quality_provenance(
                    payload["quality_info"], call.simulation_id
                )
            if payload.get("delivery_info") is not None:
                _validate_delivery_provenance(
                    payload["delivery_info"], call.simulation_id
                )


def _verify_judged_legacy(manifest: GenericJudgeManifest) -> None:
    """Require all three exact frozen contracts after the suite completes."""
    _verify_legacy_identity_audio(manifest)
    legacy_root = Path(manifest.workspace) / LEGACY_DIRNAME
    for cell in manifest.cells:
        cell_root = legacy_root / cell.source_relative_path
        for call in cell.calls:
            payload = json.loads((cell_root / call.simulation_path).read_text())
            _validate_nativeness(payload.get("nativeness_info"), call.simulation_id)
            _validate_quality(payload.get("quality_info"), call.simulation_id)
            _validate_delivery(payload.get("delivery_info"), call.simulation_id)


def prepare_frozen_judge_workspace(
    *,
    source_root: Path,
    workspace: Path,
    contract: GenericJudgeCohortContract = CANONICAL_GENERIC_COHORT,
) -> GenericJudgeManifest:
    """Atomically materialize mutable legacy copies after a strict preflight."""
    manifest = plan_frozen_judge_workspace(
        source_root=source_root, workspace=workspace, contract=contract
    )
    destination = Path(manifest.workspace)
    if destination.exists():
        manifest_path = destination / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ValueError(f"existing judge workspace has no manifest: {destination}")
        existing = GenericJudgeManifest.model_validate_json(manifest_path.read_text())
        if existing != manifest:
            raise ValueError("existing judge workspace manifest differs from preflight")
        _verify_sources(existing)
        _verify_legacy_materialization(existing)
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for cell in manifest.cells:
            _stage_cell(Path(manifest.source_root), temporary, cell)
        _write_json(temporary / MANIFEST_FILENAME, manifest)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _verify_sources(manifest)
    _verify_legacy_materialization(manifest)
    return manifest


def load_generic_judge_manifest(path: Path) -> GenericJudgeManifest:
    """Load either a workspace or its manifest and reproduce its fingerprint."""
    path = path.expanduser().resolve()
    manifest_path = path / MANIFEST_FILENAME if path.is_dir() else path
    manifest = GenericJudgeManifest.model_validate_json(manifest_path.read_text())
    expected = _sha256_value(
        {
            "cohort": manifest.cohort.model_dump(mode="json"),
            "judges": manifest.judges.model_dump(mode="json"),
            "cells": [cell.model_dump(mode="json") for cell in manifest.cells],
        }
    )
    if expected != manifest.cohort_fingerprint_sha256:
        raise ValueError("generic judge manifest fingerprint does not reproduce")
    return manifest


def verify_frozen_worktree(path: Path) -> FrozenWorktreeIdentity:
    """Require the exact frozen commit and a completely clean worktree."""
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"frozen judge worktree is missing: {path}")
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != FROZEN_GENERIC_SUITE_COMMIT:
        raise ValueError(
            f"frozen judge worktree is {commit}, expected {FROZEN_GENERIC_SUITE_COMMIT}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("frozen judge worktree has local changes")
    return FrozenWorktreeIdentity(path=str(path), commit=commit, clean=True)


def suite_execution_plan(
    *,
    manifest: GenericJudgeManifest,
    frozen_worktree: Path,
    python_executable: Path,
) -> SuiteExecutionPlan:
    """Build the exact resumable suite command without executing it."""
    frozen_worktree = frozen_worktree.expanduser().resolve()
    roots = [
        str(Path(manifest.workspace) / LEGACY_DIRNAME / cell.source_relative_path)
        for cell in manifest.cells
    ]
    command = [
        str(python_executable),
        "-m",
        "tau2.cli",
        "judges",
        "suite",
        *roots,
        "--concurrency",
        "10",
        "--text-processes",
        "8",
        "--audio-processes",
        "5",
        "--trials",
        "0",
    ]
    return SuiteExecutionPlan(
        frozen_worktree=str(frozen_worktree),
        frozen_commit=FROZEN_GENERIC_SUITE_COMMIT,
        pythonpath=str(frozen_worktree / "src"),
        command=command,
        calls=manifest.counts.calls,
        cells=manifest.counts.cells,
    )


def _frozen_environment(frozen_worktree: Path) -> dict[str, str]:
    environment = dict(os.environ)
    frozen_pythonpath = str(frozen_worktree / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        frozen_pythonpath
        if not existing_pythonpath
        else f"{frozen_pythonpath}{os.pathsep}{existing_pythonpath}"
    )
    return environment


def _materialize_roundtrip_probe(
    manifest: GenericJudgeManifest, temporary: Path
) -> list[Path]:
    """Copy only JSON into a disposable tree and mark every call for save proof."""
    roots: list[Path] = []
    legacy_root = Path(manifest.workspace) / LEGACY_DIRNAME
    for cell in manifest.cells:
        source_cell = legacy_root / cell.source_relative_path
        probe_cell = temporary / cell.source_relative_path
        probe_cell.mkdir(parents=True)
        shutil.copy2(source_cell / "results.json", probe_cell / "results.json")
        for call in cell.calls:
            source_sim = source_cell / call.simulation_path
            probe_sim = probe_cell / call.simulation_path
            payload = json.loads(source_sim.read_text())
            payload[ROUNDTRIP_PROBE_FIELD] = call.simulation_id
            _write_json(probe_sim, payload)
        roots.append(probe_cell)
    return roots


def probe_frozen_schema_roundtrip(
    *,
    manifest: GenericJudgeManifest,
    frozen_worktree: Path,
    python_executable: Path,
) -> FrozenSchemaRoundTripReport:
    """Run an API-free frozen load/save probe over all staged simulations.

    The frozen ``convert-results`` CLI round-trips each staged root from sharded
    to monolithic JSON and back.  A disposable marker proves that each file was
    actually serialized; the locked judge-evidence hash then proves that no
    transcript, tick, environment, or other judge input was lost.  No audio is
    copied, no judge is invoked, and no external API can be called.
    """
    identity = verify_frozen_worktree(frozen_worktree)
    _verify_sources(manifest)
    _verify_runnable_legacy(manifest)
    temporary = Path(
        tempfile.mkdtemp(prefix=".frozen-schema-probe.", dir=manifest.workspace)
    )
    try:
        roots = _materialize_roundtrip_probe(manifest, temporary)
        commands: list[list[str]] = []
        for root in roots:
            commands.extend(
                [
                    [
                        str(python_executable),
                        "-m",
                        "tau2.cli",
                        "convert-results",
                        str(root),
                        "--to",
                        "json",
                        "--no-backup",
                    ],
                    [
                        str(python_executable),
                        "-m",
                        "tau2.cli",
                        "convert-results",
                        str(root / "results.json"),
                        "--to",
                        "dir",
                        "--no-backup",
                    ],
                ]
            )
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=identity.path,
                env=_frozen_environment(Path(identity.path)),
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "API-free frozen-schema round-trip probe exited "
                    f"{completed.returncode}: {command}"
                )
        observed_calls = 0
        for cell, probe_cell in zip(manifest.cells, roots, strict=True):
            expected_ids = {call.simulation_id for call in cell.calls}
            observed_ids = {
                path.stem for path in (probe_cell / "simulations").glob("*.json")
            }
            if observed_ids != expected_ids:
                raise ValueError(
                    f"round-trip simulation inventory drifted: "
                    f"{cell.source_relative_path}"
                )
            for call in cell.calls:
                payload = json.loads((probe_cell / call.simulation_path).read_text())
                staged_payload = json.loads(
                    (
                        Path(manifest.workspace)
                        / LEGACY_DIRNAME
                        / cell.source_relative_path
                        / call.simulation_path
                    ).read_text()
                )
                if ROUNDTRIP_PROBE_FIELD in payload:
                    raise ValueError(
                        f"frozen serializer did not save call: {call.simulation_id}"
                    )
                if judge_evidence_sha256(payload) != judge_evidence_sha256(
                    staged_payload
                ):
                    drifted_fields = [
                        field
                        for field in JUDGE_EVIDENCE_FIELDS
                        if _sha256_value(payload.get(field))
                        != _sha256_value(staged_payload.get(field))
                    ]
                    raise ValueError(
                        f"round-trip judge evidence drifted: {call.simulation_id}; "
                        f"fields={drifted_fields}"
                    )
                observed_calls += 1
        if observed_calls != manifest.counts.calls:
            raise ValueError(
                f"round-trip covered {observed_calls} calls; "
                f"expected {manifest.counts.calls}"
            )
        _verify_sources(manifest)
        return FrozenSchemaRoundTripReport(
            frozen_worktree=identity.path,
            frozen_commit=identity.commit,
            commands=commands,
            cells=len(roots),
            calls=observed_calls,
            judge_evidence_mismatches=0,
            source_hash_mismatches=0,
            external_api_calls=False,
        )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def preflight_frozen_judge_suite(
    *,
    manifest: GenericJudgeManifest,
    frozen_worktree: Path,
    python_executable: Path,
) -> SuitePreflightReport:
    """Perform the complete API-free preflight and return the paid command."""
    roundtrip = probe_frozen_schema_roundtrip(
        manifest=manifest,
        frozen_worktree=frozen_worktree,
        python_executable=python_executable,
    )
    plan = suite_execution_plan(
        manifest=manifest,
        frozen_worktree=Path(roundtrip.frozen_worktree),
        python_executable=python_executable,
    )
    return SuitePreflightReport(plan=plan, frozen_schema_roundtrip=roundtrip)


def run_frozen_judge_suite(
    *,
    manifest: GenericJudgeManifest,
    frozen_worktree: Path,
    python_executable: Path,
) -> SuiteExecutionReceipt:
    """Execute/resume the exact frozen suite against isolated copies only."""
    preflight = preflight_frozen_judge_suite(
        manifest=manifest,
        frozen_worktree=frozen_worktree,
        python_executable=python_executable,
    )
    completed = subprocess.run(
        preflight.plan.command,
        cwd=preflight.plan.frozen_worktree,
        env=_frozen_environment(Path(preflight.plan.frozen_worktree)),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"frozen judge suite exited {completed.returncode}; rerun the same "
            "command to fill only remaining gaps"
        )
    _verify_sources(manifest)
    _verify_judged_legacy(manifest)
    receipt = SuiteExecutionReceipt(
        schema_version=GENERIC_COHORT_SCHEMA_VERSION,
        finished_at=_now(),
        plan=preflight.plan,
        frozen_schema_roundtrip=preflight.frozen_schema_roundtrip,
        returncode=0,
    )
    _write_json(Path(manifest.workspace) / EXECUTION_RECEIPT_FILENAME, receipt)
    return receipt


def _validate_nativeness_provenance(info: object, simulation_id: str) -> NativenessInfo:
    """Validate the frozen nativeness identity, including partial attempts."""
    parsed = NativenessInfo.model_validate(info)
    if (
        parsed.rubric_version != NATIVENESS_RUBRIC_VERSION
        or parsed.judge_model != NATIVENESS_MODEL
        or parsed.judge_args != NATIVENESS_ARGS
        or parsed.judge_prompt_version != NATIVENESS_PROMPT_VERSION
    ):
        raise ValueError(f"wrong frozen nativeness contract: {simulation_id}")
    return parsed


def _validate_nativeness(info: object, simulation_id: str) -> NativenessInfo:
    parsed = _validate_nativeness_provenance(info, simulation_id)
    if (
        parsed.num_errors != 0
        or parsed.num_deferred != 0
        or any(
            check.outcome in {JudgeOutcome.ERROR, JudgeOutcome.DEFERRED}
            for check in parsed.factor_checks
        )
    ):
        raise ValueError(f"incomplete frozen nativeness contract: {simulation_id}")
    return parsed


def _validate_quality_provenance(info: object, simulation_id: str) -> QualityInfo:
    """Validate exact v7/r4/v5 factor provenance, allowing retryable gaps."""
    parsed = QualityInfo.model_validate(info)
    expected_ids = QUALITY_DETERMINISTIC_FACTORS | frozenset(QUALITY_FACTOR_ARGS)
    by_id = {check.id: check for check in parsed.factor_checks}
    if (
        parsed.rubric_version != QUALITY_RUBRIC_VERSION
        or parsed.metrics_version != QUALITY_METRICS_VERSION
        or set(by_id) != expected_ids
        or len(parsed.factor_checks) != len(expected_ids)
    ):
        raise ValueError(f"wrong frozen quality contract: {simulation_id}")
    for factor_id in QUALITY_DETERMINISTIC_FACTORS:
        check = by_id[factor_id]
        if (
            check.evaluator != "deterministic"
            or check.judge_model is not None
            or check.judge_args is not None
            or check.judge_prompt_version is not None
        ):
            raise ValueError(f"wrong deterministic quality provenance: {simulation_id}")
    for factor_id, expected_args in QUALITY_FACTOR_ARGS.items():
        check = by_id[factor_id]
        # Hybrid factors can be deterministically preclassified with no model call.
        if check.judge_model is None:
            allowed_unstamped = QUALITY_PRECLASSIFIED_OUTCOMES[factor_id] | {
                JudgeOutcome.ERROR,
                JudgeOutcome.DEFERRED,
            }
            if (
                check.outcome not in allowed_unstamped
                or check.judge_args is not None
                or check.judge_prompt_version is not None
            ):
                raise ValueError(f"unstamped frozen quality factor: {simulation_id}")
            continue
        if (
            check.judge_model != QUALITY_MODEL
            or check.judge_args != expected_args
            or check.judge_prompt_version != QUALITY_PROMPT_VERSION
        ):
            raise ValueError(f"wrong frozen quality factor contract: {simulation_id}")
    return parsed


def _validate_quality(info: object, simulation_id: str) -> QualityInfo:
    parsed = _validate_quality_provenance(info, simulation_id)
    if (
        parsed.num_errors != 0
        or parsed.num_deferred != 0
        or any(
            check.outcome in {JudgeOutcome.ERROR, JudgeOutcome.DEFERRED}
            for check in parsed.factor_checks
        )
    ):
        raise ValueError(f"incomplete frozen quality contract: {simulation_id}")
    return parsed


def _validate_delivery_provenance(info: object, simulation_id: str) -> DeliveryInfo:
    """Validate exact Gemini v5 delivery provenance, including failed attempts."""
    parsed = DeliveryInfo.model_validate(info)
    if (
        parsed.judge_model != DELIVERY_MODEL
        or parsed.judge_args != DELIVERY_ARGS
        or parsed.judge_prompt_version != DELIVERY_PROMPT_VERSION
        or parsed.sample_rate != DELIVERY_SAMPLE_RATE
        or parsed.max_segments != DELIVERY_MAX_SEGMENTS
    ):
        raise ValueError(f"wrong frozen delivery contract: {simulation_id}")
    return parsed


def _validate_delivery(info: object, simulation_id: str) -> DeliveryInfo:
    parsed = _validate_delivery_provenance(info, simulation_id)
    rows = parsed.utterance_results
    if not rows or rows[0].utterance_idx != 0:
        raise ValueError(f"delivery inventory lacks greeting: {simulation_id}")
    indices = [row.utterance_idx for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f"duplicate delivery utterance index: {simulation_id}")
    if any(
        row.outcome not in {JudgeOutcome.PASS, JudgeOutcome.FAIL, JudgeOutcome.ERROR}
        for row in rows
    ):
        raise ValueError(f"unsupported delivery outcome: {simulation_id}")
    if parsed.num_judged != len(rows):
        raise ValueError(f"delivery judged count drifted: {simulation_id}")
    if parsed.num_errors != sum(row.outcome is JudgeOutcome.ERROR for row in rows):
        raise ValueError(f"delivery error count drifted: {simulation_id}")
    if parsed.num_flagged != sum(row.flag_for_review for row in rows):
        raise ValueError(f"delivery flagged count drifted: {simulation_id}")
    return parsed


def _validate_quality_summary(info: QualityInfo, simulation_id: str) -> None:
    """Require stored quality counters and score to reproduce exactly."""
    outcomes = Counter(check.outcome for check in info.factor_checks)
    expected = {
        JudgeOutcome.PASS: info.num_pass,
        JudgeOutcome.FAIL: info.num_fail,
        JudgeOutcome.NO_OPPORTUNITY: info.num_no_opportunity,
        JudgeOutcome.DEFERRED: info.num_deferred,
        JudgeOutcome.ERROR: info.num_errors,
    }
    if any(outcomes[outcome] != count for outcome, count in expected.items()):
        raise ValueError(f"quality counters drifted: {simulation_id}")
    scored = info.num_pass + info.num_fail
    expected_score = info.num_pass / scored if scored else None
    expected_coverage = scored / len(info.factor_checks)
    if (info.score is None) != (expected_score is None) or (
        info.score is not None
        and expected_score is not None
        and not math.isclose(info.score, expected_score, abs_tol=1e-12)
    ):
        raise ValueError(f"quality score drifted: {simulation_id}")
    if not math.isclose(info.score_coverage, expected_coverage, abs_tol=1e-12):
        raise ValueError(f"quality coverage drifted: {simulation_id}")


def _validate_paper_quality(info: object, simulation_id: str) -> QualityInfo:
    """Validate the exact seven-deterministic plus three-learned paper slice."""
    parsed = QualityInfo.model_validate(info)
    by_id = {check.id: check for check in parsed.factor_checks}
    if (
        parsed.rubric_version != QUALITY_RUBRIC_VERSION
        or parsed.metrics_version != QUALITY_METRICS_VERSION
        or set(by_id) != PAPER_QUALITY_FACTOR_IDS
        or len(parsed.factor_checks) != len(PAPER_QUALITY_FACTOR_IDS)
        or set(by_id) & EXCLUDED_QUALITY_FACTOR_IDS
    ):
        raise ValueError(f"wrong selected quality contract: {simulation_id}")
    for factor_id in QUALITY_DETERMINISTIC_FACTORS:
        check = by_id[factor_id]
        if (
            check.evaluator != "deterministic"
            or check.judge_model is not None
            or check.judge_args is not None
            or check.judge_prompt_version is not None
        ):
            raise ValueError(f"wrong deterministic quality provenance: {simulation_id}")
    for factor_id, expected_args in PAPER_QUALITY_FACTOR_ARGS.items():
        check = by_id[factor_id]
        expected_evaluator = (
            "hybrid"
            if factor_id in {"agent_caused_tool_error", "auth_arg_mismatch"}
            else "llm"
        )
        if check.evaluator != expected_evaluator:
            raise ValueError(f"wrong selected quality evaluator: {simulation_id}")
        if check.judge_model is None:
            if (
                check.outcome not in QUALITY_PRECLASSIFIED_OUTCOMES[factor_id]
                or check.judge_args is not None
                or check.judge_prompt_version is not None
            ):
                raise ValueError(f"unstamped selected quality factor: {simulation_id}")
        elif (
            check.judge_model != QUALITY_MODEL
            or check.judge_args != expected_args
            or check.judge_prompt_version != QUALITY_PROMPT_VERSION
        ):
            raise ValueError(f"wrong selected quality provenance: {simulation_id}")
    if (
        parsed.num_errors
        or parsed.num_deferred
        or any(
            check.outcome in {JudgeOutcome.ERROR, JudgeOutcome.DEFERRED}
            for check in parsed.factor_checks
        )
    ):
        raise ValueError(f"incomplete selected quality contract: {simulation_id}")
    _validate_quality_summary(parsed, simulation_id)
    return parsed


def _validate_nativeness_summary(info: NativenessInfo, simulation_id: str) -> None:
    """Require one selected nativeness slice to reproduce its stored summary."""
    scored_checks = [check for check in info.factor_checks if not check.shadow]
    outcomes = Counter(check.outcome for check in scored_checks)
    expected = {
        JudgeOutcome.PASS: info.num_pass,
        JudgeOutcome.FAIL: info.num_fail,
        JudgeOutcome.NO_OPPORTUNITY: info.num_no_opportunity,
        JudgeOutcome.DEFERRED: info.num_deferred,
        JudgeOutcome.ERROR: info.num_errors,
    }
    if any(outcomes[outcome] != count for outcome, count in expected.items()):
        raise ValueError(f"nativeness counters drifted: {simulation_id}")
    scored = info.num_pass + info.num_fail
    expected_score = info.num_pass / scored if scored else None
    expected_coverage = scored / len(scored_checks)
    if (info.score is None) != (expected_score is None) or (
        info.score is not None
        and expected_score is not None
        and not math.isclose(info.score, expected_score, abs_tol=1e-12)
    ):
        raise ValueError(f"nativeness score drifted: {simulation_id}")
    if not math.isclose(info.score_coverage, expected_coverage, abs_tol=1e-12):
        raise ValueError(f"nativeness coverage drifted: {simulation_id}")


def _validate_modal_nativeness(info: object, simulation_id: str) -> NativenessInfo:
    """Validate the exact Mandarin modal-particle diagnostic contract."""
    parsed = NativenessInfo.model_validate(info)
    by_id = {check.id: check for check in parsed.factor_checks}
    if (
        parsed.language != "zh"
        or parsed.script != "hans"
        or parsed.rubric_version != NATIVENESS_RUBRIC_VERSION
        or set(by_id) != PAPER_MODAL_FACTOR_IDS
        or len(parsed.factor_checks) != len(PAPER_MODAL_FACTOR_IDS)
    ):
        raise ValueError(f"wrong modal-particle contract: {simulation_id}")
    stamped = (
        parsed.judge_model == NATIVENESS_MODEL
        and parsed.judge_args == NATIVENESS_ARGS
        and parsed.judge_prompt_version == NATIVENESS_PROMPT_VERSION
    )
    unstamped_no_call = (
        parsed.judge_model is None
        and parsed.judge_args is None
        and parsed.judge_prompt_version is None
        and all(
            check.outcome is JudgeOutcome.NO_OPPORTUNITY
            and check.evidence is None
            and check.quote is None
            and check.observed_severity == 0
            and check.violation_count == 0
            and not check.unit_results
            for check in parsed.factor_checks
        )
    )
    if not (stamped or unstamped_no_call):
        raise ValueError(f"wrong modal-particle judge stamps: {simulation_id}")
    for factor_id in PAPER_MODAL_DETERMINISTIC_FACTOR_IDS:
        if by_id[factor_id].evaluation_level != "deterministic":
            raise ValueError(f"wrong modal deterministic provenance: {simulation_id}")
    if by_id["modal_particles"].evaluation_level != "call":
        raise ValueError(f"wrong modal-particle evaluation level: {simulation_id}")
    if (
        parsed.num_errors
        or parsed.num_deferred
        or any(
            check.outcome in {JudgeOutcome.ERROR, JudgeOutcome.DEFERRED}
            for check in parsed.factor_checks
        )
    ):
        raise ValueError(f"incomplete modal-particle contract: {simulation_id}")
    _validate_nativeness_summary(parsed, simulation_id)
    return parsed


def _validate_composed_delivery(
    info: object, simulation_id: str, language: str
) -> DeliveryInfo:
    """Validate complete frozen delivery output plus internal counters."""
    parsed = _validate_delivery(info, simulation_id)
    if parsed.language != language:
        raise ValueError(f"delivery language drifted: {simulation_id}")
    return parsed


def _manifest_lock_payload(manifest: GenericJudgeManifest) -> dict:
    """Return the shared cohort lock, excluding only its mutable workspace path."""
    payload = manifest.model_dump(mode="json")
    payload.pop("workspace")
    return payload


def _manifest_lock_sha256(manifest: GenericJudgeManifest) -> str:
    return _sha256_value(_manifest_lock_payload(manifest))


def _load_compose_manifest(workspace: Path) -> tuple[GenericJudgeManifest, Path]:
    manifest_path = workspace / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(f"judge workspace has no manifest: {workspace}")
    return (
        GenericJudgeManifest.model_validate_json(manifest_path.read_text()),
        manifest_path,
    )


def _verify_overlay_inventory(
    *, manifest: GenericJudgeManifest, workspace: Path
) -> None:
    """Verify an explicit legacy root without trusting manifest.workspace."""
    legacy_root = workspace / LEGACY_DIRNAME
    for cell in manifest.cells:
        cell_root = legacy_root / cell.source_relative_path
        metadata_path = cell_root / "results.json"
        if not metadata_path.is_file():
            raise ValueError(f"overlay cell is missing results.json: {cell_root}")
        metadata = json.loads(metadata_path.read_text())
        expected_ids = {call.simulation_id for call in cell.calls}
        observed_ids = {
            str(entry.get("id")) for entry in metadata.get("simulation_index", [])
        }
        observed_files = {
            path.stem for path in (cell_root / "simulations").glob("*.json")
        }
        if observed_ids != expected_ids or observed_files != expected_ids:
            raise ValueError(
                f"overlay inventory drifted at {workspace}: {cell.source_relative_path}"
            )
        for call in cell.calls:
            payload = json.loads((cell_root / call.simulation_path).read_text())
            if (
                payload.get("id") != call.simulation_id
                or str(payload.get("task_id")) != call.task_id
                or payload.get("trial") != call.trial
            ):
                raise ValueError(f"overlay identity drifted: {call.simulation_id}")
            if sha256_file(cell_root / call.audio_path) != call.audio_sha256:
                raise ValueError(f"overlay audio drifted: {call.simulation_id}")


def _compose_input_identity(
    *,
    role: Literal["quality", "delivery", "modal_particles"],
    imported_field: Literal["quality_info", "delivery_info", "nativeness_info"],
    imported_calls: int,
    workspace: Path,
    manifest: GenericJudgeManifest,
    manifest_path: Path,
) -> ComposeInputIdentity:
    declared = Path(manifest.workspace).expanduser().resolve()
    return ComposeInputIdentity(
        role=role,
        workspace=str(workspace),
        manifest_path=str(manifest_path),
        manifest_sha256=sha256_file(manifest_path),
        declared_workspace=str(declared),
        declared_workspace_matches=declared == workspace,
        cohort_lock_sha256=_manifest_lock_sha256(manifest),
        imported_field=imported_field,
        imported_calls=imported_calls,
    )


def _validate_compose_roots(
    *,
    source_root: Path,
    workspaces: tuple[Path, Path, Path],
    output_workspace: Path,
) -> None:
    if output_workspace == output_workspace.parent:
        raise ValueError("compose output cannot be a filesystem root")
    protected = (source_root, *workspaces)
    for path in protected:
        if output_workspace == path or output_workspace.is_relative_to(path):
            raise ValueError(f"compose output overlaps an input: {path}")
        if path.is_relative_to(output_workspace):
            raise ValueError(f"compose output would contain an input: {path}")


def _compose_cell(
    *,
    manifest: GenericJudgeManifest,
    cell: LockedJudgeCell,
    quality_workspace: Path,
    delivery_workspace: Path,
    modal_workspace: Path,
    merged_root: Path,
) -> tuple[int, int, int, int]:
    source_cell = Path(manifest.source_root) / cell.source_relative_path
    quality_cell = quality_workspace / LEGACY_DIRNAME / cell.source_relative_path
    delivery_cell = delivery_workspace / LEGACY_DIRNAME / cell.source_relative_path
    modal_cell = modal_workspace / LEGACY_DIRNAME / cell.source_relative_path
    merged_cell = merged_root / cell.source_relative_path
    index = []
    modal_calls = 0
    preserved_nativeness_calls = 0
    delivery_errors = 0
    for call in cell.calls:
        source_payload = json.loads((source_cell / call.simulation_path).read_text())
        quality_payload = json.loads((quality_cell / call.simulation_path).read_text())
        delivery_payload = json.loads(
            (delivery_cell / call.simulation_path).read_text()
        )
        quality = _validate_paper_quality(
            quality_payload.get("quality_info"), call.simulation_id
        )
        delivery = _validate_composed_delivery(
            delivery_payload.get("delivery_info"), call.simulation_id, cell.language
        )
        delivery_errors += delivery.num_errors
        merged_payload = dict(source_payload)
        merged_payload["quality_info"] = quality.model_dump(mode="json")
        merged_payload["delivery_info"] = delivery.model_dump(mode="json")
        if cell.language == "zh":
            modal_payload = json.loads((modal_cell / call.simulation_path).read_text())
            nativeness = _validate_modal_nativeness(
                modal_payload.get("nativeness_info"), call.simulation_id
            )
            merged_payload["nativeness_info"] = nativeness.model_dump(mode="json")
            modal_calls += 1
        else:
            if merged_payload.get("nativeness_info") != source_payload.get(
                "nativeness_info"
            ):
                raise ValueError(
                    f"source nativeness changed before compose: {call.simulation_id}"
                )
            preserved_nativeness_calls += 1
        if nonjudge_sha256(merged_payload) != call.nonjudge_sha256:
            raise ValueError(f"compose changed a non-judge field: {call.simulation_id}")
        simulation = SimulationRun.model_validate(merged_payload)
        _write_json(merged_cell / call.simulation_path, merged_payload)
        index.append(Results.index_entry(simulation).model_dump(mode="json"))
        _relative_symlink(
            source_cell / call.audio_path,
            merged_cell / call.audio_path,
        )
    metadata = json.loads((source_cell / "results.json").read_text())
    metadata["simulation_index"] = index
    _write_json(merged_cell / "results.json", metadata)
    return len(cell.calls), modal_calls, preserved_nativeness_calls, delivery_errors


def _verify_composed_cell(
    *,
    manifest: GenericJudgeManifest,
    cell: LockedJudgeCell,
    quality_workspace: Path,
    delivery_workspace: Path,
    modal_workspace: Path,
    merged_root: Path,
) -> int:
    source_cell = Path(manifest.source_root) / cell.source_relative_path
    quality_cell = quality_workspace / LEGACY_DIRNAME / cell.source_relative_path
    delivery_cell = delivery_workspace / LEGACY_DIRNAME / cell.source_relative_path
    modal_cell = modal_workspace / LEGACY_DIRNAME / cell.source_relative_path
    merged_cell = merged_root / cell.source_relative_path
    metadata = Results.load_metadata(merged_cell)
    if len(metadata.simulation_index or []) != len(cell.calls):
        raise ValueError(f"composed index count drifted: {cell.source_relative_path}")
    expected_ids = {call.simulation_id for call in cell.calls}
    if {entry.id for entry in metadata.simulation_index or []} != expected_ids:
        raise ValueError(
            f"composed index identity drifted: {cell.source_relative_path}"
        )
    merged_metadata = json.loads((merged_cell / "results.json").read_text())
    source_metadata = json.loads((source_cell / "results.json").read_text())
    if _results_nonindex_sha256(merged_metadata) != _results_nonindex_sha256(
        source_metadata
    ):
        raise ValueError(f"composed metadata drifted: {cell.source_relative_path}")
    delivery_errors = 0
    for call in cell.calls:
        source_payload = json.loads((source_cell / call.simulation_path).read_text())
        merged_payload = json.loads((merged_cell / call.simulation_path).read_text())
        if nonjudge_sha256(merged_payload) != call.nonjudge_sha256:
            raise ValueError(
                f"composed non-judge payload drifted: {call.simulation_id}"
            )
        quality_payload = json.loads((quality_cell / call.simulation_path).read_text())
        expected_quality = _validate_paper_quality(
            quality_payload.get("quality_info"), call.simulation_id
        )
        observed_quality = _validate_paper_quality(
            merged_payload.get("quality_info"), call.simulation_id
        )
        if observed_quality != expected_quality:
            raise ValueError(f"composed quality drifted: {call.simulation_id}")
        delivery_payload = json.loads(
            (delivery_cell / call.simulation_path).read_text()
        )
        expected_delivery = _validate_composed_delivery(
            delivery_payload.get("delivery_info"), call.simulation_id, cell.language
        )
        observed_delivery = _validate_composed_delivery(
            merged_payload.get("delivery_info"), call.simulation_id, cell.language
        )
        if observed_delivery != expected_delivery:
            raise ValueError(f"composed delivery drifted: {call.simulation_id}")
        delivery_errors += observed_delivery.num_errors
        if cell.language == "zh":
            modal_payload = json.loads((modal_cell / call.simulation_path).read_text())
            expected_modal = _validate_modal_nativeness(
                modal_payload.get("nativeness_info"), call.simulation_id
            )
            observed_modal = _validate_modal_nativeness(
                merged_payload.get("nativeness_info"), call.simulation_id
            )
            if observed_modal != expected_modal:
                raise ValueError(
                    f"composed modal verdict drifted: {call.simulation_id}"
                )
        elif merged_payload.get("nativeness_info") != source_payload.get(
            "nativeness_info"
        ):
            raise ValueError(
                f"composed Korean nativeness was overwritten: {call.simulation_id}"
            )
        SimulationRun.model_validate(merged_payload)
        if sha256_file(merged_cell / call.audio_path) != call.audio_sha256:
            raise ValueError(f"composed audio drifted: {call.simulation_id}")
    return delivery_errors


def _composed_fingerprints(
    manifest: GenericJudgeManifest, merged_root: Path
) -> tuple[str, str]:
    """Hash the complete promoted results and simulation JSON inventories."""
    results: list[tuple[str, str]] = []
    simulations: list[tuple[str, str, str]] = []
    for cell in manifest.cells:
        results_path = merged_root / cell.source_relative_path / "results.json"
        results.append((cell.source_relative_path, sha256_file(results_path)))
        for call in cell.calls:
            path = merged_root / cell.source_relative_path / call.simulation_path
            simulations.append(
                (cell.source_relative_path, call.simulation_id, sha256_file(path))
            )
    return _sha256_value(results), _sha256_value(simulations)


def verify_cross_workspace_composition(
    *,
    report: CrossWorkspaceComposeReport,
    manifest: GenericJudgeManifest,
    quality_workspace: Path,
    delivery_workspace: Path,
    modal_workspace: Path,
    merged_root: Path,
) -> None:
    """Re-verify every composed field against its explicit source workspace."""
    if report.cohort_lock_sha256 != _manifest_lock_sha256(manifest):
        raise ValueError("compose report cohort lock drifted")
    if report.cohort_fingerprint_sha256 != manifest.cohort_fingerprint_sha256:
        raise ValueError("compose report cohort fingerprint drifted")
    if report.merged_root != str(merged_root):
        raise ValueError("compose report merged-root identity drifted")
    delivery_errors = 0
    for cell in manifest.cells:
        delivery_errors += _verify_composed_cell(
            manifest=manifest,
            cell=cell,
            quality_workspace=quality_workspace,
            delivery_workspace=delivery_workspace,
            modal_workspace=modal_workspace,
            merged_root=merged_root,
        )
    if delivery_errors != report.delivery_error_utterances:
        raise ValueError("composed delivery ERROR count drifted")
    results_fingerprint, simulations_fingerprint = _composed_fingerprints(
        manifest, merged_root
    )
    if results_fingerprint != report.merged_results_fingerprint_sha256:
        raise ValueError("composed results fingerprint drifted")
    if simulations_fingerprint != report.merged_simulations_fingerprint_sha256:
        raise ValueError("composed simulations fingerprint drifted")


def compose_cross_workspace_judgments(
    *,
    quality_workspace: Path,
    delivery_workspace: Path,
    modal_workspace: Path,
    output_workspace: Path,
) -> CrossWorkspaceComposeReport:
    """Promote exact judge fields from three roots into one isolated corpus.

    The quality workspace is the authoritative source/cohort lock.  The
    delivery and modal manifests must describe the identical lock after
    removing only ``workspace``.  In particular, this function never follows
    the modal manifest's declared workspace: the production modal run was
    cloned from the quality workspace and intentionally retains that stale
    declaration.  Every input root is therefore explicit.

    Natural-word-choice v16 artifacts remain a separate utterance sidecar and
    are not embedded by this operation.
    """
    quality_workspace = quality_workspace.expanduser().resolve()
    delivery_workspace = delivery_workspace.expanduser().resolve()
    modal_workspace = modal_workspace.expanduser().resolve()
    output_workspace = output_workspace.expanduser().resolve()
    quality_manifest, quality_manifest_path = _load_compose_manifest(quality_workspace)
    delivery_manifest, delivery_manifest_path = _load_compose_manifest(
        delivery_workspace
    )
    modal_manifest, modal_manifest_path = _load_compose_manifest(modal_workspace)
    lock_sha256 = _manifest_lock_sha256(quality_manifest)
    if _manifest_lock_sha256(delivery_manifest) != lock_sha256:
        raise ValueError("delivery workspace does not share the quality cohort lock")
    if _manifest_lock_sha256(modal_manifest) != lock_sha256:
        raise ValueError("modal workspace does not share the quality cohort lock")
    if Path(quality_manifest.workspace).expanduser().resolve() != quality_workspace:
        raise ValueError("quality manifest declares a different workspace")
    if Path(delivery_manifest.workspace).expanduser().resolve() != delivery_workspace:
        raise ValueError("delivery manifest declares a different workspace")
    declared_modal = Path(modal_manifest.workspace).expanduser().resolve()
    if declared_modal not in {modal_workspace, quality_workspace}:
        raise ValueError("modal manifest has an unrecognized stale workspace")
    source_root = Path(quality_manifest.source_root).expanduser().resolve()
    _validate_compose_roots(
        source_root=source_root,
        workspaces=(quality_workspace, delivery_workspace, modal_workspace),
        output_workspace=output_workspace,
    )
    _verify_sources(quality_manifest)
    for workspace in (quality_workspace, delivery_workspace, modal_workspace):
        _verify_overlay_inventory(manifest=quality_manifest, workspace=workspace)
    inputs = (
        _compose_input_identity(
            role="quality",
            imported_field="quality_info",
            imported_calls=quality_manifest.counts.calls,
            workspace=quality_workspace,
            manifest=quality_manifest,
            manifest_path=quality_manifest_path,
        ),
        _compose_input_identity(
            role="delivery",
            imported_field="delivery_info",
            imported_calls=quality_manifest.counts.calls,
            workspace=delivery_workspace,
            manifest=delivery_manifest,
            manifest_path=delivery_manifest_path,
        ),
        _compose_input_identity(
            role="modal_particles",
            imported_field="nativeness_info",
            imported_calls=sum(
                len(cell.calls)
                for cell in quality_manifest.cells
                if cell.language == "zh"
            ),
            workspace=modal_workspace,
            manifest=modal_manifest,
            manifest_path=modal_manifest_path,
        ),
    )
    merged_root = output_workspace / MERGED_DIRNAME
    if output_workspace.exists():
        report_path = output_workspace / COMPOSE_REPORT_FILENAME
        if not report_path.is_file():
            raise ValueError("compose output exists without its typed report")
        report = CrossWorkspaceComposeReport.model_validate_json(
            report_path.read_text()
        )
        expected_input_hashes = tuple(row.manifest_sha256 for row in inputs)
        observed_input_hashes = tuple(row.manifest_sha256 for row in report.inputs)
        if expected_input_hashes != observed_input_hashes:
            raise ValueError("compose output was built from different input manifests")
        verify_cross_workspace_composition(
            report=report,
            manifest=quality_manifest,
            quality_workspace=quality_workspace,
            delivery_workspace=delivery_workspace,
            modal_workspace=modal_workspace,
            merged_root=merged_root,
        )
        return report
    output_workspace.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_workspace.name}.", dir=output_workspace.parent
        )
    )
    temporary_merged = temporary / MERGED_DIRNAME
    try:
        calls = 0
        modal_calls = 0
        preserved_nativeness_calls = 0
        delivery_errors = 0
        for cell in quality_manifest.cells:
            cell_calls, cell_modal, cell_preserved, cell_delivery_errors = (
                _compose_cell(
                    manifest=quality_manifest,
                    cell=cell,
                    quality_workspace=quality_workspace,
                    delivery_workspace=delivery_workspace,
                    modal_workspace=modal_workspace,
                    merged_root=temporary_merged,
                )
            )
            calls += cell_calls
            modal_calls += cell_modal
            preserved_nativeness_calls += cell_preserved
            delivery_errors += cell_delivery_errors
        results_fingerprint, simulations_fingerprint = _composed_fingerprints(
            quality_manifest, temporary_merged
        )
        report = CrossWorkspaceComposeReport(
            schema_version=COMPOSE_SCHEMA_VERSION,
            created_at=_now(),
            output_root=str(output_workspace),
            merged_root=str(merged_root),
            source_root=str(source_root),
            authoritative_manifest_path=str(quality_manifest_path),
            authoritative_manifest_sha256=sha256_file(quality_manifest_path),
            cohort_fingerprint_sha256=(quality_manifest.cohort_fingerprint_sha256),
            cohort_lock_sha256=lock_sha256,
            inputs=inputs,
            cells=len(quality_manifest.cells),
            calls=calls,
            imported_quality_calls=calls,
            imported_delivery_calls=calls,
            delivery_error_utterances=delivery_errors,
            imported_modal_nativeness_calls=modal_calls,
            preserved_source_nativeness_calls=preserved_nativeness_calls,
            quality_factor_ids=tuple(sorted(PAPER_QUALITY_FACTOR_IDS)),
            excluded_quality_factor_ids=tuple(sorted(EXCLUDED_QUALITY_FACTOR_IDS)),
            modal_factor_ids=tuple(sorted(PAPER_MODAL_FACTOR_IDS)),
            merged_results_fingerprint_sha256=results_fingerprint,
            merged_simulations_fingerprint_sha256=simulations_fingerprint,
            source_hash_mismatches=0,
            nonjudge_hash_mismatches=0,
            audio_hash_mismatches=0,
            refreshed_indexes=len(quality_manifest.cells),
            unselected_calls_not_copied=quality_manifest.counts.unselected_calls,
            natural_word_choice_storage="separate_sidecar",
        )
        temporary_report = report.model_copy(
            update={"merged_root": str(temporary_merged)}
        )
        verify_cross_workspace_composition(
            report=temporary_report,
            manifest=quality_manifest,
            quality_workspace=quality_workspace,
            delivery_workspace=delivery_workspace,
            modal_workspace=modal_workspace,
            merged_root=temporary_merged,
        )
        _write_json(temporary / COMPOSE_REPORT_FILENAME, report)
        temporary.replace(output_workspace)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_cross_workspace_composition(
        report=report,
        manifest=quality_manifest,
        quality_workspace=quality_workspace,
        delivery_workspace=delivery_workspace,
        modal_workspace=modal_workspace,
        merged_root=merged_root,
    )
    return report


def _relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def _merge_cell(
    *,
    manifest: GenericJudgeManifest,
    cell: LockedJudgeCell,
    temporary: Path,
) -> None:
    source_cell = Path(manifest.source_root) / cell.source_relative_path
    legacy_cell = Path(manifest.workspace) / LEGACY_DIRNAME / cell.source_relative_path
    merged_cell = temporary / cell.source_relative_path
    index = []
    for call in cell.calls:
        source_payload = json.loads((source_cell / call.simulation_path).read_text())
        legacy_payload = json.loads((legacy_cell / call.simulation_path).read_text())
        if (
            legacy_payload.get("id") != call.simulation_id
            or str(legacy_payload.get("task_id")) != call.task_id
            or legacy_payload.get("trial") != call.trial
        ):
            raise ValueError(f"judged legacy identity drifted: {call.simulation_id}")
        nativeness = _validate_nativeness(
            legacy_payload.get("nativeness_info"), call.simulation_id
        )
        quality = _validate_quality(
            legacy_payload.get("quality_info"), call.simulation_id
        )
        delivery = _validate_delivery(
            legacy_payload.get("delivery_info"), call.simulation_id
        )
        merged_payload = dict(source_payload)
        merged_payload.update(
            {
                "nativeness_info": nativeness.model_dump(mode="json"),
                "quality_info": quality.model_dump(mode="json"),
                "delivery_info": delivery.model_dump(mode="json"),
            }
        )
        if nonjudge_sha256(merged_payload) != call.nonjudge_sha256:
            raise ValueError(f"merge changed a non-judge field: {call.simulation_id}")
        simulation = SimulationRun.model_validate(merged_payload)
        _write_json(merged_cell / call.simulation_path, merged_payload)
        index.append(Results.index_entry(simulation).model_dump(mode="json"))
        _relative_symlink(
            legacy_cell / call.audio_path,
            merged_cell / call.audio_path,
        )
    metadata = json.loads((source_cell / "results.json").read_text())
    metadata["simulation_index"] = index
    _write_json(merged_cell / "results.json", metadata)


def _verify_merged(manifest: GenericJudgeManifest, merged_root: Path) -> None:
    for cell in manifest.cells:
        cell_root = merged_root / cell.source_relative_path
        source_cell = Path(manifest.source_root) / cell.source_relative_path
        metadata = Results.load_metadata(cell_root)
        if len(metadata.simulation_index or []) != len(cell.calls):
            raise ValueError(f"merged index count drifted: {cell.source_relative_path}")
        merged_metadata_payload = json.loads((cell_root / "results.json").read_text())
        source_metadata_payload = json.loads((source_cell / "results.json").read_text())
        if _results_nonindex_sha256(
            merged_metadata_payload
        ) != _results_nonindex_sha256(source_metadata_payload):
            raise ValueError(
                f"merged results metadata drifted: {cell.source_relative_path}"
            )
        expected_ids = {call.simulation_id for call in cell.calls}
        observed_ids = {entry.id for entry in metadata.simulation_index or []}
        if observed_ids != expected_ids:
            raise ValueError(
                f"merged index identity drifted: {cell.source_relative_path}"
            )
        for call in cell.calls:
            payload = json.loads((cell_root / call.simulation_path).read_text())
            if nonjudge_sha256(payload) != call.nonjudge_sha256:
                raise ValueError(
                    f"merged non-judge payload drifted: {call.simulation_id}"
                )
            _validate_nativeness(payload.get("nativeness_info"), call.simulation_id)
            _validate_quality(payload.get("quality_info"), call.simulation_id)
            _validate_delivery(payload.get("delivery_info"), call.simulation_id)
            if sha256_file(cell_root / call.audio_path) != call.audio_sha256:
                raise ValueError(f"merged audio drifted: {call.simulation_id}")


def _verify_final_window_merged(
    manifest: GenericJudgeManifest, merged_root: Path
) -> None:
    """Verify exactly the locked fields consumed by the final-window filter."""
    for cell in manifest.cells:
        cell_root = merged_root / cell.source_relative_path
        source_cell = Path(manifest.source_root) / cell.source_relative_path
        metadata = Results.load_metadata(cell_root)
        expected_ids = {call.simulation_id for call in cell.calls}
        if (
            len(metadata.simulation_index or []) != len(cell.calls)
            or {entry.id for entry in metadata.simulation_index or []} != expected_ids
        ):
            raise ValueError(
                f"final-window input index drifted: {cell.source_relative_path}"
            )
        merged_metadata = json.loads((cell_root / "results.json").read_text())
        source_metadata = json.loads((source_cell / "results.json").read_text())
        if _results_nonindex_sha256(merged_metadata) != _results_nonindex_sha256(
            source_metadata
        ):
            raise ValueError(
                f"final-window input metadata drifted: {cell.source_relative_path}"
            )
        for call in cell.calls:
            payload = json.loads((cell_root / call.simulation_path).read_text())
            if nonjudge_sha256(payload) != call.nonjudge_sha256:
                raise ValueError(
                    f"final-window input non-judge payload drifted: "
                    f"{call.simulation_id}"
                )
            _validate_delivery(payload.get("delivery_info"), call.simulation_id)
            if sha256_file(cell_root / call.audio_path) != call.audio_sha256:
                raise ValueError(
                    f"final-window input audio drifted: {call.simulation_id}"
                )


def merge_frozen_judgments(*, manifest: GenericJudgeManifest) -> GenericMergeReport:
    """Import only the three judge siblings into a new current-schema corpus."""
    _verify_sources(manifest)
    _verify_judged_legacy(manifest)
    destination = Path(manifest.workspace) / MERGED_DIRNAME
    report_path = Path(manifest.workspace) / MERGE_REPORT_FILENAME
    if destination.exists():
        _verify_merged(manifest, destination)
        if not report_path.is_file():
            raise ValueError("merged corpus exists without its merge report")
        return GenericMergeReport.model_validate_json(report_path.read_text())
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{MERGED_DIRNAME}.", dir=Path(manifest.workspace))
    )
    try:
        for cell in manifest.cells:
            _merge_cell(manifest=manifest, cell=cell, temporary=temporary)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _verify_merged(manifest, destination)
    report = GenericMergeReport(
        schema_version=GENERIC_MERGE_SCHEMA_VERSION,
        created_at=_now(),
        output_root=str(destination),
        cells=manifest.counts.cells,
        calls=manifest.counts.calls,
        imported_fields=JUDGE_FIELDS,
        source_hash_mismatches=0,
        nonjudge_hash_mismatches=0,
        refreshed_indexes=manifest.counts.cells,
    )
    _write_json(report_path, report)
    return report


def _clip_duration_seconds(
    wav_path: Path,
    span: tuple[int, int],
    tick_seconds: float,
) -> float:
    with wave.open(str(wav_path), "rb") as audio:
        rate = audio.getframerate()
        total_frames = audio.getnframes()
    start_tick, end_tick = span
    start_frame = round(start_tick * tick_seconds * rate)
    end_frame = min(round((end_tick + 1) * tick_seconds * rate), total_frames)
    if start_frame >= total_frames or end_frame <= start_frame:
        raise ValueError(f"tick span {span} falls outside {wav_path}")
    return (end_frame - start_frame) / rate


def _tick_seconds(simulation: SimulationRun) -> float:
    values = {
        tick.tick_duration_seconds
        for tick in simulation.ticks or []
        if tick.tick_duration_seconds is not None
    }
    if len(values) != 1:
        raise ValueError(f"simulation has no unique tick duration: {simulation.id}")
    return float(next(iter(values)))


def build_final_window_exclusions(
    *,
    manifest: GenericJudgeManifest,
    output: Path,
    canonical_sidecar: Path | None = None,
    merged_root: Path | None = None,
) -> GeneratedFinalWindowExclusionManifest:
    """Build the v1 exclusion sidecar, optionally replacing the old paper slice.

    With ``canonical_sidecar``, Korean/Mandarin retail rows from the original
    paper CSV are removed and replaced with the corrected calls' rows.  The
    resulting columns exactly match the Experience loader's canonical schema.
    Without it, the artifact contains only the corrected replacement slice.
    """
    _verify_sources(manifest)
    if merged_root is None:
        merged_root = Path(manifest.workspace) / MERGED_DIRNAME
        _verify_merged(manifest, merged_root)
    else:
        merged_root = merged_root.expanduser().resolve()
        _verify_final_window_merged(manifest, merged_root)
    replacement_rows: list[FinalWindowExclusionRow] = []
    results_hash_lines: list[str] = []
    for cell in manifest.cells:
        cell_root = merged_root / cell.source_relative_path
        results_path = cell_root / "results.json"
        results_hash_lines.append(
            f"{cell.source_relative_path}\t{sha256_file(results_path)}"
        )
        for call in cell.calls:
            simulation = SimulationRun.model_validate_json(
                (cell_root / call.simulation_path).read_text()
            )
            delivery = _validate_delivery(simulation.delivery_info, simulation.id)
            spans = agent_utterance_tick_spans(simulation)
            tick_seconds = _tick_seconds(simulation)
            wav_path = cell_root / call.audio_path
            for utterance in delivery.utterance_results:
                if not 0 <= utterance.utterance_idx < len(spans):
                    raise ValueError(
                        f"delivery utterance index outside tick spans: {simulation.id}"
                    )
                duration = _clip_duration_seconds(
                    wav_path, spans[utterance.utterance_idx], tick_seconds
                )
                for finding_index, finding in enumerate(utterance.findings):
                    if finding.axis != "fidelity" or not is_in_final_utterance_window(
                        finding.time_range, duration
                    ):
                        continue
                    bounds = time_range_bounds_seconds(finding.time_range)
                    assert bounds is not None and finding.time_range is not None
                    replacement_rows.append(
                        FinalWindowExclusionRow(
                            sim_id=simulation.id,
                            language=cell.language,
                            domain="retail",
                            system=PAPER_SYSTEM_LABELS.get(
                                cell.system_slug, cell.system_slug
                            ),
                            utterance_idx=utterance.utterance_idx,
                            finding_index=finding_index,
                            severity=finding.severity,
                            time_range=finding.time_range,
                            clip_duration_seconds=duration,
                            span_start_seconds=bounds[0],
                            span_end_seconds=bounds[1],
                        )
                    )
    canonical_hash = None
    if canonical_sidecar is None:
        rows = replacement_rows
        cohort = "tau-multilingual corrected ko/zh retail trial-0 calls"
    else:
        canonical_sidecar = canonical_sidecar.expanduser().resolve()
        canonical_manifest_path = canonical_sidecar.with_suffix(".json")
        canonical_csv_sha256 = sha256_file(canonical_sidecar)
        canonical_manifest_sha256 = sha256_file(canonical_manifest_path)
        if _uses_canonical_contract(manifest.cohort) and (
            canonical_csv_sha256 != CANONICAL_FINAL_WINDOW_CSV_SHA256
            or canonical_manifest_sha256 != CANONICAL_FINAL_WINDOW_MANIFEST_SHA256
        ):
            raise ValueError("paper canonical final-window sidecar identity drifted")
        with canonical_sidecar.open(newline="") as handle:
            canonical_rows = [
                FinalWindowExclusionRow.model_validate(row)
                for row in csv.DictReader(handle)
            ]
        if (
            _uses_canonical_contract(manifest.cohort)
            and len(canonical_rows) != CANONICAL_FINAL_WINDOW_ROWS
        ):
            raise ValueError(
                f"paper canonical final-window sidecar has {len(canonical_rows)} "
                f"rows; expected {CANONICAL_FINAL_WINDOW_ROWS}"
            )
        canonical_keys = {
            (row.sim_id, row.utterance_idx, row.finding_index) for row in canonical_rows
        }
        if len(canonical_keys) != len(canonical_rows):
            raise ValueError("canonical final-window sidecar has duplicate rows")
        canonical_manifest = json.loads(canonical_manifest_path.read_text())
        if (
            canonical_manifest.get("filter_version")
            != DELIVERY_FIDELITY_FINDING_FILTER_VERSION
            or float(canonical_manifest.get("window_seconds", -1))
            != DEFAULT_DELIVERY_FIDELITY_END_EXCLUSION_SECONDS
            or int(canonical_manifest.get("excluded_findings", -1))
            != len(canonical_rows)
        ):
            raise ValueError("canonical final-window sidecar manifest drifted")
        retained = [
            row
            for row in canonical_rows
            if not (row.language in {"ko", "zh"} and row.domain == "retail")
        ]
        rows = retained + replacement_rows
        cohort = "tau-multilingual corrected paper trial-0 calls"
        canonical_hash = _sha256_value(
            {
                "csv": canonical_csv_sha256,
                "manifest": canonical_manifest_sha256,
            }
        )
    rows.sort(
        key=lambda row: (
            row.language,
            row.domain,
            row.system,
            row.sim_id,
            row.utterance_idx,
            row.finding_index,
        )
    )
    keys = {(row.sim_id, row.utterance_idx, row.finding_index) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("hybrid final-window sidecar has duplicate rows")
    by_language = Counter(row.language for row in rows)
    by_system = Counter(row.system for row in rows)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(FinalWindowExclusionRow.model_fields),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(row.model_dump(mode="json") for row in rows)
    temporary.replace(output)
    sidecar = GeneratedFinalWindowExclusionManifest(
        artifact=output.name,
        filter_version=DELIVERY_FIDELITY_FINDING_FILTER_VERSION,
        window_seconds=DEFAULT_DELIVERY_FIDELITY_END_EXCLUSION_SECONDS,
        cohort=cohort,
        excluded_findings=len(rows),
        excluded_severity_2plus_findings=sum(row.severity >= 2 for row in rows),
        by_language=dict(sorted(by_language.items())),
        by_system=dict(sorted(by_system.items())),
        generation_results_files_sha256=_sha256_value(
            {
                "canonical_sidecar_sha256": canonical_hash,
                "replacement_results": results_hash_lines,
            }
        ),
        replacement_cohort_fingerprint_sha256=(manifest.cohort_fingerprint_sha256),
        replacement_calls=manifest.counts.calls,
        canonical_sidecar_sha256=canonical_hash,
    )
    _write_json(output.with_suffix(".json"), sidecar)
    return sidecar


def build_composed_final_window_exclusions(
    *,
    quality_workspace: Path,
    delivery_workspace: Path,
    modal_workspace: Path,
    composed_workspace: Path,
    output: Path,
    canonical_sidecar: Path | None = None,
) -> GeneratedFinalWindowExclusionManifest:
    """Verify a split composition and build its deterministic v1 sidecar."""
    composed_workspace = composed_workspace.expanduser().resolve()
    if not composed_workspace.is_dir():
        raise ValueError("composed workspace must already exist")
    report = compose_cross_workspace_judgments(
        quality_workspace=quality_workspace,
        delivery_workspace=delivery_workspace,
        modal_workspace=modal_workspace,
        output_workspace=composed_workspace,
    )
    manifest, _manifest_path = _load_compose_manifest(
        quality_workspace.expanduser().resolve()
    )
    return build_final_window_exclusions(
        manifest=manifest,
        output=output,
        canonical_sidecar=canonical_sidecar,
        merged_root=Path(report.merged_root),
    )


__all__ = [
    "CANONICAL_FINAL_WINDOW_CSV_SHA256",
    "CANONICAL_FINAL_WINDOW_MANIFEST_SHA256",
    "CANONICAL_FINAL_WINDOW_ROWS",
    "CANONICAL_GENERIC_COHORT",
    "COMPOSE_REPORT_FILENAME",
    "CrossWorkspaceComposeReport",
    "EXCLUDED_QUALITY_FACTOR_IDS",
    "FROZEN_GENERIC_SUITE_COMMIT",
    "FinalWindowExclusionManifest",
    "FinalWindowExclusionRow",
    "GeneratedFinalWindowExclusionManifest",
    "FrozenGenericJudgeContract",
    "FrozenSchemaRoundTripReport",
    "FrozenWorktreeIdentity",
    "GenericJudgeCell",
    "GenericJudgeCohortContract",
    "GenericJudgeManifest",
    "GenericMergeReport",
    "PAPER_MODAL_FACTOR_IDS",
    "PAPER_QUALITY_FACTOR_IDS",
    "SuiteExecutionPlan",
    "SuiteExecutionReceipt",
    "SuitePreflightReport",
    "build_composed_final_window_exclusions",
    "build_final_window_exclusions",
    "canonical_cohort_fingerprint_sha256",
    "canonical_results_files_sha256",
    "compose_cross_workspace_judgments",
    "load_generic_judge_manifest",
    "merge_frozen_judgments",
    "nonjudge_sha256",
    "plan_frozen_judge_workspace",
    "preflight_frozen_judge_suite",
    "prepare_frozen_judge_workspace",
    "probe_frozen_schema_roundtrip",
    "run_frozen_judge_suite",
    "sha256_file",
    "suite_execution_plan",
    "verify_cross_workspace_composition",
    "verify_frozen_worktree",
]
