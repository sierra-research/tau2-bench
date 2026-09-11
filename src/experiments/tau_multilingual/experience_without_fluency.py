"""Recompute tau-Multilingual Interaction and utterance Experience.

The analysis reads the frozen trial-0 cohort. Interaction is the descriptive
equal-weight mean of five call-level component failure rates. Experience is the
higher-is-better fraction of aligned agent utterances that pass both the frozen
combined naturalness judge and the material speech-fidelity check. Multiple
findings and cross-axis failures on one utterance are counted once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.data_model.simulation import JudgeOutcome
from tau2.judges.nativeness.exclusions import (
    BENCHMARK_TRANSFER_MESSAGE,
    is_benchmark_transfer_fragment,
)
from tau2.judges.nativeness.factors import (
    enabled_deterministic_factor_ids_for,
    judge_factors_for,
)
from tau2.judges.nativeness.paper_trial import (
    CANONICAL_EXPERIENCE_PATH,
    ExperienceSource,
    TrialRunManifest,
    TrialSimulationArtifact,
    TrialSourceIdentity,
    _aggregate_trial,
    frozen_judge_spec,
)

LANGUAGES = ("en", "es", "pt", "hi", "ko", "zh")
DOMAINS = ("airline", "retail", "telecom")
SYSTEMS = (
    "OpenAI minimal",
    "OpenAI xhigh",
    "Gemini minimal",
    "Gemini high",
    "xAI",
)
SYSTEM_SLUGS = {
    "OpenAI minimal": "openai_minimal",
    "OpenAI xhigh": "openai_xhigh",
    "Gemini minimal": "gemini_minimal",
    "Gemini high": "gemini_high",
    "xAI": "xai_provider_default",
}
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "hi": "Hindi",
    "ko": "Korean",
    "zh": "Mandarin",
}

TOOL_USE_FACTOR_IDS = {
    "incorrect_tool_parameters",
    "auth_arg_mismatch",
    "agent_caused_tool_error",
}
INTERACTION_COMPONENT_FACTORS = {
    "nonresponse": {"responsiveness", "yielding"},
    "interruption": {"inappropriate_interruption"},
    "selectivity": {
        "backchannel_selectivity",
        "vocal_tic_selectivity",
        "non_directed_selectivity",
    },
    "monologue": {"monologue"},
    "tool_use": TOOL_USE_FACTOR_IDS,
}
UTTERANCE_FLUENCY_FACTOR_ID = "natural_word_choice"
LOCALIZATION_DIAGNOSTIC_FACTOR_IDS = {
    "regional_consistency",
    "modal_particles",
    "name_address_conventions",
    "email_symbol_verbalization",
    "gender_agreement",
    "counting_units",
    "register_formality",
}
ALL_FLUENCY_FACTOR_IDS = frozenset(
    {UTTERANCE_FLUENCY_FACTOR_ID} | LOCALIZATION_DIAGNOSTIC_FACTOR_IDS
)

GENDER_OVERRIDE_PATHS = {
    language: Path(f"data/analysis/{language}_gender_v3_male_full_2026-09-03.csv")
    for language in ("pt", "hi")
}
FIDELITY_END_EXCLUSIONS_PATH = Path(
    "data/analysis/tau_multilingual_fidelity_final_second_exclusions_2026-09-03.csv"
)


class GenderOverrideRow(BaseModel):
    """One v3 gender verdict aligned to a frozen paper call."""

    sim_id: str = Field(description="Simulation identifier in the paper cohort.")
    outcome: JudgeOutcome = Field(description="Replacement gender-agreement verdict.")


class FidelityEndExclusionRow(BaseModel):
    """One frozen finding excluded by the runtime final-window rule."""

    sim_id: str = Field(description="Frozen simulation id.")
    language: str = Field(description="ISO language code.")
    domain: str = Field(description="Benchmark domain.")
    system: str = Field(description="Paper system label.")
    utterance_idx: int = Field(description="Delivery utterance index.")
    finding_index: int = Field(description="Index within findings.")
    severity: int = Field(description="Stored finding severity.")
    time_range: str = Field(description="Raw judge time range.")
    clip_duration_seconds: float = Field(description="Exact judged clip duration.")
    span_start_seconds: float = Field(description="Parsed approximate span start.")
    span_end_seconds: float = Field(description="Parsed approximate span end.")


class FidelityEndExclusionManifest(BaseModel):
    """Provenance and counts for the frozen-paper exclusion sidecar."""

    artifact: str
    filter_version: str
    window_seconds: float
    cohort: str
    excluded_findings: int
    excluded_severity_2plus_findings: int
    by_language: dict[str, int]
    by_system: dict[str, int]
    results_files_sha256: str


class NaturalnessUtteranceVerdict(BaseModel):
    """One atomic combined-naturalness verdict used in Experience."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_index: Annotated[int, Field(ge=0, description="Nativeness turn index.")]
    input_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Atomic input hash.")
    ]
    text: Annotated[str, Field(description="Delivered text judged for fluency.")]
    outcome: Annotated[
        Literal[
            JudgeOutcome.PASS,
            JudgeOutcome.FAIL,
            JudgeOutcome.NO_OPPORTUNITY,
        ],
        Field(description="Completed atomic combined-naturalness outcome."),
    ]


class NaturalnessSidecarCall(BaseModel):
    """One validated call and its atomic combined-naturalness verdicts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_results_path: Annotated[
        str, Field(description="Repository-relative source results path.")
    ]
    source_results_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Source results hash.")
    ]
    source_simulation_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Source simulation hash.")
    ]
    simulation_id: Annotated[str, Field(description="Source simulation id.")]
    task_id: Annotated[str, Field(description="Source task id.")]
    language: Annotated[str, Field(description="ISO language code.")]
    domain: Annotated[str, Field(description="Benchmark domain.")]
    outcome: Annotated[
        Literal[
            JudgeOutcome.PASS,
            JudgeOutcome.FAIL,
            JudgeOutcome.NO_OPPORTUNITY,
        ],
        Field(description="Completed call-level combined-naturalness outcome."),
    ]
    utterance_results: Annotated[
        list[NaturalnessUtteranceVerdict],
        Field(description="Ordered atomic verdicts in the call rollup."),
    ]

    @property
    def utterances(self) -> int:
        """Return the number of atomic verdicts in this call."""
        return len(self.utterance_results)

    @property
    def utterance_input_sha256s(self) -> list[str]:
        """Return ordered atomic input hashes for provenance validation."""
        return [result.input_sha256 for result in self.utterance_results]


class UtteranceExperienceCounts(BaseModel):
    """Deduplicated utterance-level Experience sufficient counts for one call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    eligible_utterances: Annotated[
        int, Field(ge=0, description="Utterances scored by both judges.")
    ]
    fluency_failures: Annotated[
        int, Field(ge=0, description="Eligible utterances failing naturalness.")
    ]
    speech_fidelity_failures: Annotated[
        int, Field(ge=0, description="Eligible utterances failing speech fidelity.")
    ]
    overlapping_failures: Annotated[
        int, Field(ge=0, description="Eligible utterances failing both components.")
    ]
    experience_failures: Annotated[
        int, Field(ge=0, description="Unique eligible utterances failing either.")
    ]
    pinned_greetings_excluded: Annotated[
        int, Field(ge=0, description="Injected delivery greetings removed before join.")
    ]
    unmatched_naturalness_utterances: Annotated[
        int, Field(ge=0, description="Naturalness rows without a delivery match.")
    ]
    unmatched_delivery_utterances: Annotated[
        int, Field(ge=0, description="Delivery rows without a naturalness match.")
    ]
    fidelity_findings_excluded: Annotated[
        int, Field(ge=0, description="Frozen final-window findings removed.")
    ]

    @model_validator(mode="after")
    def _counts_are_consistent(self) -> "UtteranceExperienceCounts":
        component_union = (
            self.fluency_failures
            + self.speech_fidelity_failures
            - self.overlapping_failures
        )
        if self.experience_failures != component_union:
            raise ValueError("Experience failures do not equal the component union")
        if any(
            count > self.eligible_utterances
            for count in (
                self.fluency_failures,
                self.speech_fidelity_failures,
                self.overlapping_failures,
                self.experience_failures,
            )
        ):
            raise ValueError("Experience failure counts exceed eligible utterances")
        return self


class NaturalnessSidecarInput(BaseModel):
    """Typed, internally consistent completed replay consumed by this analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    root: Annotated[Path, Field(description="Resolved sidecar artifact root.")]
    manifest_path: Annotated[Path, Field(description="Resolved sidecar manifest path.")]
    manifest_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Manifest byte hash.")
    ]
    manifest: Annotated[
        TrialRunManifest, Field(description="Completed replay manifest.")
    ]
    calls: Annotated[
        list[NaturalnessSidecarCall],
        Field(description="Exact call override inventory."),
    ]

    @model_validator(mode="after")
    def _calls_are_unique(self) -> "NaturalnessSidecarInput":
        keys = [(call.source_results_path, call.simulation_id) for call in self.calls]
        if len(keys) != len(set(keys)):
            raise ValueError("naturalness sidecar contains duplicate call identities")
        return self


class NaturalnessSidecarProvenance(BaseModel):
    """Portable provenance for the sidecar actually joined into the analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Annotated[str, Field(description="Input sidecar path.")]
    manifest_sha256: Annotated[str, Field(description="Sidecar manifest hash.")]
    identity_sha256: Annotated[str, Field(description="Replay identity hash.")]
    work_fingerprint_sha256: Annotated[
        str, Field(description="Replay work fingerprint.")
    ]
    calls: Annotated[int, Field(ge=0, description="Joined call overrides.")]
    factor_id: Literal["natural_word_choice"]
    prompt_version: Annotated[str, Field(description="Judge wrapper prompt version.")]
    rubric_version: Annotated[str, Field(description="Nativeness rubric version.")]


class InteractionComponentRates(BaseModel):
    """Call-level Interaction component failure percentages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nonresponse: Annotated[
        float, Field(ge=0, le=100, description="Non-response call failure percent.")
    ]
    interruption: Annotated[
        float, Field(ge=0, le=100, description="Interruption call failure percent.")
    ]
    selectivity: Annotated[
        float, Field(ge=0, le=100, description="Selectivity call failure percent.")
    ]
    monologue: Annotated[
        float, Field(ge=0, le=100, description="Monologue call failure percent.")
    ]
    tool_use: Annotated[
        float, Field(ge=0, le=100, description="Tool-use call failure percent.")
    ]


class ExperienceDenominators(BaseModel):
    """Exact denominators and failure counts for one reporting cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    interaction_calls: Annotated[
        int,
        Field(
            ge=0,
            description="Calls with at least one scored Interaction component.",
        ),
    ]
    experience_utterances: Annotated[
        int, Field(ge=0, description="Utterances scored by both Experience judges.")
    ]
    fluency_failures: Annotated[
        int, Field(ge=0, description="Eligible utterances failing Fluency.")
    ]
    speech_fidelity_failures: Annotated[
        int, Field(ge=0, description="Eligible utterances failing Speech fidelity.")
    ]
    overlapping_failures: Annotated[
        int, Field(ge=0, description="Eligible utterances failing both components.")
    ]
    experience_failures: Annotated[
        int, Field(ge=0, description="Unique utterances failing either component.")
    ]


class LanguageSystemSummary(BaseModel):
    """Metrics for one language-system cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_calls: Annotated[int, Field(ge=0, description="Calls in the cell.")]
    latency_seconds: Annotated[
        float | None,
        Field(
            ge=0,
            description=(
                "Equal-weight mean of event-weighted response and yield latency."
            ),
        ),
    ]
    interaction_failure: Annotated[
        float | None,
        Field(
            ge=0,
            le=100,
            description=(
                "Descriptive equal-weight mean of call-level Interaction component "
                "failure percentages."
            ),
        ),
    ]
    interaction_components: Annotated[
        InteractionComponentRates,
        Field(description="Call-level component failure percentages."),
    ]
    experience: Annotated[
        float | None,
        Field(ge=0, le=100, description="Utterance Experience pass percent."),
    ]
    experience_failure: Annotated[
        float | None,
        Field(ge=0, le=100, description="Deduplicated Experience failure percent."),
    ]
    fluency_failure: Annotated[
        float | None,
        Field(ge=0, le=100, description="Utterance Fluency failure percent."),
    ]
    speech_fidelity_failure: Annotated[
        float | None,
        Field(ge=0, le=100, description="Utterance Speech-fidelity failure percent."),
    ]
    overlapping_failure: Annotated[
        float | None,
        Field(ge=0, le=100, description="Cross-component overlap percent."),
    ]
    localization_diagnostics: Annotated[
        dict[str, float | None],
        Field(description="Separate retained localization diagnostic percentages."),
    ]
    denominators: Annotated[
        ExperienceDenominators,
        Field(description="Exact call and utterance sufficient counts."),
    ]


class ProviderSummary(BaseModel):
    """Language-macro metrics for one provider configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_calls: Annotated[int, Field(ge=0, description="Calls for the provider.")]
    latency_seconds: Annotated[
        float,
        Field(
            ge=0,
            description="Six-language macro mean response/yield latency.",
        ),
    ]
    mean_call_duration_minutes: Annotated[
        float, Field(ge=0, description="Mean complete-cohort call duration.")
    ]
    interaction_failure: Annotated[
        float,
        Field(
            ge=0,
            le=100,
            description="Six-language descriptive Interaction composite percent.",
        ),
    ]
    interaction_components: Annotated[
        InteractionComponentRates,
        Field(description="Six-language component macro percentages."),
    ]
    experience: Annotated[
        float, Field(ge=0, le=100, description="Five-language Experience percent.")
    ]
    experience_failure: Annotated[
        float, Field(ge=0, le=100, description="Five-language union failure percent.")
    ]
    fluency_failure: Annotated[
        float, Field(ge=0, le=100, description="Five-language Fluency percent.")
    ]
    speech_fidelity_failure: Annotated[
        float, Field(ge=0, le=100, description="Five-language fidelity percent.")
    ]
    overlapping_failure: Annotated[
        float, Field(ge=0, le=100, description="Five-language overlap percent.")
    ]


class LanguageSummary(BaseModel):
    """System-macro metrics for one language."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    interaction_failure: Annotated[
        float | None,
        Field(
            ge=0,
            le=100,
            description="Descriptive Interaction composite failure percent.",
        ),
    ]
    experience: Annotated[
        float | None, Field(ge=0, le=100, description="Experience pass percent.")
    ]
    experience_failure: Annotated[
        float | None, Field(ge=0, le=100, description="Experience failure percent.")
    ]
    fluency_failure: Annotated[
        float | None, Field(ge=0, le=100, description="Fluency failure percent.")
    ]
    speech_fidelity_failure: Annotated[
        float | None,
        Field(ge=0, le=100, description="Speech-fidelity failure percent."),
    ]
    overlapping_failure: Annotated[
        float | None, Field(ge=0, le=100, description="Failure overlap percent.")
    ]


class OverallSummary(BaseModel):
    """Grand macro metrics across provider summaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    latency_seconds: Annotated[
        float, Field(ge=0, description="Five-provider mean latency in seconds.")
    ]
    interaction_failure: Annotated[
        float,
        Field(
            ge=0,
            le=100,
            description="Grand descriptive Interaction composite failure percent.",
        ),
    ]
    experience: Annotated[
        float, Field(ge=0, le=100, description="Grand Experience pass percent.")
    ]
    experience_failure: Annotated[
        float, Field(ge=0, le=100, description="Grand Experience failure percent.")
    ]
    fluency_failure: Annotated[
        float, Field(ge=0, le=100, description="Grand Fluency failure percent.")
    ]
    speech_fidelity_failure: Annotated[
        float,
        Field(ge=0, le=100, description="Grand Speech-fidelity failure percent."),
    ]
    overlapping_failure: Annotated[
        float, Field(ge=0, le=100, description="Grand failure overlap percent.")
    ]


class DescriptiveSummary(BaseModel):
    """Complete-cohort descriptive metrics at every paper rollup."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language_system: Annotated[
        dict[str, dict[str, LanguageSystemSummary]],
        Field(description="Metrics keyed by language name and system."),
    ]
    language: Annotated[
        dict[str, LanguageSummary],
        Field(description="System-macro metrics keyed by language name."),
    ]
    provider: Annotated[
        dict[str, ProviderSummary],
        Field(description="Language-macro metrics keyed by system."),
    ]
    overall: Annotated[OverallSummary, Field(description="Grand macro summary.")]


class ProviderComparison(BaseModel):
    """One corrected paired provider comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    system_a: Annotated[str, Field(description="First provider label.")]
    system_b: Annotated[str, Field(description="Second provider label.")]
    delta_a_minus_b_points: Annotated[
        float, Field(description="Pooled Experience difference in points.")
    ]
    raw_p: Annotated[float, Field(ge=0, le=1, description="Raw permutation p.")]
    holm_p: Annotated[float, Field(ge=0, le=1, description="Holm-adjusted p.")]
    significant_0_05: Annotated[
        bool, Field(description="Whether adjusted p is below 0.05.")
    ]


class SignificanceSummary(BaseModel):
    """Pooled estimates and all corrected provider comparisons."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_point_estimates: Annotated[
        dict[str, float], Field(description="Pooled utterance Experience by system.")
    ]
    pairs: Annotated[
        list[ProviderComparison], Field(description="All ten pairwise comparisons.")
    ]


class ExperienceCohort(BaseModel):
    """Frozen cohort dimensions used by the analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    calls: Annotated[int, Field(ge=0, description="Total trial-0 calls.")]
    calls_per_system: Annotated[
        int, Field(ge=0, description="Trial-0 calls per system.")
    ]
    languages: Annotated[list[str], Field(description="Ordered language codes.")]
    domains: Annotated[list[str], Field(description="Ordered benchmark domains.")]
    systems: Annotated[list[str], Field(description="Ordered system labels.")]
    trial: Annotated[Literal[0], Field(description="Frozen trial index.")]


class ExperienceFormulas(BaseModel):
    """Human-readable definitions serialized with the artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    latency: Annotated[str, Field(description="Latency summary definition.")]
    interaction: Annotated[
        str, Field(description="Descriptive Interaction composite definition.")
    ]
    fluency_failure: Annotated[str, Field(description="Fluency failure definition.")]
    speech_fidelity_failure: Annotated[
        str, Field(description="Speech-fidelity failure definition.")
    ]
    experience: Annotated[str, Field(description="Experience score definition.")]
    gating: Annotated[str, Field(description="Denominator gating definition.")]


class ExperienceReportingRules(BaseModel):
    """Cohort, alignment, and deduplication reporting rules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    complete_cohort: Annotated[str, Field(description="Cohort inclusion rule.")]
    utterance_alignment: Annotated[str, Field(description="Utterance join rule.")]
    deduplication: Annotated[str, Field(description="Finding deduplication rule.")]


class UtteranceAlignmentSummary(BaseModel):
    """Global utterance alignment and Experience sufficient counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    eligible_utterances: Annotated[
        int, Field(ge=0, description="Utterances scored by both judges.")
    ]
    fluency_failures: Annotated[
        int, Field(ge=0, description="Eligible Fluency failures.")
    ]
    speech_fidelity_failures: Annotated[
        int, Field(ge=0, description="Eligible Speech-fidelity failures.")
    ]
    overlapping_failures: Annotated[
        int, Field(ge=0, description="Eligible failures shared by both components.")
    ]
    experience_failures: Annotated[
        int, Field(ge=0, description="Deduplicated Experience failures.")
    ]
    pinned_greetings_excluded: Annotated[
        int, Field(ge=0, description="Injected greetings removed before alignment.")
    ]
    unmatched_naturalness_utterances: Annotated[
        int, Field(ge=0, description="Naturalness rows without delivery matches.")
    ]
    unmatched_delivery_utterances: Annotated[
        int, Field(ge=0, description="Delivery rows without naturalness matches.")
    ]
    fidelity_findings_excluded: Annotated[
        int, Field(ge=0, description="Matched final-window findings excluded.")
    ]


class UtteranceExperienceArtifact(BaseModel):
    """Typed machine-readable trial-0 paper analysis artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instrument: Annotated[
        Literal["tau-multilingual-utterance-experience"],
        Field(description="Analysis instrument identifier."),
    ]
    instrument_version: Annotated[
        Literal["2.2.0"], Field(description="Analysis contract version.")
    ]
    formulas: Annotated[ExperienceFormulas, Field(description="Metric definitions.")]
    seed: Annotated[int, Field(description="Permutation random seed.")]
    num_permutations: Annotated[
        int, Field(gt=0, description="Monte Carlo permutations per comparison.")
    ]
    analysis_unit: Annotated[str, Field(description="Paired inferential unit.")]
    test: Annotated[str, Field(description="Inferential test definition.")]
    cohort: Annotated[ExperienceCohort, Field(description="Frozen cohort dimensions.")]
    reporting_rules: Annotated[
        ExperienceReportingRules, Field(description="Published reporting rules.")
    ]
    provenance: Annotated[
        dict[str, Any], Field(description="Input hashes and exclusion provenance.")
    ]
    descriptive_complete_cohort: Annotated[
        DescriptiveSummary, Field(description="Complete-cohort descriptive results.")
    ]
    significance_complete_matched_cohort: Annotated[
        SignificanceSummary, Field(description="Corrected provider comparisons.")
    ]
    utterance_alignment: Annotated[
        UtteranceAlignmentSummary,
        Field(description="Global alignment and deduplicated failure counts."),
    ]


STAT_FIELDS = ("experience_failed", "experience_total")


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _bucket_failure(checks: list[dict[str, Any]], factors: set[str]) -> float | None:
    outcomes = [check.get("outcome") for check in checks if check.get("id") in factors]
    if "fail" in outcomes:
        return 1.0
    if "pass" in outcomes:
        return 0.0
    return None


def _active_fluency_factor_ids(language: str) -> set[str]:
    """Resolve active fluency and localization diagnostics from the pack."""
    return {
        factor.id for factor in judge_factors_for(language) if factor.enabled
    } | enabled_deterministic_factor_ids_for(language)


def _call_level_interaction_components(
    checks_list: list[dict[str, Any]],
) -> dict[str, float | None]:
    """Return the five call-level Interaction component verdicts."""
    return {
        component: _bucket_failure(checks_list, factor_ids)
        for component, factor_ids in INTERACTION_COMPONENT_FACTORS.items()
    }


def _call_level_latency_counts(checks_list: list[dict[str, Any]]) -> dict[str, float]:
    """Return sufficient counts for event-weighted response and yield latency."""
    checks = {check["id"]: check for check in checks_list}
    result: dict[str, float] = defaultdict(float)
    for factor, total_field, prefix in (
        ("responsiveness", "response_total", "response"),
        ("yielding", "yield_total", "yield"),
    ):
        metrics = (checks.get(factor) or {}).get("metrics", {})
        total = metrics.get(total_field)
        latency = metrics.get(f"{prefix}_latency_mean")
        if total is None or latency is None:
            continue
        result[f"{prefix}_latency_seconds_sum"] += float(total) * float(latency)
        result[f"{prefix}_latency_total"] += float(total)
    return dict(result)


def _normalized_utterance_text(text: str) -> str:
    """Normalize text for strict monotonic identity alignment."""
    normalized = unicodedata.normalize("NFKC", text.replace("###STOP###", ""))
    return "".join(
        char.lower()
        for char in normalized
        if unicodedata.category(char)[0] in {"L", "M", "N"}
    )


def _utterance_match_score(left: str, right: str) -> int:
    """Score a strong shared prefix; unrelated utterances cannot match."""
    if not left or not right:
        return 0
    common_prefix = 0
    for left_char, right_char in zip(left, right, strict=False):
        if left_char != right_char:
            break
        common_prefix += 1
    shorter = min(len(left), len(right))
    if common_prefix == shorter:
        return 1_100_000 + common_prefix
    if common_prefix >= min(12, shorter):
        return 1_010_000 + common_prefix
    return 0


def _align_utterance_results(
    naturalness: list[NaturalnessUtteranceVerdict],
    delivery: list[dict[str, Any]],
) -> list[tuple[NaturalnessUtteranceVerdict, dict[str, Any]]]:
    """Align the two ordered utterance inventories without trusting their indices."""
    natural_texts = [_normalized_utterance_text(row.text) for row in naturalness]
    delivery_texts = [
        _normalized_utterance_text(str(row.get("expected_text") or ""))
        for row in delivery
    ]
    natural_count = len(naturalness)
    delivery_count = len(delivery)
    scores = [[0] * (delivery_count + 1) for _ in range(natural_count + 1)]
    actions = [[0] * (delivery_count + 1) for _ in range(natural_count + 1)]
    for natural_index in range(1, natural_count + 1):
        for delivery_index in range(1, delivery_count + 1):
            options = (
                scores[natural_index - 1][delivery_index],
                scores[natural_index][delivery_index - 1],
                scores[natural_index - 1][delivery_index - 1]
                + _utterance_match_score(
                    natural_texts[natural_index - 1],
                    delivery_texts[delivery_index - 1],
                ),
            )
            action = max(range(3), key=options.__getitem__)
            scores[natural_index][delivery_index] = options[action]
            actions[natural_index][delivery_index] = action

    natural_index = natural_count
    delivery_index = delivery_count
    matches: list[tuple[int, int]] = []
    while natural_index and delivery_index:
        action = actions[natural_index][delivery_index]
        match_score = _utterance_match_score(
            natural_texts[natural_index - 1],
            delivery_texts[delivery_index - 1],
        )
        if action == 2 and match_score:
            matches.append((natural_index - 1, delivery_index - 1))
            natural_index -= 1
            delivery_index -= 1
        elif action == 0:
            natural_index -= 1
        else:
            delivery_index -= 1
    matches.reverse()
    expected_matches = min(natural_count, delivery_count)
    if len(matches) != expected_matches:
        raise ValueError(
            "Naturalness and delivery utterance inventories do not align: "
            f"matched={len(matches)}, expected={expected_matches}, "
            f"naturalness={natural_count}, delivery={delivery_count}"
        )
    return [(naturalness[left], delivery[right]) for left, right in matches]


def _speech_fidelity_failure(
    result: dict[str, Any], exclusions: set[tuple[int, int]]
) -> tuple[bool, int]:
    """Return one deduplicated material fidelity verdict and excluded count."""
    utterance_idx = int(result["utterance_idx"])
    material_failure = False
    excluded = 0
    for finding_index, finding in enumerate(result.get("findings", [])):
        if (utterance_idx, finding_index) in exclusions:
            excluded += 1
            continue
        material_failure |= (
            finding.get("axis") == "fidelity" and int(finding.get("severity") or 0) >= 2
        )
    tone_failure = any(
        check.get("id") == "tone_meaning_flip" and check.get("outcome") == "fail"
        for check in result.get("factor_checks", [])
    )
    return material_failure or tone_failure, excluded


def _utterance_experience_counts(
    naturalness: list[NaturalnessUtteranceVerdict],
    delivery: dict[str, Any] | None,
    exclusions: set[tuple[int, int]],
) -> UtteranceExperienceCounts:
    """Join, deduplicate, and count utterance-level Experience outcomes."""
    delivery_results = list((delivery or {}).get("utterance_results", []))
    pinned_greetings = int(bool(delivery_results))
    if delivery_results:
        if int(delivery_results[0]["utterance_idx"]) != 0:
            raise ValueError("Delivery inventory does not start with pinned greeting")
        delivery_results = delivery_results[1:]
    matches = _align_utterance_results(naturalness, delivery_results)
    eligible = 0
    fluency_failures = 0
    fidelity_failures = 0
    overlapping_failures = 0
    experience_failures = 0
    excluded_findings = 0
    for naturalness_result, delivery_result in matches:
        fidelity_failure, excluded = _speech_fidelity_failure(
            delivery_result, exclusions
        )
        excluded_findings += excluded
        naturalness_scored = naturalness_result.outcome in {
            JudgeOutcome.PASS,
            JudgeOutcome.FAIL,
        }
        delivery_scored = delivery_result.get("outcome") in {"pass", "fail"}
        if not naturalness_scored or not delivery_scored:
            continue
        fluency_failure = naturalness_result.outcome is JudgeOutcome.FAIL
        eligible += 1
        fluency_failures += fluency_failure
        fidelity_failures += fidelity_failure
        overlapping_failures += fluency_failure and fidelity_failure
        experience_failures += fluency_failure or fidelity_failure
    return UtteranceExperienceCounts(
        eligible_utterances=eligible,
        fluency_failures=fluency_failures,
        speech_fidelity_failures=fidelity_failures,
        overlapping_failures=overlapping_failures,
        experience_failures=experience_failures,
        pinned_greetings_excluded=pinned_greetings,
        unmatched_naturalness_utterances=len(naturalness) - len(matches),
        unmatched_delivery_utterances=len(delivery_results) - len(matches),
        fidelity_findings_excluded=excluded_findings,
    )


def _without_benchmark_transfer_findings(
    checks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, int | None]]]:
    """Drop findings caused only by the policy-mandated English transfer line."""
    filtered_checks = []
    exclusions = []
    for check in checks:
        units = check.get("unit_results") or []
        excluded_units = [
            unit
            for unit in units
            if unit.get("violated")
            and is_benchmark_transfer_fragment(str(unit.get("quote") or ""))
        ]
        if not excluded_units:
            filtered_checks.append(check)
            continue

        exclusions.extend(
            (str(check["id"]), unit.get("unit_index")) for unit in excluded_units
        )
        excluded_ids = {id(unit) for unit in excluded_units}
        retained_units = [unit for unit in units if id(unit) not in excluded_ids]
        retained_opportunities = [
            unit for unit in retained_units if unit.get("opportunity")
        ]
        retained_violations = [
            unit for unit in retained_opportunities if unit.get("violated")
        ]

        if retained_violations:
            strongest = max(
                retained_violations,
                key=lambda unit: int(unit.get("severity") or 0),
            )
            outcome = JudgeOutcome.FAIL.value
            evidence = "; ".join(
                dict.fromkeys(
                    str(unit.get("reasoning") or "").strip()
                    for unit in retained_violations
                    if str(unit.get("reasoning") or "").strip()
                )
            )
            quote = str(strongest.get("quote") or "").strip()
            observed_severity = min(
                4,
                int(strongest.get("severity") or 0)
                + (1 if len(retained_violations) >= 2 else 0),
            )
        else:
            # The removed violation was a scored opportunity. Preserve that
            # denominator while clearing the benchmark-authored false positive.
            outcome = JudgeOutcome.PASS.value
            evidence = None
            quote = None
            observed_severity = 0

        filtered_checks.append(
            {
                **check,
                "outcome": outcome,
                "evidence": evidence,
                "quote": quote,
                "observed_severity": observed_severity,
                "violation_count": len(retained_violations),
                "unit_results": retained_units,
            }
        )
    return filtered_checks, exclusions


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latency_seconds = _mean(
        (
            sum(
                row["latency_counts"].get(f"{prefix}_latency_seconds_sum", 0.0)
                for row in rows
            )
            / latency_total
            if (
                latency_total := sum(
                    row["latency_counts"].get(f"{prefix}_latency_total", 0.0)
                    for row in rows
                )
            )
            else None
        )
        for prefix in ("response", "yield")
    )
    interaction_components = {
        component: _mean(row["interaction_components"][component] for row in rows)
        for component in INTERACTION_COMPONENT_FACTORS
    }
    interaction = _mean(interaction_components.values())
    experience_total = sum(row["experience_utterances"] for row in rows)
    experience_failed = sum(row["experience_failures"] for row in rows)
    fluency_failed = sum(row["fluency_failures"] for row in rows)
    fidelity_failed = sum(row["speech_fidelity_failures"] for row in rows)
    overlap_failed = sum(row["overlapping_failures"] for row in rows)
    experience_failure = (
        experience_failed / experience_total if experience_total else None
    )
    return {
        "n_calls": len(rows),
        "latency_seconds": latency_seconds,
        "interaction_failure": interaction,
        "interaction_components": interaction_components,
        "experience": (
            None if experience_failure is None else 1.0 - experience_failure
        ),
        "experience_failure": experience_failure,
        "fluency_failure": (
            fluency_failed / experience_total if experience_total else None
        ),
        "speech_fidelity_failure": (
            fidelity_failed / experience_total if experience_total else None
        ),
        "overlapping_failure": (
            overlap_failed / experience_total if experience_total else None
        ),
        "denominators": {
            "interaction_calls": sum(
                any(
                    value is not None
                    for value in row["interaction_components"].values()
                )
                for row in rows
            ),
            "experience_utterances": experience_total,
            "fluency_failures": fluency_failed,
            "speech_fidelity_failures": fidelity_failed,
            "overlapping_failures": overlap_failed,
            "experience_failures": experience_failed,
        },
    }


def _sufficient_stats(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            sum(row["experience_failures"] for row in rows),
            sum(row["experience_utterances"] for row in rows),
        ],
        dtype=float,
    )


def _scores_from_stats(stats: np.ndarray) -> np.ndarray:
    failed = stats[..., STAT_FIELDS.index("experience_failed")]
    total = stats[..., STAT_FIELDS.index("experience_total")]
    return 100.0 * (1.0 - failed / total)


def _reaggregated_permutation_p(
    stats_a: np.ndarray,
    stats_b: np.ndarray,
    *,
    num_permutations: int,
    seed: int,
) -> tuple[float, float, float]:
    total_a = stats_a.sum(axis=0)
    total_b = stats_b.sum(axis=0)
    score_a = float(_scores_from_stats(total_a))
    score_b = float(_scores_from_stats(total_b))
    observed = abs(score_a - score_b)
    combined = total_a + total_b
    delta = stats_a - stats_b
    rng = np.random.default_rng(seed)
    exceedances = 0
    completed = 0
    while completed < num_permutations:
        batch = min(5_000, num_permutations - completed)
        signs = rng.choice((-1.0, 1.0), size=(batch, len(stats_a)))
        pseudo_a = (combined + signs @ delta) / 2.0
        pseudo_b = combined - pseudo_a
        differences = np.abs(
            _scores_from_stats(pseudo_a) - _scores_from_stats(pseudo_b)
        )
        exceedances += int((differences >= observed - 1e-15).sum())
        completed += batch
    return score_a, score_b, (exceedances + 1) / (num_permutations + 1)


def _read_prefix(path: Path) -> dict[str, Any]:
    """Read top-level simulation metadata without loading the enormous tick list."""
    marker = b'\n  "ticks": '
    payload = bytearray()
    with path.open("rb") as handle:
        while marker not in payload:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                raise ValueError(f"No ticks field found in {path}")
            payload.extend(chunk)
    prefix = bytes(payload).split(marker, 1)[0].rstrip()
    if prefix.endswith(b","):
        prefix = prefix[:-1]
    return json.loads(prefix + b"\n}")


def _cell_directory(root: Path, language: str, domain: str) -> Path:
    if domain == "airline":
        stem = (
            "airline_en_v1"
            if language == "en"
            else (f"airline_v1_{LANGUAGE_NAMES[language].lower()}_airline")
        )
    elif domain == "retail":
        stem = f"retail_v1_{LANGUAGE_NAMES[language].lower()}_retail"
    elif language in {"en", "es", "pt"}:
        stem = f"preference_strat50_{LANGUAGE_NAMES[language].lower()}_telecom"
    else:
        stem = f"preference_runs_v1_{LANGUAGE_NAMES[language].lower()}_telecom"
    return root / stem


def _xai_directory(root: Path, language: str, domain: str) -> Path:
    return root / f"{domain}_xai_v2_{LANGUAGE_NAMES[language].lower()}_{domain}"


def _result_path(root: Path, language: str, domain: str, system: str) -> Path:
    parent = (
        _xai_directory(root, language, domain)
        if system == "xAI"
        else _cell_directory(root, language, domain)
    )
    matches = sorted(
        parent.glob(f"*/{language}_{domain}_{SYSTEM_SLUGS[system]}/results.json")
    )
    if not matches:
        matches = sorted(
            parent.glob(f"{language}_{domain}_{SYSTEM_SLUGS[system]}/results.json")
        )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one results file for {language}/{domain}/{system}, got {matches}"
        )
    return matches[0]


def _canonical_task(task_id: str, language: str) -> str:
    marker = f"_{language}"
    return task_id.rsplit(marker, 1)[0] if marker in task_id else task_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_value(value: object) -> str:
    """Hash a JSON value with the replay runner's canonical serialization."""
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _typed_identity_sha256(manifest: TrialRunManifest) -> str:
    """Recompute the replay identity digest using the runner's canonical form."""
    return _sha256_value(manifest.identity.model_dump(mode="json"))


def _validate_sidecar_simulation(artifact: TrialSimulationArtifact, path: Path) -> None:
    """Reject a call rollup that disagrees with any atomic utterance verdict."""
    child_outcomes: list[JudgeOutcome] = []
    child_units = []
    seen_turns: set[int] = set()
    turn_indices = [child.turn.index for child in artifact.utterances]
    if turn_indices != sorted(turn_indices) or len(turn_indices) != len(
        set(turn_indices)
    ):
        raise ValueError(f"sidecar utterances are not uniquely turn-ordered at {path}")
    for child in artifact.utterances:
        if (
            child.source != artifact.source
            or child.simulation_id != artifact.simulation_id
            or child.task_id != artifact.task_id
            or child.trial != artifact.trial
            or child.language != artifact.language
            or child.domain != artifact.domain
            or child.source_simulation_sha256 != artifact.source_simulation_sha256
            or child.judge != artifact.judge
        ):
            raise ValueError(f"sidecar child identity drift at {path}")
        if child.check.outcome in {JudgeOutcome.ERROR, JudgeOutcome.DEFERRED}:
            raise ValueError(f"sidecar contains an incomplete utterance at {path}")
        if len(child.check.unit_results) != 1:
            raise ValueError(f"sidecar utterance lacks one unit verdict at {path}")
        unit = child.check.unit_results[0]
        if unit.unit_index != child.turn.index or child.turn.index in seen_turns:
            raise ValueError(f"sidecar utterance turn identity drift at {path}")
        seen_turns.add(child.turn.index)
        expected_child = (
            JudgeOutcome.NO_OPPORTUNITY
            if not unit.opportunity
            else (JudgeOutcome.FAIL if unit.violated else JudgeOutcome.PASS)
        )
        if child.check.outcome is not expected_child:
            raise ValueError(f"sidecar utterance rollup drift at {path}")
        expected_input_sha256 = _sha256_value(
            {
                "schema_version": child.schema_version,
                "source": child.source.model_dump(mode="json"),
                "simulation_id": child.simulation_id,
                "task_id": child.task_id,
                "trial": child.trial,
                "language": child.language,
                "domain": child.domain,
                "source_simulation_sha256": child.source_simulation_sha256,
                "judge": child.judge.model_dump(mode="json"),
                "turn": child.turn.model_dump(mode="json"),
            }
        )
        if child.input_sha256 != expected_input_sha256:
            raise ValueError(f"sidecar utterance input identity drift at {path}")
        child_outcomes.append(child.check.outcome)
        child_units.append(unit)

    expected_call = (
        JudgeOutcome.FAIL
        if JudgeOutcome.FAIL in child_outcomes
        else (
            JudgeOutcome.PASS
            if JudgeOutcome.PASS in child_outcomes
            else JudgeOutcome.NO_OPPORTUNITY
        )
    )
    if artifact.check.outcome is not expected_call:
        raise ValueError(f"sidecar call rollup drift at {path}")
    if artifact.check.unit_results != child_units:
        raise ValueError(f"sidecar call units drift from atomic verdicts at {path}")
    if artifact.input_sha256 != _sha256_value(
        [child.input_sha256 for child in artifact.utterances]
    ):
        raise ValueError(f"sidecar call input identity drift at {path}")


def _load_naturalness_sidecar(path: Path) -> NaturalnessSidecarInput:
    """Load one complete replay and validate its internal artifact inventory."""
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"naturalness sidecar is not a directory: {root}")
    manifest_path = root / "manifest.json"
    manifest = TrialRunManifest.model_validate_json(manifest_path.read_text())
    if manifest.status != "complete":
        raise ValueError(
            f"naturalness sidecar must be complete, got {manifest.status!r}"
        )
    if manifest.identity_sha256 != _typed_identity_sha256(manifest):
        raise ValueError("naturalness sidecar identity hash does not reproduce")
    counts = manifest.counts
    if (
        counts.error_utterances
        or counts.error_simulations
        or counts.complete_utterances != counts.utterances
        or counts.complete_simulations != counts.simulations
    ):
        raise ValueError("naturalness sidecar manifest contains incomplete work")
    if manifest.aggregate is None or (
        manifest.aggregate.overall.calls != counts.simulations
        or manifest.aggregate.overall.error_calls
    ):
        raise ValueError("naturalness sidecar aggregate disagrees with manifest counts")

    selected = {
        source.results_path: source for source in manifest.identity.selected_sources
    }
    if len(selected) != len(manifest.identity.selected_sources):
        raise ValueError("naturalness sidecar has duplicate selected source paths")
    simulation_root = root / "simulations"
    artifact_paths = (
        sorted(simulation_root.rglob("*.json")) if simulation_root.is_dir() else []
    )
    if len(artifact_paths) != counts.simulations:
        raise ValueError(
            "naturalness sidecar simulation inventory differs from manifest counts"
        )

    artifacts: list[TrialSimulationArtifact] = []
    calls: list[NaturalnessSidecarCall] = []
    utterance_count = 0
    zero_utterance_count = 0
    for artifact_path in artifact_paths:
        artifact = TrialSimulationArtifact.model_validate_json(
            artifact_path.read_text()
        )
        source = selected.get(artifact.source.results_path)
        if source is None or source != artifact.source:
            raise ValueError(
                f"sidecar artifact has an unselected source: {artifact_path}"
            )
        if artifact.language != source.language or artifact.domain != source.domain:
            raise ValueError(f"sidecar source dimensions drift at {artifact_path}")
        if artifact.judge != manifest.identity.judges.get(artifact.language):
            raise ValueError(f"sidecar judge identity drift at {artifact_path}")
        if artifact.check.id != "natural_word_choice":
            raise ValueError(
                f"sidecar artifact contains the wrong factor: {artifact_path}"
            )
        _validate_sidecar_simulation(artifact, artifact_path)
        artifacts.append(artifact)
        utterance_count += len(artifact.utterances)
        zero_utterance_count += not artifact.utterances
        calls.append(
            NaturalnessSidecarCall(
                source_results_path=source.results_path,
                source_results_sha256=source.results_sha256,
                source_simulation_sha256=artifact.source_simulation_sha256,
                simulation_id=artifact.simulation_id,
                task_id=artifact.task_id,
                language=artifact.language,
                domain=artifact.domain,
                outcome=artifact.check.outcome,
                utterance_results=[
                    NaturalnessUtteranceVerdict(
                        turn_index=child.turn.index,
                        input_sha256=child.input_sha256,
                        text=child.turn.text,
                        outcome=child.check.outcome,
                    )
                    for child in artifact.utterances
                ],
            )
        )
    if utterance_count != counts.utterances:
        raise ValueError("naturalness sidecar utterance inventory drifted")
    if zero_utterance_count != counts.zero_utterance_simulations:
        raise ValueError("naturalness sidecar zero-utterance count drifted")
    if manifest.aggregate != _aggregate_trial(artifacts):
        raise ValueError("naturalness sidecar aggregate drifted from call artifacts")
    return NaturalnessSidecarInput(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        manifest=manifest,
        calls=calls,
    )


def _sidecar_provenance_path(root: Path, repo_root: Path) -> str:
    """Prefer a portable repository-relative path when the sidecar is local."""
    try:
        return root.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(root)


def _validate_sidecar_cohort(
    sidecar: NaturalnessSidecarInput,
    *,
    repo_root: Path,
    sources: list[dict[str, Any]],
    selected_sources: list[TrialSourceIdentity],
    used_calls: set[tuple[str, str]],
    work_simulations: list[tuple[str, str, str]],
    work_utterances: list[str],
) -> NaturalnessSidecarProvenance:
    """Match a complete replay exactly to the cohort discovered by analysis."""
    identity = sidecar.manifest.identity
    if identity.experience_path != CANONICAL_EXPERIENCE_PATH.as_posix():
        raise ValueError("naturalness sidecar uses a non-canonical Experience path")
    if identity.trial != 0:
        raise ValueError("naturalness sidecar must contain trial 0")
    expected_sources = [ExperienceSource.model_validate(source) for source in sources]
    if identity.all_sources != expected_sources:
        raise ValueError("naturalness sidecar all-source identity drifted")
    fingerprint_payload = "\n".join(
        f"{row.path}\t{row.sha256}\t{row.trial_0_calls}" for row in expected_sources
    )
    expected_fingerprint = hashlib.sha256(fingerprint_payload.encode()).hexdigest()
    if identity.cohort_fingerprint_sha256 != expected_fingerprint:
        raise ValueError("naturalness sidecar cohort fingerprint drifted")
    if identity.selected_sources != selected_sources:
        raise ValueError("naturalness sidecar selected-source identity drifted")
    if sidecar.manifest.counts.verified_roots != len(expected_sources) or (
        sidecar.manifest.counts.selected_roots != len(selected_sources)
    ):
        raise ValueError("naturalness sidecar root counts drifted")

    expected_judges = {
        language: frozen_judge_spec(language)
        for language in LANGUAGES
        if language != "en"
    }
    if identity.judges != expected_judges:
        raise ValueError("naturalness sidecar does not use the frozen paper judges")

    sidecar_keys = {
        (call.source_results_path, call.simulation_id) for call in sidecar.calls
    }
    missing = used_calls - sidecar_keys
    extra = sidecar_keys - used_calls
    if missing or extra:
        raise ValueError(
            "naturalness sidecar call inventory drifted: "
            f"missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}"
        )
    if identity.work_fingerprint_sha256 != _sha256_value(
        {"simulations": work_simulations, "utterances": work_utterances}
    ):
        raise ValueError("naturalness sidecar work fingerprint drifted")

    judges = list(identity.judges.values())
    prompt_versions = {judge.prompt_version for judge in judges}
    rubric_versions = {judge.rubric_version for judge in judges}
    if len(prompt_versions) != 1 or len(rubric_versions) != 1:
        raise ValueError("naturalness sidecar judge versions differ across languages")
    return NaturalnessSidecarProvenance(
        path=_sidecar_provenance_path(sidecar.root, repo_root),
        manifest_sha256=sidecar.manifest_sha256,
        identity_sha256=sidecar.manifest.identity_sha256,
        work_fingerprint_sha256=identity.work_fingerprint_sha256,
        calls=len(sidecar.calls),
        factor_id="natural_word_choice",
        prompt_version=prompt_versions.pop(),
        rubric_version=rubric_versions.pop(),
    )


def _load_gender_overrides(
    repo_root: Path,
) -> tuple[dict[str, dict[str, JudgeOutcome]], dict[str, Any]]:
    overrides_by_language: dict[str, dict[str, JudgeOutcome]] = {}
    provenance: dict[str, Any] = {}
    for language, relative_path in GENDER_OVERRIDE_PATHS.items():
        path = repo_root / relative_path
        with path.open(newline="") as handle:
            rows = [
                GenderOverrideRow.model_validate(row) for row in csv.DictReader(handle)
            ]
        overrides = {row.sim_id: row.outcome for row in rows}
        if len(overrides) != len(rows):
            raise ValueError(f"Duplicate simulation ids in {path}")
        overrides_by_language[language] = overrides
        provenance[language] = {
            "path": str(path.relative_to(repo_root)),
            "sha256": _sha256(path),
            "rows": len(rows),
            "factor": "gender_agreement",
            "judge_spec": "v3",
            "operative_agent_gender": "male",
        }
    return overrides_by_language, provenance


def _load_fidelity_end_exclusions(
    repo_root: Path,
) -> tuple[dict[str, set[tuple[int, int]]], dict[str, Any], set[tuple[str, int, int]]]:
    path = repo_root / FIDELITY_END_EXCLUSIONS_PATH
    with path.open(newline="") as handle:
        rows = [
            FidelityEndExclusionRow.model_validate(row)
            for row in csv.DictReader(handle)
        ]
    keys = {(row.sim_id, row.utterance_idx, row.finding_index) for row in rows}
    if len(keys) != len(rows):
        raise ValueError(f"Duplicate fidelity exclusion rows in {path}")
    manifest_path = path.with_suffix(".json")
    manifest = FidelityEndExclusionManifest.model_validate_json(
        manifest_path.read_text()
    )
    if manifest.excluded_findings != len(rows):
        raise ValueError(
            f"Fidelity exclusion manifest says {manifest.excluded_findings} rows, "
            f"but {path} has {len(rows)}"
        )
    by_sim: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in rows:
        by_sim[row.sim_id].add((row.utterance_idx, row.finding_index))
    provenance = {
        "path": str(path.relative_to(repo_root)),
        "sha256": _sha256(path),
        "manifest_path": str(manifest_path.relative_to(repo_root)),
        "manifest_sha256": _sha256(manifest_path),
        **manifest.model_dump(mode="json"),
    }
    return dict(by_sim), provenance, keys


def _load_calls(
    repo_root: Path,
    *,
    naturalness_sidecar: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = repo_root / "data/simulations/paper_runs/tau-multi/main_runs"
    sidecar = _load_naturalness_sidecar(naturalness_sidecar)
    sidecar_calls = {
        (call.source_results_path, call.simulation_id): call for call in sidecar.calls
    }
    used_sidecar_calls: set[tuple[str, str]] = set()
    sidecar_work_simulations: list[tuple[str, str, str]] = []
    sidecar_work_utterances: list[str] = []
    expected_sidecar_sources: list[TrialSourceIdentity] = []
    gender_overrides, gender_override_provenance = _load_gender_overrides(repo_root)
    fidelity_exclusions, fidelity_exclusion_provenance, all_exclusion_keys = (
        _load_fidelity_end_exclusions(repo_root)
    )
    used_exclusion_keys: set[tuple[str, int, int]] = set()
    used_gender_overrides: dict[str, set[str]] = {
        language: set() for language in gender_overrides
    }
    retired_factor_exclusions: list[tuple[str, str, str, str]] = []
    transfer_exclusions: list[tuple[str, str, str, int | None]] = []
    calls: list[dict[str, Any]] = []
    sources = []
    for language in LANGUAGES:
        for domain in DOMAINS:
            for system in SYSTEMS:
                results_path = _result_path(root, language, domain, system)
                results = json.loads(results_path.read_text())
                rows = [row for row in results["simulation_index"] if row["trial"] == 0]
                if len(rows) != 50:
                    raise ValueError(f"Expected 50 trial-0 rows in {results_path}")
                source_path = results_path.relative_to(repo_root).as_posix()
                source_sha256 = _sha256(results_path)
                sources.append(
                    {
                        "path": source_path,
                        "sha256": source_sha256,
                        "trial_0_calls": len(rows),
                    }
                )
                if language != "en":
                    expected_sidecar_sources.append(
                        TrialSourceIdentity(
                            results_path=source_path,
                            results_sha256=source_sha256,
                            source_key=hashlib.sha256(source_path.encode()).hexdigest()[
                                :16
                            ],
                            language=language,
                            domain=domain,
                            system_slug=SYSTEM_SLUGS[system],
                            trial_0_calls=len(rows),
                        )
                    )
                for row in rows:
                    sim_path = results_path.parent / "simulations" / f"{row['id']}.json"
                    sim = _read_prefix(sim_path)
                    sidecar_call: NaturalnessSidecarCall | None = None
                    quality_checks = (sim.get("quality_info") or {}).get(
                        "factor_checks", []
                    )
                    nativeness_checks = (sim.get("nativeness_info") or {}).get(
                        "factor_checks", []
                    )
                    active_fluency_ids = (
                        _active_fluency_factor_ids(language) & ALL_FLUENCY_FACTOR_IDS
                    )
                    retired_ids = ALL_FLUENCY_FACTOR_IDS - active_fluency_ids
                    for check in nativeness_checks:
                        if check.get("id") in retired_ids:
                            retired_factor_exclusions.append(
                                (
                                    row["id"],
                                    language,
                                    str(check.get("id")),
                                    str(check.get("outcome")),
                                )
                            )
                    nativeness_checks = [
                        check
                        for check in nativeness_checks
                        if check.get("id") not in retired_ids
                    ]
                    nativeness_checks, removed_transfer_units = (
                        _without_benchmark_transfer_findings(nativeness_checks)
                    )
                    transfer_exclusions.extend(
                        (row["id"], language, factor, unit_index)
                        for factor, unit_index in removed_transfer_units
                    )
                    if language != "en":
                        key = (source_path, row["id"])
                        sidecar_call = sidecar_calls.get(key)
                        if sidecar_call is None:
                            raise ValueError(
                                "Missing naturalness sidecar call for "
                                f"{source_path}/{row['id']}"
                            )
                        simulation_sha256 = _sha256(sim_path)
                        if (
                            sidecar_call.source_results_sha256 != source_sha256
                            or sidecar_call.task_id != str(row["task_id"])
                            or sidecar_call.language != language
                            or sidecar_call.domain != domain
                            or sidecar_call.source_simulation_sha256
                            != simulation_sha256
                        ):
                            raise ValueError(
                                "Naturalness sidecar call identity drift for "
                                f"{source_path}/{row['id']}"
                            )
                        used_sidecar_calls.add(key)
                        sidecar_work_simulations.append(
                            (source_path, row["id"], simulation_sha256)
                        )
                        sidecar_work_utterances.extend(
                            sidecar_call.utterance_input_sha256s
                        )
                    if language in gender_overrides:
                        override = gender_overrides[language].get(row["id"])
                        if override is None:
                            raise ValueError(
                                f"Missing {language} v3 gender override for {row['id']}"
                            )
                        if not any(
                            check.get("id") == "gender_agreement"
                            for check in nativeness_checks
                        ):
                            raise ValueError(
                                f"No stored gender_agreement check for {row['id']}"
                            )
                        nativeness_checks = [
                            {
                                **check,
                                "outcome": override.value,
                            }
                            if check.get("id") == "gender_agreement"
                            else check
                            for check in nativeness_checks
                        ]
                        used_gender_overrides[language].add(row["id"])
                    interaction_components = _call_level_interaction_components(
                        quality_checks
                    )
                    latency_counts = _call_level_latency_counts(quality_checks)
                    localization_diagnostics = {
                        factor_id: _bucket_failure(nativeness_checks, {factor_id})
                        for factor_id in sorted(
                            active_fluency_ids - {UTTERANCE_FLUENCY_FACTOR_ID}
                        )
                    }
                    delivery = sim.get("delivery_info")
                    call_exclusions = fidelity_exclusions.get(row["id"], set())
                    used_exclusion_keys.update(
                        (row["id"], utterance_idx, finding_index)
                        for utterance_idx, finding_index in call_exclusions
                    )
                    experience_counts = (
                        _utterance_experience_counts(
                            sidecar_call.utterance_results,
                            delivery,
                            call_exclusions,
                        )
                        if sidecar_call is not None
                        else UtteranceExperienceCounts(
                            eligible_utterances=0,
                            fluency_failures=0,
                            speech_fidelity_failures=0,
                            overlapping_failures=0,
                            experience_failures=0,
                            pinned_greetings_excluded=0,
                            unmatched_naturalness_utterances=0,
                            unmatched_delivery_utterances=0,
                            fidelity_findings_excluded=0,
                        )
                    )
                    calls.append(
                        {
                            "sim_id": row["id"],
                            "language": language,
                            "domain": domain,
                            "canonical_task": _canonical_task(row["task_id"], language),
                            "system": system,
                            "duration_seconds": float(
                                row.get("duration") or sim.get("duration") or 0.0
                            ),
                            "latency_counts": latency_counts,
                            "interaction_components": interaction_components,
                            "localization_diagnostics": localization_diagnostics,
                            "experience_utterances": (
                                experience_counts.eligible_utterances
                            ),
                            "fluency_failures": experience_counts.fluency_failures,
                            "speech_fidelity_failures": (
                                experience_counts.speech_fidelity_failures
                            ),
                            "overlapping_failures": (
                                experience_counts.overlapping_failures
                            ),
                            "experience_failures": (
                                experience_counts.experience_failures
                            ),
                            "pinned_greetings_excluded": (
                                experience_counts.pinned_greetings_excluded
                            ),
                            "unmatched_naturalness_utterances": (
                                experience_counts.unmatched_naturalness_utterances
                            ),
                            "unmatched_delivery_utterances": (
                                experience_counts.unmatched_delivery_utterances
                            ),
                            "fidelity_findings_excluded": (
                                experience_counts.fidelity_findings_excluded
                            ),
                        }
                    )
    for language, overrides in gender_overrides.items():
        if used_gender_overrides[language] != set(overrides):
            unused = sorted(set(overrides) - used_gender_overrides[language])
            raise ValueError(
                f"{language} v3 gender overrides do not match the paper cohort; "
                f"unused ids: {unused[:5]}"
            )
    if used_exclusion_keys != all_exclusion_keys:
        unused = sorted(all_exclusion_keys - used_exclusion_keys)
        raise ValueError(
            "Fidelity exclusions do not match the paper cohort; unused rows: "
            f"{unused[:5]}"
        )
    fingerprint_payload = "\n".join(
        f"{row['path']}\t{row['sha256']}\t{row['trial_0_calls']}" for row in sources
    )
    active_factors_by_language = {
        language: sorted(_active_fluency_factor_ids(language) & ALL_FLUENCY_FACTOR_IDS)
        for language in LANGUAGES
        if language != "en"
    }
    provenance = {
        "cohort_fingerprint_sha256": hashlib.sha256(
            fingerprint_payload.encode()
        ).hexdigest(),
        "results_files": sources,
        "gender_overrides": gender_override_provenance,
        "fidelity_end_exclusions": fidelity_exclusion_provenance,
        "retired_nativeness_exclusions": {
            "rule": (
                "Only the enabled combined-naturalness factor enters utterance "
                "Experience. Other enabled localization factors remain separate "
                "diagnostics; retired rubrics and deterministic checkers remain "
                "available but are not scored."
            ),
            "experience_fluency_factor_id": UTTERANCE_FLUENCY_FACTOR_ID,
            "active_factors_by_language": active_factors_by_language,
            "selection_sha256": _sha256_value(active_factors_by_language),
            "checks_removed": len(retired_factor_exclusions),
            "failed_checks_removed": sum(
                outcome == JudgeOutcome.FAIL.value
                for _sim_id, _language, _factor, outcome in retired_factor_exclusions
            ),
            "checks_removed_by_language": {
                language: sum(
                    excluded_language == language
                    for _sim_id, excluded_language, _factor, _outcome in retired_factor_exclusions
                )
                for language in LANGUAGES
                if language != "en"
            },
            "failed_checks_removed_by_language": {
                language: sum(
                    excluded_language == language and outcome == JudgeOutcome.FAIL.value
                    for _sim_id, excluded_language, _factor, outcome in retired_factor_exclusions
                )
                for language in LANGUAGES
                if language != "en"
            },
            "checks_removed_by_factor": {
                factor: sum(
                    excluded_factor == factor
                    for _sim_id, _language, excluded_factor, _outcome in retired_factor_exclusions
                )
                for factor in sorted(
                    {factor for _, _, factor, _ in retired_factor_exclusions}
                )
            },
            "failed_checks_removed_by_factor": {
                factor: sum(
                    excluded_factor == factor and outcome == JudgeOutcome.FAIL.value
                    for _sim_id, _language, excluded_factor, outcome in retired_factor_exclusions
                )
                for factor in sorted(
                    {factor for _, _, factor, _ in retired_factor_exclusions}
                )
            },
        },
        "benchmark_transfer_exclusions": {
            "rule": (
                "The policy-mandated English transfer line is benchmark-authored "
                "and not scored as model language."
            ),
            "message": BENCHMARK_TRANSFER_MESSAGE,
            "findings_removed": len(transfer_exclusions),
            "by_language": {
                LANGUAGE_NAMES[language]: sum(
                    excluded_language == language
                    for _sim_id, excluded_language, _factor, _unit in transfer_exclusions
                )
                for language in LANGUAGES
            },
            "by_factor": {
                factor: sum(
                    excluded_factor == factor
                    for _sim_id, _language, excluded_factor, _unit in transfer_exclusions
                )
                for factor in sorted(
                    {
                        excluded_factor
                        for _, _, excluded_factor, _ in transfer_exclusions
                    }
                )
            },
        },
    }
    provenance["naturalness_sidecar"] = _validate_sidecar_cohort(
        sidecar,
        repo_root=repo_root,
        sources=sources,
        selected_sources=expected_sidecar_sources,
        used_calls=used_sidecar_calls,
        work_simulations=sidecar_work_simulations,
        work_utterances=sidecar_work_utterances,
    ).model_dump(mode="json")
    return calls, provenance


def _holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, p_values[index] * (len(p_values) - rank))
        adjusted[index] = min(1.0, running)
    return adjusted


def _percentage(value: float | None) -> float | None:
    """Convert a unit-interval rate to percentage points."""
    return None if value is None else 100.0 * value


def _localization_diagnostic_rates(
    rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    """Aggregate retained non-Experience diagnostics at their native call rollup."""
    factor_ids = sorted(
        {factor_id for row in rows for factor_id in row["localization_diagnostics"]}
    )
    return {
        factor_id: _mean(row["localization_diagnostics"].get(factor_id) for row in rows)
        for factor_id in factor_ids
    }


def _summaries(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the complete cohort without language-specific exclusions."""
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        by_cell[(call["language"], call["system"])].append(call)

    language_system: dict[str, dict[str, Any]] = {}
    for (language, system), cell in sorted(by_cell.items()):
        aggregate = _aggregate(cell)
        language_system.setdefault(LANGUAGE_NAMES[language], {})[system] = {
            "n_calls": len(cell),
            "latency_seconds": aggregate["latency_seconds"],
            "interaction_failure": _percentage(aggregate["interaction_failure"]),
            "interaction_components": {
                key: _percentage(value)
                for key, value in aggregate["interaction_components"].items()
            },
            "experience": _percentage(aggregate["experience"]),
            "experience_failure": _percentage(aggregate["experience_failure"]),
            "fluency_failure": _percentage(aggregate["fluency_failure"]),
            "speech_fidelity_failure": _percentage(
                aggregate["speech_fidelity_failure"]
            ),
            "overlapping_failure": _percentage(aggregate["overlapping_failure"]),
            "localization_diagnostics": {
                key: _percentage(value)
                for key, value in _localization_diagnostic_rates(cell).items()
            },
            "denominators": aggregate["denominators"],
        }

    provider: dict[str, dict[str, Any]] = {}
    for system in SYSTEMS:
        rows = [row for row in calls if row["system"] == system]
        all_cells = [_aggregate(by_cell[(language, system)]) for language in LANGUAGES]
        localized_cells = all_cells[1:]
        provider[system] = {
            "n_calls": len(rows),
            "latency_seconds": float(
                np.mean([cell["latency_seconds"] for cell in all_cells])
            ),
            "mean_call_duration_minutes": float(
                np.mean([row["duration_seconds"] for row in rows]) / 60.0
            ),
            "interaction_failure": 100.0
            * float(np.mean([cell["interaction_failure"] for cell in all_cells])),
            "interaction_components": {
                key: 100.0
                * float(
                    np.mean([cell["interaction_components"][key] for cell in all_cells])
                )
                for key in INTERACTION_COMPONENT_FACTORS
            },
            "experience": 100.0
            * float(np.mean([cell["experience"] for cell in localized_cells])),
            "experience_failure": 100.0
            * float(np.mean([cell["experience_failure"] for cell in localized_cells])),
            "fluency_failure": 100.0
            * float(np.mean([cell["fluency_failure"] for cell in localized_cells])),
            "speech_fidelity_failure": 100.0
            * float(
                np.mean([cell["speech_fidelity_failure"] for cell in localized_cells])
            ),
            "overlapping_failure": 100.0
            * float(np.mean([cell["overlapping_failure"] for cell in localized_cells])),
        }

    language: dict[str, dict[str, float | None]] = {}
    for language_code in LANGUAGES:
        language_name = LANGUAGE_NAMES[language_code]
        cells = [language_system[language_name][system] for system in SYSTEMS]
        language[language_name] = {
            metric: _mean(cell[metric] for cell in cells)
            for metric in (
                "interaction_failure",
                "experience",
                "experience_failure",
                "fluency_failure",
                "speech_fidelity_failure",
                "overlapping_failure",
            )
        }

    return {
        "language_system": language_system,
        "language": language,
        "provider": provider,
        "overall": {
            metric: float(np.mean([value[metric] for value in provider.values()]))
            for metric in (
                "latency_seconds",
                "interaction_failure",
                "experience",
                "experience_failure",
                "fluency_failure",
                "speech_fidelity_failure",
                "overlapping_failure",
            )
        },
    }


def analyze(
    repo_root: Path,
    *,
    seed: int = 42,
    naturalness_sidecar: Path,
) -> UtteranceExperienceArtifact:
    calls, provenance = _load_calls(repo_root, naturalness_sidecar=naturalness_sidecar)
    num_permutations = 100_000
    by_cluster_system: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for call in calls:
        by_cluster_system[
            (call["domain"], call["canonical_task"], call["system"])
        ].append(call)
    cluster_stats: dict[str, np.ndarray] = {}
    cluster_order = sorted({(call["domain"], call["canonical_task"]) for call in calls})
    for system in SYSTEMS:
        values = []
        for domain, task in cluster_order:
            cluster_calls = by_cluster_system[(domain, task, system)]
            if len(cluster_calls) != len(LANGUAGES):
                raise ValueError(f"Unbalanced cluster {domain}/{task}/{system}")
            values.append(_sufficient_stats(cluster_calls))
        cluster_stats[system] = np.stack(values)

    pairs = []
    raw_p = []
    for system_a, system_b in combinations(SYSTEMS, 2):
        score_a, score_b, p_value = _reaggregated_permutation_p(
            cluster_stats[system_a],
            cluster_stats[system_b],
            num_permutations=num_permutations,
            seed=seed,
        )
        raw_p.append(p_value)
        pairs.append(
            {
                "system_a": system_a,
                "system_b": system_b,
                "delta_a_minus_b_points": score_a - score_b,
                "raw_p": p_value,
            }
        )
    for row, adjusted in zip(pairs, _holm(raw_p), strict=True):
        row["holm_p"] = adjusted
        row["significant_0_05"] = adjusted < 0.05

    descriptive = _summaries(calls)
    return UtteranceExperienceArtifact.model_validate(
        {
            "instrument": "tau-multilingual-utterance-experience",
            "instrument_version": "2.2.0",
            "formulas": {
                "latency": (
                    "six-language macro mean of the equal-weight mean of "
                    "event-weighted response and yield latency"
                ),
                "interaction": (
                    "descriptive equal-weight mean of five call-level component failure "
                    "rates: non-response, interruption, selectivity, monologue, and "
                    "validated tool use"
                ),
                "fluency_failure": (
                    "aligned utterance fails the frozen combined naturalness judge; "
                    "natural word choice, translationese, and verb morphology are one "
                    "binary verdict"
                ),
                "speech_fidelity_failure": (
                    "aligned utterance has any retained fidelity finding with severity "
                    ">=2 or a tone-meaning-flip failure"
                ),
                "experience": (
                    "100 * (1 - unique utterances failing fluency OR speech fidelity / "
                    "utterances successfully evaluated by both)"
                ),
                "gating": (
                    "only pass/fail verdicts aligned across both judges enter Experience; "
                    "no-opportunity, deferred, error, and unmatched utterances do not"
                ),
            },
            "seed": seed,
            "num_permutations": num_permutations,
            "analysis_unit": (
                "150 canonical domain-task clusters; Experience uses the five localized "
                "trial-0 calls for one system because English has no validated fluency judge"
            ),
            "test": (
                "Two-sided paired label-swap permutation; the same swap is applied "
                "to the five localized calls in a domain-task cluster, then deduplicated "
                "utterance counts are reaggregated into Experience; +1 Monte Carlo "
                "correction; Holm correction across ten provider pairs"
            ),
            "cohort": {
                "calls": len(calls),
                "calls_per_system": len(calls) // len(SYSTEMS),
                "languages": list(LANGUAGES),
                "domains": list(DOMAINS),
                "systems": list(SYSTEMS),
                "trial": 0,
            },
            "reporting_rules": {
                "complete_cohort": (
                    "All 150 calls per language-system cell are retained. Interaction "
                    "is a descriptive component average over all six languages; utterance "
                    "Experience uses ES/PT/HI/KO/ZH. Provider and language summaries "
                    "macro-average language cells."
                ),
                "utterance_alignment": (
                    "The structurally injected delivery greeting is excluded, then "
                    "naturalness and delivery utterances are joined monotonically by "
                    "strong normalized text prefix. The join must achieve the maximum "
                    "possible inventory overlap or analysis fails."
                ),
                "deduplication": (
                    "Any number of fluency or fidelity findings on one aligned utterance "
                    "contribute at most one Experience failure."
                ),
            },
            "provenance": provenance,
            "descriptive_complete_cohort": descriptive,
            "significance_complete_matched_cohort": {
                "provider_point_estimates": {
                    system: float(_scores_from_stats(stats.sum(axis=0)))
                    for system, stats in cluster_stats.items()
                },
                "pairs": pairs,
            },
            "utterance_alignment": {
                "eligible_utterances": sum(
                    call["experience_utterances"] for call in calls
                ),
                "fluency_failures": sum(call["fluency_failures"] for call in calls),
                "speech_fidelity_failures": sum(
                    call["speech_fidelity_failures"] for call in calls
                ),
                "overlapping_failures": sum(
                    call["overlapping_failures"] for call in calls
                ),
                "experience_failures": sum(
                    call["experience_failures"] for call in calls
                ),
                "pinned_greetings_excluded": sum(
                    call["pinned_greetings_excluded"] for call in calls
                ),
                "unmatched_naturalness_utterances": sum(
                    call["unmatched_naturalness_utterances"] for call in calls
                ),
                "unmatched_delivery_utterances": sum(
                    call["unmatched_delivery_utterances"] for call in calls
                ),
                "fidelity_findings_excluded": sum(
                    call["fidelity_findings_excluded"] for call in calls
                ),
            },
        }
    )


def write_analysis(
    repo_root: Path,
    output: Path,
    *,
    naturalness_sidecar: Path,
    seed: int = 42,
) -> UtteranceExperienceArtifact:
    """Recompute and write the typed paper artifact."""
    repo_root = repo_root.expanduser().resolve()
    destination = output.expanduser()
    if not destination.is_absolute():
        destination = repo_root / destination
    result = analyze(
        repo_root,
        seed=seed,
        naturalness_sidecar=naturalness_sidecar.expanduser().resolve(),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result.model_dump_json(indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--naturalness-sidecar",
        type=Path,
        required=True,
        help=(
            "Completed trial-0 combined-naturalness replay. Its exact call and "
            "utterance inventories are joined to delivery judgments; any drift fails "
            "analysis."
        ),
    )
    args = parser.parse_args()
    if args.output:
        write_analysis(
            args.repo_root,
            args.output,
            naturalness_sidecar=args.naturalness_sidecar,
        )
    else:
        result = analyze(
            args.repo_root.resolve(),
            naturalness_sidecar=args.naturalness_sidecar,
        )
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
