#!/usr/bin/env python3
"""Reproduce τ-Multilingual task-success inference from locked result indexes.

The analysis keeps the canonical 90-cell voice cohort and 36-cell text cohort
unless ``--replacement-root`` is supplied.  In that mode, only the ten Korean
and Mandarin retail voice cells and four matched text cells are replaced by the
corrected name-role runs.  All inputs are hashed into the output artifact.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tau2.data_model.simulation import Results
from tau2.judges.nativeness.paper_trial import _atomic_write
from tau2.judges.nativeness.validation import sha256_file

SCHEMA_VERSION = "tau-multi-task-success-significance-v1"
LANGUAGES = ("en", "es", "pt", "hi", "ko", "zh")
LOCALIZED_LANGUAGES = ("es", "pt", "hi", "ko", "zh")
DOMAINS = ("airline", "retail", "telecom")
REPEATED_SYSTEMS = (
    "openai_minimal",
    "openai_xhigh",
    "gemini_minimal",
    "gemini_high",
)
VOICE_SYSTEMS = (*REPEATED_SYSTEMS, "xai_provider_default")
TEXT_SYSTEMS = ("gpt55_xhigh", "gemini31pro_high")
LANGUAGE_LABELS = {
    "en": "english",
    "es": "spanish",
    "pt": "portuguese",
    "hi": "hindi",
    "ko": "korean",
    "zh": "mandarin",
}


class TaskSuccessAnalysisConfig(BaseModel):
    """Filesystem and Monte Carlo inputs for one deterministic analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_root: Annotated[
        Path,
        Field(
            description="Canonical tau-multi root containing main_runs/text_channel."
        ),
    ]
    replacement_root: Annotated[
        Optional[Path],
        Field(description="Corrected simulations root, or null for canonical replay."),
    ] = None
    output: Annotated[Path, Field(description="New JSON artifact path.")]
    seed: Annotated[int, Field(description="Shared Monte Carlo RNG seed.")] = 42
    permutations: Annotated[
        int, Field(gt=0, description="Sign permutations per comparison.")
    ] = 100_000
    bootstrap_resamples: Annotated[
        int, Field(gt=0, description="Paired BCa bootstrap resamples.")
    ] = 10_000


class TaskResultSource(BaseModel):
    """One hashed results index used by the analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cohort: Literal["voice", "text"]
    language: str
    domain: str
    system: str
    replacement: bool
    path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    rows: int
    trials: list[int]


class TaskComparison(BaseModel):
    """One paired localized-language comparison with English."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str
    n_clusters: int
    english_rate: float
    language_rate: float
    delta_points: float
    ci_low_points: float
    ci_high_points: float
    raw_p: float
    holm_p: float
    significant_0_05: bool


class TaskComparisonFamily(BaseModel):
    """Five English-relative comparisons sharing one Holm family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    system: str
    comparisons: list[TaskComparison]


class TrialStabilityRow(BaseModel):
    """Descriptive repeated-run stability for one language."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str
    trial_0_rate: float
    trial_1_rate: float
    two_trial_mean: float
    max_absolute_system_gap_points: float


class OverallTrialStability(BaseModel):
    """Single paired trial-1-minus-trial-0 comparison over task clusters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_clusters: int
    trial_0_rate: float
    trial_1_rate: float
    delta_points: float
    raw_p: float


class TaskSuccessDescriptive(BaseModel):
    """Paper-facing trial-0 task success and matched text-control rates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    voice_language_system: dict[str, dict[str, float]]
    voice_language_mean: dict[str, float]
    voice_system_mean: dict[str, float]
    voice_domain_mean: dict[str, float]
    text_language_system: dict[str, dict[str, float]]
    text_language_mean: dict[str, float]
    text_voice_gap_points: dict[str, float]


class TaskSuccessAnalysisArtifact(BaseModel):
    """Complete typed task-success and trial-stability result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-task-success-significance-v1"]
    seed: int
    permutations: int
    bootstrap_resamples: int
    analysis_unit: str
    test: str
    sources_fingerprint_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    sources: list[TaskResultSource]
    descriptive_trial_0: TaskSuccessDescriptive
    per_system_significance: list[TaskComparisonFamily]
    fixed_system_panel: TaskComparisonFamily
    trial_stability: list[TrialStabilityRow]
    overall_trial_stability: OverallTrialStability


class _Observation(BaseModel):
    """One binary reward addressed by its full matched design key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cohort: Literal["voice", "text"]
    language: str
    domain: str
    system: str
    task: str
    trial: int
    reward: float


def _canonical_task_id(task_id: str, language: str) -> str:
    for suffix in (
        f"_{language}_identity_native",
        f"_{language}_identity",
        f"_{language}",
    ):
        if task_id.endswith(suffix):
            return task_id[: -len(suffix)]
    return task_id


def _voice_group(language: str, domain: str, system: str) -> str:
    label = LANGUAGE_LABELS[language]
    if system == "xai_provider_default":
        return f"{domain}_xai_v2_{label}_{domain}"
    if domain == "airline":
        return "airline_en_v1" if language == "en" else f"airline_v1_{label}_airline"
    if domain == "retail":
        return f"retail_v1_{label}_retail"
    prefix = (
        "preference_strat50" if language in {"en", "es", "pt"} else "preference_runs_v1"
    )
    return f"{prefix}_{label}_telecom"


def _voice_path(
    canonical_root: Path,
    replacement_root: Optional[Path],
    language: str,
    domain: str,
    system: str,
) -> tuple[Path, bool]:
    if replacement_root is not None and language in {"ko", "zh"} and domain == "retail":
        label = LANGUAGE_LABELS[language]
        group = (
            f"retail_name_roles_xai_v1_{label}_retail"
            if system == "xai_provider_default"
            else f"retail_name_roles_v1_{label}_retail"
        )
        return (
            replacement_root / group / f"{language}_{domain}_{system}" / "results.json",
            True,
        )
    group = _voice_group(language, domain, system)
    return (
        canonical_root
        / "main_runs"
        / group
        / f"{language}_{domain}_{system}"
        / "results.json",
        False,
    )


def _text_path(
    canonical_root: Path,
    replacement_root: Optional[Path],
    language: str,
    domain: str,
    system: str,
) -> tuple[Path, bool]:
    label = LANGUAGE_LABELS[language]
    if replacement_root is not None and language in {"ko", "zh"} and domain == "retail":
        return (
            replacement_root
            / f"multilingual_text_retail_name_roles_v1_{label}_retail"
            / f"{language}_{domain}_{system}"
            / "results.json",
            True,
        )
    return (
        canonical_root
        / "text_channel"
        / f"multilingual_text_v1_{label}_{domain}"
        / f"{language}_{domain}_{system}"
        / "results.json",
        False,
    )


def _load_cell(
    *,
    path: Path,
    replacement: bool,
    cohort: Literal["voice", "text"],
    language: str,
    domain: str,
    system: str,
) -> tuple[list[_Observation], TaskResultSource]:
    if not path.is_file():
        raise ValueError(f"task-success results file is missing: {path}")
    results = Results.load_metadata(path)
    if results.simulation_index is None:
        raise ValueError(f"results file has no simulation index: {path}")
    expected_trials = (
        {0, 1} if cohort == "voice" and system in REPEATED_SYSTEMS else {0}
    )
    trials = {row.trial for row in results.simulation_index}
    expected_rows = 50 * len(expected_trials)
    if trials != expected_trials or len(results.simulation_index) != expected_rows:
        raise ValueError(
            f"task-success cell shape drifted at {path}: "
            f"rows={len(results.simulation_index)}, trials={sorted(trials)}"
        )
    observations: list[_Observation] = []
    seen: set[tuple[str, int]] = set()
    for row in results.simulation_index:
        if row.reward not in {0.0, 1.0}:
            raise ValueError(f"non-binary or missing reward at {path}/{row.id}")
        task = _canonical_task_id(str(row.task_id), language)
        key = (task, row.trial)
        if key in seen:
            raise ValueError(f"duplicate task/trial at {path}: {key}")
        seen.add(key)
        observations.append(
            _Observation(
                cohort=cohort,
                language=language,
                domain=domain,
                system=system,
                task=task,
                trial=row.trial,
                reward=float(row.reward),
            )
        )
    if any(sum(row.trial == trial for row in observations) != 50 for trial in trials):
        raise ValueError(f"task-success trial count drifted at {path}")
    return observations, TaskResultSource(
        cohort=cohort,
        language=language,
        domain=domain,
        system=system,
        replacement=replacement,
        path=str(path.resolve()),
        sha256=sha256_file(path),
        rows=len(observations),
        trials=sorted(trials),
    )


def paired_permutation_test(
    paired_a: np.ndarray,
    paired_b: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> float:
    """Frozen tau-Voice two-sided sign-permutation implementation."""
    diffs = paired_a - paired_b
    observed = np.abs(np.mean(diffs))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_permutations, len(diffs)))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    return float(np.mean(perm_means >= observed))


def _norm_ppf(value: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(np.clip(value, 1e-10, 1 - 1e-10)))


def _norm_cdf(value: float) -> float:
    from scipy.stats import norm

    return float(norm.cdf(value))


def bootstrap_delta_ci(
    paired_a: np.ndarray,
    paired_b: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Frozen tau-Voice paired BCa interval on mean(A - B)."""
    diffs = paired_a - paired_b
    size = len(diffs)
    observed = np.mean(diffs)
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, size, size=(n_bootstrap, size))
    boot_means = diffs[boot_indices].mean(axis=1)
    z0 = _norm_ppf(np.mean(boot_means < observed))
    jack_means = np.array([np.mean(np.delete(diffs, index)) for index in range(size)])
    jack_mean = jack_means.mean()
    numerator = np.sum((jack_mean - jack_means) ** 3)
    denominator = 6.0 * np.sum((jack_mean - jack_means) ** 2) ** 1.5
    acceleration = numerator / denominator if denominator != 0 else 0.0
    z_low = _norm_ppf(alpha / 2)
    z_high = _norm_ppf(1 - alpha / 2)
    alpha_low = _norm_cdf(z0 + (z0 + z_low) / (1 - acceleration * (z0 + z_low)))
    alpha_high = _norm_cdf(z0 + (z0 + z_high) / (1 - acceleration * (z0 + z_high)))
    alpha_low = np.clip(alpha_low, 1 / n_bootstrap, 1 - 1 / n_bootstrap)
    alpha_high = np.clip(alpha_high, 1 / n_bootstrap, 1 - 1 / n_bootstrap)
    sorted_boots = np.sort(boot_means)
    return (
        float(np.percentile(sorted_boots, alpha_low * 100)),
        float(np.percentile(sorted_boots, alpha_high * 100)),
    )


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Frozen tau-Voice Holm adjustment."""
    order = sorted(enumerate(p_values), key=lambda row: row[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, (index, value) in enumerate(order):
        running = max(running, value * (len(p_values) - rank))
        adjusted[index] = min(running, 1.0)
    return adjusted


def _sources_fingerprint(sources: list[TaskResultSource]) -> str:
    payload = "\n".join(
        f"{row.cohort}\t{row.language}\t{row.domain}\t{row.system}\t"
        f"{row.path}\t{row.sha256}\t{row.rows}"
        for row in sources
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _paired_family(
    *,
    system: str,
    arrays: dict[str, np.ndarray],
    config: TaskSuccessAnalysisConfig,
) -> TaskComparisonFamily:
    english = arrays["en"]
    preliminary: list[tuple[str, np.ndarray, float, tuple[float, float]]] = []
    p_values: list[float] = []
    for language in LOCALIZED_LANGUAGES:
        target = arrays[language]
        if len(target) != 150 or len(english) != 150:
            raise ValueError(f"paired task inventory drifted for {system}/{language}")
        raw_p = paired_permutation_test(
            target,
            english,
            n_permutations=config.permutations,
            seed=config.seed,
        )
        interval = bootstrap_delta_ci(
            target,
            english,
            n_bootstrap=config.bootstrap_resamples,
            seed=config.seed,
        )
        preliminary.append((language, target, raw_p, interval))
        p_values.append(raw_p)
    adjusted = holm_bonferroni(p_values)
    return TaskComparisonFamily(
        system=system,
        comparisons=[
            TaskComparison(
                language=language,
                n_clusters=len(target),
                english_rate=float(np.mean(english)),
                language_rate=float(np.mean(target)),
                delta_points=100 * float(np.mean(target - english)),
                ci_low_points=100 * interval[0],
                ci_high_points=100 * interval[1],
                raw_p=raw_p,
                holm_p=holm_p,
                significant_0_05=holm_p < 0.05,
            )
            for (language, target, raw_p, interval), holm_p in zip(
                preliminary, adjusted, strict=True
            )
        ],
    )


def analyze_task_success(
    config: TaskSuccessAnalysisConfig,
) -> TaskSuccessAnalysisArtifact:
    """Load, validate, and analyze the canonical or corrected cohort."""
    canonical_root = config.canonical_root.expanduser().resolve()
    replacement_root = (
        config.replacement_root.expanduser().resolve()
        if config.replacement_root is not None
        else None
    )
    observations: list[_Observation] = []
    sources: list[TaskResultSource] = []
    for language in LANGUAGES:
        for domain in DOMAINS:
            for system in VOICE_SYSTEMS:
                path, replacement = _voice_path(
                    canonical_root, replacement_root, language, domain, system
                )
                rows, source = _load_cell(
                    path=path,
                    replacement=replacement,
                    cohort="voice",
                    language=language,
                    domain=domain,
                    system=system,
                )
                observations.extend(rows)
                sources.append(source)
            for system in TEXT_SYSTEMS:
                path, replacement = _text_path(
                    canonical_root, replacement_root, language, domain, system
                )
                rows, source = _load_cell(
                    path=path,
                    replacement=replacement,
                    cohort="text",
                    language=language,
                    domain=domain,
                    system=system,
                )
                observations.extend(rows)
                sources.append(source)

    voice = [row for row in observations if row.cohort == "voice"]
    text = [row for row in observations if row.cohort == "text"]
    matched: dict[tuple[str, str, str, str, int], float] = {}
    for row in voice:
        key = (row.language, row.domain, row.system, row.task, row.trial)
        if key in matched:
            raise ValueError(f"duplicate voice matched key: {key}")
        matched[key] = row.reward

    domain_tasks: dict[str, set[str]] = defaultdict(set)
    for row in voice:
        domain_tasks[row.domain].add(row.task)
    if any(len(tasks) != 50 for tasks in domain_tasks.values()):
        raise ValueError("canonical domain task frames are not 50 tasks each")
    cluster_order = [
        (domain, task) for domain in DOMAINS for task in sorted(domain_tasks[domain])
    ]

    two_trial_arrays: dict[str, dict[str, np.ndarray]] = {}
    for system in REPEATED_SYSTEMS:
        two_trial_arrays[system] = {}
        for language in LANGUAGES:
            two_trial_arrays[system][language] = np.array(
                [
                    np.mean(
                        [
                            matched[(language, domain, system, task, trial)]
                            for trial in (0, 1)
                        ]
                    )
                    for domain, task in cluster_order
                ],
                dtype=float,
            )
    per_system = [
        _paired_family(system=system, arrays=two_trial_arrays[system], config=config)
        for system in REPEATED_SYSTEMS
    ]
    panel_arrays = {
        language: np.mean(
            np.stack(
                [two_trial_arrays[system][language] for system in REPEATED_SYSTEMS]
            ),
            axis=0,
        )
        for language in LANGUAGES
    }
    fixed_panel = _paired_family(
        system="four_system_mean", arrays=panel_arrays, config=config
    )

    stability: list[TrialStabilityRow] = []
    trial_cluster_arrays: dict[int, list[float]] = {0: [], 1: []}
    for language in LANGUAGES:
        rates_by_trial: dict[int, float] = {}
        gaps: list[float] = []
        for trial in (0, 1):
            values = [
                matched[(language, domain, system, task, trial)]
                for domain, task in cluster_order
                for system in REPEATED_SYSTEMS
            ]
            rates_by_trial[trial] = float(np.mean(values))
        for system in REPEATED_SYSTEMS:
            trial_rates = [
                float(
                    np.mean(
                        [
                            matched[(language, domain, system, task, trial)]
                            for domain, task in cluster_order
                        ]
                    )
                )
                for trial in (0, 1)
            ]
            gaps.append(abs(trial_rates[1] - trial_rates[0]))
        stability.append(
            TrialStabilityRow(
                language=language,
                trial_0_rate=rates_by_trial[0],
                trial_1_rate=rates_by_trial[1],
                two_trial_mean=np.mean(list(rates_by_trial.values())),
                max_absolute_system_gap_points=100 * max(gaps),
            )
        )
    for trial in (0, 1):
        trial_cluster_arrays[trial] = [
            float(
                np.mean(
                    [
                        matched[(language, domain, system, task, trial)]
                        for language in LANGUAGES
                        for system in REPEATED_SYSTEMS
                    ]
                )
            )
            for domain, task in cluster_order
        ]
    trial_0 = np.array(trial_cluster_arrays[0])
    trial_1 = np.array(trial_cluster_arrays[1])
    overall_stability = OverallTrialStability(
        n_clusters=len(trial_0),
        trial_0_rate=float(np.mean(trial_0)),
        trial_1_rate=float(np.mean(trial_1)),
        delta_points=100 * float(np.mean(trial_1 - trial_0)),
        raw_p=paired_permutation_test(
            trial_1,
            trial_0,
            n_permutations=config.permutations,
            seed=config.seed,
        ),
    )

    def mean_reward(
        rows: list[_Observation],
        *,
        language: Optional[str] = None,
        domain: Optional[str] = None,
        system: Optional[str] = None,
    ) -> float:
        values = [
            row.reward
            for row in rows
            if row.trial == 0
            and (language is None or row.language == language)
            and (domain is None or row.domain == domain)
            and (system is None or row.system == system)
        ]
        if not values:
            raise ValueError("empty descriptive task-success cell")
        return float(np.mean(values))

    voice_language_system = {
        language: {
            system: mean_reward(voice, language=language, system=system)
            for system in VOICE_SYSTEMS
        }
        for language in LANGUAGES
    }
    text_language_system = {
        language: {
            system: mean_reward(text, language=language, system=system)
            for system in TEXT_SYSTEMS
        }
        for language in LANGUAGES
    }
    voice_language_mean = {
        language: mean_reward(voice, language=language) for language in LANGUAGES
    }
    text_language_mean = {
        language: mean_reward(text, language=language) for language in LANGUAGES
    }
    descriptive = TaskSuccessDescriptive(
        voice_language_system=voice_language_system,
        voice_language_mean=voice_language_mean,
        voice_system_mean={
            system: mean_reward(voice, system=system) for system in VOICE_SYSTEMS
        },
        voice_domain_mean={
            domain: mean_reward(voice, domain=domain) for domain in DOMAINS
        },
        text_language_system=text_language_system,
        text_language_mean=text_language_mean,
        text_voice_gap_points={
            language: 100
            * (text_language_mean[language] - voice_language_mean[language])
            for language in LANGUAGES
        },
    )
    return TaskSuccessAnalysisArtifact(
        schema_version=SCHEMA_VERSION,
        seed=config.seed,
        permutations=config.permutations,
        bootstrap_resamples=config.bootstrap_resamples,
        analysis_unit="150 paired canonical domain-task clusters",
        test=(
            "two-sided paired sign permutation on task-level differences; trials "
            "averaged within task; Holm across five languages; paired BCa interval"
        ),
        sources_fingerprint_sha256=_sources_fingerprint(sources),
        sources=sources,
        descriptive_trial_0=descriptive,
        per_system_significance=per_system,
        fixed_system_panel=fixed_panel,
        trial_stability=stability,
        overall_trial_stability=overall_stability,
    )


def write_task_success_analysis(
    config: TaskSuccessAnalysisConfig,
) -> TaskSuccessAnalysisArtifact:
    """Run the analysis and atomically write its typed artifact."""
    artifact = analyze_task_success(config)
    output = config.output.expanduser().resolve()
    if output.exists():
        existing = TaskSuccessAnalysisArtifact.model_validate_json(output.read_text())
        if existing != artifact:
            raise ValueError(
                "existing task-success analysis was built from other inputs"
            )
        return existing
    _atomic_write(output, artifact)
    return artifact


def _parse_args() -> TaskSuccessAnalysisConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--replacement-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return TaskSuccessAnalysisConfig(**vars(parser.parse_args()))


def main() -> None:
    artifact = write_task_success_analysis(_parse_args())
    print(artifact.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
