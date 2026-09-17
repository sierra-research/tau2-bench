"""Paired significance analysis for the canonical tau-Elicitation provider study.

The analysis unit is one of the 200 aligned tasks. Each task contributes three
prespecified environment realizations for each system. The implementation
matches the paired sign-permutation and BCa interval procedure used in the
tau-Voice paper, and adds an omnibus Cochran Q test for realization effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2, norm

from tau2.paper.elicitation_scoring import (
    DEFAULT_ARTIFACT,
    corrected_reward,
    load_correction_map,
)

RELEASE_ROOT = Path("papers/tau-intake/v1/reproduction")
REALIZATIONS = ("regular", "chanheavy", "speechheavy")
SYSTEM_TOKENS = {
    "GPT minimal": "openai_minimal",
    "GPT xhigh": "openai_xhigh",
    "Gemini high": "gemini_high",
    "Grok": "xai_10",
}


def _transcript_input(system_token: str, realization: str) -> str:
    date = "2026-09-02"
    if system_token == "openai_minimal" and realization != "regular":
        date = "2026-09-05"
    return str(
        RELEASE_ROOT
        / "transcripts"
        / f"main_runs__modeb_{system_token}_{realization}_{date}.jsonl"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, p_values[index] * (len(p_values) - rank))
        adjusted[index] = min(1.0, running)
    return adjusted


def _paired_permutation_p(
    differences: np.ndarray,
    *,
    num_permutations: int,
    seed: int,
) -> float:
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    exceedances = 0
    completed = 0
    batch_size = 5_000
    while completed < num_permutations:
        batch = min(batch_size, num_permutations - completed)
        signs = rng.choice((-1.0, 1.0), size=(batch, len(differences)))
        statistics = np.abs((signs * differences).mean(axis=1))
        exceedances += int((statistics >= observed - 1e-15).sum())
        completed += batch
    return (exceedances + 1) / (num_permutations + 1)


def _bca_interval(
    values: np.ndarray,
    *,
    num_resamples: int,
    seed: int,
) -> list[float]:
    values = np.asarray(values, dtype=float)
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(num_resamples, len(values)))
    bootstrap_means = values[indices].mean(axis=1)

    proportion = np.clip(
        np.mean(bootstrap_means < observed),
        1e-10,
        1 - 1e-10,
    )
    bias = norm.ppf(proportion)
    jackknife = (values.sum() - values) / (len(values) - 1)
    jackknife_mean = jackknife.mean()
    numerator = np.sum((jackknife_mean - jackknife) ** 3)
    denominator = 6 * np.sum((jackknife_mean - jackknife) ** 2) ** 1.5
    acceleration = numerator / denominator if denominator else 0.0

    quantiles = []
    for tail in (0.025, 0.975):
        z_tail = norm.ppf(tail)
        quantile = norm.cdf(
            bias + (bias + z_tail) / (1 - acceleration * (bias + z_tail))
        )
        quantiles.append(
            float(np.clip(quantile, 1 / num_resamples, 1 - 1 / num_resamples))
        )
    return [float(np.quantile(bootstrap_means, quantile)) for quantile in quantiles]


def _cochran_q(outcomes: np.ndarray) -> tuple[float, float]:
    num_realizations = outcomes.shape[1]
    column_totals = outcomes.sum(axis=0)
    row_totals = outcomes.sum(axis=1)
    total = column_totals.sum()
    denominator = num_realizations * total - np.sum(row_totals**2)
    statistic = 0.0
    if denominator:
        statistic = (
            (num_realizations - 1)
            * (num_realizations * np.sum(column_totals**2) - total**2)
            / denominator
        )
    return float(statistic), float(chi2.sf(statistic, num_realizations - 1))


def _load_inputs(repo_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    task_order: tuple[str, ...] | None = None
    release_root = repo_root / RELEASE_ROOT
    corrections = load_correction_map(release_root)
    for system, token in SYSTEM_TOKENS.items():
        by_realization: dict[str, dict[str, float]] = {}
        sources: dict[str, Any] = {}
        for realization in REALIZATIONS:
            relative_path = _transcript_input(token, realization)
            result_path = repo_root / relative_path
            rows = [
                json.loads(line)
                for line in result_path.read_text().splitlines()
                if line
            ]
            rewards = {
                str(row["task_id"]): corrected_reward(row, corrections) for row in rows
            }
            if len(rows) != 200 or len(rewards) != 200:
                raise ValueError(f"Expected 200 unique tasks in {relative_path}")
            by_realization[realization] = rewards
            sources[realization] = {
                "path": relative_path,
                "sha256": _sha256(result_path),
            }
        current_order = tuple(sorted(by_realization["regular"]))
        if task_order is None:
            task_order = current_order
        elif current_order != task_order:
            raise ValueError(f"Task alignment differs for {system}")
        assert task_order is not None
        outcomes = np.asarray(
            [
                [by_realization[realization][task_id] for realization in REALIZATIONS]
                for task_id in task_order
            ],
            dtype=float,
        )
        if outcomes.shape != (200, 3):
            raise ValueError(
                f"Expected a 200 x 3 matrix for {system}, got {outcomes.shape}"
            )
        if not np.isin(outcomes, (0.0, 1.0)).all():
            raise ValueError(f"Non-binary or null reward for {system}")
        arrays[system] = outcomes
        provenance[system] = {
            "realizations": list(REALIZATIONS),
            "sources": sources,
        }
    return arrays, provenance


def analyze(repo_root: Path, *, seed: int = 42) -> dict[str, Any]:
    num_permutations = 100_000
    num_resamples = 10_000
    arrays, provenance = _load_inputs(repo_root)
    release_root = repo_root / RELEASE_ROOT
    correction_path = release_root / DEFAULT_ARTIFACT
    corrections = load_correction_map(release_root)

    systems: dict[str, Any] = {}
    realization_p_values: list[float] = []
    for system, outcomes in arrays.items():
        regular_outcomes = outcomes[:, 0]
        robust_outcomes = outcomes.min(axis=1)
        gap = regular_outcomes - robust_outcomes
        statistic, p_value = _cochran_q(outcomes)
        realization_p_values.append(p_value)
        systems[system] = {
            "per_realization_passes": outcomes.sum(axis=0).astype(int).tolist(),
            "per_realization_rates": outcomes.mean(axis=0).tolist(),
            "pass_at_1": float(regular_outcomes.mean()),
            "pass_at_1_bca_95": _bca_interval(
                regular_outcomes, num_resamples=num_resamples, seed=seed
            ),
            "pass_robust_3": float(robust_outcomes.mean()),
            "pass_robust_3_bca_95": _bca_interval(
                robust_outcomes, num_resamples=num_resamples, seed=seed
            ),
            "pass_at_1_minus_robust": float(gap.mean()),
            "gap_bca_95": _bca_interval(gap, num_resamples=num_resamples, seed=seed),
            "tasks_changing_outcome": int(
                (outcomes.min(axis=1) != outcomes.max(axis=1)).sum()
            ),
            "cochran_q": statistic,
            "cochran_p": p_value,
        }
    for system, adjusted in zip(systems, _holm(realization_p_values), strict=True):
        systems[system]["cochran_p_holm"] = adjusted

    provider_pairs: list[dict[str, Any]] = []
    pass_p_values: list[float] = []
    robust_p_values: list[float] = []
    for system_a, system_b in combinations(arrays, 2):
        outcomes_a = arrays[system_a]
        outcomes_b = arrays[system_b]
        pass_difference = outcomes_a[:, 0] - outcomes_b[:, 0]
        robust_difference = outcomes_a.min(axis=1) - outcomes_b.min(axis=1)
        pass_p = _paired_permutation_p(
            pass_difference,
            num_permutations=num_permutations,
            seed=seed,
        )
        robust_p = _paired_permutation_p(
            robust_difference,
            num_permutations=num_permutations,
            seed=seed,
        )
        pass_p_values.append(pass_p)
        robust_p_values.append(robust_p)
        provider_pairs.append(
            {
                "system_a": system_a,
                "system_b": system_b,
                "delta_pass_at_1_a_minus_b": float(pass_difference.mean()),
                "delta_pass_at_1_bca_95": _bca_interval(
                    pass_difference,
                    num_resamples=num_resamples,
                    seed=seed,
                ),
                "pass_permutation_p": pass_p,
                "delta_robust_a_minus_b": float(robust_difference.mean()),
                "delta_robust_bca_95": _bca_interval(
                    robust_difference,
                    num_resamples=num_resamples,
                    seed=seed,
                ),
                "robust_permutation_p": robust_p,
            }
        )
    for row, pass_adjusted, robust_adjusted in zip(
        provider_pairs,
        _holm(pass_p_values),
        _holm(robust_p_values),
        strict=True,
    ):
        row["pass_permutation_p_holm"] = pass_adjusted
        row["robust_permutation_p_holm"] = robust_adjusted

    return {
        "instrument": "tau-elicit-main-significance",
        "instrument_version": "1.3.0",
        "seed": seed,
        "num_permutations": num_permutations,
        "num_bootstrap_resamples": num_resamples,
        "analysis_unit": (
            "200 aligned agent-directed tasks with regular, noise-heavy, and "
            "speech-heavy realizations"
        ),
        "provider_method": (
            "Two-sided paired sign-permutation test on regular-realization Pass@1 "
            "or all-realization pass; Holm correction across six provider pairs; "
            "+1 Monte Carlo correction; paired BCa 95% intervals."
        ),
        "realization_method": (
            "Cochran Q across three paired binary realization outcomes within "
            "each system; Holm correction across four systems."
        ),
        "scoring_correction": {
            "path": str(DEFAULT_ARTIFACT),
            "sha256": _sha256(correction_path),
            "corrected_calls": len(corrections),
        },
        "inputs": provenance,
        "systems": systems,
        "provider_pairs": provider_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.repo_root.resolve())
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        output = args.output
        if not output.is_absolute():
            output = args.repo_root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
