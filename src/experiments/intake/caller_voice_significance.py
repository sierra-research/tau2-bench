"""Reproduce the tau-Elicitation caller-voice significance analysis.

The paper compares exact task success across five randomly assigned caller
voices within each regular, agent-directed system.  For each system, this
module builds the 5 x 2 voice-by-outcome table and runs SciPy's two-sided
Fisher--Freeman--Halton Monte Carlo test.  The implementation intentionally
matches the historical analysis call, including row ordering, 100,000
resamples, NumPy seed 42, and SciPy's +1 Monte Carlo correction.

The same module also reproduces the ten two-sided pairwise Fisher exact tests
within Gemini and their Holm correction.  Inputs are the compact reviewer
transcripts, so no detached audio corpus is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import MonteCarloMethod, fisher_exact

ANALYSIS_VERSION = "1.0.0"
DEFAULT_MONTE_CARLO_RESAMPLES = 100_000
DEFAULT_SEED = 42
DEFAULT_RELEASE_ROOT = Path("papers/tau-intake/v1/reproduction")


class System(str, Enum):
    """Agent-directed systems in the caller-voice analysis."""

    GPT_XHIGH = "GPT xhigh"
    GEMINI_HIGH = "Gemini high"
    GROK = "Grok"


TRANSCRIPT_PATHS: dict[System, str] = {
    System.GPT_XHIGH: (
        "transcripts/main_runs__modeb_openai_xhigh_regular_2026-09-02.jsonl"
    ),
    System.GEMINI_HIGH: (
        "transcripts/main_runs__modeb_gemini_high_regular_2026-09-02.jsonl"
    ),
    System.GROK: "transcripts/main_runs__modeb_xai_10_regular_2026-09-02.jsonl",
}
EXPECTED_VOICES = frozenset(
    {
        "arjun_roy",
        "mamadou_diallo",
        "mildred_kaplan",
        "priya_patil",
        "wei_lin",
    }
)


class SourceRun(BaseModel):
    """Provenance for one compact regular-condition transcript."""

    system: Annotated[System, Field(description="Paper system label.")]
    transcript_path: Annotated[
        str, Field(description="Path relative to the reviewer release root.")
    ]
    sha256: Annotated[str, Field(description="SHA-256 of the compact transcript.")]
    calls: Annotated[int, Field(gt=0, description="Validated call rows.")]


class VoiceOutcome(BaseModel):
    """One caller voice's binary task outcomes."""

    voice: Annotated[str, Field(description="Stable caller persona name.")]
    successes: Annotated[int, Field(ge=0, description="Successful calls.")]
    failures: Annotated[int, Field(ge=0, description="Failed calls.")]
    calls: Annotated[int, Field(gt=0, description="Total calls.")]
    pass_at_1: Annotated[
        float, Field(ge=0, le=1, description="Exact task success rate.")
    ]


class OmnibusResult(BaseModel):
    """One system's 5 x 2 Fisher--Freeman--Halton result."""

    system: Annotated[System, Field(description="Paper system label.")]
    table: Annotated[
        list[VoiceOutcome],
        Field(description="Voice rows in the order passed to SciPy."),
    ]
    monte_carlo_p: Annotated[
        float, Field(ge=0, le=1, description="Raw two-sided Monte Carlo p-value.")
    ]
    holm_p: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="Holm-adjusted p-value across the three systems.",
        ),
    ]


class PairwiseResult(BaseModel):
    """One two-sided Fisher exact comparison between Gemini voices."""

    voice_a: Annotated[str, Field(description="First caller voice.")]
    voice_b: Annotated[str, Field(description="Second caller voice.")]
    fisher_p: Annotated[
        float, Field(ge=0, le=1, description="Raw two-sided Fisher exact p-value.")
    ]
    holm_p: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="Holm-adjusted p-value across ten Gemini voice pairs.",
        ),
    ]


class CallerVoiceArtifact(BaseModel):
    """Versioned, provenance-bearing caller-voice analysis artifact."""

    instrument: Annotated[
        str, Field(description="Stable analysis instrument name.")
    ] = "tau-intake-caller-voice-significance"
    instrument_version: Annotated[
        str, Field(description="Version of the analysis contract.")
    ] = ANALYSIS_VERSION
    seed: Annotated[int, Field(description="NumPy Monte Carlo RNG seed.")]
    monte_carlo_resamples: Annotated[
        int, Field(gt=0, description="Monte Carlo tables drawn per omnibus test.")
    ]
    analysis_unit: Annotated[str, Field(description="Unit entering each test.")]
    omnibus_method: Annotated[str, Field(description="Omnibus test specification.")]
    pairwise_method: Annotated[str, Field(description="Pairwise test specification.")]
    inputs: Annotated[
        list[SourceRun], Field(description="Hashed compact-transcript inputs.")
    ]
    omnibus: Annotated[
        list[OmnibusResult], Field(description="Three system-level omnibus tests.")
    ]
    gemini_pairwise: Annotated[
        list[PairwiseResult],
        Field(description="Ten post-hoc comparisons within Gemini."),
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, p_values[index] * (len(p_values) - rank))
        adjusted[index] = min(1.0, running)
    return adjusted


def _load_counts(
    release_root: Path,
) -> tuple[dict[System, dict[str, tuple[int, int]]], list[SourceRun]]:
    by_system: dict[System, dict[str, tuple[int, int]]] = {}
    sources: list[SourceRun] = []
    for system, relative_path in TRANSCRIPT_PATHS.items():
        path = release_root / relative_path
        counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        task_ids: set[str] = set()
        calls = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                reward = row.get("reward")
                if reward not in (0, 0.0, 1, 1.0):
                    raise ValueError(
                        f"Non-binary reward in {relative_path}:{line_number}"
                    )
                task_id = str(row.get("task_id") or "")
                if not task_id or task_id in task_ids:
                    raise ValueError(
                        f"Missing or duplicate task id in {relative_path}:{line_number}"
                    )
                task_ids.add(task_id)
                voice = str((row.get("speech_environment") or {}).get("persona_name"))
                if voice not in EXPECTED_VOICES:
                    raise ValueError(
                        f"Unexpected caller voice {voice!r} in {relative_path}"
                    )
                success = int(reward)
                counts[voice][0] += success
                counts[voice][1] += 1 - success
                calls += 1
        if calls != 200 or set(counts) != EXPECTED_VOICES:
            raise ValueError(
                f"Expected 200 calls across five voices in {relative_path}; "
                f"found {calls} calls across {len(counts)} voices"
            )
        by_system[system] = {
            voice: (values[0], values[1]) for voice, values in counts.items()
        }
        sources.append(
            SourceRun(
                system=system,
                transcript_path=relative_path,
                sha256=_sha256(path),
                calls=calls,
            )
        )
    return by_system, sources


def _ordered_table(counts: dict[str, tuple[int, int]]) -> list[VoiceOutcome]:
    # This is the historical row order: descending number of assigned calls,
    # with the voice name as an explicit deterministic tie-breaker.
    voices = sorted(counts, key=lambda voice: (-sum(counts[voice]), voice))
    return [
        VoiceOutcome(
            voice=voice,
            successes=counts[voice][0],
            failures=counts[voice][1],
            calls=sum(counts[voice]),
            pass_at_1=counts[voice][0] / sum(counts[voice]),
        )
        for voice in voices
    ]


def _omnibus(
    system: System,
    counts: dict[str, tuple[int, int]],
    *,
    resamples: int,
    seed: int,
) -> OmnibusResult:
    table = _ordered_table(counts)
    values = np.asarray(
        [[row.successes, row.failures] for row in table], dtype=np.int64
    )
    method = MonteCarloMethod(
        n_resamples=resamples,
        rng=np.random.default_rng(seed),
    )
    p_value = float(fisher_exact(values, method=method).pvalue)
    return OmnibusResult(
        system=system,
        table=table,
        monte_carlo_p=p_value,
        holm_p=0.0,
    )


def _gemini_pairwise(counts: dict[str, tuple[int, int]]) -> list[PairwiseResult]:
    rows = []
    for voice_a, voice_b in combinations(sorted(counts), 2):
        p_value = float(fisher_exact([counts[voice_a], counts[voice_b]]).pvalue)
        rows.append(
            PairwiseResult(
                voice_a=voice_a,
                voice_b=voice_b,
                fisher_p=p_value,
                holm_p=0.0,
            )
        )
    for row, adjusted in zip(
        rows,
        _holm([row.fisher_p for row in rows]),
        strict=True,
    ):
        row.holm_p = adjusted
    return rows


def analyze(
    repo_root: Path,
    *,
    release_root: Path = DEFAULT_RELEASE_ROOT,
    resamples: int = DEFAULT_MONTE_CARLO_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> CallerVoiceArtifact:
    """Run the caller-voice omnibus and Gemini pairwise analyses."""
    resolved_release_root = (
        release_root if release_root.is_absolute() else repo_root / release_root
    )
    counts, sources = _load_counts(resolved_release_root)
    omnibus = [
        _omnibus(system, counts[system], resamples=resamples, seed=seed)
        for system in System
    ]
    for result, adjusted in zip(
        omnibus,
        _holm([result.monte_carlo_p for result in omnibus]),
        strict=True,
    ):
        result.holm_p = adjusted
    return CallerVoiceArtifact(
        seed=seed,
        monte_carlo_resamples=resamples,
        analysis_unit=(
            "One regular-condition, agent-directed call with binary exact task "
            "success and one randomly assigned caller voice."
        ),
        omnibus_method=(
            "Two-sided 5 x 2 Fisher--Freeman--Halton test via scipy.stats."
            "fisher_exact and MonteCarloMethod; each system resets NumPy's "
            "default_rng to the recorded seed; voice rows are ordered by "
            "descending assigned-call count; SciPy applies the +1 Monte Carlo "
            "correction; Holm correction spans the three systems."
        ),
        pairwise_method=(
            "Two-sided Fisher exact tests for all ten Gemini voice pairs; Holm "
            "correction spans those ten comparisons."
        ),
        inputs=sources,
        omnibus=omnibus,
        gemini_pairwise=_gemini_pairwise(counts[System.GEMINI_HIGH]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce caller-voice omnibus and pairwise tests."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--monte-carlo-resamples",
        type=int,
        default=DEFAULT_MONTE_CARLO_RESAMPLES,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    artifact = analyze(
        repo_root,
        release_root=args.release_root,
        resamples=args.monte_carlo_resamples,
        seed=args.seed,
    )
    rendered = artifact.model_dump_json(indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
        return
    output = args.output
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)


if __name__ == "__main__":
    main()
