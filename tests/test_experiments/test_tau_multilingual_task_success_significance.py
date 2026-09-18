# Copyright Sierra
"""Deterministic τ-Multilingual task-success inference helpers."""

import json
from pathlib import Path

import numpy as np

from experiments.tau_multilingual.task_success_significance import (
    TaskSuccessAnalysisArtifact,
    _canonical_task_id,
    _sources_fingerprint,
    bootstrap_delta_ci,
    holm_bonferroni,
    paired_permutation_test,
)


def test_task_ids_are_canonicalized_without_truncating_unrelated_suffixes():
    assert _canonical_task_id("retail-63_ko_identity_native", "ko") == "retail-63"
    assert _canonical_task_id("retail-63_ko_identity", "ko") == "retail-63"
    assert _canonical_task_id("retail-63_ko", "ko") == "retail-63"
    assert _canonical_task_id("retail-63_en", "ko") == "retail-63_en"


def test_frozen_permutation_and_bca_helpers_are_seeded():
    target = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    reference = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])

    assert (
        paired_permutation_test(
            target,
            reference,
            n_permutations=1_000,
            seed=42,
        )
        == 0.628
    )
    assert bootstrap_delta_ci(
        target,
        reference,
        n_bootstrap=1_000,
        seed=42,
    ) == (-2 / 3, 2 / 3)


def test_holm_adjustment_enforces_rank_monotonicity():
    assert holm_bonferroni([0.04, 0.01, 0.03, 0.2]) == [0.09, 0.04, 0.09, 0.2]


def test_frozen_task_and_experience_artifacts_share_voice_sources():
    repo_root = Path(__file__).resolve().parents[2]
    task_success = TaskSuccessAnalysisArtifact.model_validate_json(
        (
            repo_root / "data/analysis/"
            "tau_multilingual_task_success_significance_2026-09-18.json"
        ).read_text()
    )
    experience = json.loads(
        (
            repo_root / "data/analysis/"
            "tau_multilingual_experience_without_fluency_2026-09-18.json"
        ).read_text()
    )

    task_voice_sources = {
        source.path: source.sha256
        for source in task_success.sources
        if source.cohort == "voice"
    }
    experience_voice_sources = {
        source["path"]: source["sha256"]
        for source in experience["provenance"]["results_files"]
    }

    assert len(task_voice_sources) == 90
    assert task_voice_sources == experience_voice_sources
    assert task_success.sources_fingerprint_sha256 == _sources_fingerprint(
        task_success.sources
    )
