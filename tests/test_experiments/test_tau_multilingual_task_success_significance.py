# Copyright Sierra
"""Deterministic τ-Multilingual task-success inference helpers."""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from experiments.tau_multilingual.task_success_significance import (
    TaskSuccessAnalysisArtifact,
    _canonical_task_id,
    _sources_fingerprint,
    _text_path,
    _voice_path,
    bootstrap_delta_ci,
    holm_bonferroni,
    paired_permutation_test,
)
from tau2.paper.cli import add_paper_args


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


def test_task_success_paths_have_move_stable_canonical_identities(tmp_path):
    physical, logical, replacement = _voice_path(
        tmp_path / "active",
        None,
        "ko",
        "retail",
        "openai_xhigh",
    )
    assert physical == (
        tmp_path
        / "active/main_runs/retail_v1_korean_retail/ko_retail_openai_xhigh/results.json"
    )
    assert logical.as_posix() == (
        "main_runs/retail_v1_korean_retail/ko_retail_openai_xhigh/results.json"
    )
    assert replacement is True

    staged, logical, replacement = _text_path(
        tmp_path / "old",
        tmp_path / "replacement",
        "zh",
        "retail",
        "gemini31pro_high",
    )
    assert staged == (
        tmp_path / "replacement/multilingual_text_retail_name_roles_v1_mandarin_retail/"
        "zh_retail_gemini31pro_high/results.json"
    )
    assert logical.as_posix() == (
        "text_channel/multilingual_text_v1_mandarin_retail/"
        "zh_retail_gemini31pro_high/results.json"
    )
    assert replacement is True


def test_task_success_cli_owns_the_typed_reproduction_verb():
    parser = ArgumentParser()
    add_paper_args(parser)
    args = parser.parse_args(
        [
            "multilingual-task-success",
            "--evidence-root",
            "/tmp/evidence",
            "--out",
            "/tmp/task-success.json",
        ]
    )

    assert args.func.__name__ == "run_multilingual_task_success"
    assert args.permutations == 100_000
    assert args.bootstrap_resamples == 10_000


def test_experience_cli_accepts_a_detached_evidence_root():
    parser = ArgumentParser()
    add_paper_args(parser)
    args = parser.parse_args(
        [
            "multilingual-experience",
            "--repo-root",
            "/tmp/reviewer-repository",
            "--evidence-root",
            "/tmp/tau-multi",
            "--naturalness-sidecar",
            "/tmp/tau-multi/judge_outputs/utterance_naturalness_v16_trial0",
            "--out",
            "/tmp/experience.json",
        ]
    )

    assert args.func.__name__ == "run_multilingual_experience"
    assert args.evidence_root == Path("/tmp/tau-multi")
