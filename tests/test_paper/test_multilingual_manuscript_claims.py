# Copyright Sierra
"""Keep the tau-Multilingual manuscript's reported values artifact-backed."""

import csv
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from experiments.tau_multilingual.experience_without_fluency import (
    UtteranceExperienceArtifact,
)
from experiments.tau_multilingual.task_success_significance import (
    TaskComparison,
    TaskSuccessAnalysisArtifact,
)
from tau2.paper.latency import MultilingualLatencyArtifact
from tau2.paper.retail_replacement import RetailAblationReplacementManifest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "papers" / "tau-multilingual"
MAIN_TEX = PAPER_ROOT / "v3" / "main.tex"
REPRODUCTION_ROOT = PAPER_ROOT / "reproduction"
ANALYSIS_ROOT = REPO_ROOT / "data" / "analysis"
VALIDATION_ROOT = (
    REPO_ROOT
    / "data"
    / "simulations"
    / "paper_runs"
    / "tau-multi"
    / "validation_runs"
    / "pre-retail-name-role-v1"
    / "human_annotations"
    / "validations"
)


def _normalized_tex() -> str:
    return re.sub(r"\s+", " ", MAIN_TEX.read_text()).strip()


def _one_decimal(value: float) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _task_success() -> TaskSuccessAnalysisArtifact:
    return TaskSuccessAnalysisArtifact.model_validate_json(
        (
            ANALYSIS_ROOT / "tau_multilingual_task_success_significance_2026-09-18.json"
        ).read_text()
    )


def _experience() -> UtteranceExperienceArtifact:
    return UtteranceExperienceArtifact.model_validate_json(
        (
            ANALYSIS_ROOT
            / "tau_multilingual_experience_without_fluency_2026-09-18.json"
        ).read_text()
    )


def _loss_interval(comparison: TaskComparison) -> tuple[float, float, float]:
    assert comparison.delta_points < 0
    assert comparison.ci_low_points < comparison.ci_high_points < 0
    return (
        -comparison.delta_points,
        -comparison.ci_high_points,
        -comparison.ci_low_points,
    )


def test_task_success_claims_match_typed_artifact() -> None:
    artifact = _task_success()
    trial_zero = artifact.descriptive_trial_0
    tex = _normalized_tex()

    english = trial_zero.voice_language_mean["en"]
    preserved_gap = max(
        abs(trial_zero.voice_language_mean[language] - english)
        for language in ("es", "pt", "hi")
    )
    korean_gap = english - trial_zero.voice_language_mean["ko"]
    mandarin_gap = english - trial_zero.voice_language_mean["zh"]
    assert (
        f"within {preserved_gap * 100:.1f} task-completion points of English, "
        f"but Korean and Mandarin fall by {korean_gap * 100:.1f} and "
        f"{mandarin_gap * 100:.1f} points."
    ) in tex

    comparisons = {
        comparison.language: comparison
        for comparison in artifact.fixed_system_panel.comparisons
    }
    ko_loss, ko_low, ko_high = _loss_interval(comparisons["ko"])
    zh_loss, zh_low, zh_high = _loss_interval(comparisons["zh"])
    assert (
        "confirmatory English-relative losses for Korean and Mandarin are "
        f"{_one_decimal(ko_loss)} points "
        f"(95\\% CI: {_one_decimal(ko_low)}--{_one_decimal(ko_high)}) and "
        f"{_one_decimal(zh_loss)} points "
        f"({_one_decimal(zh_low)}--{_one_decimal(zh_high)}), respectively"
    ) in tex
    assert comparisons["ko"].holm_p == comparisons["zh"].holm_p == 0.0

    gaps = trial_zero.text_voice_gap_points
    assert (
        "The voice--text gap favors pooled text systems in every language by "
        f"{min(gaps.values()):.1f}--{max(gaps.values()):.1f} points."
    ) in tex
    assert (
        f"text reaches {trial_zero.text_language_mean['ko'] * 100:.1f}\\% "
        f"versus {trial_zero.voice_language_mean['ko'] * 100:.1f}\\% voice."
    ) in tex


def test_corrected_mandarin_ablation_table_matches_replacement_manifest() -> None:
    manifest = RetailAblationReplacementManifest.model_validate_json(
        (REPRODUCTION_ROOT / "retail_ablation_name_role_replacement.json").read_text()
    )
    with (REPRODUCTION_ROOT / "retail_ablations.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    mandarin = {row["system"]: row for row in rows if row["language"] == "zh"}
    assert set(mandarin) == {"openai_xhigh", "gemini_high"}
    corrected = {
        (cell.condition, cell.system): cell.replacement.trial_success["0"] * 100
        for cell in manifest.cells
    }
    for system, row in mandarin.items():
        assert float(row["source"]) == pytest.approx(
            corrected[("source_english_identity", system)]
        )
        assert float(row["native"]) == pytest.approx(
            corrected[("native_script_database", system)]
        )

    labels = {"openai_xhigh": "GPT xhigh", "gemini_high": "Gemini high"}
    tex = MAIN_TEX.read_text()
    for system, label in labels.items():
        row = mandarin[system]
        values = " & ".join(
            f"{float(row[column]):.1f}"
            for column in ("localized", "source", "romanized", "native")
        )
        language = "Mandarin" if system == "openai_xhigh" else ""
        assert f"{language} & {label} & {values} \\\\" in tex


def test_model_latency_and_significance_claims_match_typed_artifacts() -> None:
    task_success = _task_success().descriptive_trial_0
    experience = _experience()
    latency = MultilingualLatencyArtifact.model_validate_json(
        (REPRODUCTION_ROOT / "latency.json").read_text()
    )
    tex = _normalized_tex()
    providers = experience.descriptive_complete_cohort.provider

    assert (
        "with a six-language mean of "
        f"{task_success.voice_system_mean['xai_provider_default'] * 100:.1f}\\%, "
        f"but has the lowest Generation score ({providers['xAI'].experience:.1f}\\%)."
    ) in tex
    assert (
        f"GPT xhigh leads Generation ({providers['OpenAI xhigh'].experience:.1f}\\%) "
        f"at {task_success.voice_system_mean['openai_xhigh'] * 100:.1f}\\% task "
        "completion"
    ) in tex
    assert (
        "GPT minimal leads Interaction "
        f"({100 - providers['OpenAI minimal'].interaction_failure:.1f}\\%)"
    ) in tex

    localized_codes = ("es", "pt", "hi", "ko", "zh")
    grok_values = [
        task_success.voice_language_system[language]["xai_provider_default"] * 100
        for language in localized_codes
    ]
    localized_names = ("Spanish", "Portuguese", "Hindi", "Korean", "Mandarin")
    gemini_high_interaction = [
        100
        - experience.descriptive_complete_cohort.language_system[language][
            "Gemini high"
        ].interaction_failure
        for language in localized_names
    ]
    assert (
        "Grok's "
        f"{max(grok_values) - min(grok_values):.1f} points in task completion and "
        "Gemini high's "
        f"{max(gemini_high_interaction) - min(gemini_high_interaction):.1f} points "
        "in Interaction."
    ) in tex

    latency_by_provider = {
        name: rollup.latency_seconds
        for name, rollup in latency.descriptive_complete_cohort.provider.items()
    }
    gpt_latency = [
        latency_by_provider["OpenAI minimal"],
        latency_by_provider["OpenAI xhigh"],
    ]
    gemini_latency = [
        latency_by_provider["Gemini minimal"],
        latency_by_provider["Gemini high"],
    ]
    non_xai_durations = [
        summary.mean_call_duration_minutes
        for name, summary in providers.items()
        if name != "xAI"
    ]
    assert (
        "turn-taking latency "
        f"(GPT: {min(gpt_latency):.2f}--{max(gpt_latency):.2f} s; "
        f"Gemini: {min(gemini_latency):.2f}--{max(gemini_latency):.2f} s). "
        f"Mean call durations are {min(non_xai_durations):.1f}--"
        f"{max(non_xai_durations):.1f} min for GPT/Gemini versus "
        f"{providers['xAI'].mean_call_duration_minutes:.1f} min for Grok."
    ) in tex

    interaction_tests = experience.interaction_language_significance.comparisons
    assert all(comparison.significant_0_05 for comparison in interaction_tests)
    declines = [
        abs(comparison.delta_vs_english_points) for comparison in interaction_tests
    ]
    decline_text = ", ".join(f"{value:.1f}" for value in declines[:-1])
    decline_text += f", and {declines[-1]:.1f}"
    max_interaction_p = max(comparison.holm_p for comparison in interaction_tests)
    assert (
        f"All English-relative Interaction declines are significant: {decline_text} "
        "points for Spanish, Portuguese, Hindi, Korean, and Mandarin "
        f"(Holm-adjusted $p\\leq{max_interaction_p:.5f}$).".replace("0.", ".")
    ) in tex

    provider_tests = experience.significance_complete_matched_cohort.pairs
    assert len(provider_tests) == 10
    assert all(comparison.significant_0_05 for comparison in provider_tests)
    weakest = max(provider_tests, key=lambda comparison: comparison.holm_p)
    assert (weakest.system_a, weakest.system_b) == ("Gemini minimal", "Gemini high")
    assert (
        "weakest is Gemini minimal versus high "
        f"({abs(weakest.delta_a_minus_b_points):.1f} points; Holm-adjusted "
        f"$p={weakest.holm_p:.5f}$).".replace("0.", ".")
    ) in tex


def test_fidelity_denominator_and_pooled_validation_f1_match_artifacts() -> None:
    experience = _experience()
    tex = _normalized_tex()
    providers = experience.descriptive_complete_cohort.provider
    other_fidelity = [
        summary.standalone_fidelity_cleanliness
        for name, summary in providers.items()
        if name != "xAI"
    ]
    denominator = experience.utterance_alignment.standalone_fidelity_utterances
    assert (
        "Speech fidelity reveals an all-language provider pattern: "
        f"{providers['xAI'].standalone_fidelity_cleanliness:.1f}\\% of Grok "
        "utterances contain no material audible error, versus "
        f"{min(other_fidelity):.1f}--{max(other_fidelity):.1f}\\% for the other "
        f"systems. Across {denominator:,} scored utterances"
    ) in tex

    with (VALIDATION_ROOT / "index.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    pooled_f1 = {}
    for measure in ("tool_use", "naturalness", "speech_fidelity"):
        measure_rows = [row for row in rows if row["measure_id"] == measure]
        assert measure_rows
        assert all(row["gate_status"] == "passes" for row in measure_rows)
        tp = sum(int(row["tp"]) for row in measure_rows)
        fn = sum(int(row["fn"]) for row in measure_rows)
        fp = sum(int(row["fp"]) for row in measure_rows)
        pooled_f1[measure] = 2 * tp / (2 * tp + fp + fn)

    assert (
        "Pooled validation micro-F1 is "
        f"{pooled_f1['tool_use'] * 100:.1f}\\% for tool use, "
        f"{pooled_f1['naturalness'] * 100:.1f}\\% for naturalness, and "
        f"{pooled_f1['speech_fidelity'] * 100:.1f}\\% for speech fidelity"
    ) in tex
