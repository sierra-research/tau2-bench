# Copyright Sierra
"""Offline tests for τ-Multilingual paper reproduction contracts."""

import hashlib
import json

from tau2.paper.multilingual import (
    PAPER_TASK_SUCCESS,
    PAPER_TEXT_VOICE_GAPS,
    ResultCell,
    TableRow,
    _canonical_task_id,
    _cohort_fingerprint,
    _text_voice_gap_table,
    _trial_stability_table,
    _validate_cells,
    _voice_result_path,
)


def _cell(**updates) -> ResultCell:
    payload = dict(
        cohort="voice",
        language="en",
        domain="airline",
        system="openai_minimal",
        results_path="main_runs/example/results.json",
        sha256="a" * 64,
        git_commit="b" * 40,
        rows=100,
        unique_tasks=50,
        trials=[0, 1],
        trial_success={0: 0.5, 1: 0.6},
        provider="openai",
        model="gpt-realtime-2",
        reasoning_effort="minimal",
        task_subset="airline_50",
        task_ids_sha256="c" * 64,
        matches_expected_subset=True,
    )
    payload.update(updates)
    return ResultCell(**payload)


def test_canonical_task_id_removes_only_language_variant_suffix():
    assert _canonical_task_id("task_1_hi_identity", "hi") == "task_1"
    assert _canonical_task_id("task_1_hi_identity_native", "hi") == "task_1"
    assert _canonical_task_id("task_1_hi", "hi") == "task_1"
    assert _canonical_task_id("task_1_en", "hi") == "task_1_en"


def test_explicit_voice_paths_select_new_xai_v2_cohort(tmp_path):
    path = _voice_result_path(tmp_path, "pt", "telecom", "xai_provider_default")
    assert path == (
        tmp_path
        / "main_runs"
        / "telecom_xai_v2_portuguese_telecom"
        / "pt_telecom_xai_provider_default"
        / "results.json"
    )


def test_fingerprint_has_a_fully_specified_serialization():
    cell = _cell(results_path="b/results.json", sha256="1" * 64, rows=100)
    other = _cell(results_path="a/results.json", sha256="2" * 64, rows=50)
    expected = hashlib.sha256(
        (f"a/results.json\t{'2' * 64}\t50\nb/results.json\t{'1' * 64}\t100\n").encode()
    ).hexdigest()
    assert _cohort_fingerprint([cell, other]) == expected


def test_missing_subset_metadata_is_warning_when_ids_prove_the_frame():
    [finding] = _validate_cells([_cell(task_subset=None)], voice=True)
    assert finding.status == "warning"
    assert "task ids match" in finding.message


def test_wrong_xai_model_fails():
    finding = _validate_cells(
        [
            _cell(
                system="xai_provider_default",
                rows=50,
                trials=[0],
                trial_success={0: 0.5},
                provider="xai",
                reasoning_effort="provider_default",
                model="grok-voice-think-fast",
            )
        ],
        voice=True,
    )[0]
    assert finding.status == "fail"
    assert "grok-voice-think-fast-1.0" in finding.message


def test_claim_table_has_six_languages_and_five_systems():
    assert set(PAPER_TASK_SUCCESS) == {"en", "es", "pt", "hi", "ko", "zh"}
    assert all(len(row) == 5 for row in PAPER_TASK_SUCCESS.values())
    json.dumps(PAPER_TASK_SUCCESS)


def test_text_voice_gap_uses_balanced_five_system_voice_mean():
    task_rows = []
    text_rows = []
    for language, values in PAPER_TASK_SUCCESS.items():
        task_rows.append(
            TableRow(
                language=language,
                values={
                    name: value
                    for name, value in zip(
                        (
                            "openai_minimal",
                            "openai_xhigh",
                            "gemini_minimal",
                            "gemini_high",
                            "xai_provider_default",
                        ),
                        values,
                        strict=True,
                    )
                },
            )
        )
        voice = sum(values) / len(values)
        text_rows.append(
            TableRow(
                language=language,
                values={
                    "gpt55_xhigh": 0.0,
                    "gemini31pro_high": 0.0,
                    "pooled": voice + PAPER_TEXT_VOICE_GAPS[language],
                },
            )
        )
    observed = _text_voice_gap_table(task_rows, text_rows)
    assert {
        row.language: round(row.values["gap"], 1) for row in observed
    } == PAPER_TEXT_VOICE_GAPS


def test_trial_stability_pools_domains_before_computing_system_gap():
    cells = []
    for domain, trial_1 in zip(
        ("airline", "retail", "telecom"), (0.0, 1.0, 0.5), strict=True
    ):
        for system in (
            "openai_minimal",
            "openai_xhigh",
            "gemini_minimal",
            "gemini_high",
        ):
            cells.append(
                _cell(
                    language="en",
                    domain=domain,
                    system=system,
                    trial_success={0: 0.5, 1: trial_1},
                )
            )
    # Fill the remaining language keys with identical synthetic cells so the
    # table's closed six-language contract is exercised.
    for language in ("es", "pt", "hi", "ko", "zh"):
        cells.extend(
            cell.model_copy(update={"language": language}) for cell in cells[:12]
        )
    row = _trial_stability_table(cells)[0]
    assert row.values["trial_0"] == 50.0
    assert row.values["trial_1"] == 50.0
    assert row.values["max_system_gap"] == 0.0
