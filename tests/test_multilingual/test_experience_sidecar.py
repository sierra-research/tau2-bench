# Copyright Sierra
"""Strict sidecar integration for the tau-Multilingual Experience analysis."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.tau_multilingual.experience_without_fluency import (
    INTERACTION_COMPONENT_FACTORS,
    INTERACTION_STAT_FIELDS,
    STAT_FIELDS,
    NaturalnessSidecarInput,
    NaturalnessUtteranceVerdict,
    _aggregate,
    _call_level_interaction_components,
    _interaction_pass_from_stats,
    _interaction_sufficient_stats,
    _load_naturalness_sidecar,
    _scores_from_stats,
    _standalone_fidelity_counts,
    _utterance_experience_counts,
    _validate_sidecar_cohort,
)
from tau2.data_model.simulation import (
    JudgeOutcome,
    NativenessFactorCheck,
    NativenessJudgeUnitResult,
)
from tau2.judges.nativeness.harness import AgentTurn
from tau2.judges.nativeness.paper_trial import (
    CANONICAL_EXPERIENCE_PATH,
    ExperienceSource,
    TrialCounts,
    TrialRunIdentity,
    TrialRunManifest,
    TrialSimulationArtifact,
    TrialSourceIdentity,
    TrialUtteranceArtifact,
    _aggregate_trial,
    frozen_judge_spec,
)
from tau2.judges.nativeness.validation import runtime_combined_factor


def _value_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _check(outcome: JudgeOutcome, *, turn_index: int = 0) -> NativenessFactorCheck:
    factor = runtime_combined_factor("hi")
    violated = outcome is JudgeOutcome.FAIL
    unit = NativenessJudgeUnitResult(
        unit_index=turn_index,
        opportunity=True,
        violated=violated,
        severity=1 if violated else 0,
        reasoning="fixture verdict",
        quote="अटपटा" if violated else "",
    )
    return NativenessFactorCheck(
        id="natural_word_choice",
        category=factor.category,
        severity=factor.severity,
        outcome=outcome,
        shadow=factor.shadow,
        evaluation_level="utterance",
        observed_severity=unit.severity,
        violation_count=int(violated),
        unit_results=[unit],
    )


def _write_sidecar(
    tmp_path: Path, *, drift_manifest_aggregate: bool = False
) -> tuple[Path, ExperienceSource, TrialSourceIdentity, TrialSimulationArtifact]:
    results_path = "data/simulations/paper_runs/tau-multi/main_runs/hi/results.json"
    results_sha = "a" * 64
    source = TrialSourceIdentity(
        results_path=results_path,
        results_sha256=results_sha,
        source_key=hashlib.sha256(results_path.encode()).hexdigest()[:16],
        language="hi",
        domain="airline",
        system_slug="openai_minimal",
        trial_0_calls=1,
    )
    experience_source = ExperienceSource(
        path=results_path,
        sha256=results_sha,
        trial_0_calls=1,
    )
    judge = frozen_judge_spec("hi")
    turn = AgentTurn(index=0, text="यह एक परीक्षण है।")
    check = _check(JudgeOutcome.PASS)
    simulation_sha256 = "c" * 64
    utterance_input_sha256 = _value_hash(
        {
            "schema_version": "tau-multi-naturalness-trial0-v1",
            "source": source.model_dump(mode="json"),
            "simulation_id": "sim-1",
            "task_id": "task-1_hi",
            "trial": 0,
            "language": "hi",
            "domain": "airline",
            "source_simulation_sha256": simulation_sha256,
            "judge": judge.model_dump(mode="json"),
            "turn": turn.model_dump(mode="json"),
        }
    )
    utterance = TrialUtteranceArtifact(
        schema_version="tau-multi-naturalness-trial0-v1",
        created_at="2026-09-04T00:00:00+00:00",
        input_sha256=utterance_input_sha256,
        source=source,
        simulation_id="sim-1",
        task_id="task-1_hi",
        trial=0,
        language="hi",
        domain="airline",
        source_simulation_sha256=simulation_sha256,
        judge=judge,
        turn=turn,
        check=check,
    )
    simulation = TrialSimulationArtifact(
        schema_version="tau-multi-naturalness-trial0-v1",
        created_at=utterance.created_at,
        input_sha256=_value_hash([utterance_input_sha256]),
        source=source,
        simulation_id=utterance.simulation_id,
        task_id=utterance.task_id,
        trial=0,
        language="hi",
        domain="airline",
        source_simulation_sha256=utterance.source_simulation_sha256,
        judge=judge,
        utterances=[utterance],
        check=check,
    )
    fingerprint_payload = (
        f"{experience_source.path}\t{experience_source.sha256}\t"
        f"{experience_source.trial_0_calls}"
    )
    identity = TrialRunIdentity(
        schema_version="tau-multi-naturalness-trial0-v1",
        experience_path=CANONICAL_EXPERIENCE_PATH.as_posix(),
        experience_sha256="e" * 64,
        cohort_fingerprint_sha256=hashlib.sha256(
            fingerprint_payload.encode()
        ).hexdigest(),
        work_fingerprint_sha256=_value_hash(
            {
                "simulations": [(source.results_path, "sim-1", simulation_sha256)],
                "utterances": [utterance_input_sha256],
            }
        ),
        validation_evidence=None,
        runtime_source_fingerprint_sha256="1" * 64,
        all_sources=[experience_source],
        selected_sources=[source],
        judges={
            language: frozen_judge_spec(language)
            for language in ("es", "pt", "hi", "ko", "zh")
        },
        trial=0,
    )
    aggregate_source = simulation
    if drift_manifest_aggregate:
        aggregate_source = simulation.model_copy(
            update={"check": _check(JudgeOutcome.FAIL)}
        )
    manifest = TrialRunManifest(
        schema_version="tau-multi-naturalness-trial0-v1",
        created_at="2026-09-04T00:00:00+00:00",
        updated_at="2026-09-04T00:00:01+00:00",
        status="complete",
        identity_sha256=_value_hash(identity.model_dump(mode="json")),
        identity=identity,
        counts=TrialCounts(
            verified_roots=1,
            selected_roots=1,
            simulations=1,
            utterances=1,
            zero_utterance_simulations=0,
            complete_utterances=1,
            error_utterances=0,
            complete_simulations=1,
            error_simulations=0,
        ),
        aggregate=_aggregate_trial([aggregate_source]),
        invocations=[],
    )
    root = tmp_path / "naturalness-sidecar"
    artifact_path = root / "simulations" / source.source_key / "sim-1.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(simulation.model_dump_json(indent=2) + "\n")
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return root, experience_source, source, simulation


def test_interaction_components_are_call_level_and_include_tool_use():
    assert INTERACTION_COMPONENT_FACTORS["tool_use"] == {
        "incorrect_tool_parameters",
        "auth_arg_mismatch",
        "agent_caused_tool_error",
    }
    assert all(
        "redundancy" not in factors
        for factors in INTERACTION_COMPONENT_FACTORS.values()
    )

    components = _call_level_interaction_components(
        [
            {"id": "responsiveness", "outcome": "pass"},
            {"id": "yielding", "outcome": "pass"},
            {"id": "inappropriate_interruption", "outcome": "pass"},
            {"id": "backchannel_selectivity", "outcome": "pass"},
            {"id": "vocal_tic_selectivity", "outcome": "pass"},
            {"id": "non_directed_selectivity", "outcome": "pass"},
            {"id": "monologue", "outcome": "pass"},
            {"id": "incorrect_tool_parameters", "outcome": "fail"},
            {"id": "auth_arg_mismatch", "outcome": "pass"},
            {"id": "agent_caused_tool_error", "outcome": "no_opportunity"},
        ]
    )

    assert components == {
        "nonresponse": 0.0,
        "interruption": 0.0,
        "selectivity": 0.0,
        "monologue": 0.0,
        "tool_use": 1.0,
    }


def test_utterance_experience_deduplicates_across_judges_and_axes():
    naturalness = [
        NaturalnessUtteranceVerdict(
            turn_index=0,
            input_sha256="a" * 64,
            text="पहला उत्तर",
            outcome=JudgeOutcome.FAIL,
        ),
        NaturalnessUtteranceVerdict(
            turn_index=1,
            input_sha256="b" * 64,
            text="दूसरा उत्तर",
            outcome=JudgeOutcome.PASS,
        ),
    ]
    delivery = {
        "utterance_results": [
            {
                "utterance_idx": 0,
                "expected_text": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
                "outcome": "pass",
                "findings": [],
                "factor_checks": [],
            },
            {
                "utterance_idx": 8,
                "expected_text": "पहला उत्तर",
                "outcome": "fail",
                "findings": [
                    {"axis": "fidelity", "severity": 2},
                    {"axis": "fidelity", "severity": 3},
                ],
                "factor_checks": [],
            },
            {
                "utterance_idx": 13,
                "expected_text": "दूसरा उत्तर",
                "outcome": "fail",
                "findings": [
                    {"axis": "fidelity", "severity": 2},
                    {"axis": "intonation", "severity": 3},
                ],
                "factor_checks": [{"id": "tone_meaning_flip", "outcome": "fail"}],
            },
        ]
    }

    counts = _utterance_experience_counts(naturalness, delivery, set())

    assert counts.eligible_utterances == 2
    assert counts.fluency_failures == 1
    assert counts.speech_fidelity_failures == 2
    assert counts.overlapping_failures == 1
    assert counts.experience_failures == 2
    assert counts.pinned_greetings_excluded == 1
    assert counts.unmatched_naturalness_utterances == 0
    assert counts.unmatched_delivery_utterances == 0


def test_standalone_fidelity_uses_delivery_denominator_without_naturalness():
    delivery = {
        "utterance_results": [
            {
                "utterance_idx": 0,
                "expected_text": "Pinned greeting",
                "outcome": "pass",
                "findings": [],
                "factor_checks": [],
            },
            {
                "utterance_idx": 2,
                "expected_text": "A scored failure",
                "outcome": "fail",
                "findings": [
                    {"axis": "fidelity", "severity": 2},
                    {"axis": "intonation", "severity": 3},
                ],
                "factor_checks": [],
            },
            {
                "utterance_idx": 5,
                "expected_text": "An excluded fidelity finding",
                "outcome": "pass",
                "findings": [{"axis": "fidelity", "severity": 3}],
                "factor_checks": [],
            },
            {
                "utterance_idx": 8,
                "expected_text": "Judge error",
                "outcome": "error",
                "findings": [{"axis": "fidelity", "severity": 3}],
                "factor_checks": [],
            },
        ]
    }

    counts = _standalone_fidelity_counts(delivery, {(5, 0)})

    assert counts.eligible_utterances == 2
    assert counts.failures == 1
    assert counts.pinned_greetings_excluded == 1
    assert counts.findings_excluded == 1


def test_utterance_experience_aligns_truncated_text_and_gates_unscored_rows():
    naturalness = [
        NaturalnessUtteranceVerdict(
            turn_index=0,
            input_sha256="a" * 64,
            text="यह एक लंबा स्वाभाविक उत्तर है जो आगे जारी रहता है",
            outcome=JudgeOutcome.NO_OPPORTUNITY,
        ),
        NaturalnessUtteranceVerdict(
            turn_index=1,
            input_sha256="b" * 64,
            text="दूसरा उत्तर",
            outcome=JudgeOutcome.PASS,
        ),
    ]
    delivery = {
        "utterance_results": [
            {
                "utterance_idx": 0,
                "expected_text": "नमस्ते",
                "outcome": "pass",
                "findings": [],
                "factor_checks": [],
            },
            {
                "utterance_idx": 2,
                "expected_text": "यह एक लंबा स्वाभाविक उत्तर है",
                "outcome": "pass",
                "findings": [],
                "factor_checks": [],
            },
            {
                "utterance_idx": 5,
                "expected_text": "दूसरा उत्तर",
                "outcome": "error",
                "findings": [],
                "factor_checks": [],
            },
        ]
    }

    counts = _utterance_experience_counts(naturalness, delivery, set())

    assert counts.eligible_utterances == 0
    assert counts.pinned_greetings_excluded == 1
    assert counts.unmatched_naturalness_utterances == 0
    assert counts.unmatched_delivery_utterances == 0


def test_utterance_experience_rejects_unrelated_equal_size_inventories():
    naturalness = [
        NaturalnessUtteranceVerdict(
            turn_index=0,
            input_sha256="a" * 64,
            text="पहला उत्तर",
            outcome=JudgeOutcome.PASS,
        )
    ]
    delivery = {
        "utterance_results": [
            {
                "utterance_idx": 0,
                "expected_text": "नमस्ते",
                "outcome": "pass",
                "findings": [],
                "factor_checks": [],
            },
            {
                "utterance_idx": 3,
                "expected_text": "इसका कोई संबंध नहीं है",
                "outcome": "pass",
                "findings": [],
                "factor_checks": [],
            },
        ]
    }

    with pytest.raises(ValueError, match="inventories do not align"):
        _utterance_experience_counts(naturalness, delivery, set())


def test_experience_score_is_the_utterance_union_complement():
    assert STAT_FIELDS == ("experience_failed", "experience_total")
    stats = np.asarray([1.0, 4.0])
    assert float(_scores_from_stats(stats)) == pytest.approx(75.0)

    aggregate = _aggregate(
        [
            {
                "interaction_components": {
                    "nonresponse": 0.0,
                    "interruption": 0.0,
                    "selectivity": 0.0,
                    "monologue": 0.0,
                    "tool_use": 1.0,
                },
                "experience_utterances": 4,
                "experience_failures": 1,
                "fluency_failures": 1,
                "speech_fidelity_failures": 1,
                "overlapping_failures": 1,
                "standalone_fidelity_utterances": 4,
                "standalone_fidelity_failures": 1,
            }
        ]
    )
    assert aggregate["interaction_failure"] == pytest.approx(0.2)
    assert aggregate["experience"] == pytest.approx(0.75)
    assert aggregate["experience_failure"] == pytest.approx(0.25)
    assert aggregate["fluency_failure"] == pytest.approx(0.25)
    assert aggregate["speech_fidelity_failure"] == pytest.approx(0.25)
    assert aggregate["standalone_fidelity_cleanliness"] == pytest.approx(0.75)


def test_interaction_stats_reaggregate_component_denominators():
    rows = [
        {
            "interaction_components": {
                "nonresponse": 1.0,
                "interruption": 0.0,
                "selectivity": 0.0,
                "monologue": None,
                "tool_use": 1.0,
            }
        },
        {
            "interaction_components": {
                "nonresponse": 0.0,
                "interruption": 0.0,
                "selectivity": 1.0,
                "monologue": 0.0,
                "tool_use": None,
            }
        },
    ]

    stats = _interaction_sufficient_stats(rows)

    assert len(stats) == len(INTERACTION_STAT_FIELDS)
    assert _interaction_pass_from_stats(stats) == pytest.approx(60.0)


def test_sidecar_loads_and_matches_exact_cohort(tmp_path):
    root, experience_source, source, _simulation = _write_sidecar(tmp_path)
    sidecar = _load_naturalness_sidecar(root)
    provenance = _validate_sidecar_cohort(
        sidecar,
        repo_root=tmp_path,
        sources=[experience_source.model_dump(mode="json")],
        selected_sources=[source],
        used_calls={(source.results_path, "sim-1")},
        work_simulations=[
            (
                source.results_path,
                sidecar.calls[0].simulation_id,
                sidecar.calls[0].source_simulation_sha256,
            )
        ],
        work_utterances=sidecar.calls[0].utterance_input_sha256s,
    )

    assert provenance.calls == 1
    assert provenance.factor_id == "natural_word_choice"
    assert sidecar.calls[0].outcome is JudgeOutcome.PASS


def test_sidecar_rejects_manifest_aggregate_drift(tmp_path):
    root, *_ = _write_sidecar(tmp_path, drift_manifest_aggregate=True)
    with pytest.raises(ValueError, match="aggregate drifted"):
        _load_naturalness_sidecar(root)


def test_sidecar_rejects_atomic_input_identity_drift(tmp_path):
    root, *_ = _write_sidecar(tmp_path)
    artifact_path = next((root / "simulations").rglob("*.json"))
    artifact = json.loads(artifact_path.read_text())
    artifact["utterances"][0]["input_sha256"] = "0" * 64
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    with pytest.raises(ValueError, match="utterance input identity drift"):
        _load_naturalness_sidecar(root)


@pytest.mark.parametrize(
    "used_calls",
    [
        set(),
        {
            (
                "data/simulations/paper_runs/tau-multi/main_runs/hi/results.json",
                "sim-1",
            ),
            ("missing/results.json", "missing-sim"),
        },
    ],
)
def test_sidecar_rejects_extra_or_missing_call(tmp_path, used_calls):
    root, experience_source, source, _simulation = _write_sidecar(tmp_path)
    sidecar = _load_naturalness_sidecar(root)
    with pytest.raises(ValueError, match="call inventory drifted"):
        _validate_sidecar_cohort(
            sidecar,
            repo_root=tmp_path,
            sources=[experience_source.model_dump(mode="json")],
            selected_sources=[source],
            used_calls=used_calls,
            work_simulations=[
                (
                    source.results_path,
                    sidecar.calls[0].simulation_id,
                    sidecar.calls[0].source_simulation_sha256,
                )
            ],
            work_utterances=sidecar.calls[0].utterance_input_sha256s,
        )


def test_sidecar_rejects_duplicate_calls(tmp_path):
    root, *_ = _write_sidecar(tmp_path)
    sidecar = _load_naturalness_sidecar(root)
    with pytest.raises(ValueError, match="duplicate call identities"):
        NaturalnessSidecarInput(
            root=sidecar.root,
            manifest_path=sidecar.manifest_path,
            manifest_sha256=sidecar.manifest_sha256,
            manifest=sidecar.manifest,
            calls=[sidecar.calls[0], sidecar.calls[0]],
        )


def test_sidecar_rejects_source_drift_and_error_status(tmp_path):
    root, experience_source, source, _simulation = _write_sidecar(tmp_path)
    sidecar = _load_naturalness_sidecar(root)
    drifted_source = experience_source.model_copy(update={"sha256": "2" * 64})
    with pytest.raises(ValueError, match="all-source identity drifted"):
        _validate_sidecar_cohort(
            sidecar,
            repo_root=tmp_path,
            sources=[drifted_source.model_dump(mode="json")],
            selected_sources=[source],
            used_calls={(source.results_path, "sim-1")},
            work_simulations=[
                (
                    source.results_path,
                    sidecar.calls[0].simulation_id,
                    sidecar.calls[0].source_simulation_sha256,
                )
            ],
            work_utterances=sidecar.calls[0].utterance_input_sha256s,
        )

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "complete_with_errors"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ValueError, match="must be complete"):
        _load_naturalness_sidecar(root)


def test_sidecar_rejects_nonfrozen_judge_and_work_identity(tmp_path):
    root, experience_source, source, _simulation = _write_sidecar(tmp_path)
    sidecar = _load_naturalness_sidecar(root)
    common = {
        "repo_root": tmp_path,
        "sources": [experience_source.model_dump(mode="json")],
        "selected_sources": [source],
        "used_calls": {(source.results_path, "sim-1")},
        "work_simulations": [
            (
                source.results_path,
                sidecar.calls[0].simulation_id,
                sidecar.calls[0].source_simulation_sha256,
            )
        ],
        "work_utterances": sidecar.calls[0].utterance_input_sha256s,
    }

    wrong_judges = dict(sidecar.manifest.identity.judges)
    wrong_judges.pop("es")
    wrong_identity = sidecar.manifest.identity.model_copy(
        update={"judges": wrong_judges}
    )
    wrong_sidecar = sidecar.model_copy(
        update={
            "manifest": sidecar.manifest.model_copy(update={"identity": wrong_identity})
        }
    )
    with pytest.raises(ValueError, match="frozen paper judges"):
        _validate_sidecar_cohort(wrong_sidecar, **common)

    wrong_identity = sidecar.manifest.identity.model_copy(
        update={"work_fingerprint_sha256": "0" * 64}
    )
    wrong_sidecar = sidecar.model_copy(
        update={
            "manifest": sidecar.manifest.model_copy(update={"identity": wrong_identity})
        }
    )
    with pytest.raises(ValueError, match="work fingerprint drifted"):
        _validate_sidecar_cohort(wrong_sidecar, **common)
