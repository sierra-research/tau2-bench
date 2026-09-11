"""Focused contracts for the tau-Elicitation reviewer archive."""

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from tau2.paper.elicitation import (
    DETACHED_EVIDENCE_URL,
    FIDELITY_VALIDATION,
    HUMAN_FAILURE_VALIDATION,
    PAPER_AGENT_DIRECTED,
    PAPER_ANALYSIS_INPUTS,
    PAPER_SCAFFOLDED,
    REALISM_EVENT_VERSION,
    FidelityValidationRow,
    HumanFailureValidationRow,
    TranscriptRecord,
    TranscriptTurn,
    _cell_metadata,
    _full_duplex_turns,
    _human_validation_leaks,
    _parse_user_snapshot,
    _read_validation_artifacts,
    _read_validation_rows,
    _realism_event_record,
    _runtime_caller_guidelines,
    _task_source_paths,
    verify_release,
)


def test_release_includes_detached_evidence_and_realism_analysis() -> None:
    assert DETACHED_EVIDENCE_URL.startswith("https://drive.google.com/drive/folders/")
    assert "intake_realism_effects_2026-09-07.json" in PAPER_ANALYSIS_INPUTS
    assert (
        "intake_realism_effects_agent_directed_2026-09-11.json" in PAPER_ANALYSIS_INPUTS
    )
    assert "intake_realism_effects_scaffolded_2026-09-11.json" in PAPER_ANALYSIS_INPUTS


def test_realism_event_record_preserves_events_needed_by_paper() -> None:
    transcript = TranscriptRecord(
        cell="main_runs/test/results.json",
        cohort="paper_agent_directed",
        simulation_id="sim-1",
        source_simulation_sha256="a" * 64,
        task_id="task-1",
        trial=0,
        seed=9401,
        reward=1.0,
        evaluation_fields={},
        termination_reason="completed",
        duration_seconds=42.5,
        tick_count=10,
        complication={"kind": "spell_correction"},
        speech_environment={},
        turns=[
            TranscriptTurn(
                order=0,
                role="tool",
                source="user_tool_event",
                tool_name="note_spell_request",
            )
        ],
    )
    row = _realism_event_record(
        transcript,
        {
            "effect_timeline": {
                "events": [
                    {"effect_type": "spell_out", "params": {"restarts": 1}},
                    {"effect_type": "spell_out", "params": {"restarts": 2}},
                ]
            }
        },
        system="gpt_xhigh",
        condition="regular",
    )

    assert row.schema_version == REALISM_EVENT_VERSION
    assert (row.spell_requests, row.spell_events, row.spell_restarts) == (1, 2, 3)
    assert row.duration_seconds == 42.5


def test_paper_cell_roster_is_exact() -> None:
    assert len(PAPER_AGENT_DIRECTED) == 12
    assert len(PAPER_SCAFFOLDED) == 12
    assert PAPER_AGENT_DIRECTED.isdisjoint(PAPER_SCAFFOLDED)


def test_cell_metadata_includes_completed_minimal_stress_cells() -> None:
    paper = _cell_metadata(
        "main_runs/modeb_openai_minimal_regular_2026-09-02/results.json", {}
    )
    noise_heavy = _cell_metadata(
        "main_runs/modeb_openai_minimal_chanheavy_2026-09-05/results.json", {}
    )
    assert paper == ("paper_agent_directed", "gpt_minimal", "regular")
    assert noise_heavy == (
        "paper_agent_directed",
        "gpt_minimal",
        "noise_heavy",
    )


def test_parse_user_snapshot_uses_active_chunks() -> None:
    value = (
        '<message uuid="m1" active="2">'
        '<chunk id="0">A &amp; </chunk><chunk id="1">B</chunk></message>'
    )
    assert _parse_user_snapshot(value) == ("m1", 2, "A & B")


def test_full_duplex_transcript_orders_speech_and_tools() -> None:
    ticks = [
        {
            "agent_chunk": {"content": "Hello"},
            "user_chunk": {"audio_script_gold": ""},
            "agent_tool_calls": [],
            "agent_tool_results": [],
        },
        {
            "agent_chunk": {"content": " there"},
            "user_chunk": {"audio_script_gold": ""},
            "agent_tool_calls": [],
            "agent_tool_results": [],
        },
        {
            "agent_chunk": {"content": ""},
            "user_chunk": {
                "audio_script_gold": (
                    '<message uuid="u1" active="1">'
                    '<chunk id="0">Hi</chunk><chunk id="1">!</chunk></message>'
                )
            },
            "agent_tool_calls": [
                {"id": "c1", "name": "get_callback_order", "arguments": {}}
            ],
            "agent_tool_results": [{"id": "c1", "content": '{"record_id":"R1"}'}],
            "user_tool_calls": [
                {
                    "id": "u1",
                    "name": "note_readback",
                    "arguments": {"field": "name", "i_affirmed": False},
                }
            ],
            "user_tool_results": [{"id": "u1", "content": "Noted read-back."}],
        },
    ]
    turns = _full_duplex_turns({"ticks": ticks})
    assert [(row.role, row.text, row.tool_name) for row in turns] == [
        ("agent", "Hello there", None),
        ("user", "Hi!", None),
        ("tool", "", "get_callback_order"),
        ("tool", "", "note_readback"),
    ]
    assert turns[-1].source == "user_tool_event"


def test_release_validation_projection_is_exact_and_sanitized() -> None:
    root = Path(__file__).resolve().parents[2]
    canonical = root / "data/simulations/paper_runs/tau-elicit/judge_validation"
    release = root / "papers/tau-intake/v1/reproduction/judge_validation"
    expected = {
        HUMAN_FAILURE_VALIDATION: {"README.md", "calls.csv", "metrics.json"},
        FIDELITY_VALIDATION: {"README.md", "utterances.csv", "metrics.json"},
    }

    for directory, filenames in expected.items():
        canonical_files = {
            path.name for path in (canonical / directory).iterdir() if path.is_file()
        }
        release_files = {
            path.name for path in (release / directory).iterdir() if path.is_file()
        }
        assert canonical_files == release_files == filenames
        for filename in filenames:
            assert (canonical / directory / filename).read_bytes() == (
                release / directory / filename
            ).read_bytes()

    human_rows = _read_validation_rows(
        release / HUMAN_FAILURE_VALIDATION / "calls.csv",
        HumanFailureValidationRow,
    )
    fidelity_rows = _read_validation_rows(
        release / FIDELITY_VALIDATION / "utterances.csv",
        FidelityValidationRow,
    )
    assert len(human_rows) == len({row.simulation_id for row in human_rows}) == 90
    assert all(row.reward == 0 for row in human_rows)
    assert len(fidelity_rows) == len({row.utterance_id for row in fidelity_rows}) == 60
    assert list(HumanFailureValidationRow.model_fields) == [
        "schema_version",
        "validation_set_id",
        "provider",
        "simulation_id",
        "task_id",
        "bank",
        "tier",
        "reward",
        "error_source",
        "error_subtype",
    ]
    assert Counter(row.error_source for row in human_rows) == {
        "agent": 81,
        "user": 2,
        "no_error": 3,
        "unresolved": 4,
    }
    assert list(FidelityValidationRow.model_fields) == [
        "utterance_id",
        "provider",
        "simulation_id",
        "task_id",
        "utterance_idx",
        "human_fidelity_positive",
        "judge_any_finding_positive",
        "judge_max_fidelity_severity",
        "judge_severity_ge_2_positive",
        "confusion_severity_ge_2",
        "confusion_any_finding",
    ]
    assert _human_validation_leaks(root / "papers/tau-intake/v1/reproduction") == []
    assert not (canonical / "intake_review_100_2026-08-27").exists()
    assert not (release / "validation.csv").exists()
    assert not (release / "metrics.json").exists()


def test_human_validation_leak_check_ignores_automated_judge_findings(
    tmp_path: Path,
) -> None:
    automated = tmp_path / "speech_judgments" / "utterances.jsonl"
    automated.parent.mkdir(parents=True)
    automated.write_text('{"findings":[],"summary":"automated LLM output"}\n')
    assert _human_validation_leaks(tmp_path) == []

    leaked = tmp_path / "raw_review.csv"
    with leaked.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["simulation_id", "fidelity_notes"])
        writer.writeheader()
        writer.writerow({"simulation_id": "sim-1", "fidelity_notes": "private note"})
    assert _human_validation_leaks(tmp_path) == ["raw_review.csv"]


def test_validation_contract_rejects_extra_files_and_note_fields(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    canonical = root / "data/simulations/paper_runs/tau-elicit/judge_validation"
    validation = tmp_path / "judge_validation"
    shutil.copytree(canonical, validation)

    extra = validation / FIDELITY_VALIDATION / "raw_notes.csv"
    extra.write_text("simulation_id,note\nsim-1,private\n")
    with pytest.raises(ValueError, match="Unexpected entries"):
        _read_validation_artifacts(validation)

    extra.unlink()
    metrics_path = validation / FIDELITY_VALIDATION / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["error_notes"] = "private note"
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _read_validation_artifacts(validation)


@pytest.mark.parametrize(
    "trace",
    [
        "Ian",
        "Niko",
        "rater consensus",
        "adjudicated",
        "post_discussion",
        "single_rater_positive",
        "source_split",
        "data/annotation/",
    ],
)
def test_validation_contract_rejects_pre_resolution_trace_text(
    tmp_path: Path, trace: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    canonical = root / "data/simulations/paper_runs/tau-elicit/judge_validation"
    validation = tmp_path / "judge_validation"
    shutil.copytree(canonical, validation)
    readme = validation / FIDELITY_VALIDATION / "README.md"
    readme.write_text(readme.read_text() + f"\n{trace}\n")

    with pytest.raises(ValueError, match="pre-resolution annotation material"):
        _read_validation_artifacts(validation)


def test_release_retains_all_automated_speech_judgments() -> None:
    root = Path(__file__).resolve().parents[2]
    release = root / "papers/tau-intake/v1/reproduction"
    judgments = release / "speech_judgments/utterances.jsonl"
    manifest = json.loads((release / "speech_judgments/manifest.json").read_text())

    assert sum(1 for _ in judgments.open(encoding="utf-8")) == 6_422
    assert manifest["total_rows"] == 6_422
    assert hashlib.sha256(judgments.read_bytes()).hexdigest() == (
        "edff37432f125b8333508cbf0b981b250e635925241d85d90840a33abda34724"
    )


def test_checked_release_passes_offline_verification() -> None:
    root = Path(__file__).resolve().parents[2]
    report = verify_release(root / "papers/tau-intake/v1/reproduction")
    assert report.ok, report.summary


def test_runtime_caller_prompt_is_outbound_and_free_arm_removes_strategy_nudge() -> (
    None
):
    common = {"audio_native_config": {"provider": "openai"}}
    scaffolded = _runtime_caller_guidelines(
        {**common, "environment_info": {"domain_name": "intake"}}
    )
    free = _runtime_caller_guidelines(
        {**common, "environment_info": {"domain_name": "intake_free"}}
    )
    assert "ANSWERING a VOICE CALL" in scaffolded
    assert "offer to spell it out" in scaffolded
    assert "ANSWERING a VOICE CALL" in free
    assert "offer to spell it out" not in free
    assert not scaffolded.endswith("\n")


def test_task_snapshot_paths_follow_the_frozen_split() -> None:
    assert _task_source_paths(
        {
            "environment_info": {"domain_name": "intake"},
            "task_split_name": "compose_n2",
        }
    ) == ("data/tau2/domains/intake/bands/compose_n2/tasks.json",)
    assert _task_source_paths(
        {
            "environment_info": {"domain_name": "intake_staged"},
            "task_split_name": "base",
        }
    ) == (
        "data/tau2/domains/intake/bands/chain_n2/tasks.json",
        "data/tau2/domains/intake/bands/chain_n3/tasks.json",
    )


def test_xai_factory_preserves_explicit_paper_model() -> None:
    from tau2.voice.audio_native.adapter import create_adapter

    adapter, resolved = create_adapter(
        "xai",
        tick_duration_ms=200,
        model="grok-voice-think-fast-1.0",
    )
    assert resolved == "grok-voice-think-fast-1.0"
    assert adapter.model == "grok-voice-think-fast-1.0"


def test_xai_paper_arm_sends_no_reasoning_setting() -> None:
    from tau2.voice.audio_native.adapter import create_adapter

    adapter, _ = create_adapter(
        "xai",
        tick_duration_ms=200,
        model="grok-voice-think-fast-1.0",
        reasoning_effort="provider_default",
    )
    assert adapter.reasoning_effort is None


def test_matched_one_field_ledger_reproduces_paper_row() -> None:
    root = Path(__file__).resolve().parents[2]
    artifact = json.loads(
        (
            root / "papers/tau-intake/v1/reproduction/analysis_inputs/"
            "composition_one_field_matched.json"
        ).read_text()
    )
    assert artifact["counts"] == {
        "matched_parent_instances": 210,
        "observations": 630,
        "passes": 486,
        "unique_parent_tasks": 127,
    }
    assert artifact["task_pass_at_1"] == 486 / 630
    assert artifact["field_pass_at_1"] == 486 / 630
    assert len(artifact["rows"]) == 210
    assert all(len(row["source_outcomes"]) == 3 for row in artifact["rows"])


def test_protocol_archive_contains_only_observed_workflows() -> None:
    root = Path(__file__).resolve().parents[2]
    artifact = json.loads(
        (
            root / "papers/tau-intake/v1/reproduction/analysis_inputs/"
            "composition_protocol_recomputed.json"
        ).read_text()
    )
    assert artifact["schema_version"] == "tau-elicit-composition-protocol-v2"
    assert set(artifact["protocol"]) == {
        "joint_submission_without_validation",
        "field_by_field_validation_and_retry",
    }
    joint = artifact["protocol"]["joint_submission_without_validation"]
    assert (joint["task_passes"], joint["observations"]) == (173, 270)
    assert (joint["field_passes"], joint["fields"]) == (492, 630)
    validated = artifact["protocol"]["field_by_field_validation_and_retry"]
    assert (validated["task_passes"], validated["observations"]) == (222, 270)
    assert (validated["field_passes"], validated["fields"]) == (544, 630)
    assert artifact["source_gaps"] == []


def test_same_vs_crossed_pass3_ledger_reproduces_paper_claim() -> None:
    root = Path(__file__).resolve().parents[2]
    artifact = json.loads(
        (
            root / "papers/tau-intake/v1/reproduction/analysis_inputs/"
            "pass3_same_vs_crossed.json"
        ).read_text()
    )
    assert artifact["schema_version"] == "tau-elicit-same-vs-crossed-pass3-v1"
    assert artifact["same_environment"] == {
        "passes": 103,
        "total": 200,
        "rate": 0.515,
    }
    assert artifact["crossed_environment"] == {
        "passes": 77,
        "total": 200,
        "rate": 0.385,
    }
    assert artifact["difference_points"] == -13.0
    assert len(artifact["rows"]) == 200
    assert sum(row["same_environment_pass3"] for row in artifact["rows"]) == 103
    assert sum(row["crossed_environment_pass3"] for row in artifact["rows"]) == 77
