"""Focused contracts for the tau-Elicitation reviewer archive."""

import json
from pathlib import Path

from tau2.paper.elicitation import (
    PAPER_AGENT_DIRECTED,
    PAPER_SCAFFOLDED,
    _cell_metadata,
    _full_duplex_turns,
    _kappa,
    _parse_user_snapshot,
    _runtime_caller_guidelines,
    _task_source_paths,
)


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


def test_kappa_matches_binary_decision_contract() -> None:
    assert _kappa([True, True, False, False], [True, False, False, False]) == 0.5


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
