# Copyright Sierra
"""Tests for compact retail localization-ablation transcripts."""

import json

import pytest

import tau2.paper.ablation_transcripts as export
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.simulation import (
    AgentInfo,
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.environment.environment import EnvironmentInfo


def _simulation(language: str, suffix: str) -> SimulationRun:
    return SimulationRun(
        id=f"sim-{language}-{suffix}",
        task_id=f"1_{language}",
        trial=0,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60,
        termination_reason=TerminationReason.AGENT_STOP,
        reward_info=RewardInfo(reward=1),
        messages=[
            UserMessage(role="user", content=f"caller {language}"),
            AssistantMessage(role="assistant", content=f"agent {language}"),
        ],
    )


def _write_run(path, simulation: SimulationRun) -> None:
    results = Results(
        info=Info(
            git_commit="a" * 40,
            num_trials=1,
            max_steps=50,
            max_errors=10,
            user_info=UserInfo(implementation="voice_user"),
            agent_info=AgentInfo(implementation="audio_agent"),
            environment_info=EnvironmentInfo(domain_name="retail", policy="test"),
        ),
        tasks=[],
        simulations=[simulation],
    )
    results.save(path, format="dir")


def test_export_and_verify_compact_ablation_transcripts(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    subset = data_dir / "tau2" / "task_subsets" / "retail_30.json"
    subset.parent.mkdir(parents=True)
    subset.write_text(json.dumps({"task_ids": ["1"]}))
    monkeypatch.setattr(export, "DATA_DIR", data_dir)

    evidence = tmp_path / "tau-multi"
    for language in export.LANGUAGES:
        for system in export.SYSTEMS:
            for condition in export.CONDITIONS:
                path = export._condition_path(evidence, language, system, condition)
                _write_run(
                    path,
                    _simulation(language, f"{system}-{condition}"),
                )

    out = tmp_path / "out"
    manifest = export.export_retail_ablation_transcripts(evidence, out)

    assert manifest.rows == 12
    assert manifest.rows_by_language == {"hi": 6, "zh": 6}
    assert manifest.rows_by_condition == {
        "baseline": 4,
        "source_entities": 4,
        "native_script_entities_and_db": 4,
    }
    assert len(manifest.source_runs) == 12
    rows = [
        json.loads(line)
        for line in (out / "transcripts.jsonl").read_text().splitlines()
    ]
    assert all(len(row["turns"]) == 2 for row in rows)
    assert all(
        row["table_columns"] == ["localized", "romanized"]
        for row in rows
        if row["condition"] == "baseline"
    )
    assert export.verify_retail_ablation_transcripts(out) == manifest

    (out / "transcripts.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="hash differs"):
        export.verify_retail_ablation_transcripts(out)
