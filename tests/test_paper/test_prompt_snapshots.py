# Copyright Sierra
"""Tests for immutable τ-Multilingual prompt snapshots."""

import json

import tau2.paper.prompt_snapshots as snapshots


def _result_payload() -> dict[str, object]:
    return {
        "info": {
            "git_commit": "a" * 40,
            "agent_info": {"implementation": "agent", "llm": "model"},
            "audio_native_config": {"provider": "openai"},
            "environment_info": {
                "domain_name": "retail",
                "policy": "Exact agent policy.\n",
            },
            "seed": 42,
            "task_set_name": "retail_hi_identity",
            "user_info": {
                "implementation": "voice_user",
                "global_simulation_guidelines": "Exact user guidelines.\n",
            },
            "user_persona_id": "hi",
        },
        "simulation_index": [
            {
                "id": "sim-1",
                "task_id": "task_hi_identity",
                "trial": 0,
                "seed": 42,
            }
        ],
        "tasks": [
            {
                "id": "task_hi_identity",
                "user_scenario": {"instructions": {"reason_for_call": "Help."}},
            }
        ],
    }


def _write_result_fixture(result):
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps(_result_payload()))
    simulations = result.parent / "simulations"
    simulations.mkdir()
    (simulations / "sim-1.json").write_text(
        json.dumps(
            {
                "id": "sim-1",
                "task_id": "task_hi_identity",
                "trial": 0,
                "seed": 42,
                "policy": "Exact agent policy.\n",
                "speech_environment": {
                    "persona_id": "rishika_hindi_v1",
                    "locale": "IN-DL",
                },
            }
        )
    )


def _stub_renderer(_repo, _commit, requests):
    return {
        request.key: snapshots._HistoricalPromptPair(
            agent="Exact rendered agent prompt.\n",
            user="Exact rendered user prompt.\n",
            persona_id="rishika_hindi_v1",
        )
        for request in requests
    }


def test_prompt_export_uses_recorded_inputs_and_historical_runtime_pack(
    tmp_path, monkeypatch
):
    evidence = tmp_path / "evidence"
    result = evidence / "cell" / "results.json"
    _write_result_fixture(result)
    spec = snapshots._ResultSpec(
        cohort="voice",
        language="hi",
        domain="retail",
        system="openai_xhigh",
        path=result,
    )
    monkeypatch.setattr(snapshots, "_result_specs", lambda _root: [spec])
    pack = b"""\
language: hi
display_name: Hindi
agent_greeting: \"historical greeting\"
personas: {}
nativeness:
  judge_factors: []
"""
    monkeypatch.setattr(snapshots, "_git_show", lambda *_args: pack)
    monkeypatch.setattr(snapshots, "_render_historical_prompts", _stub_renderer)

    out = tmp_path / "prompts"
    manifest = snapshots.export_prompt_snapshots(evidence, out, repo=tmp_path)

    assert len(manifest.cells) == 1
    assert manifest.cells[0].source_pack_sha256 == snapshots._sha256_bytes(pack)
    by_kind = {obj.kind: obj for obj in manifest.objects}
    assert (out / by_kind["agent_policy"].file).read_text() == ("Exact agent policy.\n")
    runtime_pack = json.loads((out / by_kind["runtime_language_pack"].file).read_text())
    assert runtime_pack["agent_greeting"] == "historical greeting"
    assert "nativeness" not in runtime_pack
    assert manifest.cells[0].simulations[0].simulation_id == "sim-1"
    assert "agent_system_prompt" in by_kind
    assert "user_system_prompt" in by_kind
    assert snapshots.verify_prompt_snapshot_export(out).ok
    assert snapshots.verify_prompt_snapshot_sources(evidence, out, repo=tmp_path).ok


def test_prompt_verification_detects_changed_object(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    result = evidence / "cell" / "results.json"
    _write_result_fixture(result)
    spec = snapshots._ResultSpec(
        cohort="voice",
        language="hi",
        domain="retail",
        system="openai_xhigh",
        path=result,
    )
    monkeypatch.setattr(snapshots, "_result_specs", lambda _root: [spec])
    monkeypatch.setattr(
        snapshots,
        "_git_show",
        lambda *_args: b"language: hi\ndisplay_name: Hindi\npersonas: {}\n",
    )
    monkeypatch.setattr(snapshots, "_render_historical_prompts", _stub_renderer)
    out = tmp_path / "prompts"
    manifest = snapshots.export_prompt_snapshots(evidence, out, repo=tmp_path)
    target = out / manifest.objects[0].file
    target.write_text("changed")

    report = snapshots.verify_prompt_snapshot_export(out)

    assert not report.ok
    assert any("mismatch" in problem for problem in report.problems)


def test_source_verification_detects_changed_frozen_result(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    result = evidence / "cell" / "results.json"
    _write_result_fixture(result)
    spec = snapshots._ResultSpec(
        cohort="voice",
        language="hi",
        domain="retail",
        system="openai_xhigh",
        path=result,
    )
    monkeypatch.setattr(snapshots, "_result_specs", lambda _root: [spec])
    monkeypatch.setattr(
        snapshots,
        "_git_show",
        lambda *_args: b"language: hi\ndisplay_name: Hindi\npersonas: {}\n",
    )
    monkeypatch.setattr(snapshots, "_render_historical_prompts", _stub_renderer)
    out = tmp_path / "prompts"
    snapshots.export_prompt_snapshots(evidence, out, repo=tmp_path)
    payload = _result_payload()
    payload["info"]["environment_info"]["policy"] = "Changed policy.\n"
    result.write_text(json.dumps(payload))

    report = snapshots.verify_prompt_snapshot_sources(evidence, out, repo=tmp_path)

    assert not report.ok
    assert any("result hash mismatch" in problem for problem in report.problems)


def test_retained_requests_are_the_exact_prompt_source(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    result = evidence / "cell" / "results.json"
    _write_result_fixture(result)
    spec = snapshots._ResultSpec(
        cohort="voice",
        language="hi",
        domain="retail",
        system="xai_provider_default",
        path=result,
    )
    monkeypatch.setattr(snapshots, "_result_specs", lambda _root: [spec])
    monkeypatch.setattr(
        snapshots,
        "_git_show",
        lambda *_args: b"language: hi\ndisplay_name: Hindi\npersonas: {}\n",
    )
    monkeypatch.setattr(snapshots, "_render_historical_prompts", _stub_renderer)
    debug = (
        result.parent
        / "artifacts"
        / "task_task_hi_identity"
        / "sim_sim-1"
        / "llm_debug"
    )
    debug.mkdir(parents=True)
    (debug / "user_streaming_response.json").write_text(
        json.dumps(
            {
                "request": {
                    "messages": [
                        {"role": "system", "content": ["Logged", "user prompt."]}
                    ]
                }
            }
        )
    )
    (debug.parent / "task.log").write_text(
        '    "instructions": "Logged agent prompt.\\n"\n'
    )

    out = tmp_path / "prompts"
    manifest = snapshots.export_prompt_snapshots(evidence, out, repo=tmp_path)
    simulation = manifest.cells[0].simulations[0]
    objects = {(obj.kind, obj.sha256): obj for obj in manifest.objects}

    user = objects[("user_system_prompt", simulation.user_system_prompt_sha256)]
    agent = objects[("agent_system_prompt", simulation.agent_system_prompt_sha256)]
    assert (out / user.file).read_text() == "Logged\nuser prompt."
    assert (out / agent.file).read_text() == "Logged agent prompt.\n"
    source = snapshots.verify_prompt_snapshot_sources(evidence, out, repo=tmp_path)
    assert source.ok
    assert source.logged_user_prompts_compared == 1
    assert source.logged_agent_prompts_compared == 1
