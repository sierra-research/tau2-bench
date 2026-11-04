"""Offline end-to-end diagnostics over ordinary saved Results."""

import argparse
import json

import pytest

from tau2.data_model.benchmark import BenchmarkRunInfo, encode_conditions
from tau2.data_model.judges import MeasureVerdict, PlanAction
from tau2.data_model.live_config import LiveConfig
from tau2.data_model.message import AssistantMessage, Tick
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import (
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    TextRunConfig,
    VoiceRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.data_model.transcript import LiveTranscriptFragment, LiveTranscriptTick
from tau2.judges.catalog import load_catalog
from tau2.judges.cli import add_judges_args
from tau2.judges.run_evidence import SavedRun, progress_path
from tau2.judges.suite import JudgeExecutionConfig, plan_suite
from tau2.judges.summary import build_export as export_run
from tau2.runner.helpers import get_info


@pytest.fixture(params=["json", "dir"])
def saved_run(tmp_path, request):
    info = get_info(TextRunConfig(domain="retail", language="pt", num_trials=2))
    info.benchmark = BenchmarkRunInfo(
        benchmark_version="voice-v2",
        frame_revision="fixture",
        language="pt",
        domain="retail",
        binding_revision="fixture",
        canonical_task_ids=[str(i) for i in range(50)],
        selected_task_ids=[str(i) for i in range(50)],
        trials=[0, 1],
        condition_fingerprint="a" * 64,
        conditions=encode_conditions(
            VoiceRunConfig(language="pt", audio_native_config=AudioNativeConfig())
        ),
    )
    task = Task(id="localized-0", user_scenario={"instructions": "Fixture caller."})
    results = Results(
        info=info,
        tasks=[task],
        simulations=[
            SimulationRun(
                id=f"sim-{trial}",
                task_id=task.id,
                start_time="start",
                end_time="end",
                duration=1,
                termination_reason=TerminationReason.USER_STOP,
                trial=trial,
                reward_info=RewardInfo(reward=float(trial)),
                messages=[],
            )
            for trial in (0, 1)
        ],
    )
    path = tmp_path / ("run.json" if request.param == "json" else "run")
    if request.param == "dir":
        path.mkdir()
    results.save(path, format=request.param)
    return path


def test_live_transcript_allows_text_judging_without_audio(tmp_path):
    config = VoiceRunConfig(
        domain="retail",
        language="es",
        audio_native_config=AudioNativeConfig(
            provider="openai_live", live_config=LiveConfig(backend_model="gpt-6-astra")
        ),
    )
    simulation = SimulationRun(
        id="live-text-only",
        task_id="0",
        trial=0,
        start_time="start",
        end_time="end",
        duration=0.2,
        termination_reason=TerminationReason.AGENT_STOP,
        messages=[],
        ticks=[
            Tick(
                tick_id=0,
                timestamp="2026-09-28T00:00:00",
                tick_duration_seconds=0.2,
                agent_chunk=AssistantMessage(
                    role="assistant",
                    content="Hola",
                    contains_speech=True,
                    raw_data={
                        "live_transcript": LiveTranscriptTick(
                            session_id="recorded-session",
                            fragments=[
                                LiveTranscriptFragment(
                                    speaker="agent",
                                    text="Hola",
                                    start_ms=0,
                                    end_ms=200,
                                )
                            ],
                            speech_intervals_ms=[(0, 200)],
                        ).model_dump()
                    },
                ),
            )
        ],
    )
    path = tmp_path / "live.json"
    Results(
        info=get_info(config),
        tasks=[Task(id="0", user_scenario={"instructions": "Say hello."})],
        simulations=[simulation],
    ).save(path, format="json")

    saved = SavedRun(path)
    item = saved.inputs[0]
    assert not item.has_audio
    assert saved.load(item).simulation.id == simulation.id
    plan = plan_suite(
        saved.inputs, load_catalog(), [], only=["naturalness", "speech_fidelity"]
    )
    assert {entry.measure: entry.action for entry in plan.items} == {
        "naturalness": PlanAction.MISSING,
        "speech_fidelity": PlanAction.MISSING_INPUT,
    }

    legacy = simulation.model_copy(deep=True)
    legacy.ticks[0].agent_chunk.raw_data = {"tick_number": 0}
    Results(
        info=get_info(config),
        tasks=[Task(id="0", user_scenario={"instructions": "Say hello."})],
        simulations=[legacy],
    ).save(tmp_path / "legacy.json", format="json")
    with pytest.raises(ValueError, match="Recorded Live audio is required"):
        SavedRun(tmp_path / "legacy.json").inputs


def args(*values):
    parser = argparse.ArgumentParser()
    add_judges_args(parser)
    return parser.parse_args(values)


def build_export(options):
    return export_run(
        options.results,
        load_catalog(),
        only=options.only,
        trials=options.trials,
        force=getattr(options, "force", False),
        execute=options.judge_command == "suite" and not options.dry_run,
        execution=JudgeExecutionConfig(
            unit_retries=getattr(
                options, "unit_retries", JudgeExecutionConfig().unit_retries
            )
        ),
    )


def all_source_bytes(path):
    if path.is_file():
        return {str(path): path.read_bytes()}
    return {
        str(p): p.read_bytes()
        for p in path.rglob("*.json")
        if "diagnostics" not in p.parts
    }


def backend(item, contract, measure):
    return MeasureVerdict.model_validate(
        {
            "units": [
                {
                    "unit_id": "call",
                    "opportunity": True,
                    "violation": item.trial == 1,
                    "evidence": "Fixture evidence",
                }
            ]
        }
    )


def test_cli_dry_run_and_readers_never_call_backend_or_modify_results(
    saved_run, monkeypatch
):
    import tau2.judges.suite as suite

    monkeypatch.setattr(
        suite,
        "create_judge",
        lambda *_: pytest.fail("read-only command reached backend"),
    )
    before = all_source_bytes(saved_run)
    for command in ("suite", "summarize", "export"):
        options = [command, str(saved_run), "--only", "naturalness,tool_use"]
        if command == "suite":
            options.append("--dry-run")
        report = build_export(args(*options))
        assert len(report.plan.items) == 4
        assert all(summary.not_judged_calls == 2 for summary in report.summary)
        assert all(summary.observed_pass_fraction is None for summary in report.summary)
    assert not progress_path(saved_run.resolve()).exists()
    assert all_source_bytes(saved_run) == before


def test_cli_production_no_speech_is_explicit_no_opportunity(saved_run, monkeypatch):
    monkeypatch.setattr(
        "tau2.judges.criteria.generate",
        lambda **kw: pytest.fail("no speech must not call a model"),
    )
    before = all_source_bytes(saved_run)
    report = build_export(args("suite", str(saved_run), "--only", "naturalness"))
    assert report.summary[0].judged_calls == 2
    assert report.summary[0].no_opportunity_units == 2
    assert all_source_bytes(saved_run) == before


def test_cli_fixture_backend_resume_force_filter_stale_and_reward_invariance(
    saved_run, monkeypatch
):
    import tau2.judges.suite as suite

    calls = []

    def recorded(item, contract, measure):
        calls.append(item.trial)
        return backend(item, contract, measure)

    from types import SimpleNamespace

    monkeypatch.setattr(
        suite,
        "create_judge",
        lambda contract: SimpleNamespace(
            evaluate=lambda evidence, measure: recorded(
                evidence.simulation, contract, measure
            )
        ),
    )
    before = all_source_bytes(saved_run)
    report = build_export(args("suite", str(saved_run), "--only", "naturalness"))
    assert calls == [0, 1]
    assert (
        report.summary[0].complete and report.summary[0].observed_pass_fraction == 0.5
    )
    build_export(args("suite", str(saved_run), "--only", "naturalness"))
    assert calls == [0, 1]
    build_export(
        args(
            "suite", str(saved_run), "--only", "naturalness", "--force", "--trials", "1"
        )
    )
    assert calls == [0, 1, 1]
    assert all_source_bytes(saved_run) == before
    # Input edits invalidate saved diagnostics; a reward-only edit does not.
    original = SavedRun(saved_run).inputs
    result = Results.load(saved_run)
    result.simulations[0].reward_info.reward = 0.5
    result.save(saved_run, format="dir" if saved_run.is_dir() else "json")
    assert SavedRun(saved_run).inputs == original
    result.tasks[0].user_scenario.instructions = "Changed caller context."
    result.save(saved_run, format="dir" if saved_run.is_dir() else "json")
    stale = build_export(
        args("suite", str(saved_run), "--only", "naturalness", "--dry-run")
    )
    assert all(item.action == PlanAction.STALE for item in stale.plan.items)


def test_cli_malformed_fixture_reply_remains_an_explicit_error(saved_run, monkeypatch):
    import tau2.judges.suite as suite

    monkeypatch.setattr(
        suite,
        "create_judge",
        lambda *_: __import__("types").SimpleNamespace(evaluate=lambda *_: "not typed"),
    )
    report = build_export(args("suite", str(saved_run), "--only", "naturalness"))
    assert report.summary[0].error_calls == 2
    assert report.summary[0].observed_pass_fraction is None
    assert all(row.error for row in report.verdicts)


def test_cli_has_no_rejudge_and_rejects_unvalidated_measure(saved_run):
    with pytest.raises(SystemExit):
        args("rejudge", "run.json")
    with pytest.raises(ValueError, match="absent from the judge registry"):
        build_export(
            args("suite", str(saved_run), "--only", "regional_consistency", "--dry-run")
        )


def test_cli_production_tool_use_adapter_runs_without_api_when_no_tools(
    saved_run, monkeypatch
):
    import tau2.judges.tool_use as module

    monkeypatch.setattr(
        module,
        "generate",
        lambda **_: pytest.fail("No-tool call must be preclassified"),
    )
    report = build_export(args("suite", str(saved_run), "--only", "tool_use"))
    assert report.summary[0].judged_calls == 2
    assert report.summary[0].no_opportunity_units == 2
    assert report.summary[0].observed_pass_fraction is None


def test_cli_verified_missing_audio_is_explicit(saved_run):
    report = build_export(
        args("suite", str(saved_run), "--only", "speech_fidelity", "--dry-run")
    )
    assert report.summary[0].missing_input_calls == 2


def test_saved_run_skips_infrastructure_errors(saved_run):
    results = Results.load(saved_run)
    results.simulations[1].termination_reason = TerminationReason.INFRASTRUCTURE_ERROR
    results.save(saved_run, format="dir" if saved_run.is_dir() else "json")

    assert [item.simulation_id for item in SavedRun(saved_run).inputs] == ["sim-0"]


def test_backend_rejects_changed_source_before_model_call(saved_run):
    from tau2.judges.run_evidence import SavedRun

    item = SavedRun(saved_run).inputs[0]
    reader = SavedRun(saved_run)
    assert reader.inputs[0] == item
    result = Results.load(saved_run)
    result.tasks[0].user_scenario.instructions = "Changed caller instructions."
    result.save(saved_run, format="dir" if saved_run.is_dir() else "json")
    with pytest.raises(ValueError, match="metadata changed"):
        reader.load(item)


def test_saved_production_naturalness_errors_resume_and_cached_success(
    saved_run, monkeypatch
):
    from types import SimpleNamespace

    from tau2.data_model.message import AssistantMessage, UserMessage

    results = Results.load(saved_run)
    for simulation in results.simulations:
        simulation.messages = [
            UserMessage(role="user", content="Pode ajudar?"),
            AssistantMessage(role="assistant", content="Claro, posso ajudar."),
        ]
    results.save(saved_run, format="dir" if saved_run.is_dir() else "json")
    before = all_source_bytes(saved_run)
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(content='{"results": []}')
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": "naturalness",
                            "opportunity": True,
                            "violated": False,
                            "severity": 0,
                            "reasoning": "Natural phrasing.",
                            "quote": "",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr("tau2.judges.criteria.generate", generate)
    # Without in-run retries the failed unit stays recorded for the next run.
    no_retry = ("--only", "naturalness", "--unit-retries", "0")
    report = build_export(args("suite", str(saved_run), *no_retry))
    assert report.summary[0].error_calls == 1
    assert report.summary[0].judged_calls == 1
    report = build_export(args("suite", str(saved_run), *no_retry))
    assert len(calls) == 3
    assert report.summary[0].complete
    assert report.summary[0].observed_pass_fraction == 1
    build_export(args("suite", str(saved_run), "--only", "naturalness"))
    assert len(calls) == 3
    assert all_source_bytes(saved_run) == before
    assert all(call["model"] == "gpt-5.5" for call in calls)
    assert all(call["reasoning_effort"] == "xhigh" for call in calls)


def test_a_failed_reply_is_asked_again_in_the_same_suite_run(saved_run, monkeypatch):
    from types import SimpleNamespace

    from tau2.data_model.message import AssistantMessage, UserMessage

    results = Results.load(saved_run)
    for simulation in results.simulations:
        simulation.messages = [
            UserMessage(role="user", content="Pode ajudar?"),
            AssistantMessage(role="assistant", content="Claro, posso ajudar."),
        ]
    results.save(saved_run, format="dir" if saved_run.is_dir() else "json")
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(content='{"results": []}')
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": "naturalness",
                            "opportunity": True,
                            "violated": False,
                            "severity": 0,
                            "reasoning": "Natural phrasing.",
                            "quote": "",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr("tau2.judges.criteria.generate", generate)
    report = build_export(args("suite", str(saved_run), "--only", "naturalness"))
    assert len(calls) == 3
    assert report.summary[0].complete
    assert report.summary[0].error_calls == 0
    assert all(call["model"] == "gpt-5.5" for call in calls)


def test_evidence_reader_is_lazy_and_reuses_metadata(saved_run, monkeypatch):
    from tau2.judges.run_evidence import SavedRun

    metadata_reads = []
    load = Results.load_metadata

    def counted(path, **kwargs):
        metadata_reads.append(path)
        return load(path, **kwargs)

    monkeypatch.setattr(Results, "load_metadata", counted)
    saved = SavedRun(saved_run)
    assert metadata_reads == []
    item = saved.inputs[0]
    evidence = saved.load(item)
    assert len(metadata_reads) == 1
    assert evidence.simulation.id == item.simulation_id
    assert evidence.language == item.language
    assert evidence.agent_gender == item.agent_gender
    assert evidence.task.id == evidence.simulation.task_id
    assert evidence.policy == load(saved_run).info.environment_info.policy


def test_suite_consumes_typed_verdict_without_json_round_trip(saved_run, monkeypatch):
    from tau2.data_model.judges import MeasureVerdict
    from tau2.judges.catalog import load_catalog
    from tau2.judges.suite import judge_saved_run

    monkeypatch.setattr(
        MeasureVerdict,
        "model_validate_json",
        lambda *_: pytest.fail("Typed judge verdict must not be reparsed as JSON"),
    )
    report = judge_saved_run(saved_run, load_catalog(), only=["tool_use"], execute=True)
    assert all(row.verdict is not None for row in report.verdicts)
    assert all(row.error is None for row in report.verdicts)


def test_directory_call_edits_are_rejected_before_judging(saved_run):
    if not saved_run.is_dir():
        return
    from tau2.judges.run_evidence import SavedRun

    reader = SavedRun(saved_run)
    item = reader.inputs[0]
    _, directory = Results._resolve_paths(saved_run)
    path = directory / f"{item.simulation_id}.json"
    simulation = SimulationRun.model_validate_json(path.read_text())
    simulation.policy = "Changed call policy."
    path.write_text(simulation.model_dump_json())
    with pytest.raises(ValueError, match="inputs changed"):
        reader.load(item)


def test_suite_and_summarize_print_compact_tables_by_default(
    saved_run, monkeypatch, capsys, tmp_path
):
    from tau2.judges.cli import run_judges
    from tau2.judges.summary import DiagnosticExport

    monkeypatch.setattr(
        "tau2.judges.suite.create_judge",
        lambda *_: pytest.fail("read-only command reached backend"),
    )
    selection = ["--only", "naturalness,tool_use"]

    run_judges(args("summarize", str(saved_run), *selection))
    table = capsys.readouterr().out
    assert table.splitlines()[0].split() == [
        "measure",
        "applicable",
        "judged",
        "errors",
        "not_judged",
        "pass",
        "fail",
        "pass_frac",
        "complete",
    ]
    assert table.splitlines()[2].split() == [
        "naturalness", "2", "0", "0", "2", "0", "0", "-", "no"
    ]  # fmt: skip
    assert "{" not in table

    run_judges(args("suite", str(saved_run), "--only", "tool_use", "--dry-run"))
    plan = capsys.readouterr().out
    assert plan.splitlines()[0].split() == [
        "measure",
        "judge",
        "reuse",
        "not_applicable",
        "missing_input",
    ]
    assert plan.splitlines()[2].split() == ["tool_use", "2", "0", "0", "0"]
    assert "Verdicts to judge: 2; reused: 0." in plan
    # Tool use asks at most one question per calibrated criterion per call.
    assert "Estimated model calls (upper bound): 6." in plan

    # --json restores the full typed export, identical to the library result.
    run_judges(args("summarize", str(saved_run), *selection, "--json"))
    exported = DiagnosticExport.model_validate_json(capsys.readouterr().out)
    assert exported == export_run(
        saved_run, load_catalog(), only=selection[1].split(",")
    )

    # --output saves the same JSON while the terminal keeps the table.
    output = tmp_path / "out" / "diagnostics.json"
    run_judges(
        args("suite", str(saved_run), *selection, "--dry-run", "--output", str(output))
    )
    printed = capsys.readouterr().out
    assert printed.startswith("measure")
    assert f"Full JSON written to {output}" in printed
    assert DiagnosticExport.model_validate_json(output.read_text()).plan == (
        exported.plan
    )
    assert not progress_path(saved_run.resolve()).exists()


def test_export_command_stays_json(saved_run, capsys):
    from tau2.judges.cli import run_judges
    from tau2.judges.summary import DiagnosticExport

    run_judges(args("export", str(saved_run), "--only", "tool_use"))
    DiagnosticExport.model_validate_json(capsys.readouterr().out)
    with pytest.raises(SystemExit):
        args("export", str(saved_run), "--json")
