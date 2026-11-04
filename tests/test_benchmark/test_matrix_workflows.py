"""Offline matrix discovery shared by viewing, diagnostics, and submissions."""

import argparse

import pytest

from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import (
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    TextRunConfig,
    VoiceRunConfig,
)
from tau2.multilingual.runtime import load_language
from tau2.runner.helpers import get_info
from tau2.runner.presets import load_frame, resolve_configs


@pytest.fixture
def saved_matrix(tmp_path):
    root = tmp_path / "matrix"
    root.mkdir()
    frame = load_frame()
    (root / "benchmark-frame.json").write_text(frame.model_dump_json())
    plan = resolve_configs(
        frame,
        VoiceRunConfig(audio_native_config=AudioNativeConfig()),
        root,
        languages=["en", "pt"],
        domains=["retail"],
    )
    for config in plan.configs:
        loaded = load_language(config.language, config.domain, mode="voice")
        info = get_info(config)
        tasks = [task for task in loaded.tasks if task.id in config.task_ids]
        simulation = SimulationRun(
            id="matrix-call",
            task_id=tasks[0].id,
            trial=0,
            start_time="start",
            end_time="end",
            duration=1,
            termination_reason=TerminationReason.USER_STOP,
            reward_info=RewardInfo(reward=1),
            language_context=info.language_tasks[tasks[0].id],
        )
        Results(info=info, tasks=tasks, simulations=[simulation]).save(
            root / f"{config.language}-retail" / "results.json", format="dir"
        )
    return root


def test_text_matrix_counts_inline_simulations(tmp_path):
    """Text cells keep simulations inside results.json; all must be counted."""
    from tau2.metrics.benchmark import load_saved_benchmark

    root = tmp_path / "text-matrix"
    root.mkdir()
    frame = load_frame("text")
    (root / "benchmark-frame.json").write_text(frame.model_dump_json())
    plan = resolve_configs(
        frame,
        TextRunConfig(workers=0),
        root,
        languages=["en", "pt"],
        domains=["retail"],
    )
    for config in plan.configs:
        loaded = load_language(config.language, config.domain, mode="text")
        info = get_info(config)
        tasks = [task for task in loaded.tasks if task.id in config.task_ids]
        task = tasks[0]
        simulation = SimulationRun(
            id="text-call",
            task_id=task.id,
            trial=0,
            start_time="start",
            end_time="end",
            duration=1,
            termination_reason=TerminationReason.USER_STOP,
            reward_info=RewardInfo(reward=1),
            language_context=info.language_tasks[task.id],
        )
        Results(info=info, tasks=tasks, simulations=[simulation]).save(
            root / f"{config.language}-retail" / "results.json", format="json"
        )

    summary = load_saved_benchmark(root).submission.summary
    assert summary.scored_trials == 2
    assert summary.missing_trials == 898


def test_matrix_discovery_validates_membership_and_ignores_unreleased_dirs(
    saved_matrix,
):
    from tau2.metrics.benchmark import load_saved_benchmark

    (saved_matrix / "fr-retail").mkdir()
    (saved_matrix / "fr-retail" / "results.json").write_text("unreleased")
    matrix = load_saved_benchmark(saved_matrix)
    assert matrix.preset == "voice"
    assert len(matrix.cells) == 18
    assert sum(cell.results_path is not None for cell in matrix.cells) == 2
    assert matrix.submission.summary.scored_trials == 2
    assert matrix.submission.summary.missing_trials == 898
    assert matrix.submission.summary.score is None
    result_path = saved_matrix / "en-retail"
    result = Results.load(result_path)
    result.info.user_info.llm = "tampered"
    result.save(result_path, format="dir")
    with pytest.raises(ValueError, match="Effective saved settings"):
        load_saved_benchmark(saved_matrix)


def test_matrix_judges_dry_run_and_summary_preserve_cells_without_model_calls(
    saved_matrix, monkeypatch, capsys
):
    from tau2.judges import cli, suite
    from tau2.judges.summary import MatrixDiagnosticExport

    monkeypatch.setattr(
        suite,
        "create_judge",
        lambda *_: pytest.fail("Read-only command called backend"),
    )
    parser = argparse.ArgumentParser()
    cli.add_judges_args(parser)
    for command in ("suite", "summarize", "export"):
        values = [command, str(saved_matrix), "--only", "naturalness"]
        if command == "suite":
            values.append("--dry-run")
        if command != "export":
            capsys.readouterr()
            cli.run_judges(parser.parse_args(values))
            table = capsys.readouterr().out.splitlines()
            assert table[0].split()[:2] == ["cell", "measure"]
            assert [row.split()[:2] for row in table if "naturalness" in row] == [
                ["en/retail", "naturalness"],
                ["pt/retail", "naturalness"],
            ]
            assert sum("(no results)" in row for row in table) == 16
            values.append("--json")
        capsys.readouterr()
        cli.run_judges(parser.parse_args(values))
        report = MatrixDiagnosticExport.model_validate_json(capsys.readouterr().out)
        assert len(report.cells) == 18
        reports = [cell.diagnostics for cell in report.cells if cell.diagnostics]
        assert len(reports) == 2
        assert reports[0].plan.items[0].input.language == "en"
        assert reports[0].summary[0].not_judged_calls == 1
        assert reports[1].plan.items[0].input.language == "pt"
        assert reports[1].summary[0].not_judged_calls == 1
    assert not list(saved_matrix.rglob("diagnostics"))


def test_submit_prepare_autodetects_matrix_and_refuses_incomplete(
    saved_matrix, tmp_path, submitter_file
):
    from tau2.scripts.leaderboard.cli import add_submit_args

    parser = argparse.ArgumentParser()
    add_submit_args(parser)
    args = parser.parse_args(
        ["prepare", str(saved_matrix), "--output", str(tmp_path / "submission")]
    )
    with pytest.raises(SystemExit, match="require --metadata-file"):
        args.func(args)
    args.metadata_file = submitter_file
    with pytest.raises(SystemExit, match="898/900 calls lack a scored reward"):
        args.func(args)
    assert not (tmp_path / "submission").exists()
    args.no_verify = True
    with pytest.raises(ValueError, match="full verification"):
        args.func(args)


def test_matrix_view_shows_official_missing_and_cell_drilldown(
    saved_matrix, monkeypatch
):
    from rich.console import Console

    from tau2.scripts import view_simulations as viewer

    console = Console(record=True, width=140)
    monkeypatch.setattr(viewer.ConsoleDisplay, "console", console)
    selected = []
    monkeypatch.setattr(viewer, "main", lambda **kwargs: selected.append(kwargs))
    choices = iter([1, 0])
    monkeypatch.setattr(viewer.IntPrompt, "ask", lambda *a, **kw: next(choices))
    viewer.view_benchmark(saved_matrix)
    output = console.export_text()
    assert "Official pass@1: incomplete" in output
    assert "898 missing" in output
    assert "en" in output and "retail" in output
    assert selected[0]["sim_file"] == str(saved_matrix / "en-retail" / "results.json")


def test_default_view_discovery_keeps_matrix_roots(saved_matrix):
    from tau2.scripts.view_simulations import get_available_simulations

    assert get_available_simulations(saved_matrix.parent) == [saved_matrix]


def test_matrix_reader_rejects_unpublished_frame_before_any_judging(
    saved_matrix, monkeypatch
):
    from tau2.judges import cli, suite

    frame = load_frame()
    frame.cells[0].binding_revision = "unpublished-change"
    (saved_matrix / "benchmark-frame.json").write_text(frame.model_dump_json())
    monkeypatch.setattr(
        suite, "create_judge", lambda *_: pytest.fail("Invalid frame reached backend")
    )
    parser = argparse.ArgumentParser()
    cli.add_judges_args(parser)
    args = parser.parse_args(["suite", str(saved_matrix)])
    with pytest.raises(ValueError, match="installed published benchmark"):
        cli.run_judges(args)
