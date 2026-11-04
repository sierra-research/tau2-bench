"""English Voice V1 remains an explicit, complete current-leaderboard frame."""

import argparse

import pytest

from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import (
    Results,
    SimulationRun,
    TerminationReason,
    VoiceRunConfig,
)
from tau2.metrics.benchmark import _authoritative_language
from tau2.multilingual.runtime import load_language, load_run_language, resolve_task
from tau2.runner.cli import check_benchmark_selection
from tau2.runner.helpers import get_info
from tau2.runner.presets import (
    BenchmarkRequest,
    load_cell_language,
    load_frame,
    reload_recorded_call,
    resolve_benchmark,
    uses_canonical_tasks,
)
from tau2.scripts.leaderboard.prepare_submission import validate_submission_traj_set
from tau2.task_subsets.store import load_frame as load_canonical_frame


def test_voice_v1_frame_is_complete_english_base(tmp_path):
    frame = load_frame("voice-v1")
    assert frame.benchmark_version == "voice-v1"
    assert [
        (cell.language, cell.domain, len(cell.canonical_task_ids))
        for cell in frame.cells
    ] == [
        ("en", "airline", 50),
        ("en", "retail", 114),
        ("en", "telecom", 114),
    ]
    plan = resolve_benchmark(
        BenchmarkRequest(preset="voice-v1", output=tmp_path / "v1")
    )
    assert not plan.partial
    assert len(plan.configs) == 3
    assert all(
        config.benchmark.benchmark_version == "voice-v1" for config in plan.configs
    )
    assert all(config.language == "en" for config in plan.configs)
    assert [len(config.task_ids) for config in plan.configs] == [50, 114, 114]
    for config in plan.configs:
        loaded = load_language("en", config.domain, mode="voice", canonical_tasks=True)
        canonical = load_canonical_frame(config.domain).tasks
        assert [loaded.task_index[task.id].model_dump() for task in canonical] == [
            task.model_dump() for task in canonical
        ]
    assert (
        load_language("en", "retail", mode="voice").tasks_sha256
        != load_language(
            "en", "retail", mode="voice", canonical_tasks=True
        ).tasks_sha256
    )
    assert not (tmp_path / "v1").exists()
    assert load_frame("voice").benchmark_version == "voice-v2"


def test_voice_v1_cli_dry_run_uses_existing_controller_path(monkeypatch, tmp_path):
    import sys

    from tau2 import cli
    from tau2.runner import cli as runner_cli

    captured = []
    monkeypatch.setattr(
        runner_cli,
        "execute_plan",
        lambda plan, output, settings, **kwargs: captured.append((plan, kwargs)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tau2",
            "run",
            "--benchmark",
            "voice-v1",
            "--domain",
            "retail",
            "--audio-native",
            "--dry-run",
            "--save-to",
            str(tmp_path / "retail"),
        ],
    )
    cli.main()
    plan, options = captured[0]
    assert len(plan.configs) == 1
    assert plan.configs[0].benchmark.benchmark_version == "voice-v1"
    assert len(plan.configs[0].task_ids) == 114
    assert options == {"dry_run": True}
    assert not (tmp_path / "retail").exists()


@pytest.mark.parametrize("override", [{"num_tasks": 1}, {"task_ids": ["0"]}])
def test_voice_v1_rejects_task_slices(tmp_path, override):
    with pytest.raises(ValueError, match="complete task frame"):
        resolve_benchmark(
            BenchmarkRequest(
                preset="voice-v1",
                output=tmp_path / "sample",
                template=VoiceRunConfig(
                    audio_native_config=AudioNativeConfig(), **override
                ),
            )
        )


@pytest.mark.parametrize(
    "override", [{"task_set_name": "retail"}, {"task_split_name": "train"}]
)
def test_voice_v1_rejects_custom_task_sources(tmp_path, override):
    with pytest.raises(ValueError, match="tasks are frozen"):
        resolve_benchmark(
            BenchmarkRequest(
                preset="voice-v1",
                output=tmp_path / "custom",
                template=VoiceRunConfig(
                    audio_native_config=AudioNativeConfig(), **override
                ),
            )
        )


@pytest.mark.parametrize(
    "override",
    [
        {"language": ["es"]},
        {"num_tasks": 1},
    ],
)
def test_voice_v1_cli_rejects_nonrelease_selection(override):
    args = argparse.Namespace(
        benchmark="voice-v1",
        audio_native=True,
        domain=["retail"],
        language=None,
        user_persona=None,
        task_ids=None,
        num_tasks=None,
    )
    for key, value in override.items():
        setattr(args, key, value)
    with pytest.raises(ValueError):
        check_benchmark_selection(args)


def test_voice_v1_submission_requires_the_recorded_full_domain(tmp_path):
    plan = resolve_benchmark(
        BenchmarkRequest(
            preset="voice-v1",
            output=tmp_path / "retail",
            domains=["airline"],
        )
    )
    config = plan.configs[0]
    loaded = load_language("en", "airline", mode="voice")
    tasks = [loaded.task_index[task_id] for task_id in config.task_ids]
    info = get_info(config)
    simulations = [
        SimulationRun(
            id=f"sim-{index}",
            task_id=task.id,
            trial=0,
            start_time="start",
            end_time="end",
            duration=1,
            termination_reason=TerminationReason.USER_STOP,
        )
        for index, task in enumerate(tasks)
    ]
    results = Results(info=info, tasks=tasks, simulations=simulations)
    assert validate_submission_traj_set([results])[0]
    assert not validate_submission_traj_set(
        [results.model_copy(update={"simulations": simulations[:-1]})]
    )[0]
    for extra in (
        simulations[0].model_copy(update={"id": "extra-trial", "trial": 1}),
        simulations[0].model_copy(update={"id": "extra-task", "task_id": "foreign"}),
        simulations[0].model_copy(update={"id": "duplicate-attempt"}),
    ):
        changed = results.model_copy(update={"simulations": [*simulations, extra]})
        assert "exactly one result" in validate_submission_traj_set([changed])[1]
    sliced = results.model_copy(deep=True)
    sliced.info.benchmark = sliced.info.benchmark.model_copy(
        update={"selected_task_ids": ["0"]}
    )
    assert not validate_submission_traj_set([sliced])[0]
    unmarked = results.model_copy(deep=True)
    unmarked.info.benchmark = None
    assert "mix benchmark modes" in validate_submission_traj_set([results, unmarked])[1]
    assert validate_submission_traj_set([unmarked])[0]


@pytest.mark.parametrize("domain", ["airline", "retail", "telecom"])
def test_voice_v1_calls_and_export_reload_the_pinned_binding(domain):
    """Voice V1 pins the canonical task set, also where the pack publishes a
    tasks file (retail, telecom); the per-call reload and the authoritative
    export must load that same binding, and never guess between the two."""
    frame = load_frame("voice-v1")
    assert uses_canonical_tasks(frame)
    cell = next(cell for cell in frame.cells if cell.domain == domain)
    plan = load_cell_language(
        cell.language, cell.domain, frame.mode, canonical_tasks=True
    )
    context = resolve_task(plan, plan.tasks[0], 300)
    reloaded = reload_recorded_call(context, cell.domain, canonical_tasks=True)
    assert reloaded.artifact_pins == context.artifact_pins
    if plan.binding.tasks_file is not None:
        with pytest.raises(ValueError, match="differ from the recorded call"):
            reload_recorded_call(context, cell.domain)
    authoritative = _authoritative_language(
        cell.language,
        cell.domain,
        cell.binding_revision,
        frame.mode,
        canonical_tasks=True,
    )
    assert authoritative.artifact_pins == plan.artifact_pins


def test_voice_v1_run_configs_reload_their_planned_binding(tmp_path):
    """The batch path derives the reload rule from each planned run config."""
    plan = resolve_benchmark(
        BenchmarkRequest(preset="voice-v1", output=tmp_path / "v1")
    )
    for config in plan.configs:
        assert uses_canonical_tasks(config.benchmark)
        loaded = load_run_language(config)
        context = resolve_task(loaded, loaded.tasks[0], 300)
        reloaded = reload_recorded_call(
            context,
            config.domain,
            canonical_tasks=uses_canonical_tasks(config.benchmark),
        )
        assert reloaded.artifact_pins == loaded.artifact_pins
