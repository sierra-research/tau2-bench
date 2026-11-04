"""Default voice execution selects a released frame, never installed languages."""

import pytest

from tau2.data_model.live_config import LiveConfig
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import VoiceRunConfig
from tau2.runner.presets import resolve_configs


def test_sample_uses_same_frozen_ids_and_keeps_official_denominator(
    published_frame, tmp_path
):
    template = VoiceRunConfig(num_tasks=2, audio_native_config=AudioNativeConfig())
    plan = resolve_configs(published_frame, template, tmp_path, domains=["retail"])
    assert len(plan.configs) == 6
    assert len(plan.frame.cells) == 18
    assert plan.partial
    assert all(
        config.benchmark.selected_task_ids == ["0", "1"] for config in plan.configs
    )
    assert all(
        len(config.benchmark.canonical_task_ids) == 50 for config in plan.configs
    )
    assert all(
        config.task_ids == ["retail-localized-0", "retail-localized-1"]
        for config in plan.configs
    )


def test_canonical_ids_are_resolved_explicitly(published_frame, tmp_path):
    template = VoiceRunConfig(
        task_ids=["9", "2"], audio_native_config=AudioNativeConfig()
    )
    plan = resolve_configs(published_frame, template, tmp_path, domains=["retail"])
    assert all(
        config.benchmark.selected_task_ids == ["2", "9"] for config in plan.configs
    )
    with pytest.raises(ValueError, match="one domain"):
        resolve_configs(published_frame, template, tmp_path)
    with pytest.raises(ValueError, match="frozen"):
        resolve_configs(
            published_frame,
            VoiceRunConfig(
                task_ids=["foreign"], audio_native_config=AudioNativeConfig()
            ),
            tmp_path,
            domains=["retail"],
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_tasks": 0},
        {"num_tasks": 51},
        {"num_tasks": 2, "task_ids": ["0"]},
        {"task_set_name": "other"},
        {"task_split_name": "train"},
    ],
)
def test_selection_overrides_are_not_silently_discarded(
    published_frame, tmp_path, kwargs
):
    with pytest.raises(ValueError):
        resolve_configs(
            published_frame,
            VoiceRunConfig(audio_native_config=AudioNativeConfig(), **kwargs),
            tmp_path,
        )


@pytest.mark.parametrize(
    "flags, cells, calls",
    [
        ([], 18, 900),
        (["--domain", "retail"], 6, 300),
        (["--language", "en"], 3, 150),
        (["--domain", "retail", "--language", "en"], 1, 50),
        (["--domain", "retail", "--num-tasks", "2"], 6, 12),
    ],
)
def test_normal_voice_cli_resolves_released_default(
    monkeypatch, tmp_path, capsys, flags, cells, calls
):
    import sys

    from tau2 import cli
    from tau2.runner import cli as runner_cli

    captured = []
    monkeypatch.setattr(
        runner_cli,
        "execute_plan",
        lambda plan, output, settings, **kwargs: captured.append(
            (plan, settings, kwargs)
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tau2",
            "run",
            "--audio-native",
            "--dry-run",
            "--save-to",
            str(tmp_path / "matrix"),
            *flags,
        ],
    )
    cli.main()
    plan, settings, options = captured[0]
    assert len(plan.configs) == cells
    assert sum(len(config.task_ids) for config in plan.configs) == calls
    assert len(plan.frame.cells) == 18
    assert settings.workers == 1
    assert options == {"dry_run": True}
    assert all(config.llm_user == "openai/gpt-5.5" for config in plan.configs)
    assert all(
        config.llm_args_user == {"reasoning_effort": "low"} for config in plan.configs
    )
    assert all(
        config.audio_native_config.max_steps_seconds == 2400 for config in plan.configs
    )
    assert all(config.audio_native_config.voice == "marin" for config in plan.configs)
    assert all(
        config.communicate_scoring.judge.model == "gpt-5.5" for config in plan.configs
    )
    assert f"{calls} calls" in capsys.readouterr().out
    assert not (tmp_path / "matrix").exists()


def test_installing_a_pack_does_not_change_released_membership(tmp_path, monkeypatch):
    import shutil

    import yaml

    import tau2.utils
    from tau2.multilingual.runtime import list_languages
    from tau2.runner.presets import BenchmarkRequest, resolve_benchmark

    original = tau2.utils.DATA_DIR
    before = resolve_benchmark(BenchmarkRequest(output=tmp_path / "before"))
    released = tmp_path / "installation"
    shutil.copytree(original / "tau2" / "benchmarks", released / "tau2" / "benchmarks")
    shutil.copytree(
        original / "tau2" / "task_subsets", released / "tau2" / "task_subsets"
    )
    shutil.copytree(
        original / "tau2" / "multilingual", released / "tau2" / "multilingual"
    )
    for domain in ("airline", "retail", "telecom"):
        shutil.copytree(
            original / "tau2" / "domains" / domain,
            released / "tau2" / "domains" / domain,
        )
    extra = released / "tau2" / "multilingual" / "fr"
    shutil.copytree(released / "tau2" / "multilingual" / "en", extra)
    pack_path = extra / "pack.yaml"
    pack = yaml.safe_load(pack_path.read_text())
    pack["language"] = "fr"
    pack_path.write_text(yaml.safe_dump(pack, allow_unicode=True))
    monkeypatch.setattr(tau2.utils, "DATA_DIR", released)
    assert "fr" in list_languages()
    after = resolve_benchmark(BenchmarkRequest(output=tmp_path / "after"))
    assert before.frame == after.frame
    assert {config.language for config in after.configs} == {
        "en",
        "es",
        "pt",
        "hi",
        "zh",
        "ko",
    }
    assert sum(len(config.task_ids) for config in after.configs) == 900


def test_resume_expands_requested_tasks_without_changing_conditions(
    published_frame, tmp_path, monkeypatch
):
    from tau2.data_model.benchmark import BenchmarkPlan
    from tau2.runner import batch
    from tau2.runner.presets import ExecutionSettings, execute_plan

    tmp_path = tmp_path / "matrix"
    runs = []
    monkeypatch.setattr(
        batch, "run_domains", lambda configs, **kwargs: runs.append(configs)
    )
    for task_ids in (["0"], ["1"]):
        plan = resolve_configs(
            published_frame,
            VoiceRunConfig(task_ids=task_ids, audio_native_config=AudioNativeConfig()),
            tmp_path,
            domains=["retail"],
            languages=["en"],
        )
        execute_plan(plan, tmp_path, ExecutionSettings(workers=1))
    assert runs[-1][0].benchmark.selected_task_ids == ["0", "1"]
    saved = BenchmarkPlan.model_validate_json(
        (tmp_path / "benchmark-plan.json").read_text()
    )
    assert saved.selection.cells[0].canonical_task_ids == ["0", "1"]
    changed = resolve_configs(
        published_frame,
        VoiceRunConfig(
            num_tasks=3, llm_user="different", audio_native_config=AudioNativeConfig()
        ),
        tmp_path,
        domains=["retail"],
        languages=["en"],
    )
    before = (tmp_path / "benchmark-plan.json").read_bytes()
    with pytest.raises(ValueError, match="effective benchmark conditions"):
        execute_plan(changed, tmp_path, ExecutionSettings(workers=1))
    assert (tmp_path / "benchmark-plan.json").read_bytes() == before


def test_sample_scores_and_checkpoint_extension_are_explicit(
    published_frame, tmp_path, monkeypatch
):
    from tau2.data_model.simulation import (
        Results,
        RewardInfo,
        SimulationRun,
        TerminationReason,
    )
    from tau2.metrics.benchmark import export_submission
    from tau2.multilingual.runtime import load_language
    from tau2.runner import batch
    from tau2.runner.checkpoint import try_resume
    from tau2.runner.helpers import get_info
    from tau2.runner.presets import ExecutionSettings, execute_plan

    root = tmp_path / "matrix"
    calls = []
    monkeypatch.setattr(
        batch, "run_domains", lambda configs, **kwargs: calls.append(configs)
    )
    template = VoiceRunConfig(num_tasks=1, audio_native_config=AudioNativeConfig())
    plan = resolve_configs(
        published_frame, template, root, domains=["retail"], languages=["en"]
    )
    execute_plan(plan, root, ExecutionSettings(workers=1))
    config = calls[-1][0]
    loaded = load_language("en", "retail", mode="voice")
    info = get_info(config)
    tasks = [task for task in loaded.tasks if task.id in config.task_ids]
    sim = SimulationRun(
        id="complete-one",
        task_id=tasks[0].id,
        trial=0,
        seed=4,
        start_time="start",
        end_time="end",
        duration=1,
        termination_reason=TerminationReason.USER_STOP,
        reward_info=RewardInfo(reward=1),
        language_context=info.language_tasks[tasks[0].id],
    )
    path = root / "en-retail" / "results.json"
    Results(info=info, tasks=tasks, simulations=[sim]).save(path, format="dir")
    submission = export_submission(root, published_frame)
    assert (
        submission.summary.score is None and submission.summary.expected_trials == 900
    )
    assert submission.requested.complete and submission.requested.score == 1
    assert submission.requested.expected_trials == 1

    # Expanding the requested tasks keeps the first completed call and grows metadata.
    template = VoiceRunConfig(num_tasks=2, audio_native_config=AudioNativeConfig())
    plan = resolve_configs(
        published_frame, template, root, domains=["retail"], languages=["en"]
    )
    execute_plan(plan, root, ExecutionSettings(workers=3))
    config = calls[-1][0]
    tasks = [task for task in loaded.tasks if task.id in config.task_ids]
    current = Results(info=get_info(config), tasks=tasks, simulations=[])
    resumed, done, _ = try_resume(
        path, current, tasks, 1, auto_resume=True, results_format="dir"
    )
    assert done == {(0, sim.task_id, 4)}
    assert resumed.simulations == [sim]
    assert resumed.info.benchmark.selected_task_ids == ["0", "1"]
    submission = export_submission(root, published_frame)
    assert submission.requested.expected_trials == 2
    assert submission.requested.scored_trials == 1
    assert submission.requested.score is None
    assert submission.summary.missing_trials == 899


@pytest.mark.parametrize(
    "provider", ["openai", "gemini", "xai", "nova", "qwen", "livekit"]
)
def test_python_request_uses_provider_model_defaults(
    published_frame, tmp_path, provider
):
    from tau2.config import DEFAULT_AUDIO_NATIVE_MODELS

    plan = resolve_configs(
        published_frame,
        VoiceRunConfig(audio_native_config=AudioNativeConfig(provider=provider)),
        tmp_path,
        languages=["en"],
        domains=["retail"],
    )
    assert (
        plan.configs[0].audio_native_config.model
        == DEFAULT_AUDIO_NATIVE_MODELS[provider]
    )


@pytest.mark.parametrize(
    "provider, expected_model",
    [
        ("openai_live", "gpt-live-1"),
        ("gemini", "gemini-3.8-live-extended-thinking"),
        ("xai", "grok-voice-think-fast-2.0"),
    ],
)
def test_frontier_provider_defaults(
    published_frame, tmp_path, provider, expected_model
):
    live_config = (
        LiveConfig(backend_model="gpt-6-astra") if provider == "openai_live" else None
    )
    plan = resolve_configs(
        published_frame,
        VoiceRunConfig(
            audio_native_config=AudioNativeConfig(
                provider=provider, live_config=live_config
            )
        ),
        tmp_path,
        languages=["en"],
        domains=["retail"],
    )
    assert plan.configs[0].audio_native_config.model == expected_model


def test_cli_preserves_explicit_user_voice_duration_and_worker_overrides(
    monkeypatch, tmp_path, capsys
):
    import sys

    from tau2 import cli
    from tau2.runner import cli as runner_cli

    captured = []
    monkeypatch.setattr(
        runner_cli,
        "execute_plan",
        lambda plan, output, settings, **kwargs: captured.append((plan, settings)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tau2",
            "run",
            "--audio-native",
            "--language",
            "en",
            "--domain",
            "retail",
            "--dry-run",
            "--save-to",
            str(tmp_path),
            "--user-llm",
            "custom",
            "--user-llm-args",
            '{"temperature":0.7}',
            "--voice",
            "cedar",
            "--max-steps-seconds",
            "90",
            "--audio-native-model",
            "custom-model",
            "--reasoning-effort",
            "low",
            "--workers",
            "3",
            "--max-concurrency",
            "10",
            "--global-limit",
            "20",
            "--provider-limit",
            "openai=15",
        ],
    )
    cli.main()
    plan, settings = captured[0]
    config = plan.configs[0]
    assert (config.llm_user, config.llm_args_user) == ("custom", {"temperature": 0.7})
    assert config.audio_native_config.voice == "cedar"
    assert config.audio_native_config.max_steps_seconds == 90
    assert config.audio_native_config.model == "custom-model"
    assert config.audio_native_config.reasoning_effort == "low"
    assert config.max_concurrency == 10
    assert settings.workers == 3 and settings.global_limit == 20
    assert settings.provider_limits == {"openai": 15}
    assert "50 calls" in capsys.readouterr().out


def test_custom_voice_escape_preserves_intake_defaults(monkeypatch):
    import sys

    from tau2 import cli
    from tau2.config import (
        DEFAULT_ELICITATION_LLM_USER,
        DEFAULT_ELICITATION_MAX_STEPS_SECONDS,
    )

    captured = []
    monkeypatch.setattr(cli, "run_domain", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tau2",
            "run",
            "--audio-native",
            "--benchmark",
            "none",
            "--domain",
            "intake",
            "--num-tasks",
            "2",
        ],
    )
    cli.main()
    config = captured[0]
    assert config.benchmark is None
    assert config.language == "en"
    assert config.domain == "intake" and config.num_tasks == 2
    assert config.llm_user == DEFAULT_ELICITATION_LLM_USER
    assert (
        config.audio_native_config.max_steps_seconds
        == DEFAULT_ELICITATION_MAX_STEPS_SECONDS
    )


@pytest.mark.parametrize("flags", [["--domain", "intake"], ["--language", "fr"]])
def test_unreleased_selections_fail_with_custom_run_guidance(monkeypatch, flags):
    import sys

    from tau2 import cli

    monkeypatch.setattr(
        sys, "argv", ["tau2", "run", "--audio-native", "--dry-run", *flags]
    )
    with pytest.raises(ValueError, match="--benchmark none"):
        cli.main()


def test_unreleased_domain_is_rejected_before_config_build(monkeypatch):
    """Elicitation flags must not mask the frame error with a config error."""
    import sys

    from tau2 import cli

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tau2",
            "run",
            "--audio-native",
            "--dry-run",
            "--domain",
            "intake",
            "--elicitation-strategy",
            "scaffolded",
        ],
    )
    with pytest.raises(ValueError, match="not part of the released.*--benchmark none"):
        cli.main()


@pytest.mark.parametrize(
    "requested,expected,warns", [(None, 1, False), (0, 1, True), (3, 3, False)]
)
def test_benchmark_workers_warn_when_overridden(requested, expected, warns):
    from loguru import logger

    from tau2.runner.cli import benchmark_workers

    messages = []
    handler = logger.add(messages.append, level="WARNING")
    try:
        assert benchmark_workers(requested) == expected
    finally:
        logger.remove(handler)
    assert bool(messages) == warns


@pytest.mark.parametrize(
    "raw", ['{"model":"fixture-judge","model_args":{"temperature":0}}', "null"]
)
def test_custom_cli_records_semantic_override(monkeypatch, raw):
    import json
    import sys

    from tau2 import cli

    seen = []
    monkeypatch.setattr(cli, "run_domain", lambda config: seen.append(config))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tau2",
            "run",
            "--benchmark",
            "none",
            "--domain",
            "retail",
            "--language",
            "es",
            *(
                ["--communicate-scorer", "substring"]
                if raw == "null"
                else ["--communicate-judge", raw]
            ),
        ],
    )
    cli.main()
    if raw == "null":
        assert seen[0].communicate_scoring.scorer == "substring"
        assert seen[0].communicate_scoring.judge is None
    else:
        judge = seen[0].communicate_scoring.judge
        assert seen[0].communicate_scoring.scorer == "semantic"
        assert judge.model == json.loads(raw)["model"]
        assert judge.model_args == {"temperature": 0}


def test_released_cli_rejects_disabling_semantic_scoring(monkeypatch):
    import sys

    from tau2 import cli

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tau2",
            "run",
            "--audio-native",
            "--communicate-scorer",
            "substring",
            "--dry-run",
        ],
    )
    with pytest.raises(ValueError, match="(?i)semantic"):
        cli.main()


@pytest.mark.parametrize("voice", [False, True])
def test_custom_cli_can_keep_language_with_canonical_scenarios(monkeypatch, voice):
    import sys

    from tau2 import cli

    captured = []
    monkeypatch.setattr(cli, "run_domain", captured.append)
    flags = [
        "tau2",
        "run",
        "--domain",
        "retail",
        "--language",
        "ko",
        "--no-localize-db",
    ]
    if voice:
        flags += ["--audio-native", "--benchmark", "none"]
    monkeypatch.setattr(sys, "argv", flags)
    cli.main()
    config = captured[0]
    assert config.language == "ko"
    assert config.localize_db is False
    assert config.benchmark is None
    if voice:
        assert config.audio_native_config.language == "ko"


def test_released_benchmark_rejects_canonical_scenario_ablation(
    published_frame, tmp_path
):
    with pytest.raises(ValueError, match="--benchmark none"):
        resolve_configs(
            published_frame,
            VoiceRunConfig(localize_db=False, audio_native_config=AudioNativeConfig()),
            tmp_path,
        )
