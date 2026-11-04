"""Generic and crossed Pass^3 retain distinct, reproducible contracts."""

import argparse
from types import SimpleNamespace

import pytest

from tau2.cli import add_run_args
from tau2.data_model.persona import InterruptTendency, PersonaConfig
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import Results, SimulationIndexEntry, VoiceRunConfig
from tau2.data_model.tasks import Task
from tau2.data_model.voice import SynthesisConfig
from tau2.runner.cli import add_pass3_args
from tau2.runner.pass3 import (
    CROSSED_CONDITIONS,
    summarize_crossed_pass3,
    validate_regular_source,
)
from tau2.user_simulation_voice_presets import (
    VoiceSelection,
    get_or_load_task_voice_config,
    sample_voice_config,
)


def test_pass3_default_is_generic_and_crossed_accepts_other_voice_domains():
    parser = argparse.ArgumentParser()
    seen = []
    add_pass3_args(parser, add_run_args=add_run_args, run_command=seen.append)

    generic = parser.parse_args(["--domain", "airline", "--dry-run"])
    generic.func(generic)
    assert len(seen) == 1
    assert seen[0].num_trials == 3
    assert seen[0].domain == ["airline"]

    seen.clear()
    crossed = parser.parse_args(
        [
            "--pass3-mode",
            "crossed",
            "--audio-native",
            "--domain",
            "airline",
            "--save-to",
            "fixture",
            "--dry-run",
        ]
    )
    crossed.func(crossed)
    assert [call.pass3_condition for call in seen] == [
        condition.label for condition in CROSSED_CONDITIONS
    ]
    assert [call.seed for call in seen] == [42, 401, 16319]
    assert all(call.voice_sampling_seed is None for call in seen)
    assert all(call.domain == ["airline"] for call in seen)


def test_crossed_voice_modes_change_acoustics_and_selected_caller_behavior():
    source = SynthesisConfig()
    selection = VoiceSelection(
        persona_name="reviewed",
        persona_config=PersonaConfig(interrupt_tendency=InterruptTendency.WAITS),
        vocal_tics=[],
        non_directed_phrases=[],
        speech_insert_events_per_minute=0.7,
    )
    regular = sample_voice_config(42, source, selection=selection)
    channel = sample_voice_config(
        401,
        source,
        selection=selection,
        channel_effects_mode="heavy",
        speech_effects_mode="light",
    )
    speech = sample_voice_config(
        16319,
        source,
        selection=selection,
        channel_effects_mode="light",
        speech_effects_mode="heavy",
    )

    assert regular.speech_effects_config.speech_insert_events_per_minute == 0.7
    assert channel.channel_effects_config.frame_drop_rate == 0.03
    assert channel.source_effects_config.noise_snr_db == 10
    assert channel.speech_effects_config.speech_insert_events_per_minute == 0
    assert channel.backchannel_min_threshold is None
    assert not channel.enable_interruptions
    assert speech.channel_effects_config.frame_drop_rate == 0
    assert speech.speech_effects_config.speech_insert_events_per_minute == 1.5
    assert speech.persona_config.interrupt_tendency == InterruptTendency.INTERRUPTS
    assert speech.enable_interruptions


def test_crossed_seeds_redraw_the_run_as_in_the_robustness_catalog():
    from tau2.runner.batch import _load_run_tasks
    from tau2.runner.helpers import get_info

    def configs(domain, language):
        return [
            VoiceRunConfig(
                domain=domain,
                language=language,
                num_tasks=1,
                seed=condition.seed,
                audio_native_config=AudioNativeConfig(),
            )
            for condition in CROSSED_CONDITIONS
        ]

    intake = configs("intake", "en")
    _load_run_tasks(intake[0])
    assert len({config.seed for config in intake}) == 3
    assert all(config.voice_sampling_seed is None for config in intake)

    retail = configs("retail", "ko")
    task = _load_run_tasks(retail[0])[0]
    contexts = [
        get_info(config, tasks=[task]).language_tasks[task.id] for config in retail
    ]
    assert contexts[0] == contexts[1] == contexts[2]


def test_crossed_regular_resamples_instead_of_loading_a_presampled_file(
    monkeypatch, tmp_path
):
    from tau2 import user_simulation_voice_presets as presets

    saved = sample_voice_config(42, SynthesisConfig())
    file = tmp_path / "presampled.json"
    file.touch()
    monkeypatch.setattr(presets, "get_task_voice_configs_path", lambda _: file)
    monkeypatch.setattr(
        presets,
        "load_task_voice_configs",
        lambda _: SimpleNamespace(get_config=lambda *args: saved),
    )

    actual = get_or_load_task_voice_config(
        "airline", "task", 42, "regular", SynthesisConfig(), force_sample=True
    )
    assert actual is not saved
    assert (
        actual.model_dump() == sample_voice_config(42, SynthesisConfig()).model_dump()
    )


def _result(condition, rewards, *, ordinary_regular=False):
    info = SimpleNamespace(
        environment_info=SimpleNamespace(domain_name="airline"),
        language=None,
        git_commit="commit",
        domain_tools_version="tools",
        voice_instruction_prompt_version="voice",
        pack_extension=None,
        localize_db=True,
        language_artifacts=None,
        language_tasks=None,
        communicate_scoring=None,
        benchmark=None,
        elicitation=None,
        elicitation_scoring_version="scoring",
        elicitation_prompt_version="prompt",
        elicitation_catalog_version="catalog",
        elicitation_spell_detection_version="spell",
        agent_info="agent",
        user_info="user",
        audio_native_config="audio",
        retrieval_config=None,
        retrieval_config_kwargs=None,
        seed=condition.seed,
        voice_sampling_seed=None,
        pass3_condition=None if ordinary_regular else condition.label,
        num_trials=1,
        max_steps=100,
        max_errors=1,
        text_streaming_config=None,
        speech_complexity="regular",
        channel_effects_mode=condition.channel_effects_mode,
        speech_effects_mode=condition.speech_effects_mode,
    )
    index = [
        SimulationIndexEntry(
            id=f"{condition.label}-{task}",
            task_id=task,
            trial=0,
            reward=reward,
            termination_reason=(
                "infrastructure_error" if reward is None else "agent_stop"
            ),
        )
        for task, reward in rewards.items()
    ]
    return Results.model_construct(
        info=info,
        tasks=[Task.model_construct(id=task) for task in rewards],
        simulation_index=index,
    )


def test_crossed_pass3_scores_only_aligned_complete_conditions():
    results = {
        condition.label: _result(
            condition,
            {"one": 1.0, "two": 0.0 if condition.label == "speech_heavy" else 1.0},
        )
        for condition in CROSSED_CONDITIONS
    }
    paths = {
        condition.label: f"/{condition.label}/results.json"
        for condition in CROSSED_CONDITIONS
    }
    report = summarize_crossed_pass3(results, paths)
    assert report.passed_all == 1
    assert report.total_tasks == 2
    assert report.pass3 == 0.5
    assert report.scored_tasks == 2
    assert report.infrastructure_error_task_ids == []
    assert report.source_paths == paths

    results["speech_heavy"].info.seed = 301
    with pytest.raises(ValueError, match="run seed does not match"):
        summarize_crossed_pass3(results, paths)
    results["speech_heavy"].info.seed = CROSSED_CONDITIONS[2].seed

    results["speech_heavy"].simulation_index = results["speech_heavy"].simulation_index[
        :1
    ]
    with pytest.raises(ValueError, match="missing or unexpected"):
        summarize_crossed_pass3(results, paths)

    results["speech_heavy"] = _result(CROSSED_CONDITIONS[2], {"one": 1.0, "two": 0.0})
    results["speech_heavy"].simulation_index[0].reward = None
    with pytest.raises(ValueError, match="unscored non-infrastructure"):
        summarize_crossed_pass3(results, paths)


def test_crossed_pass3_reuses_ordinary_regular_and_counts_infrastructure():
    results = {
        condition.label: _result(
            condition,
            {"one": None if condition.label == "regular" else 1.0, "two": 1.0},
            ordinary_regular=condition.label == "regular",
        )
        for condition in CROSSED_CONDITIONS
    }
    results["channel_heavy"].info.git_commit = "later-commit"
    report = summarize_crossed_pass3(
        results, {label: f"/{label}/results.json" for label in results}
    )
    assert report.passed_all == 1
    assert report.total_tasks == 2
    assert report.scored_tasks == 1
    assert report.infrastructure_error_task_ids == ["one"]
    assert report.source_git_commits["channel_heavy"] == "later-commit"
    assert report.pass3 == 1.0
    assert report.tasks[0].passes == (None, True, True)


def test_regular_source_rejects_scoring_and_seed_drift():
    regular = CROSSED_CONDITIONS[0]
    result = _result(regular, {"one": 1.0}, ordinary_regular=True)
    expected = _result(regular, {"one": 1.0}, ordinary_regular=True)
    validate_regular_source(result, expected.info, expected.tasks)
    result.info.elicitation_scoring_version = "older"
    with pytest.raises(ValueError, match="settings, scoring, pins or tasks"):
        validate_regular_source(result, expected.info, expected.tasks)
    result.info.elicitation_scoring_version = "scoring"
    result.info.seed = 9401
    with pytest.raises(ValueError, match="ordinary seed-42"):
        validate_regular_source(result, expected.info, expected.tasks)


def test_regular_source_requires_unique_indexed_calls():
    regular = CROSSED_CONDITIONS[0]
    result = _result(regular, {"one": 1.0, "two": None}, ordinary_regular=True)
    validate_regular_source(result, result.info, result.tasks)
    result.simulation_index = result.simulation_index[:1]
    with pytest.raises(ValueError, match="one indexed call per task"):
        validate_regular_source(result, result.info, result.tasks)


def test_crossed_cli_reuses_regular_without_running_it(monkeypatch, tmp_path):
    from tau2.runner import batch, helpers, pass3

    regular = _result(CROSSED_CONDITIONS[0], {"one": 1.0}, ordinary_regular=True)
    source = tmp_path / "results.json"
    source.touch()
    monkeypatch.setattr(Results, "load", lambda _: regular)
    monkeypatch.setattr(batch, "_load_run_tasks", lambda _: regular.tasks)
    monkeypatch.setattr(helpers, "get_info", lambda *_args, **_kwargs: regular.info)
    checked = []
    monkeypatch.setattr(
        pass3, "validate_regular_source", lambda *args: checked.append(args)
    )

    seen = []

    def run_command(args):
        seen.append(args)
        return VoiceRunConfig(domain="intake", audio_native_config=AudioNativeConfig())

    parser = argparse.ArgumentParser()
    add_pass3_args(parser, add_run_args=add_run_args, run_command=run_command)
    args = parser.parse_args(
        [
            "--pass3-mode",
            "crossed",
            "--audio-native",
            "--domain",
            "intake",
            "--save-to",
            "fixture",
            "--regular-results",
            str(source),
            "--dry-run",
        ]
    )
    args.func(args)
    assert len(checked) == 1
    assert [(call.seed, call.pass3_condition) for call in seen] == [
        (42, None),
        (401, "channel_heavy"),
        (16319, "speech_heavy"),
    ]
