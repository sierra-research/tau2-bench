"""Offline checks for explicit, provider-independent language-pack loading."""

import json

import pytest
import yaml

from tau2.data_model.language import LanguagePack
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import TextRunConfig, VoiceRunConfig
from tau2.multilingual.runtime import load_language, resolve_persona, resolve_task
from tau2.user.guidelines import language_directive, render_simulation_guidelines


@pytest.fixture
def language_tree(tmp_path, monkeypatch, english_prompt_content):
    import tau2.utils

    monkeypatch.setattr(tau2.utils, "DATA_DIR", tmp_path)
    root = tmp_path / "tau2" / "multilingual" / "xx"
    root.mkdir(parents=True)
    task = {
        "id": "localized-a",
        "user_scenario": {"instructions": "Your name is Alice."},
    }
    (root / "tasks.json").write_text(json.dumps([task]))
    (root / "metadata.json").write_text(
        json.dumps(
            {"localized-a": {"canonical_task_id": "a", "caller_gender": "female"}}
        )
    )
    content = english_prompt_content["prompt_content"]
    pack = {
        "schema_version": 3,
        "language": "xx",
        "revision": "test-v1",
        "display_name": "Example",
        "agent_greeting": "Reviewed Example greeting.",
        "agent_language_clause": "Speak Example.",
        "personas": {
            "speaker": {
                "gender": "female",
                "gender_guidance": "You are a female customer (she/her).",
                "locale": "Example region",
                "variety": "Example variety",
                "language_directive": {"origin": "Example region"},
                "pragmatics_clauses": ["Use short natural sentences."],
                "voice": {
                    "voice_id": "voice-fixture",
                    "voice_examples": content["voice_examples"],
                    "backchannel_phrases": ["Reviewed acknowledgment"],
                    "vocal_tics": ["Reviewed filler"],
                    "non_directed_phrases": ["Reviewed aside"],
                    "interrupt_tendency": "interrupts",
                },
            }
        },
        "voice": {
            "spellout_section": english_prompt_content["spellout_section"],
            "speech_insert_events_per_minute": 1.0,
            "backchannel_level": "medium",
        },
        "domains": {
            "retail": {
                "revision": "tasks-v1",
                "tasks_file": "tasks.json",
                "task_metadata_file": "metadata.json",
            }
        },
    }
    (root / "pack.yaml").write_text(yaml.safe_dump(pack))
    return root, pack


def test_explicit_binding_and_content_provenance(language_tree):
    root, _ = language_tree
    loaded = load_language("xx", "retail", mode="voice")
    resolved = resolve_task(loaded, loaded.tasks[0], seed=4)
    assert resolved.canonical_task_id == "a"
    assert resolved.persona_id == "speaker"
    assert resolved.persona.voice.voice_id == "voice-fixture"
    assert resolved.pack_sha256 == loaded.pack_sha256
    assert resolved.task_sha256
    assert resolved.cross_lingual and resolved.mode == "voice"
    directive = language_directive(resolved.persona)
    assert "native Example speaker from Example region" in directive
    assert directive in render_simulation_guidelines(resolved.persona, use_tools=False)
    assert "female customer" in resolved.persona.to_guidelines_text("voice")
    before = loaded.tasks_sha256
    (root / "tasks.json").write_text(
        (root / "tasks.json").read_text().replace("Alice", "Bob")
    )
    assert load_language("xx", "retail", mode="voice").tasks_sha256 != before


def test_mismatch_missing_gender_and_escape_rejected(language_tree):
    root, pack = language_tree
    (root / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="metadata"):
        load_language("xx", "retail", mode="voice")
    pack["domains"]["retail"]["tasks_file"] = "../tasks.json"
    with pytest.raises(ValueError, match="relative"):
        LanguagePack.model_validate(pack)


@pytest.mark.parametrize("language", [None, "en", "xx"])
def test_language_selection_does_not_change_run_conditions(language):
    from tau2.config import DEFAULT_LLM_ARGS_USER, DEFAULT_LLM_USER

    selected = {} if language is None else {"language": language}
    text = TextRunConfig(domain="retail", **selected)
    assert text.language == (language or "en")
    assert text.llm_user == DEFAULT_LLM_USER
    assert text.llm_args_user == DEFAULT_LLM_ARGS_USER
    voice = VoiceRunConfig(
        domain="retail", **selected, audio_native_config=AudioNativeConfig()
    )
    assert voice.llm_user == text.llm_user
    assert voice.llm_args_user == text.llm_args_user
    assert (
        voice.audio_native_config.max_steps_seconds
        == AudioNativeConfig().max_steps_seconds
    )
    assert voice.audio_native_config.voice is None
    assert voice.audio_native_config.transcription_language == (language or "en")
    assert voice.audio_native_config.language == (language or "en")
    pinned = VoiceRunConfig(
        domain="retail",
        **selected,
        llm_user="explicit",
        llm_args_user={"reasoning_effort": "high"},
        audio_native_config=AudioNativeConfig(voice="cedar", max_steps_seconds=77),
    )
    assert pinned.llm_user == "explicit"
    assert pinned.llm_args_user["reasoning_effort"] == "high"
    assert pinned.audio_native_config.voice == "cedar"
    assert pinned.audio_native_config.max_steps_seconds == 77


def test_language_resolution_does_not_mutate_shared_audio_settings():
    audio = AudioNativeConfig()
    english = VoiceRunConfig(domain="retail", language="en", audio_native_config=audio)
    korean = VoiceRunConfig(domain="retail", language="ko", audio_native_config=audio)
    assert english.audio_native_config.transcription_language == "en"
    assert korean.audio_native_config.transcription_language == "ko"
    assert audio.transcription_language is None


def test_english_uses_requested_spelling_and_has_no_language_contradiction(
    language_tree,
):
    _, raw = language_tree
    raw["language"] = "en"
    raw["display_name"] = "English"
    del raw["personas"]["speaker"]["language_directive"]
    pack = LanguagePack.model_validate(raw)
    for mode in ("voice", "text"):
        persona = resolve_persona(
            pack, "speaker", mode=mode, binding=pack.binding("retail")
        )
        rendered = render_simulation_guidelines(persona, use_tools=False)
        assert "LANGUAGE OF THE" not in rendered
        assert "never in English" not in persona.to_guidelines_text(mode)
        assert "Use short natural sentences." in persona.to_guidelines_text(mode)
    voice = resolve_persona(pack, "speaker", mode="voice")
    rendered = render_simulation_guidelines(voice, use_tools=False)
    assert pack.voice.spellout_section.rstrip() in rendered
    assert "## Scenario Adherence" in rendered
    assert "Keep comma-separated digits." not in rendered


def test_text_and_voice_build_resolve_same_persona_without_provider_calls(
    language_tree, voice_extra, monkeypatch
):
    from tau2.runner import build
    from tau2.runner.build import build_text_orchestrator, build_voice_orchestrator
    from tau2.runner.helpers import get_info
    from tau2.voice.voice_gender import selected_agent_voice

    sampled_ids = []
    sample = build.get_or_load_task_voice_config

    def tracked_sample(**kwargs):
        sampled_ids.append(kwargs["task_id"])
        return sample(**kwargs)

    monkeypatch.setattr(build, "get_or_load_task_voice_config", tracked_sample)

    tasks = load_language("xx", "retail", mode="voice").tasks
    text = TextRunConfig(domain="retail", language="xx", seed=9)
    voice = VoiceRunConfig(
        domain="retail",
        language="xx",
        seed=9,
        audio_native_config=AudioNativeConfig(voice="marin"),
    )
    a = build_text_orchestrator(text, tasks[0])
    b = build_voice_orchestrator(voice, tasks[0])
    try:
        assert (
            a.call_context.language.persona_id
            == b.call_context.language.persona_id
            == "speaker"
        )
        assert a.call_context.language.mode == "text"
        assert b.call_context.language.mode == "voice"
        assert "Speak Example." in a.agent.system_prompt
        assert "Speak Example." in b.agent.system_prompt
        assert "Your gender is female." in b.agent.system_prompt
        assert "perceived as" not in b.agent.system_prompt
        assert b.agent.voice == "marin"
        assert b.agent.transcription_language == "xx"
        assert sampled_ids == ["a"]
        assert b.agent._adapter is None
        assert b.user.enable_user_initiated_interruption
        env = b.user.voice_settings.speech_environment
        assert env.voice_id == "voice-fixture"
        assert env.persona_name == "speaker"
        assert env.channel_effects_config.frame_drop_rate == 0.02
        assert env.speech_effects_config.speech_insert_events_per_minute == 1.0
        assert env.speech_effects_config.vocal_tics[0].text == "Reviewed filler"
        assert (
            env.speech_effects_config.non_directed_phrases[0].text == "Reviewed aside"
        )
        assert b.user.backchannel.phrases == ["Reviewed acknowledgment"]
        assert b.user.backchannel.language_name == "Example"
        info = get_info(voice)
        assert info.voice_instruction_prompt_version == "voice-instructions-v2"
        assert info.language_tasks[tasks[0].id].canonical_task_id == "a"
        assert (
            type(info)
            .model_validate_json(info.model_dump_json())
            .language_tasks[tasks[0].id]
            .persona.language
            == "xx"
        )
        assert selected_agent_voice(voice) == ("marin", "female")
        override = voice.model_copy(deep=True)
        override.audio_native_config.voice = "unknown"
        assert selected_agent_voice(override) == ("unknown", None)
    finally:
        if hasattr(b.user, "_executor"):
            b.user._executor.shutdown(wait=False)


def test_live_backend_configuration_is_preserved(language_tree, voice_extra):
    from tau2.runner.build import build_voice_orchestrator
    from tau2.voice.voice_gender import selected_agent_voice

    config = VoiceRunConfig(
        domain="retail",
        language="xx",
        audio_native_config=AudioNativeConfig(
            provider="openai_live",
            model="gpt-live-1",
            live_config={"backend_model": "gpt-6-astra", "voice": "marin"},
            reasoning_effort="medium",
        ),
    )
    built = build_voice_orchestrator(
        config, load_language("xx", "retail", mode="voice").tasks[0]
    )
    try:
        assert built.agent.live_config.backend_model == "gpt-6-astra"
        assert built.agent.live_config.voice == "marin"
        assert built.agent.reasoning_effort == "medium"
        assert built.agent.voice is None
        assert selected_agent_voice(config) == ("marin", "female")
    finally:
        if hasattr(built.user, "_executor"):
            built.user._executor.shutdown(wait=False)


def test_telecom_initialization_preserves_distinct_user_database():
    from tau2.data_model.tasks import InitializationData
    from tau2.domains.telecom.environment import get_environment

    env = get_environment()
    user_type = type(env.user_tools.db)
    customer = env.tools.db.customers[0].model_copy(
        update={"full_name": "Mamadou Diallo"}
    )
    customers = [
        customer.model_dump(mode="json"),
        *[c.model_dump(mode="json") for c in env.tools.db.customers[1:]],
    ]
    env.set_state(
        initialization_data=InitializationData(agent_data={"customers": customers}),
        initialization_actions=None,
        message_history=[],
    )
    assert isinstance(env.user_tools.db, user_type)
    assert env.user_tools.db is not env.tools.db
    assert env.tools.db.customers[0].full_name == "Mamadou Diallo"


def test_registered_path_rejects_task_override_before_construction(
    language_tree, monkeypatch
):
    from tau2.runner import build

    task = (
        load_language("xx", "retail", mode="voice")
        .tasks[0]
        .model_copy(update={"id": "unpublished"})
    )
    monkeypatch.setattr(
        build,
        "build_environment",
        lambda *a, **k: pytest.fail(
            "constructed environment before validating language task"
        ),
    )
    with pytest.raises(ValueError, match="published"):
        build.build_text_orchestrator(
            TextRunConfig(domain="retail", language="xx"), task
        )


def test_pinned_artifacts_reject_mutation_and_resume(language_tree):
    from tau2.multilingual.runtime import (
        pin_language_artifacts,
        task_context,
        validate_language_resume,
    )
    from tau2.runner.helpers import get_info

    root, _ = language_tree
    config = TextRunConfig(domain="retail", language="xx")
    pin_language_artifacts(config)
    task = load_language("xx", "retail", mode="text").tasks[0]
    context = task_context(config, task)
    assert context.pack_sha256 == config.language_artifacts.pack_sha256
    original = get_info(config)
    validate_language_resume(original, original.model_copy(deep=True))
    assert original.language_tasks[task.id].prompt_version == "language-runtime-v8"
    from tau2.user.guidelines import GUIDELINE_TEMPLATE_VERSION

    assert (
        original.language_tasks[task.id].guideline_template_version
        == GUIDELINE_TEMPLATE_VERSION
    )
    old_scaffold = original.model_copy(deep=True)
    old_scaffold.language_tasks[task.id].guideline_template_version = "older"
    with pytest.raises(ValueError, match="resume changed"):
        validate_language_resume(old_scaffold, original)
    old_framing = original.model_copy(deep=True)
    old_framing.language_tasks[task.id].prompt_version = "language-runtime-v2"
    with pytest.raises(ValueError, match="resume changed"):
        validate_language_resume(old_framing, original)
    changed = original.model_copy(deep=True)
    changed.language_tasks[task.id].pack_revision = "changed"
    with pytest.raises(ValueError, match="resume changed"):
        validate_language_resume(original, changed)
    pre_schema = original.model_copy(deep=True, update={"language": None})
    with pytest.raises(ValueError, match="predate the language schema"):
        validate_language_resume(pre_schema, original)
    other_language = original.model_copy(deep=True, update={"language": "en"})
    with pytest.raises(ValueError, match="different language"):
        validate_language_resume(other_language, original)
    (root / "pack.yaml").write_text((root / "pack.yaml").read_text() + "\n")
    with pytest.raises(ValueError, match="changed during"):
        task_context(config, task)


def test_localized_spelling_applies_only_to_voice(language_tree, voice_extra):
    from tau2.runner.build import build_text_orchestrator, build_voice_orchestrator

    root, pack = language_tree
    pack["voice"]["spellout_section"] = (
        "## Speaking Special Characters and Numbers\nReviewed localized digits."
    )
    (root / "pack.yaml").write_text(yaml.safe_dump(pack))
    task = load_language("xx", "retail", mode="voice").tasks[0]
    text = build_text_orchestrator(TextRunConfig(domain="retail", language="xx"), task)
    voice = build_voice_orchestrator(
        VoiceRunConfig(
            domain="retail", language="xx", audio_native_config=AudioNativeConfig()
        ),
        task,
    )
    try:
        assert "Reviewed localized digits." not in text.user.system_prompt
        assert "Reviewed localized digits." in voice.user.system_prompt
    finally:
        if hasattr(voice.user, "_executor"):
            voice.user._executor.shutdown(wait=False)


@pytest.mark.parametrize("language", [None, "xx"])
@pytest.mark.parametrize("audio", [False, True])
def test_cli_preserves_omitted_and_explicit_defaults(monkeypatch, language, audio):
    import sys

    from tau2 import cli
    from tau2.config import DEFAULT_LLM_USER, DEFAULT_MAX_STEPS_SECONDS

    captured = []
    monkeypatch.setattr(cli, "run_domain", lambda config: captured.append(config))
    argv = ["tau2", "run", "--domain", "retail"]
    if language:
        argv += ["--language", language]
    if audio:
        argv += ["--audio-native", "--benchmark", "none"]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    config = captured.pop()
    assert config.llm_user == DEFAULT_LLM_USER
    if audio:
        assert config.audio_native_config.max_steps_seconds == (
            DEFAULT_MAX_STEPS_SECONDS
        )
    monkeypatch.setattr(
        sys,
        "argv",
        argv
        + [
            "--user-llm",
            "explicit",
            "--user-llm-args",
            '{"temperature": 0.3}',
            "--max-steps-seconds",
            "77",
        ],
    )
    cli.main()
    explicit = captured.pop()
    assert explicit.llm_user == "explicit"
    assert explicit.llm_args_user == {"temperature": 0.3}
    if audio:
        assert explicit.audio_native_config.max_steps_seconds == 77


@pytest.mark.parametrize("storage", ["json", "dir"])
def test_language_resume_extension_records_every_task_context(
    language_tree, tmp_path, storage
):
    from tau2.data_model.simulation import Results
    from tau2.runner.checkpoint import try_resume
    from tau2.runner.helpers import get_info

    root, _ = language_tree
    rows = json.loads((root / "tasks.json").read_text())
    rows.append(
        {"id": "localized-b", "user_scenario": {"instructions": "Your name is Beth."}}
    )
    (root / "tasks.json").write_text(json.dumps(rows))
    metadata = json.loads((root / "metadata.json").read_text())
    metadata["localized-b"] = {"canonical_task_id": "b", "caller_gender": "female"}
    (root / "metadata.json").write_text(json.dumps(metadata))
    tasks = load_language("xx", "retail", mode="voice").tasks
    first = TextRunConfig(domain="retail", language="xx", task_ids=[tasks[0].id])
    expanded = TextRunConfig(domain="retail", language="xx")
    path = tmp_path / "checkpoint" / "results.json"
    path.parent.mkdir()
    Results(info=get_info(first), tasks=tasks[:1]).save(path, format=storage)
    current = Results(info=get_info(expanded), tasks=tasks)
    resumed, _, _ = try_resume(path, current, tasks, 1, auto_resume=True)
    assert resumed.info.language_tasks == current.info.language_tasks
    assert (
        Results.load_metadata(path).info.language_tasks == current.info.language_tasks
    )
    # The worker reads this map even when a provider fails before producing a result.
    assert resumed.info.language_tasks[tasks[1].id].canonical_task_id == "b"
    # A subsequent resume must still recognize the expanded selection.
    again, _, _ = try_resume(path, current, tasks, 1, auto_resume=True)
    assert again.info.language_tasks == current.info.language_tasks


@pytest.mark.parametrize(
    "condition",
    [
        "voice",
        "transcription_language",
        "reasoning_effort",
        "max_steps_seconds",
        "user_model",
        "agent_model",
        "policy",
        "seed",
        "speech_complexity",
    ],
)
def test_language_resume_rejects_changed_call_conditions_before_writing(
    language_tree, tmp_path, condition
):
    from tau2.data_model.simulation import Results
    from tau2.runner.checkpoint import try_resume
    from tau2.runner.helpers import get_info

    config = VoiceRunConfig(
        domain="retail",
        language="xx",
        audio_native_config=AudioNativeConfig(reasoning_effort="minimal"),
    )
    tasks = load_language("xx", "retail", mode="voice").tasks
    previous = Results(info=get_info(config), tasks=tasks)
    current = previous.model_copy(deep=True)
    audio = current.info.audio_native_config
    if condition == "user_model":
        current.info.user_info.llm = "changed-customer"
    elif condition == "agent_model":
        audio.model = "changed-agent"
    elif condition == "policy":
        current.info.environment_info.policy += "\nChanged policy."
    elif condition == "seed":
        current.info.seed += 1
    elif condition == "speech_complexity":
        current.info.speech_complexity = "complex"
    else:
        setattr(
            audio,
            condition,
            {
                "voice": "cedar",
                "transcription_language": "es",
                "reasoning_effort": "low",
                "max_steps_seconds": 1800,
            }[condition],
        )
    path = tmp_path / "results.json"
    previous.save(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="language run conditions changed"):
        try_resume(path, current, tasks, 1, auto_resume=True)
    assert path.read_bytes() == before


def test_language_resume_allows_new_trials_and_git_revision(language_tree, tmp_path):
    from tau2.data_model.simulation import Results
    from tau2.runner.checkpoint import try_resume
    from tau2.runner.helpers import get_info

    config = TextRunConfig(domain="retail", language="xx")
    tasks = load_language("xx", "retail", mode="voice").tasks
    previous = Results(info=get_info(config), tasks=tasks)
    current = previous.model_copy(deep=True)
    current.info.num_trials = 2
    current.info.git_commit = "changed-code-provenance"
    path = tmp_path / "results.json"
    previous.save(path)
    resumed, _, _ = try_resume(path, current, tasks, 2, auto_resume=True)
    assert resumed.info.num_trials == Results.load_metadata(path).info.num_trials == 2
    assert resumed.info.git_commit == previous.info.git_commit
    with pytest.raises(ValueError, match="fewer trials"):
        try_resume(path, previous, tasks, 1, auto_resume=True)


def test_english_resume_is_as_strict_as_every_language(tmp_path):
    from tau2.data_model.simulation import Results
    from tau2.runner.checkpoint import try_resume
    from tau2.runner.helpers import get_info

    config = TextRunConfig(domain="retail", num_tasks=1)
    info = get_info(config)
    task = load_language("en", "retail", mode="text").tasks[0]
    assert task.id in info.language_tasks
    previous = Results(info=info, tasks=[task])
    current = previous.model_copy(deep=True)
    current.info.agent_info.llm = "changed-model"
    path = tmp_path / "results.json"
    previous.save(path)
    with pytest.raises(ValueError, match="language run conditions changed"):
        try_resume(path, current, [task], 1, auto_resume=True)


def test_prepared_context_checks_binding_without_reassigning_persona(
    language_tree, monkeypatch
):
    from tau2.multilingual import runtime

    root, _ = language_tree
    loaded = load_language("xx", "retail", mode="text")
    task = loaded.tasks[0]
    context = resolve_task(loaded, task, seed=0)
    config = TextRunConfig(domain="retail", language="xx", seed=0)

    def forbidden(*args, **kwargs):
        raise AssertionError("Prepared calls must retain the assigned persona")

    monkeypatch.setattr(runtime, "resolve_task", forbidden)
    assert runtime.task_context(config, task, context=context) == context
    altered = task.model_copy(deep=True)
    altered.user_scenario.instructions = "Different task"
    with pytest.raises(ValueError, match="task"):
        runtime.task_context(config, altered, context=context)
    (root / "metadata.json").write_text((root / "metadata.json").read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        runtime.task_context(config, task, context=context)


def test_info_resolves_only_actual_selected_tasks_once(language_tree, monkeypatch):
    from tau2.multilingual import runtime
    from tau2.runner.helpers import get_info

    config = TextRunConfig(domain="retail", language="xx", seed=0)
    loaded = runtime.load_run_language(config)
    calls = []
    original = runtime.load_language

    def load(*args, **kwargs):
        calls.append(args)
        assert kwargs == {
            "mode": "text",
            "localize_db": True,
            "selection": runtime.run_task_selection(config),
            "pack_extension": None,
        }
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "load_language", load)
    info = get_info(config, tasks=loaded.tasks)
    assert calls == [("xx", "retail")]
    assert info.language_tasks == {
        task.id: resolve_task(loaded, task, 0) for task in loaded.tasks
    }
    calls.clear()
    get_info(config)
    assert calls == [("xx", "retail")]


@pytest.mark.parametrize("artifact", ["pack.yaml", "tasks.json", "metadata.json"])
def test_prepared_context_rejects_each_artifact_drift_before_construction(
    language_tree, monkeypatch, artifact
):
    from tau2.multilingual.runtime import task_context
    from tau2.runner import build

    root, _ = language_tree
    config = TextRunConfig(domain="retail", language="xx", seed=0)
    task = load_language("xx", "retail", mode="voice").tasks[0]
    context = task_context(config, task)
    path = root / artifact
    path.write_text(path.read_text() + "\n")
    monkeypatch.setattr(
        build,
        "build_environment",
        lambda *a, **k: pytest.fail("environment constructed"),
    )
    with pytest.raises(ValueError, match="changed"):
        build.build_text_orchestrator(config, task, language_context=context)


def test_batch_passes_recorded_assignment_across_trial_seeds(
    language_tree, monkeypatch
):
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.runner import batch

    task = load_language("xx", "retail", mode="voice").tasks[0]
    config = TextRunConfig(
        domain="retail", language="xx", seed=0, num_trials=2, max_concurrency=1
    )
    calls = []

    def run(config, task, *, seed, language_context, **kwargs):
        calls.append((seed, language_context))
        return SimulationRun(
            id=f"fixture-{seed}",
            task_id=task.id,
            seed=seed,
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:00:01",
            duration=1,
            termination_reason=TerminationReason.USER_STOP,
            messages=[],
        )

    monkeypatch.setattr(batch, "run_single_task", run)
    results = batch.run_tasks(config, [task], console_display=False)
    assert len(calls) == 2
    assert calls[0][0] != calls[1][0]
    assert all(context == results.info.language_tasks[task.id] for _, context in calls)
    assert all(
        simulation.language_context == results.info.language_tasks[task.id]
        for simulation in results.simulations
    )


@pytest.mark.parametrize("config_type", [TextRunConfig, VoiceRunConfig])
def test_communicate_scoring_defaults_semantic_and_substring_is_explicit(
    config_type,
):
    from tau2.data_model.semantic_communication import (
        CommunicateScorer,
        CommunicateScoring,
    )

    kwargs = (
        {"audio_native_config": AudioNativeConfig()}
        if config_type is VoiceRunConfig
        else {}
    )
    for language in ("en", "es"):
        config = config_type(language=language, **kwargs)
        assert config.communicate_scoring == CommunicateScoring()
        assert config.communicate_scoring.scorer == CommunicateScorer.SEMANTIC
    explicit = config_type(
        communicate_scoring=CommunicateScoring(scorer="substring"), **kwargs
    )
    assert (
        config_type.model_validate_json(explicit.model_dump_json()).communicate_scoring
        == explicit.communicate_scoring
    )
    assert explicit.communicate_scoring.scorer == CommunicateScorer.SUBSTRING


def test_custom_french_pack_uses_ordinary_voice_config(language_tree):
    root, pack = language_tree
    pack["language"] = "fr"
    (root / "pack.yaml").write_text(yaml.safe_dump(pack))
    root.rename(root.with_name("fr"))
    loaded = load_language("fr", "retail", mode="voice")
    config = VoiceRunConfig(
        domain="retail",
        language="fr",
        audio_native_config=AudioNativeConfig(provider="livekit"),
    )
    assert loaded.pack.language == "fr"
    assert config.audio_native_config.cascaded_config.stt.language == "fr"
    assert config.audio_native_config.cascaded_config.tts.language == "fr"
    assert config.communicate_scoring.scorer == "semantic"
    assert config.benchmark is None


@pytest.mark.parametrize("voice", [False, True])
def test_build_then_run_keeps_language_evaluator(
    language_tree, monkeypatch, voice, request
):
    if voice:
        request.getfixturevalue("voice_extra")

    from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
    from tau2.runner import build, simulation

    config = (
        VoiceRunConfig(
            domain="retail", language="xx", audio_native_config=AudioNativeConfig()
        )
        if voice
        else TextRunConfig(domain="retail", language="xx")
    )
    task = load_language("xx", "retail", mode="voice").tasks[0]
    orchestrator = build.build_orchestrator(config, task)
    result = SimulationRun(
        id="fixture",
        task_id=task.id,
        start_time="start",
        end_time="end",
        duration=1,
        termination_reason=TerminationReason.USER_STOP,
        messages=[],
    )
    monkeypatch.setattr(orchestrator, "run", lambda: result)
    seen = []

    def evaluate(**kwargs):
        seen.append(kwargs)
        return RewardInfo(reward=1)

    monkeypatch.setattr(simulation, "evaluate_simulation", evaluate)
    simulation.run_simulation(orchestrator)
    assert seen[0]["customer_language"] == "xx"
    assert seen[0]["communicate_scoring"] == config.communicate_scoring


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"source": "registry"}, None),
        ({"source": "registry", "task_metadata_file": "m.json"}, None),
        (
            {
                "source": "registry",
                "tasks_file": "t.json",
                "task_metadata_file": "m.json",
            },
            None,
        ),
        (
            {"source": "registry", "tasks_file": "t.json"},
            "replacement tasks_file requires task_metadata_file",
        ),
        (
            {
                "source": "registry",
                "regional_bundle_file": "r.json",
                "task_metadata_file": "m.json",
            },
            "cannot carry a regional scenario",
        ),
        ({"source": "file", "task_metadata_file": "m.json"}, "exactly one of"),
        ({"source": "file", "tasks_file": "t.json"}, "require published task metadata"),
    ],
)
def test_domain_binding_task_sources(fields, error):
    from pydantic import ValidationError

    from tau2.data_model.language import DomainBinding

    if error is None:
        DomainBinding(revision="r1", **fields)
    else:
        with pytest.raises(ValidationError, match=error):
            DomainBinding(revision="r1", **fields)
