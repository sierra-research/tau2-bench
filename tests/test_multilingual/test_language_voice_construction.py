"""Language identity is selected before construction; canonical acoustics stay fixed."""

import hashlib
from types import SimpleNamespace

import pytest

from tau2.data_model.language import (
    LanguageDirective,
    LanguageTaskInfo,
    PersonaVoice,
    RuntimePersona,
    VoiceExamples,
)
from tau2.data_model.persona import PersonaConfig
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.tasks import Task
from tau2.data_model.voice import SampledVoiceConfig, SynthesisConfig, VoiceSettings


@pytest.fixture
def context():
    # Prompt-content rendering has separate contract tests; this fixture exercises
    # construction with an already resolved typed persona, without generating prose.
    persona = RuntimePersona(
        persona_id="reviewed-persona",
        mode="voice",
        gender="female",
        locale="Fixture",
        variety="Fixture",
        language="xx",
        display_language="Fixture language",
        gender_guidance="Fixture gender guidance.",
        language_directive=LanguageDirective(origin="Fixture region"),
        interrupt_tendency="waits",
        verbosity="standard",
        spellout_section="Fixture spelling section.",
        voice=PersonaVoice(
            voice_id="reviewed-voice",
            interrupt_tendency="waits",
            backchannel_phrases=["Reviewed acknowledgment"],
            vocal_tics=["Reviewed filler"],
            non_directed_phrases=["Reviewed aside"],
            voice_examples=VoiceExamples.model_validate(
                {name: "Fixture" for name in VoiceExamples.model_fields}
            ),
        ),
    )
    return LanguageTaskInfo(
        language="xx",
        mode="voice",
        cross_lingual=True,
        pack_revision="fixture",
        pack_sha256="fixture",
        domain_revision="fixture",
        tasks_sha256="fixture",
        metadata_sha256="fixture",
        task_sha256="fixture",
        canonical_task_id="canonical-a",
        persona=persona,
        caller_gender="female",
        speech_insert_events_per_minute=1.25,
        backchannel_level="medium",
        backchannel_prompt_version="fixture",
        agent_greeting="Reviewed greeting",
        agent_language_clause="Fixture language",
        guideline_template_version="fixture",
    )


def test_language_user_receives_selected_identity_and_behavior_at_construction(
    context, monkeypatch, tmp_path, voice_extra
):
    from tau2 import user_simulation_voice_presets as presets
    from tau2.runner import build
    from tau2.user import user_simulator_streaming

    monkeypatch.setattr(
        presets, "get_task_voice_configs_path", lambda _: tmp_path / "absent"
    )

    def reject_legacy_voice(*args, **kwargs):
        pytest.fail("Language construction resolved a legacy persona/voice")

    monkeypatch.setattr(
        SampledVoiceConfig, "to_speech_environment", reject_legacy_voice
    )
    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        user_simulator_streaming, "VoiceStreamingUserSimulator", constructor
    )
    base = VoiceSettings(transcription_config=None, synthesis_config=SynthesisConfig())
    before = base.model_dump()
    build.build_voice_user(
        SimpleNamespace(
            get_domain_name=lambda: "retail", get_user_tools=lambda **_: []
        ),
        Task(id="localized-a", user_scenario={"instructions": "Fixture caller"}),
        AudioNativeConfig(),
        language_context=context,
        voice_settings=base,
        persona_config=PersonaConfig(verbosity="minimal"),
        seed=0,
    )
    assert captured["persona_config"] is context.persona
    assert captured["backchannel"].phrases == context.persona.voice.backchannel_phrases
    assert captured["backchannel"].language_name == "Fixture language"
    settings = captured["voice_settings"]
    assert settings.speech_environment.persona_name == context.persona_id
    assert settings.speech_environment.voice_id == context.persona.voice.voice_id
    assert settings.speech_environment.voice_seed == (
        int.from_bytes(
            hashlib.sha256(context.canonical_task_id.encode()).digest()[:8], "big"
        )
        % 1000000
    )
    assert (
        settings.synthesis_config.provider_config.voice_id
        == context.persona.voice.voice_id
    )
    assert settings.synthesis_config.provider_config.language_code == "xx"
    effects = settings.synthesis_config.speech_effects_config
    assert [i.text for i in effects.vocal_tics] == ["Reviewed filler"]
    assert [i.text for i in effects.non_directed_phrases] == ["Reviewed aside"]
    assert effects.speech_insert_events_per_minute == 1.25
    assert not settings.speech_environment.enable_interruptions
    assert base.model_dump() == before


@pytest.mark.parametrize(
    ("speech_mode", "expected_tendency", "expected_interruptions"),
    [("light", "waits", False), ("heavy", "interrupts", True)],
)
def test_crossed_mode_reaches_localized_caller_behavior(
    context,
    monkeypatch,
    speech_mode,
    expected_tendency,
    expected_interruptions,
    voice_extra,
):
    from tau2.runner import build
    from tau2.user import user_simulator_streaming

    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        user_simulator_streaming, "VoiceStreamingUserSimulator", constructor
    )
    build.build_voice_user(
        SimpleNamespace(
            get_domain_name=lambda: "retail", get_user_tools=lambda **_: []
        ),
        Task(id="localized-a", user_scenario={"instructions": "Fixture caller"}),
        AudioNativeConfig(),
        language_context=context,
        persona_config=context.persona,
        speech_effects_mode=speech_mode,
        force_sample_voice_config=True,
    )
    assert captured["persona_config"].interrupt_tendency == expected_tendency
    assert (
        captured["voice_settings"].speech_environment.enable_interruptions
        == expected_interruptions
    )


@pytest.mark.parametrize(
    "complexity", ["control", "regular", "control_audio", "control_behavior"]
)
@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize("base_voice_pin", [False, True])
def test_selected_persona_preserves_canonical_acoustics_and_insert_scheduling(
    context, monkeypatch, tmp_path, complexity, preloaded, base_voice_pin
):
    from tau2 import user_simulation_voice_presets as presets
    from tau2.data_model.audio_effects import UserSpeechInsert

    source = SynthesisConfig()
    if base_voice_pin:
        from tau2.data_model.voice_personas import get_elevenlabs_voice_id

        source.provider_config.voice_id = get_elevenlabs_voice_id(
            presets.REGULAR_PERSONA_NAMES[0]
        )
    source.source_effects_config.noise_snr_db = 23
    source.speech_effects_config.min_words_for_vocal_tics = 11
    seed = 391
    # Make acoustic file selection observable even in a data-free test install.
    beds, bursts = tmp_path / "beds", tmp_path / "bursts"
    beds.mkdir()
    bursts.mkdir()
    for preset in presets.ENVIRONMENT_PRESETS.values():
        for name in preset["background_noise_files"]:
            (beds / name).touch()
        for name in preset["burst_noise_files"]:
            (bursts / name).touch()
    monkeypatch.setattr(presets, "BACKGROUND_NOISE_CONTINUOUS_DIR", beds)
    monkeypatch.setattr(presets, "BURST_NOISE_DIR", bursts)
    expected = presets.sample_voice_config(seed, source, complexity)
    path = tmp_path / "acoustics.json"
    if preloaded:
        path.touch()
        monkeypatch.setattr(
            presets,
            "load_task_voice_configs",
            lambda _: SimpleNamespace(get_config=lambda *args: expected),
        )
    monkeypatch.setattr(presets, "get_task_voice_configs_path", lambda _: path)
    selection = presets.VoiceSelection(
        persona_name=context.persona_id,
        persona_config=context.persona,
        vocal_tics=[UserSpeechInsert(text="Reviewed filler", type="vocal_tic")],
        non_directed_phrases=[
            UserSpeechInsert(text="Reviewed aside", type="non_directed_phrase")
        ],
        speech_insert_events_per_minute=1.25,
    )
    real_choice = presets.random.Random.choice

    def no_legacy_persona_choice(rng, options):
        assert options not in (
            presets.CONTROL_PERSONA_NAMES,
            presets.REGULAR_PERSONA_NAMES,
        )
        return real_choice(rng, options)

    monkeypatch.setattr(presets.random.Random, "choice", no_legacy_persona_choice)
    actual = presets.get_or_load_task_voice_config(
        "retail", "canonical-a", seed, complexity, source, selection=selection
    )
    excluded = {
        "persona_name",
        "persona_config",
        "enable_interruptions",
        "speech_effects_config",
    }
    assert actual.model_dump(exclude=excluded) == expected.model_dump(exclude=excluded)
    inserts = {"vocal_tics", "non_directed_phrases", "speech_insert_events_per_minute"}
    assert actual.speech_effects_config.model_dump(exclude=inserts) == (
        expected.speech_effects_config.model_dump(exclude=inserts)
    )
    assert actual.persona_name == context.persona_id
    assert actual.persona_config is context.persona
    assert not actual.enable_interruptions
    assert actual.speech_effects_config.speech_insert_events_per_minute == 1.25
    assert source.source_effects_config.noise_snr_db == 23


def test_language_builder_hands_the_typed_call_context_to_the_orchestrator(
    context, monkeypatch
):
    from tau2.data_model.simulation import VoiceRunConfig
    from tau2.multilingual import runtime
    from tau2.runner import build

    monkeypatch.setattr(runtime, "task_context", lambda *_, **__: context)
    monkeypatch.setattr(runtime, "agent_language_guidance", lambda *_, **__: "")
    monkeypatch.setattr(build, "build_environment", lambda *a, **kw: object())
    monkeypatch.setattr(build, "_build_env_kwargs", lambda *a, **kw: {})
    monkeypatch.setattr(build, "build_agent", lambda *a, **kw: object())
    monkeypatch.setattr(build, "build_voice_user", lambda *a, **kw: object())
    monkeypatch.setattr(
        build, "FullDuplexOrchestrator", lambda **kw: SimpleNamespace(**kw)
    )
    config = VoiceRunConfig(
        domain="retail", seed=0, audio_native_config=AudioNativeConfig()
    )
    task = Task(id="fixture", user_scenario={"instructions": "Fixture caller"})
    result = build.build_voice_orchestrator(config, task, seed=0)
    assert result.call_context.language is context
    assert result.call_context.communicate_scoring == config.communicate_scoring
    assert not hasattr(result, "initial_agent_message")
