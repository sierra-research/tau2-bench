# Copyright Sierra
"""Regression tests for run-level persona wiring in the batch runner.

Caught by the staged smoke run: ``run_tasks`` pre-computed a plain
``PersonaConfig`` from the speech-complexity preset for every voice run and
passed it down, which shadowed the sampled ``MultilingualPersonaConfig`` in
``build_voice_user`` — silently disabling the localized guidelines, persona
pragmatics, and backchannel localization for ``--user-persona-id`` runs (the
backchannel decision prompt fell back to English "uh-huh"). The smoke
transcript showed Devanagari turns only because the localized task prose led
the LLM there.

The contract under test: with a persona override the run level must defer
(pass None) so the per-task sampled persona config — the language-pack persona
— wins; without an override the complexity preset still provides the knobs.
"""

from tau2.data_model.persona import InterruptTendency, PersonaConfig, Verbosity
from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
from tau2.multilingual.schema import MultilingualPersonaConfig
from tau2.runner.batch import run_level_user_persona_config
from tau2.user_simulation_voice_presets import COMPLEXITY_CONFIGS


def make_voice_config(**overrides) -> VoiceRunConfig:
    return VoiceRunConfig(
        domain="airline",
        audio_native_config=AudioNativeConfig(),
        **overrides,
    )


def test_persona_override_defers_to_sampled_config():
    """With --user-persona-id, the run level must NOT pre-compute a persona."""
    config = make_voice_config(user_persona_id="rishika_hindi_v1")
    assert run_level_user_persona_config(config) is None


def test_no_override_uses_complexity_preset():
    config = make_voice_config(speech_complexity="regular")
    persona = run_level_user_persona_config(config)
    assert isinstance(persona, PersonaConfig)
    assert not isinstance(persona, MultilingualPersonaConfig)
    preset = COMPLEXITY_CONFIGS["regular"]
    assert persona.verbosity == Verbosity(preset["verbosity"])
    assert persona.interrupt_tendency == InterruptTendency(preset["interrupt_tendency"])


def test_multilingual_persona_survives_build_when_run_level_defers():
    """End-to-end through build_voice_user: with run-level None, the sampled
    language-pack persona (and its localization) reaches the user simulator."""
    from tau2.registry import registry
    from tau2.runner.build import build_voice_user

    config = make_voice_config(user_persona_id="imran_hindi_v1")
    environment = registry.get_env_constructor("airline")()
    task = next(t for t in registry.get_tasks_loader("airline_hi")() if t.id == "7_hi")

    user = build_voice_user(
        environment,
        task,
        config.audio_native_config,
        llm="gpt-4o-mini",
        persona_config=run_level_user_persona_config(config),
        speech_complexity="regular",
        seed=42,
        domain="airline",
        user_persona_id=config.user_persona_id,
    )

    assert isinstance(user.persona_config, MultilingualPersonaConfig)
    assert user.persona_config.persona_id == "imran_hindi_v1"
    # The localized backchannel machinery is active, not the English default.
    # imran's inventory is the pack's single pure continuer (हम्म); the
    # acknowledgments it used to carry (जी / ठीक है) now live in the
    # localization block's conversational_confirmations palette.
    assert user.backchannel_phrases == ["हम्म"]
    assert "हम्म" in user.backchannel_decision_prompt
    # The prompt is ENGLISH: the English guidelines are selected (the target
    # language rides on the fixed directive, see
    # tau2.multilingual.english_prompts), with the persona slot still present
    # (persona substitution happens in the system-prompt build).
    guidelines = user.global_simulation_guidelines
    assert "(Hindi)" not in guidelines
    assert "<PERSONA_GUIDELINES>" in guidelines
    assert "## LANGUAGE OF THE CALL" in user.system_prompt


def test_voice_agent_gets_the_same_resolved_locale_as_the_user():
    """Full-duplex agent context follows per-task persona resolution."""
    from tau2.registry import registry
    from tau2.runner.build import build_voice_orchestrator

    config = make_voice_config(user_persona_id="hi")
    task = next(t for t in registry.get_tasks_loader("airline_hi")() if t.id == "7_hi")
    orchestrator = build_voice_orchestrator(config, task, seed=42)

    persona = orchestrator.user.persona_config
    assert orchestrator.agent.locale == persona.locale
    from tau2.multilingual.varieties import locale_display_name

    assert (
        f"The customer is originally from {locale_display_name(persona.locale)} "
        "and speaks Hindi — expect standard urban Hindustani."
        in orchestrator.agent.system_prompt
    )


def test_flip_roles_drops_whitespace_only_agent_chunks():
    """Regression for the stage-2 smoke retries: streamed agent transcript
    chunks can be whitespace-only (observed with Devanagari transcript
    deltas, e.g. a lone ' ' chunk). They must be dropped when flipping
    roles for the user-sim LLM call, or generate()'s message validation
    raises 'Message must have content or tool calls' mid-conversation."""
    from tau2.data_model.message import AssistantMessage, UserMessage
    from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator

    history = [
        AssistantMessage(role="assistant", content="मैं समझ रहा हूँ कि आप"),
        AssistantMessage(role="assistant", content=" "),  # whitespace-only chunk
        AssistantMessage(role="assistant", content=""),
        AssistantMessage(role="assistant", content=None),
        UserMessage(role="user", content="हाँ"),
        AssistantMessage(role="assistant", content="ठीक है।"),
    ]
    flipped = VoiceStreamingUserSimulator._flip_roles_for_llm(None, history)
    user_contents = [m.content for m in flipped if isinstance(m, UserMessage)]
    assert user_contents == ["मैं समझ रहा हूँ कि आप", "ठीक है।"]
    # Every flipped message would pass generate()'s validation.
    for m in flipped:
        assert (m.content and m.content.strip()) or getattr(m, "tool_calls", None)
