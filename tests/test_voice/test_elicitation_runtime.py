"""Domain overrides must leave the shared voice stack's defaults unchanged."""

from types import SimpleNamespace

import pytest

from tau2.agent.discrete_time_audio_native_agent import (
    AUDIO_NATIVE_SYSTEM_PROMPT_PLAIN,
    AUDIO_NATIVE_VOICE_INSTRUCTION,
    CASCADED_MODEL_INSTRUCTION,
    DiscreteTimeAudioNativeAgent,
)
from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.voice import SynthesisConfig, VoiceSettings
from tau2.domains.intake.prompts import (
    OUTBOUND_ROLE_REMINDER,
    get_agent_instructions,
    get_user_guidelines,
)
from tau2.domains.intake.spell_events import SpellEventDetector
from tau2.user.guidelines import default_persona, simulation_guidelines
from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator


def _agent(provider, **kwargs):
    return DiscreteTimeAudioNativeAgent(
        tools=[],
        domain_policy="Domain policy",
        provider=provider,
        vad_config=SimpleNamespace(),
        **kwargs,
    )


@pytest.mark.parametrize(
    "provider", ["openai", "openai_live", "gemini", "xai", "livekit"]
)
def test_agent_instruction_override_preserves_provider_defaults(provider):
    ordinary = _agent(provider)
    base_instruction = (
        CASCADED_MODEL_INSTRUCTION
        if provider == "livekit"
        else AUDIO_NATIVE_VOICE_INSTRUCTION
    )
    assert ordinary.system_prompt == AUDIO_NATIVE_SYSTEM_PROMPT_PLAIN.format(
        agent_instruction=base_instruction, domain_policy="Domain policy"
    )
    override = get_agent_instructions(provider, agent_directed=True)
    agent_directed = _agent(provider, agent_instruction_override=override)
    assert "spell it out letter by letter" not in agent_directed.system_prompt
    assert "Domain policy" in agent_directed.system_prompt
    assert get_agent_instructions(provider, agent_directed=False) is None
    assert "spell it out letter by letter" not in ordinary.system_prompt
    assert (
        "First, authenticate the customer according to the domain policy."
        in ordinary.system_prompt
    )


def test_end_call_terminates_only_explicitly_configured_voice_agents():
    message = AssistantMessage(
        role="assistant",
        tool_calls=[
            ToolCall(id="stop", name="end_call", arguments={}, requestor="assistant")
        ],
    )
    ordinary = _agent("openai")
    agent_directed = _agent(
        "openai", additional_stop_tool_names=frozenset({"end_call"})
    )
    assert not ordinary.is_stop(
        ordinary._check_if_stop_toolcall(message.model_copy(deep=True))
    )
    assert agent_directed.is_stop(
        agent_directed._check_if_stop_toolcall(message.model_copy(deep=True))
    )


def _user(**kwargs):
    return VoiceStreamingUserSimulator(
        llm="test",
        instructions="Test scenario",
        tools=[],
        voice_settings=VoiceSettings(
            transcription_config=None, synthesis_config=SynthesisConfig()
        ),
        **kwargs,
    )


def test_caller_guidelines_override_leaves_default_prompt_unchanged():
    ordinary = _user()
    assert ordinary.global_simulation_guidelines == simulation_guidelines(
        default_persona("voice"), mode="voice", use_tools=True
    )
    agent_directed = _user(global_guidelines_override=get_user_guidelines(True, True))
    assert "ANSWERING a VOICE CALL" in agent_directed.system_prompt
    assert "offer to spell it out letter by letter" not in agent_directed.system_prompt
    assert ordinary.global_guidelines_override is None
    assert ordinary.spell_event_detector is None
    assert ordinary.voice_settings.pronunciation_map is None


@pytest.mark.parametrize("elicitation", [False, True])
def test_generated_messages_use_scoped_outbound_reminder_and_spell_events(
    monkeypatch, elicitation
):
    detector = SpellEventDetector(field_skeletons={"name": ["lisa"]})
    user = _user(
        role_reminder_override=OUTBOUND_ROLE_REMINDER if elicitation else None,
        spell_event_detector=detector.detect if elicitation else None,
    )
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return AssistantMessage(role="assistant", content="L, I, S, A.")

    monkeypatch.setattr("tau2.user.user_simulator_streaming.generate", generate)
    monkeypatch.setattr(
        user, "synthesize_voice", lambda message, state, **kwargs: message
    )
    state = user.get_init_state()
    user._generate_full_duplex_voice_message(
        AssistantMessage(role="assistant", content="Spell your name."), state
    )
    assert calls[0]["call_name"] == "user_streaming_response"
    reminder = calls[0]["messages"][-1].content
    events = user.get_effect_timeline().events
    if elicitation:
        assert reminder == OUTBOUND_ROLE_REMINDER
        assert [
            (event.effect_type, event.params["letters_spoken"]) for event in events
        ] == [("spell_out", 4)]
        assert events[0].start_ms == events[0].end_ms
    else:
        assert "CUSTOMER calling for help" in reminder
        assert events == []


@pytest.mark.parametrize(
    "mapping,spoken", [(None, "Narbut"), ({"Narbut": "narbutt"}, "narbutt")]
)
def test_pronunciation_changes_only_tts_text_and_records_delivered_script(
    monkeypatch, mapping, spoken
):
    user = _user()
    user.voice_settings.pronunciation_map = mapping
    synthesized = []

    def synthesize(**kwargs):
        synthesized.append(kwargs["text"])
        return AudioData(
            data=b"\0" * 480,
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=24000),
        )

    monkeypatch.setattr("tau2.agent.base.voice.synthesize_voice", synthesize)
    monkeypatch.setattr(user, "apply_batch_effects", lambda audio, **kwargs: audio)
    message = user.synthesize_voice(
        UserMessage(role="user", content="Narbut"),
        SimpleNamespace(noise_generator=None),
        add_background_noise=False,
    )
    assert synthesized == [spoken]
    assert message.content == "Narbut"
    assert message.audio_script_gold == spoken
