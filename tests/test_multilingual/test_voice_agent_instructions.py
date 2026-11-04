"""Shared voice instructions preserve policy and provider-specific guidance."""

import pytest

from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import VoiceRunConfig
from tau2.runner.build import build_environment


@pytest.mark.parametrize("provider", ["openai", "openai_live", "livekit"])
@pytest.mark.parametrize("use_xml", [False, True])
def test_language_voice_framing_removes_coaching_and_preserves_policy(
    provider, use_xml
):
    from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent

    for domain in ("airline", "retail", "telecom"):
        policy = build_environment(domain).get_policy()
        for language in ("en", "es", "pt", "hi", "zh", "ko"):
            config = VoiceRunConfig(
                domain=domain,
                language=language,
                audio_native_config=AudioNativeConfig(
                    provider=provider,
                    live_config={"backend_model": "fixture-backend"}
                    if provider == "openai_live"
                    else None,
                ),
            )
            audio = config.audio_native_config
            agent = DiscreteTimeAudioNativeAgent(
                tools=[],
                domain_policy=policy,
                provider=provider,
                use_xml_prompt=use_xml,
                vad_config=object(),
                adapter=object(),
            )
            prompt = agent.system_prompt
            assert policy in prompt
            assert (
                "First, authenticate the customer according to the domain policy."
                in prompt
            )
            assert "# Voice Call Instructions" in prompt
            assert "# User authentication and user information collection" not in prompt
            assert "SPELL THINGS OUT" not in prompt
            assert "letter by letter" not in prompt
            if provider == "livekit":
                assert "you will see transcribed speech, not written text" in prompt
                assert (
                    "Do not use bullets (numbered or unnumbered) or markdown formatting."
                    in prompt
                )
            restored = VoiceRunConfig.model_validate_json(config.model_dump_json())
            assert restored.audio_native_config == audio


def test_registered_factory_uses_shared_voice_instructions(voice_extra):
    from tau2.agent.discrete_time_audio_native_agent import (
        create_discrete_time_audio_native_agent,
    )

    config = VoiceRunConfig(language="en", audio_native_config=AudioNativeConfig())
    agent = create_discrete_time_audio_native_agent(
        tools=[],
        domain_policy="Domain policy.",
        audio_native_config=config.audio_native_config,
    )
    assert "# Voice Call Instructions" in agent.system_prompt
    assert agent._adapter is None
