"""Elicitation configuration and stimulus isolation at the runner boundary."""

import pytest
from pydantic import ValidationError

from tau2.config import DEFAULT_LLM_ARGS_USER, DEFAULT_LLM_USER
from tau2.data_model.elicitation import ElicitationStrategy
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import TextRunConfig, VoiceRunConfig


def voice_config(domain, **kwargs):
    return VoiceRunConfig(
        domain=domain, audio_native_config=AudioNativeConfig(), **kwargs
    )


def test_entity_branding_preserves_legacy_cli_identifiers():
    import argparse

    from tau2.domains.intake.cli import add_intake_run_args

    parser = argparse.ArgumentParser()
    add_intake_run_args(parser)
    assert "τ-entity (formerly elicitation; intake only)" in parser.format_help()
    args = parser.parse_args(["--elicitation-strategy", "scaffolded"])
    assert args.elicitation_strategy == "scaffolded"


@pytest.mark.parametrize(
    "domain", ["airline", "retail", "telecom", "banking_knowledge", "mock"]
)
def test_other_domains_keep_main_defaults_and_reject_elicitation(domain):
    config = voice_config(domain)
    assert config.elicitation is None
    assert config.llm_user == DEFAULT_LLM_USER
    assert config.llm_args_user == DEFAULT_LLM_ARGS_USER
    with pytest.raises(ValidationError, match="elicitation"):
        voice_config(domain, elicitation={"rate": 1})


def test_elicitation_defaults_resolve_for_programmatic_runs_and_roundtrip():
    agent_directed = voice_config("intake")
    scaffolded = voice_config("intake", elicitation={"strategy": "scaffolded"})
    assert agent_directed.elicitation.strategy == ElicitationStrategy.AGENT_DIRECTED
    assert scaffolded.elicitation.strategy == ElicitationStrategy.SCAFFOLDED
    assert agent_directed.elicitation.rate == 1
    assert scaffolded.elicitation.rate == 1
    assert agent_directed.audio_native_config.max_steps_seconds == 240
    assert agent_directed.llm_user == "gpt-5.5"
    assert agent_directed.llm_args_user["reasoning_effort"] == "low"
    assert (
        VoiceRunConfig.model_validate_json(agent_directed.model_dump_json())
        == agent_directed
    )
    assert (
        VoiceRunConfig.model_validate_json(scaffolded.model_dump_json()) == scaffolded
    )
    assert scaffolded.model_dump(mode="json")["elicitation"]["strategy"] == "scaffolded"


@pytest.mark.parametrize("strategy", list(ElicitationStrategy))
def test_explicit_elicitation_and_user_settings_win(strategy):
    config = voice_config(
        "intake",
        elicitation={"rate": 0, "strategy": strategy},
        llm_user="custom",
        llm_args_user={},
    )
    assert config.elicitation.rate == 0
    assert config.llm_user == "custom"
    assert config.llm_args_user == {}
    with pytest.raises(ValidationError):
        voice_config("intake", elicitation={"rate": 1.1, "strategy": strategy})


def test_text_runs_require_the_scaffolded_strategy():
    with pytest.raises(ValidationError, match="voice"):
        TextRunConfig(domain="intake")
    config = TextRunConfig(domain="intake", elicitation={"strategy": "scaffolded"})
    assert config.elicitation.strategy == ElicitationStrategy.SCAFFOLDED


@pytest.mark.parametrize("strategy", list(ElicitationStrategy))
def test_explicit_catalog_rate_is_preserved(strategy):
    config = voice_config("intake", elicitation={"rate": None, "strategy": strategy})
    assert config.elicitation.rate is None


def test_removed_complication_profile_is_rejected():
    with pytest.raises(ValidationError, match="profile"):
        voice_config("intake", elicitation={"profile": "hard"})


def test_elicitation_does_not_override_explicit_audio_duration():
    config = VoiceRunConfig(
        domain="intake",
        audio_native_config=AudioNativeConfig(max_steps_seconds=60),
    )
    assert config.audio_native_config.max_steps_seconds == 60
