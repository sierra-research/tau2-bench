import argparse
import json
import sys

import pytest

from experiments.tau_voice.run_multiple import (
    ProviderSpec,
    build_command,
    build_config,
)
from tau2.cli import add_run_args
from tau2.data_model.simulation import AudioNativeConfig


def test_build_command_uses_current_python_and_user_llm_args():
    command = build_command(
        "retail",
        ProviderSpec(provider="openai", model="pine-voice-preview"),
        "regular",
        "/tmp/results",
        user_llm="gpt-5.5-2026-04-23",
        user_llm_args={"reasoning_effort": "xhigh"},
    )

    assert command[:3] == [sys.executable, "-m", "tau2.cli"]
    args_index = command.index("--user-llm-args")
    assert json.loads(command[args_index + 1]) == {"reasoning_effort": "xhigh"}


def test_build_config_sets_user_llm_args():
    config = build_config(
        "retail",
        ProviderSpec(provider="openai", model="pine-voice-preview"),
        "regular",
        "/tmp/results",
        user_llm_args={"reasoning_effort": "xhigh"},
    )

    assert config.llm_args_user == {"reasoning_effort": "xhigh"}


@pytest.mark.parametrize(
    ("realtime_generation", "expected_flag"),
    [
        (True, "--realtime-generation"),
        (False, "--no-realtime-generation"),
        (None, None),
    ],
)
def test_build_command_sets_realtime_generation_override(
    realtime_generation, expected_flag
):
    command = build_command(
        "retail",
        ProviderSpec(provider="xai", model="grok-voice-think-fast-2.0"),
        "regular",
        "/tmp/results",
        realtime_generation=realtime_generation,
    )

    flags = {"--realtime-generation", "--no-realtime-generation"}
    actual_flags = flags.intersection(command)
    assert actual_flags == ({expected_flag} if expected_flag else set())


@pytest.mark.parametrize("realtime_generation", [True, False, None])
def test_build_config_sets_realtime_generation_override(realtime_generation):
    config = build_config(
        "retail",
        ProviderSpec(provider="xai", model="grok-voice-think-fast-2.0"),
        "regular",
        "/tmp/results",
        realtime_generation=realtime_generation,
    )

    assert config.audio_native_config.realtime_generation is realtime_generation


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ([], None),
        (["--realtime-generation"], True),
        (["--no-realtime-generation"], False),
    ],
)
def test_cli_parses_realtime_generation_override(flag, expected):
    parser = argparse.ArgumentParser()
    add_run_args(parser)

    args = parser.parse_args(flag)

    assert args.realtime_generation is expected


def test_realtime_generation_has_provider_aware_default():
    assert not AudioNativeConfig(provider="xai").realtime_generation_enabled
    assert AudioNativeConfig(
        provider="openai_live",
        model="test-live",
        live_config={"backend_model": "test-backend"},
    ).realtime_generation_enabled


@pytest.mark.parametrize("override", [True, False])
def test_realtime_generation_override_wins_over_provider_default(override):
    assert (
        AudioNativeConfig(
            provider="xai", realtime_generation=override
        ).realtime_generation_enabled
        is override
    )
    assert (
        AudioNativeConfig(
            provider="openai_live",
            model="test-live",
            live_config={"backend_model": "test-backend"},
            realtime_generation=override,
        ).realtime_generation_enabled
        is override
    )
