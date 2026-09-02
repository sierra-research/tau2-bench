import json
import sys

from experiments.tau_voice.run_multiple import (
    ProviderSpec,
    build_command,
    build_config,
)


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
