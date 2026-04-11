import argparse
import sys

import pytest

from tau2 import cli
from tau2.scripts.leaderboard.verify_trajectories import VerificationMode


def test_add_run_args_defaults_and_json():
    parser = argparse.ArgumentParser()
    cli.add_run_args(parser)
    args = parser.parse_args(["--domain", "mock", "--agent-llm-args", '{"temperature": 0.2}'])
    assert args.domain == "mock"
    assert isinstance(args.agent_llm_args, dict)
    assert args.agent_llm_args == {"temperature": 0.2}
    assert args.enforce_communication_protocol is False


def test_main_run_dispatch_builds_run_config(monkeypatch):
    seen = {}

    def fake_run_domain(config):
        seen["config"] = config
        return "ok"

    monkeypatch.setattr(cli, "run_domain", fake_run_domain)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tau2",
            "run",
            "--domain",
            "mock",
            "--agent",
            "llm_agent",
            "--user",
            "user_simulator",
            "--agent-llm-args",
            '{"temperature": 0.11}',
            "--user-llm-args",
            '{"temperature": 0.22}',
            "--enforce-communication-protocol",
        ],
    )
    cli.main()
    config = seen["config"]
    assert config.domain == "mock"
    assert config.agent == "llm_agent"
    assert config.user == "user_simulator"
    assert config.llm_args_agent == {"temperature": 0.11}
    assert config.llm_args_user == {"temperature": 0.22}
    assert config.enforce_communication_protocol is True


@pytest.mark.parametrize(
    "argv,func_name",
    [
        (["tau2", "play"], "run_manual_mode"),
        (["tau2", "start"], "run_start_servers"),
        (["tau2", "check-data"], "run_check_data"),
        (["tau2", "domain", "mock"], "run_show_domain"),
        (
            ["tau2", "view", "--file", "results.json", "--only-show-failed"],
            "run_view_simulations",
        ),
        (
            ["tau2", "evaluate-trajs", "a.json", "b.json", "--output-dir", "out"],
            "run_evaluate_trajectories",
        ),
        (
            [
                "tau2",
                "submit",
                "prepare",
                "a.json",
                "b.json",
                "--output",
                "submission_out",
            ],
            "run_prepare_submission",
        ),
        (
            [
                "tau2",
                "submit",
                "validate",
                "submission_dir",
                "--mode",
                "private",
            ],
            "run_validate_submission",
        ),
        (
            [
                "tau2",
                "submit",
                "verify-trajs",
                "a.json",
                "b.json",
                "--mode",
                "private",
            ],
            "run_verify_trajectories",
        ),
    ],
)
def test_main_dispatches_subcommands(monkeypatch, argv, func_name):
    seen = {}

    def recorder(args=None):
        seen["called"] = func_name
        seen["args"] = args

    monkeypatch.setattr(cli, func_name, recorder)
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert seen["called"] == func_name


def test_submit_mode_parses_to_enum(monkeypatch):
    seen = {}

    def fake_validate(args):
        seen["mode"] = args.mode

    monkeypatch.setattr(cli, "run_validate_submission", fake_validate)
    monkeypatch.setattr(
        sys, "argv", ["tau2", "submit", "validate", "submission_dir", "--mode", "private"]
    )
    cli.main()
    assert seen["mode"] == VerificationMode.PRIVATE


def test_main_prints_help_with_no_command(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tau2"])
    cli.main()
    out = capsys.readouterr().out
    assert "usage:" in out.lower()
