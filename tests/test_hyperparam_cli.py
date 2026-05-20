import sys

import experiments.hyperparam.cli as cli
from experiments.hyperparam.cli import get_cli_parser


def test_run_evals_cli_keeps_legacy_modes_available():
    parser = get_cli_parser()
    args = parser.parse_args(
        [
            "run-evals",
            "--exp-dir",
            "test-exp",
            "--llms",
            "gpt-4.1-2025-04-14",
            "--modes",
            "default",
            "oracle-plan",
            "no-user",
            "no-user-op",
        ]
    )

    assert args.modes == ["default", "oracle-plan", "no-user", "no-user-op"]


def test_run_responses_sweep_defaults_to_default_mode_only():
    parser = get_cli_parser()
    args = parser.parse_args(["run-responses-sweep"])

    assert args.modes == ["default"]
    assert args.service_tiers == ["default", "priority"]
    assert args.max_steps == 100
    assert args.max_duration_seconds == 900.0
    assert args.responses_transports == ["http"]
    assert args.parallel_tool_calls is None


def test_run_responses_sweep_accepts_timeout_alias():
    parser = get_cli_parser()
    args = parser.parse_args(["run-responses-sweep", "--timeout", "0"])

    assert args.max_duration_seconds == 0.0


def test_run_responses_sweep_accepts_banking_knowledge_domain():
    parser = get_cli_parser()
    args = parser.parse_args(
        [
            "run-responses-sweep",
            "--domains",
            "retail",
            "banking_knowledge",
        ]
    )

    assert args.domains == ["retail", "banking_knowledge"]


def test_build_responses_report_cli_requires_exp_dir():
    parser = get_cli_parser()
    args = parser.parse_args(["build-responses-report", "--exp-dir", "sample-exp"])

    assert args.command == "build-responses-report"
    assert args.exp_dir == "sample-exp"


def test_canonicalize_responses_repair_cli_requires_target_and_sources():
    parser = get_cli_parser()
    args = parser.parse_args(
        [
            "canonicalize-responses-repair",
            "--exp-dir",
            "base-exp",
            "--repair-exp-dirs",
            "repair-exp",
        ]
    )

    assert args.command == "canonicalize-responses-repair"
    assert args.exp_dir == "base-exp"
    assert args.repair_exp_dirs == ["repair-exp"]


def test_validate_responses_resume_state_cli_requires_exp_dir():
    parser = get_cli_parser()
    args = parser.parse_args(["validate-responses-resume-state", "--exp-dir", "base-exp"])

    assert args.command == "validate-responses-resume-state"
    assert args.exp_dir == "base-exp"


def test_run_responses_sweep_accepts_domain_task_split_overrides():
    parser = get_cli_parser()
    args = parser.parse_args(
        [
            "run-responses-sweep",
            "--task-split-name",
            "base",
            "--domain-task-splits",
            '{"telecom":"full"}',
        ]
    )

    assert args.task_split_name == "base"
    assert args.domain_task_splits == '{"telecom":"full"}'


def test_run_responses_sweep_accepts_cache_sources():
    parser = get_cli_parser()
    args = parser.parse_args(
        [
            "run-responses-sweep",
            "--reuse-from-exp-dirs",
            "ofat-base-banking-telecom-full-1",
            "/tmp/other-responses-exp",
        ]
    )

    assert args.reuse_from_exp_dirs == [
        "ofat-base-banking-telecom-full-1",
        "/tmp/other-responses-exp",
    ]


def test_run_responses_sweep_accepts_failed_only_infra_errors():
    parser = get_cli_parser()
    args = parser.parse_args(
        [
            "run-responses-sweep",
            "--failed-only",
            "infrastructure_error",
        ]
    )

    assert args.failed_only == ["infrastructure_error"]


def test_run_responses_sweep_accepts_dry_run():
    parser = get_cli_parser()
    args = parser.parse_args(["run-responses-sweep", "--dry-run"])

    assert args.dry_run is True


def test_run_responses_sweep_accepts_known_variant_controls():
    parser = get_cli_parser()
    args = parser.parse_args(
        [
            "run-responses-sweep",
            "--known-variant-suite",
            "--responses-transports",
            "http",
            "websocket",
            "--parallel-tool-calls",
            "unset",
            "true",
            "false",
        ]
    )

    assert args.known_variant_suite is True
    assert args.responses_transports == ["http", "websocket"]
    assert args.parallel_tool_calls == ["unset", "true", "false"]


def test_run_responses_sweep_main_passes_max_duration(monkeypatch, tmp_path):
    captured = {}

    def fake_run_responses_sweep(**kwargs):
        captured.update(kwargs)
        return tmp_path

    monkeypatch.setattr(cli, "run_responses_sweep", fake_run_responses_sweep)
    monkeypatch.setattr(
        cli, "build_responses_report", lambda exp_dir: exp_dir / "index.html"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "run-responses-sweep",
            "--max-duration-seconds",
            "123",
        ],
    )

    cli.main()

    assert captured["max_duration_seconds"] == 123.0


def test_run_responses_sweep_main_passes_failed_only(monkeypatch, tmp_path):
    captured = {}

    def fake_run_responses_sweep(**kwargs):
        captured.update(kwargs)
        return tmp_path

    monkeypatch.setattr(cli, "run_responses_sweep", fake_run_responses_sweep)
    monkeypatch.setattr(
        cli, "build_responses_report", lambda exp_dir: exp_dir / "index.html"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "run-responses-sweep",
            "--failed-only",
            "infrastructure_error",
        ],
    )

    cli.main()

    assert captured["failed_only_termination_reasons"] == ["infrastructure_error"]


def test_run_responses_sweep_main_passes_dry_run_without_report(monkeypatch, tmp_path):
    captured = {}
    report_called = False

    def fake_run_responses_sweep(**kwargs):
        captured.update(kwargs)
        return tmp_path

    def fake_build_responses_report(exp_dir):
        nonlocal report_called
        report_called = True
        return exp_dir / "index.html"

    monkeypatch.setattr(cli, "run_responses_sweep", fake_run_responses_sweep)
    monkeypatch.setattr(cli, "build_responses_report", fake_build_responses_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "run-responses-sweep",
            "--dry-run",
        ],
    )

    cli.main()

    assert captured["dry_run"] is True
    assert report_called is False
