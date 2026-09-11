# Copyright Sierra
"""Tests for the ``tau2 run-preset`` driver (engine in
tau2.multilingual.run_preset_driver, argparse surface in tau2.multilingual.cli).

These pin the friction fixes from the 9-language batch retro plus the matrix
mode that replaced the airline matrix shell script:
- Every preset stage (smoke included) injects ``--auto-resume`` so a stale
  per-arm results.json never triggers the interactive "resume? (y/n)" prompt
  that crashes with EOFError under no stdin.
- ``_has_clean_results`` recognizes a persisted, error-free run (dir-format
  voice results) so a spurious non-zero exit is distinguishable from a defect.
- Matrix mode: langs x providers cells with the matrix matched conditions
  (gpt-realtime-2 pinned on openai only, result-dir naming, timeouts,
  --auto-resume).
"""

import argparse
import json
import shlex

import pytest

from tau2.config import (
    DEFAULT_MATRIX_MAX_STEPS_SECONDS,
    DEFAULT_MULTILINGUAL_RUN_CONCURRENCY,
    DEFAULT_MULTILINGUAL_VOICE_TIMEOUT_SECONDS,
)
from tau2.multilingual import cli as ml_cli
from tau2.multilingual import run_preset_driver as driver
from tau2.multilingual.run_presets import get_run_preset


@pytest.fixture
def parser():
    p = argparse.ArgumentParser(prog="tau2 run-preset")
    ml_cli.add_run_preset_args(p)
    return p


# ---------------------------------------------------------------------------
# Preset mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["smoke", "full"])
def test_dry_run_command_includes_auto_resume(stage, capsys):
    """The printed `tau2 run` command carries --auto-resume on every stage."""
    rc = driver.run_preset("multilingual_v1_hindi", stage=stage, dry_run=True)
    assert rc == 0
    out = capsys.readouterr().out
    # Every printed arm command (no-stop-rule path) must include --auto-resume.
    arm_lines = [line for line in out.splitlines() if "tau2 run" in line]
    assert arm_lines, f"no commands printed for stage {stage}"
    for line in arm_lines:
        assert "--auto-resume" in line, line


def test_has_clean_results_dir_format(tmp_path, monkeypatch):
    """A dir-format results.json with no too_many_errors counts as clean."""
    preset = get_run_preset("multilingual_v1_hindi")
    arm = preset.arms[0]
    save_to = preset.save_to(arm, "smoke")

    sims_dir = tmp_path / "simulations" / save_to
    sims_dir.mkdir(parents=True)
    (sims_dir / "results.json").write_text(
        json.dumps(
            {
                "simulation_index": [
                    {
                        "id": "a",
                        "task_id": 1,
                        "trial": 0,
                        "termination_reason": "user_stop",
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(driver, "DATA_DIR", tmp_path)
    assert driver._has_clean_results(save_to) is True


def test_has_clean_results_flags_too_many_errors(tmp_path, monkeypatch):
    preset = get_run_preset("multilingual_v1_hindi")
    arm = preset.arms[0]
    save_to = preset.save_to(arm, "smoke")

    sims_dir = tmp_path / "simulations" / save_to
    sims_dir.mkdir(parents=True)
    (sims_dir / "results.json").write_text(
        json.dumps(
            {
                "simulation_index": [
                    {
                        "id": "a",
                        "task_id": 1,
                        "trial": 0,
                        "termination_reason": "too_many_errors",
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(driver, "DATA_DIR", tmp_path)
    assert driver._has_clean_results(save_to) is False


def test_has_clean_results_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "DATA_DIR", tmp_path)
    assert driver._has_clean_results("nope/smoke_x") is False


def test_unknown_preset_fails_actionably(capsys):
    """An unknown preset is a clean FAILED + exit code 1 listing the options,
    not a traceback."""
    rc = driver.run_preset("no_such_preset", dry_run=True)
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "no_such_preset" in out
    assert "multilingual_v1_hindi" in out  # the available presets are listed


# ---------------------------------------------------------------------------
# Matrix mode
# ---------------------------------------------------------------------------


def test_matrix_cell_command_openai_pins_model():
    command = driver.matrix_cell_command(
        domain="airline",
        lang="hi",
        provider="openai",
        save_to="base/openai/hi",
        num_trials=1,
        max_concurrency=10,
    )
    text = shlex.join(command)
    assert "--task-set-name airline_hi" in text
    assert "--user-persona-id hi" in text
    assert f"--audio-native-model {driver.MATRIX_OPENAI_MODEL}" in text
    assert "--save-to base/openai/hi" in text
    assert f"--timeout {DEFAULT_MULTILINGUAL_VOICE_TIMEOUT_SECONDS}" in text
    assert f"--max-steps-seconds {DEFAULT_MATRIX_MAX_STEPS_SECONDS}" in text
    assert "--auto-resume" in text
    assert "--num-tasks" not in text


def test_matrix_cell_command_uses_identity_task_set_when_present():
    """Matrix mode applies the SAME identity-variant selection as preset
    generation (es has a generated airline_es_identity set in the repo)."""
    command = driver.matrix_cell_command(
        domain="airline",
        lang="es",
        provider="openai",
        save_to="base/openai/es",
        num_trials=1,
        max_concurrency=10,
    )
    assert "--task-set-name airline_es_identity" in shlex.join(command)


def test_matrix_cell_command_gemini_uses_provider_default_model():
    command = driver.matrix_cell_command(
        domain="airline",
        lang="ja",
        provider="gemini",
        save_to="base/gemini/ja",
        num_trials=1,
        max_concurrency=10,
        num_tasks=1,
    )
    text = shlex.join(command)
    assert "--audio-native-model" not in text
    assert "--audio-native-provider gemini" in text
    assert "--num-tasks 1" in text


def test_matrix_languages_excludes_english():
    langs = driver.matrix_languages()
    assert "en" not in langs
    assert "hi" in langs and "es" in langs
    assert langs == sorted(langs)


def test_matrix_dry_run_prints_langs_x_providers(capsys):
    rc = driver.run_matrix(
        providers=["openai", "gemini"], num_tasks=1, base="mbase", dry_run=True
    )
    assert rc == 0
    out = capsys.readouterr().out
    commands = [line for line in out.splitlines() if line.startswith("tau2 run")]
    langs = driver.matrix_languages()
    assert len(commands) == 2 * len(langs)
    # Result-dir naming: <base>/<provider>/<lang>.
    assert any("--save-to mbase/openai/hi" in c for c in commands)
    assert any("--save-to mbase/gemini/hi" in c for c in commands)


# ---------------------------------------------------------------------------
# CLI registration (tau2 run-preset)
# ---------------------------------------------------------------------------


def test_cli_preset_mode_dispatch(parser, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        driver, "run_preset", lambda name, **kw: seen.update(name=name, **kw) or 0
    )
    args = parser.parse_args(
        ["multilingual_v1_hindi", "--stage", "full", "--arm", "hindi", "--dry-run"]
    )
    args.func(args)
    assert seen["name"] == "multilingual_v1_hindi"
    assert seen["stage"] == "full"
    assert seen["arm"] == "hindi"
    assert seen["dry_run"] is True
    assert seen["max_concurrency"] == DEFAULT_MULTILINGUAL_RUN_CONCURRENCY


def test_cli_matrix_mode_dispatch(parser, monkeypatch):
    seen = {}
    monkeypatch.setattr(driver, "run_matrix", lambda **kw: seen.update(**kw) or 0)
    args = parser.parse_args(
        [
            "--all-languages",
            "--providers",
            "openai,gemini",
            "--lanes",
            "2",
            "--num-tasks",
            "1",
        ]
    )
    args.func(args)
    assert seen["providers"] == ["openai", "gemini"]
    assert seen["lanes"] == 2
    assert seen["num_tasks"] == 1
    assert seen["max_concurrency"] == DEFAULT_MULTILINGUAL_RUN_CONCURRENCY


def test_cli_rejects_preset_plus_all_languages(parser):
    args = parser.parse_args(["multilingual_v1_hindi", "--all-languages"])
    with pytest.raises(SystemExit):
        args.func(args)


def test_cli_requires_preset_or_matrix(parser):
    args = parser.parse_args([])
    with pytest.raises(SystemExit):
        args.func(args)


def test_cli_list_prints_presets(parser, capsys):
    """`tau2 run-preset --list` prints name + description per preset, exit 0."""
    args = parser.parse_args(["--list"])
    args.func(args)  # must not SystemExit
    out = capsys.readouterr().out
    assert "multilingual_v1_hindi" in out


def test_cli_rejects_matrix_flags_in_preset_mode(parser, capsys):
    """--providers/--num-tasks/etc. are matrix-only; naming them with a preset
    fails actionably instead of being silently ignored."""
    args = parser.parse_args(
        ["multilingual_v1_hindi", "--providers", "openai", "--num-tasks", "1"]
    )
    with pytest.raises(SystemExit) as excinfo:
        args.func(args)
    message = str(excinfo.value)
    assert "--providers" in message and "--num-tasks" in message
