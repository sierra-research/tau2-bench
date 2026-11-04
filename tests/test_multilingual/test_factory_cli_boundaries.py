"""Factory command names and cold-process stage import boundaries."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tau2.multilingual.factory.cli import add_factory_args


@pytest.mark.parametrize(
    "command", ["prepare-voices", "reuse-voices", "validate-voices"]
)
def test_voice_commands_register_without_retired_aliases(command):
    parser = argparse.ArgumentParser()
    add_factory_args(parser)
    argv = [command, "--request", "request.json"]
    if command != "validate-voices":
        argv += ["--output", "output"]
    assert parser.parse_args(argv).factory_command == command


@pytest.mark.parametrize("command", ["reuse-assets", "validate-assets"])
def test_retired_voice_commands_are_rejected(command):
    parser = argparse.ArgumentParser()
    add_factory_args(parser)
    with pytest.raises(SystemExit) as exc:
        argv = [command, "--request", "request.json"]
        if command == "reuse-assets":
            argv += ["--output", "output"]
        parser.parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("command", "required", "blocked"),
    [
        ("stage-pack", ["pipeline", "schema"], ["voice_assets", "author_form"]),
        ("validate", ["pipeline", "schema"], ["voice_assets", "author_form"]),
        ("autoform", ["author_form", "schema"], ["voice_assets", "pipeline"]),
        ("validate-voices", ["voice_assets"], ["pipeline", "author_form"]),
        ("finalize", ["voice_assets", "pipeline"], ["author_form"]),
    ],
)
def test_dispatch_imports_only_selected_stage_in_fresh_process(
    command, required, blocked, tmp_path
):
    request = tmp_path / "request.json"
    request.write_text("{}")
    code = r"""
import argparse
import importlib.abc
import importlib.util
import sys
from pydantic import ValidationError
prefix = "tau2.multilingual.factory."
blocked = sys.argv[3].split(",") + ["assets"]
class ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {prefix + name for name in blocked}:
            raise AssertionError(f"Unrelated stage imported: {fullname}")
sys.meta_path.insert(0, ImportGuard())
from tau2.multilingual.factory.cli import add_factory_args
for name in ["pipeline", "voice_assets", "schema", "author_form"]:
    assert prefix + name not in sys.modules, name
parser = argparse.ArgumentParser()
add_factory_args(parser)
argv = [sys.argv[1], "--request", sys.argv[2]]
if sys.argv[1] not in {"validate", "validate-voices"}:
    argv += ["--output", sys.argv[2] + ".out"]
args = parser.parse_args(argv)
try:
    args.func(args)
except ValidationError:
    pass  # Invalid input must fail at the selected typed boundary, before work.
else:
    raise AssertionError("Malformed request was accepted")
for name in sys.argv[4].split(","):
    assert prefix + name in sys.modules, name
sys.meta_path.pop(0)
assert importlib.util.find_spec(prefix + "assets") is None
if prefix + "voice_assets" in sys.modules:
    module = sys.modules[prefix + "voice_assets"]
    assert all(hasattr(module, name) for name in ["prepare_voices", "reuse_voices", "validate_voices"])
    assert not any(hasattr(module, name) for name in ["generate_assets", "reuse_assets", "validate_assets"])
assert not any(prefix + name in sys.modules for name in blocked)
"""
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env["TAU2_DATA_DIR"] = str(root / "data")
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            command,
            str(request),
            ",".join(blocked),
            ",".join(required),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not Path(str(request) + ".out").exists()
