# Copyright Sierra
"""`tau2 factory` verb registration + dispatch (primary and legacy surfaces).

Each verb is checked two ways: its argv shape parses to the right dispatcher,
and the dispatcher reaches the owning module's function (heavy work mocked).
The retired verbs are gone entirely — same checks, longer
argv.
"""

import argparse
from pathlib import Path

import pytest

from tau2.multilingual.factory.cli import add_factory_args


@pytest.fixture
def parser():
    p = argparse.ArgumentParser(prog="tau2 factory")
    add_factory_args(p)
    return p


@pytest.mark.parametrize(
    "argv, func_name",
    [
        (["smoke-assets"], "run_factory_smoke_assets"),
        (
            ["draft-localization", "--lang", "de", "--force"],
            "run_factory_draft_localization",
        ),
        (
            ["draft-continuers", "--lang", "it", "--force"],
            "run_factory_draft_continuers",
        ),
        (
            ["draft-nativeness", "--lang", "ro", "--reasoning", "high"],
            "run_factory_draft_nativeness",
        ),
        (
            ["apply-nativeness", "--lang", "ro"],
            "run_factory_apply_nativeness",
        ),
    ],
)
def test_new_verbs_parse_to_dispatchers(parser, argv, func_name):
    args = parser.parse_args(argv)
    assert args.func.__name__ == func_name


def test_deleted_verbs_do_not_resolve(parser):
    """The retired machinery is gone — neither at the top level nor under the
    dissolved ``legacy`` group, which no longer exists as a subcommand."""
    for argv in (
        ["legacy", "translate", "--lang", "ro", "--domain", "airline"],
        ["translate", "--lang", "ro", "--domain", "airline"],
        ["parity", "--lang", "es", "--domain", "airline", "plan"],
        ["probe-coverage", "--langs", "zh"],
        ["tasks-csv", "extract", "--domain", "airline", "--lang", "fr"],
        ["redraft-pragmatics", "--lang", "es"],
        ["draft-native-strings", "--lang", "pt"],
        ["repin-variety", "--lang", "es"],
        ["generate-beds", "--lang", "es"],
        ["translation-review", "--lang", "es", "--domain", "airline"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_smoke_assets_dispatch_renders_typed_outcome(parser, monkeypatch, capsys):
    import tau2.multilingual.factory.asset_smoke as asset_smoke

    outcome = asset_smoke.AssetSmokeOutcome(
        out_dir=Path("/tmp/smoke"),
        checks=[
            asset_smoke.EndpointCheck(name="Voice Design", passed=True, detail="ok"),
            asset_smoke.EndpointCheck(name="Sound Effects", passed=True),
        ],
    )
    monkeypatch.setattr(asset_smoke, "run_asset_smoke", lambda: outcome)
    args = parser.parse_args(["smoke-assets"])
    args.func(args)  # all-pass -> no SystemExit
    out = capsys.readouterr().out
    assert "PASS — Voice Design" in out and "OK" in out


def test_smoke_assets_dispatch_exits_nonzero_on_failure(parser, monkeypatch, capsys):
    import tau2.multilingual.factory.asset_smoke as asset_smoke

    outcome = asset_smoke.AssetSmokeOutcome(
        out_dir=Path("/tmp/smoke"),
        checks=[
            asset_smoke.EndpointCheck(
                name="Voice Design", passed=False, detail="api down"
            ),
        ],
    )
    assert outcome.ok is False
    monkeypatch.setattr(asset_smoke, "run_asset_smoke", lambda: outcome)
    args = parser.parse_args(["smoke-assets"])
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    assert exc.value.code == 1
    assert "FAILED" in capsys.readouterr().out
