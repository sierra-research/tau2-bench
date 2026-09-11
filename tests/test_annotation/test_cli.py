# Copyright Sierra
"""`tau2 annotate` surface: verbs, arg parsing, and verb bodies end to end."""

import argparse
from pathlib import Path

import pytest

from tau2.annotation.cli import add_annotate_args
from tau2.config import ReasoningEffort
from test_annotation.conftest import make_hi_results_dir

VERBS = [
    "nativeness",
    "workbook",
    "localization-sheet",
    "audit",
    "source-audit",
    "packets",
    "prompt-bed-packet",
    "calibration-packets",
    "recall-corpus",
    "feature-table",
    "ingest",
    "translation-review",
    "agreement",
    "calibration-agreement",
]


@pytest.fixture
def parser():
    p = argparse.ArgumentParser(prog="tau2 annotate")
    add_annotate_args(p)
    return p


def test_all_verbs_registered(parser):
    subparsers = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    assert list(subparsers.choices) == VERBS


@pytest.mark.parametrize(
    "argv",
    [
        [
            "nativeness",
            "results",
            "--mode",
            "both",
            "--rejudge",
            "--max-trans-chars",
            "500",
            "--max-concurrency",
            "2",
            "--out",
            "x",
        ],
        ["workbook", "a.csv", "b.csv", "--publish-to", "pub"],
        [
            "localization-sheet",
            "--domain",
            "retail",
            "--langs",
            "en,es,hi",
            "--only-subset",
            "--out",
            "out",
            "--publish-to",
            "pub",
        ],
        ["audit", "--lang", "hi", "--blind"],
        ["audit", "--generate", "--lang", "ro", "--model", "m"],
        [
            "packets",
            "--form",
            "voice_review",
            "--batch-name",
            "round1",
            "--results",
            "a",
            "b",
            "--out",
            "out",
            "--append",
            "--shuffle",
            "--seed",
            "7",
            "--max-items",
            "10",
            "--filter-reward",
            "< 1",
            "--filter-tasks",
            "t1",
            "t2",
        ],
        [
            "prompt-bed-packet",
            "--lang",
            "es",
            "--domain",
            "airline",
            "--task-index",
            "0",
            "--out",
            "out",
        ],
        [
            "feature-table",
            "results_a",
            "results_b",
            "--out",
            "table.json",
            "--langs",
            "es",
            "pt",
            "--domain",
            "telecom",
            "--long-silence-threshold",
            "2.5",
            "--max-sims",
            "5",
        ],
        ["ingest", "FILLED.csv", "--second", "other.csv", "--precision-bar", "0.9"],
        ["agreement", "--lang", "hi"],
    ],
)
def test_verb_arg_shapes_parse(parser, argv):
    args = parser.parse_args(argv)
    assert callable(args.func)


def test_nativeness_verb_runs_export(parser, tmp_path, capsys):
    run_dir = make_hi_results_dir(tmp_path)
    out_stem = tmp_path / "family" / "hi_cal"
    args = parser.parse_args(
        [
            "nativeness",
            str(run_dir),
            "--mode",
            "precision",
            "--out",
            str(out_stem),
        ]
    )
    args.func(args)
    assert (tmp_path / "family" / "hi_cal.csv").exists()
    assert (tmp_path / "family" / "hi_cal.manifest.json").exists()
    assert "manifest" in capsys.readouterr().out


def test_feature_table_verb_runs(parser, tmp_path, capsys):
    """One arm per run dir, as real runs are: the arm identity is a property
    of the RUN's audio-native config, so a two-provider results.json is not a
    shape the extractor can or should make sense of."""
    from fixtures_runs import make_hi_results

    from tau2.data_model.simulation import AudioNativeConfig
    from test_annotation.test_features import voice_sim

    run_dirs = []
    for i, (provider, model, effort) in enumerate(
        [
            ("openai", "gpt-realtime-2", ReasoningEffort.XHIGH),
            ("gemini", "flash-live", ReasoningEffort.MINIMAL),
        ]
    ):
        sims = [
            voice_sim(
                f"s-{provider}-{trial}",
                trial=trial,
                provider=provider,
                pattern="uuu" + "." * (2 + trial + 3 * i) + "aaaa",
            )
            for trial in range(2)
        ]
        run_dirs.append(
            make_hi_results(
                tmp_path,
                sims,
                name=f"run_{provider}",
                num_trials=2,
                agent_llm=f"{provider}:{model}",
                audio_native_config=AudioNativeConfig(
                    provider=provider, model=model, reasoning_effort=effort
                ),
            )
        )
    table_out = tmp_path / "table.json"
    args = parser.parse_args(
        ["feature-table", *[str(d) for d in run_dirs], "--out", str(table_out)]
    )
    args.func(args)
    assert table_out.exists()
    assert "feature table" in capsys.readouterr().out.lower()


def test_workbook_verb_exits_when_nothing_built(parser, tmp_path):
    args = parser.parse_args(["workbook", str(tmp_path / "missing.csv")])
    with pytest.raises(SystemExit, match="no workbooks built"):
        args.func(args)


def test_translation_review_dispatch_reaches_export(parser, monkeypatch, capsys):
    """Moved off the dissolved `tau2 factory legacy` group onto `tau2 annotate`."""
    import tau2.annotation.translation as annot_translation

    seen = {}

    def fake_export(lang, domain, from_existing=None, out_stem=None):
        seen.update(
            lang=lang, domain=domain, from_existing=from_existing, out_stem=out_stem
        )
        return Path("manifest.json")

    monkeypatch.setattr(annot_translation, "export_translation_review", fake_export)
    args = parser.parse_args(
        ["translation-review", "--lang", "es", "--domain", "airline"]
    )
    args.func(args)
    assert seen == {
        "lang": "es",
        "domain": "airline",
        "from_existing": None,
        "out_stem": None,
    }
    assert "manifest" in capsys.readouterr().out
