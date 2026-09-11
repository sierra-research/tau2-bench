# Copyright Sierra
"""``tau2 judges rejudge`` / ``tau2 judges export`` end-to-end on tmp fixtures.

Real Results on disk (both storage formats), judge calls mocked at the
``attach_nativeness`` seam — the code paths under test are streaming, reuse
gating, and format-preserving persistence.
"""

import json
from argparse import Namespace

import pytest

from tau2.data_model.simulation import JudgeOutcome, Results
from tau2.judges.cli import (
    run_judges_export,
    run_judges_rejudge,
)
from tau2.judges.export import judge_factor_ids
from test_judges.conftest import hi_factor_checks, hi_sim, make_hi_results


def _args(results, **over):
    base = dict(
        results=[str(p) for p in results],
        rejudge=False,
        judge_model=None,
        judge_args=None,
        max_concurrency=2,
        output=None,
    )
    base.update(over)
    return Namespace(**base)


@pytest.fixture
def patch_attach(monkeypatch):
    """Mock the judge at the attach seam: stamps complete verdicts with a
    'fresh-judge' marker; records call count + the settings model used."""
    calls = {"n": 0, "models": []}

    def attach(
        sim, task, settings=None, agent_provider=None, agent_voice=None, domain=None
    ):
        from tau2.data_model.simulation import NativenessInfo

        calls["n"] += 1
        calls["models"].append(settings.model if settings else None)
        sim.nativeness_info = NativenessInfo(
            score=1.0,
            factor_checks=hi_factor_checks(),  # complete: all PASS
            language="hi",
            script="deva",
            judge_model="fresh-judge",
        )

    import tau2.judges.attach as attach_mod

    monkeypatch.setattr(attach_mod, "attach_nativeness", attach)
    return calls


# --- rejudge -----------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["dir", "json"])
def test_rejudge_round_trips_storage_format(tmp_path, patch_attach, fmt):
    """A gap-bearing run is re-judged and saved back in its ORIGINAL format;
    the fresh verdicts survive a subsequent Results.load (no IsADirectoryError,
    no stale simulations/*.json shadowing a monolithic write)."""
    gap_id = sorted(judge_factor_ids("hi"))[0]
    sims = [
        hi_sim("s1", "t1", checks=hi_factor_checks(gap_factor=gap_id)),
        hi_sim("s2", "t2"),  # complete -> reused, no judge call
    ]
    path = make_hi_results(tmp_path, sims, format=fmt)

    run_judges_rejudge(_args([path]))

    assert patch_attach["n"] == 1  # only the gappy sim was judged
    if fmt == "dir":
        # Dir format preserved: per-sim files updated, no monolithic blob.
        sim_file = path / "simulations" / "s1.json"
        assert sim_file.is_file()
        assert "fresh-judge" in sim_file.read_text()
    loaded = Results.load(path)
    by_id = {s.id: s for s in loaded.simulations}
    assert by_id["s1"].nativeness_info.judge_model == "fresh-judge"
    checks = {c.id: c for c in by_id["s1"].nativeness_info.factor_checks}
    assert checks[gap_id].outcome == JudgeOutcome.PASS  # gap healed
    # The reused sim's stored verdicts are untouched.
    assert by_id["s2"].nativeness_info.judge_model == "stored-judge"


def test_rejudge_reuses_complete_verdicts_by_default(tmp_path, patch_attach):
    path = make_hi_results(tmp_path, [hi_sim("s1", "t1")], format="dir")
    run_judges_rejudge(_args([path]))
    assert patch_attach["n"] == 0  # complete stored verdicts -> no LLM spend
    loaded = Results.load(path)
    assert loaded.simulations[0].nativeness_info.judge_model == "stored-judge"


def test_rejudge_flag_forces_full_rejudge(tmp_path, patch_attach):
    path = make_hi_results(tmp_path, [hi_sim("s1", "t1")], format="dir")
    run_judges_rejudge(_args([path], rejudge=True))
    assert patch_attach["n"] == 1
    loaded = Results.load(path)
    assert loaded.simulations[0].nativeness_info.judge_model == "fresh-judge"


def test_rejudge_output_preserves_input_format(tmp_path, patch_attach):
    """--output to a fresh path keeps the INPUT's dir format (a fresh path
    would otherwise auto-detect as json and strand voice runs)."""
    path = make_hi_results(tmp_path, [hi_sim("s1", "t1")], format="dir")
    out = tmp_path / "rejudged_out"
    run_judges_rejudge(_args([path], rejudge=True, output=out))
    assert (out / "simulations" / "s1.json").is_file()
    loaded = Results.load(out)
    assert loaded.simulations[0].nativeness_info.judge_model == "fresh-judge"


def test_rejudge_judge_model_override_threads_through(tmp_path, patch_attach):
    path = make_hi_results(tmp_path, [hi_sim("s1", "t1")], format="dir")
    run_judges_rejudge(
        _args(
            [path],
            rejudge=True,
            judge_model="my-custom-judge",
            judge_args='{"reasoning_effort": "low"}',
        )
    )
    # The override reached the judge settings threaded into the attach seam.
    assert patch_attach["models"] == ["my-custom-judge"]


def test_rejudge_rejects_bad_judge_args(tmp_path):
    path = make_hi_results(tmp_path, [hi_sim("s1", "t1")], format="dir")
    with pytest.raises(SystemExit):
        run_judges_rejudge(_args([path], judge_args='["not", "a", "dict"]'))


# --- export ------------------------------------------------------------------


def test_export_reuses_by_default(tmp_path, patch_attach):
    path = make_hi_results(
        tmp_path, [hi_sim("s1", "t1"), hi_sim("s2", "t2")], format="dir"
    )
    out = tmp_path / "export"
    run_judges_export(_args([path], out=out))

    assert patch_attach["n"] == 0  # complete verdicts -> zero re-judging
    lines = (out / "nativeness_verdicts.jsonl").read_text().splitlines()
    assert len(lines) == 2 * len(judge_factor_ids("hi"))
    # Deterministic: rows sorted by (sim_id, factor_id).
    keys = [(r["sim_id"], r["factor_id"]) for r in (json.loads(line) for line in lines)]
    assert keys == sorted(keys)
    assert (out / "delivery_verdicts.jsonl").exists()
    assert (out / "factor_rubrics.jsonl").exists()


def test_export_persists_paid_verdicts_back_to_results(tmp_path, patch_attach):
    """When export had to judge gaps, the fresh verdicts are saved back to the
    results (format preserved) — the judging cost is never wasted."""
    gap_id = sorted(judge_factor_ids("hi"))[0]
    path = make_hi_results(
        tmp_path,
        [hi_sim("s1", "t1", checks=hi_factor_checks(gap_factor=gap_id))],
        format="dir",
    )
    out = tmp_path / "export"
    run_judges_export(_args([path], out=out))

    assert patch_attach["n"] == 1
    # Persisted back in dir format; survives a fresh load.
    loaded = Results.load(path)
    assert loaded.simulations[0].nativeness_info.judge_model == "fresh-judge"


def test_export_error_verdicts_are_healed(tmp_path, patch_attach):
    """Stored ERROR outcomes are gaps (not 'complete'), so export heals them."""
    gap_id = sorted(judge_factor_ids("hi"))[0]
    path = make_hi_results(
        tmp_path,
        [
            hi_sim(
                "s1",
                "t1",
                checks=hi_factor_checks(
                    gap_factor=gap_id, gap_outcome=JudgeOutcome.ERROR
                ),
            )
        ],
        format="dir",
    )
    run_judges_export(_args([path], out=tmp_path / "export"))
    assert patch_attach["n"] == 1
    loaded = Results.load(path)
    checks = {c.id: c for c in loaded.simulations[0].nativeness_info.factor_checks}
    assert checks[gap_id].outcome == JudgeOutcome.PASS


def test_legacy_group_owns_conversation_and_quality():
    """conversation and quality parse only under ``tau2 judges legacy`` —
    demoted verbs keep their full flag surface but leave the top level."""
    import argparse

    from tau2.judges.cli import add_judges_args

    parser = argparse.ArgumentParser(prog="tau2 judges")
    add_judges_args(parser)
    for argv in (
        ["legacy", "quality", "r", "--llm-judge", "--model", "m"],
        ["legacy", "conversation", "r", "--out", "conversation.json", "--limit", "3"],
    ):
        args = parser.parse_args(argv)
        assert callable(args.func)
    for argv in (["quality", "r"], ["conversation", "r", "--out", "c.json"]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["tau-multi-call-validation", "root"],
        ["tau-multi-speech-fidelity-validation", "root"],
        ["tau-multi-hi-gender", "validate", "root"],
        ["tau-multi-naturalness", "evaluate-validation", "root"],
        ["tau-multi-naturalness", "validate", "root"],
    ],
)
def test_obsolete_frozen_package_commands_are_not_registered(argv):
    import argparse

    from tau2.judges.cli import add_judges_args

    parser = argparse.ArgumentParser(prog="tau2 judges")
    add_judges_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_canonical_tau_multi_validation_command_is_registered():
    import argparse

    from tau2.judges.cli import add_judges_args

    parser = argparse.ArgumentParser(prog="tau2 judges")
    add_judges_args(parser)
    args = parser.parse_args(["tau-multi-validation", "archive"])
    assert args.func.__name__ == "validate_tau_multi_validation_archive"


def test_export_skips_bad_sim_and_reports_it(tmp_path, monkeypatch):
    """One failing sim is skipped, never aborting the batch; the healthy
    sims' verdicts still export."""
    gap_id = sorted(judge_factor_ids("hi"))[0]
    sims = [
        hi_sim("bad", "t1", checks=hi_factor_checks(gap_factor=gap_id)),
        hi_sim("good", "t2"),
    ]
    path = make_hi_results(tmp_path, sims, format="dir")

    def attach(
        sim, task, settings=None, agent_provider=None, agent_voice=None, domain=None
    ):
        raise RuntimeError("judge exploded")

    import tau2.judges.attach as attach_mod

    monkeypatch.setattr(attach_mod, "attach_nativeness", attach)

    out = tmp_path / "export"
    run_judges_export(_args([path], out=out))
    good_rows = [
        json.loads(line)
        for line in (out / "nativeness_verdicts.jsonl").read_text().splitlines()
    ]
    assert {r["sim_id"] for r in good_rows} == {"good"}
