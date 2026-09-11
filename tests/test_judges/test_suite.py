# Copyright Sierra
"""``tau2 judges suite`` — CLI surface, phase orchestration, provenance.

Real Results on tmp disk, judge calls mocked at the ``tau2.judges.attach``
seam (same discipline as test_cli.py): the paths under test are family
selection, trial filtering, reuse gating, sharded persistence, and the
provenance record — never LLM spend.
"""

import argparse
import json
from pathlib import Path

import pytest

from tau2.data_model.message import Tick
from tau2.data_model.simulation import (
    DeliveryInfo,
    DeliveryUtteranceResult,
    JudgeOutcome,
    QualityFactorCheck,
    QualityInfo,
)
from tau2.judges.cli import add_judges_args
from tau2.judges.delivery.judge import DELIVERY_JUDGE_PROMPT_VERSION
from tau2.judges.export import judge_factor_ids
from tau2.judges.nativeness.judge import NATIVENESS_JUDGE_PROMPT_VERSION
from tau2.judges.quality.factors import QUALITY_FACTORS, QUALITY_RUBRIC_VERSION
from tau2.judges.quality.judge import QUALITY_JUDGE_PROMPT_VERSION
from tau2.judges.suite import (
    ALL_FAMILIES,
    SuiteConfig,
    SuiteFamily,
    SuiteProvenance,
    plan_phase_shards,
    provenance_path,
    run_suite,
)
from tau2.metrics.interaction_quality import QUALITY_METRICS_VERSION
from test_judges.conftest import hi_factor_checks, hi_sim, make_hi_results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tau2 judges")
    add_judges_args(parser)
    return parser


def _config(results, **over) -> SuiteConfig:
    """A single-process, single-thread suite config (mock-friendly: worker
    processes would not see monkeypatches)."""
    base = dict(
        results=[Path(p) for p in results],
        concurrency=1,
        text_processes=1,
        audio_processes=1,
    )
    base.update(over)
    return SuiteConfig(**base)


def _complete_quality_info() -> QualityInfo:
    """Stored quality verdicts complete at the CURRENT versions (reusable)."""
    checks = [
        QualityFactorCheck(
            id=factor.id,
            category=factor.category,
            severity=factor.severity,
            evaluator=factor.evaluator,
            outcome=JudgeOutcome.PASS,
            judge_prompt_version=(
                QUALITY_JUDGE_PROMPT_VERSION
                if factor.evaluator != "deterministic"
                else None
            ),
        )
        for factor in QUALITY_FACTORS
    ]
    return QualityInfo(
        score=1.0,
        factor_checks=checks,
        rubric_version=QUALITY_RUBRIC_VERSION,
        metrics_version=QUALITY_METRICS_VERSION,
    )


def _complete_delivery_info() -> DeliveryInfo:
    """Stored delivery verdicts complete at the CURRENT prompt (reusable)."""
    return DeliveryInfo(
        judge_prompt_version=DELIVERY_JUDGE_PROMPT_VERSION,
        utterance_results=[
            DeliveryUtteranceResult(utterance_idx=0, outcome=JudgeOutcome.PASS)
        ],
    )


@pytest.fixture
def patch_attach_nativeness(monkeypatch):
    """Mock the nativeness judge at the attach seam; records the sims judged."""
    calls: list[str] = []

    def attach(sim, task, settings=None, agent_provider=None, domain=None, **_kw):
        from tau2.data_model.simulation import NativenessInfo

        calls.append(sim.id)
        sim.nativeness_info = NativenessInfo(
            score=1.0,
            factor_checks=hi_factor_checks(),
            language="hi",
            script="deva",
            judge_model="fresh-judge",
        )

    import tau2.judges.attach as attach_mod

    monkeypatch.setattr(attach_mod, "attach_nativeness", attach)
    return calls


@pytest.fixture
def patch_attach_quality(monkeypatch):
    """Mock the quality rubric at the attach seam; records the sims judged."""
    calls: list[str] = []

    def attach(sim, task, *, domain=None, settings=None):
        calls.append(sim.id)
        sim.quality_info = _complete_quality_info()

    import tau2.judges.attach as attach_mod

    monkeypatch.setattr(attach_mod, "attach_quality", attach)
    return calls


@pytest.fixture
def patch_attach_delivery(monkeypatch):
    """Mock the delivery judge at the disk-attach seam; records sims judged."""
    calls: list[str] = []

    def attach(sim, results_dir, *, settings=None):
        assert settings is not None and settings.sample_rate == 1.0
        calls.append(sim.id)
        sim.delivery_info = _complete_delivery_info()

    import tau2.judges.attach as attach_mod

    monkeypatch.setattr(attach_mod, "attach_delivery_from_disk", attach)
    return calls


# --- CLI surface ---------------------------------------------------------------


def test_suite_flag_defaults():
    args = _parser().parse_args(["suite", "some/results"])
    assert args.results == ["some/results"]
    assert args.concurrency == 10
    assert args.text_processes == 8
    assert args.audio_processes == 5
    assert args.rejudge is False
    assert args.only is None
    assert args.trials == "0"
    assert callable(args.func)


def test_suite_only_parsing():
    from tau2.judges.cli import parse_suite_families

    assert parse_suite_families(None) == ALL_FAMILIES
    assert parse_suite_families("nativeness,delivery") == frozenset(
        {SuiteFamily.NATIVENESS, SuiteFamily.DELIVERY}
    )
    assert parse_suite_families("quality") == frozenset({SuiteFamily.QUALITY})
    for bad in ("conversation", "nativeness,eva", "", " , "):
        with pytest.raises(SystemExit):
            parse_suite_families(bad)


def test_suite_trials_parsing():
    from tau2.judges.cli import parse_suite_trials

    assert parse_suite_trials("0") == frozenset({0})
    assert parse_suite_trials("0,1") == frozenset({0, 1})
    assert parse_suite_trials("all") is None
    assert parse_suite_trials("ALL") is None
    for bad in ("one", "0;1", ""):
        with pytest.raises(SystemExit):
            parse_suite_trials(bad)


# --- phase planning ------------------------------------------------------------


def test_json_root_always_runs_as_a_single_shard(tmp_path):
    """A monolithic root cannot take disjoint per-sim writes; a dir root is
    split into at most min(processes, sims) disjoint shards."""
    json_path = make_hi_results(
        tmp_path, [hi_sim("s1", "t1")], format="json", name="mono"
    )
    dir_path = make_hi_results(
        tmp_path,
        [hi_sim(f"s{i}", f"t{i}") for i in range(3)],
        format="dir",
        name="sharded",
    )
    config = _config([json_path, dir_path], text_processes=8)
    shards = plan_phase_shards(config, "text", [SuiteFamily.NATIVENESS])
    by_root = {}
    for shard in shards:
        by_root.setdefault(shard.root, []).append(shard)
    assert len(by_root[str(json_path)]) == 1
    assert len(by_root[str(dir_path)]) == 3  # capped at the sim count
    assert {s.shard_index for s in by_root[str(dir_path)]} == {0, 1, 2}


# --- orchestration (mocked judges) ----------------------------------------------


def test_suite_nativeness_fills_gaps_and_reuses(tmp_path, patch_attach_nativeness):
    gap_id = sorted(judge_factor_ids("hi"))[0]
    sims = [
        hi_sim("gappy", "t1", checks=hi_factor_checks(gap_factor=gap_id)),
        hi_sim("complete", "t2"),
    ]
    path = make_hi_results(tmp_path, sims, format="dir")

    records = run_suite(_config([path], only=frozenset({SuiteFamily.NATIVENESS})))

    assert patch_attach_nativeness == ["gappy"]
    (record,) = records
    touches = record.families["nativeness"].touches
    assert touches.judged == ["gappy"]
    assert touches.reused == ["complete"]
    # Fresh verdicts persisted back into the sim's own dir-format file.
    assert "fresh-judge" in (path / "simulations" / "gappy.json").read_text()
    assert "stored-judge" in (path / "simulations" / "complete.json").read_text()


def test_suite_nativeness_skips_en_silently(tmp_path, patch_attach_nativeness):
    """en defines zero nativeness judge factors: the family is skipped without
    a judge call, alongside sims with no resolvable language."""
    en = hi_sim("en-sim", "t1", language="en")
    no_lang = hi_sim("no-lang", "t2")
    no_lang.nativeness_info = None  # no stored info, no speech environment
    path = make_hi_results(tmp_path, [en, no_lang], format="dir")

    records = run_suite(_config([path], only=frozenset({SuiteFamily.NATIVENESS})))

    assert patch_attach_nativeness == []  # the judge never ran
    touches = records[0].families["nativeness"].touches
    assert touches.skipped == ["en-sim", "no-lang"]
    assert touches.judged == [] and touches.reused == []


def test_suite_quality_reuse_is_version_aware(tmp_path, patch_attach_quality):
    current = hi_sim("current", "t1")
    current.quality_info = _complete_quality_info()
    stale = hi_sim("stale", "t2")
    stale.quality_info = _complete_quality_info()
    stale.quality_info.rubric_version = "quality-rubric-v0"
    missing = hi_sim("missing", "t3")  # no quality_info at all
    path = make_hi_results(tmp_path, [current, stale, missing], format="dir")

    records = run_suite(_config([path], only=frozenset({SuiteFamily.QUALITY})))

    assert sorted(patch_attach_quality) == ["missing", "stale"]
    touches = records[0].families["quality"].touches
    assert touches.judged == ["missing", "stale"]
    assert touches.reused == ["current"]


def test_suite_delivery_judges_every_audio_call(tmp_path, patch_attach_delivery):
    """Tick-less sims are skipped silently; audio sims are judged (no sampling
    cap — asserted inside the attach mock) or reused when current."""
    text_sim = hi_sim("text", "t1")
    fresh = hi_sim("fresh", "t2")
    fresh.ticks = [Tick(tick_id=0, timestamp="2026-01-01T00:00:00")]
    done = hi_sim("done", "t3")
    done.ticks = [Tick(tick_id=0, timestamp="2026-01-01T00:00:00")]
    done.delivery_info = _complete_delivery_info()
    path = make_hi_results(tmp_path, [text_sim, fresh, done], format="dir")

    records = run_suite(_config([path], only=frozenset({SuiteFamily.DELIVERY})))

    assert patch_attach_delivery == ["fresh"]
    touches = records[0].families["delivery"].touches
    assert touches.judged == ["fresh"]
    assert touches.reused == ["done"]
    assert touches.skipped == ["text"]


def test_suite_rejudge_forces_full_rejudge(tmp_path, patch_attach_nativeness):
    path = make_hi_results(tmp_path, [hi_sim("s1", "t1")], format="dir")
    run_suite(_config([path], only=frozenset({SuiteFamily.NATIVENESS}), rejudge=True))
    assert patch_attach_nativeness == ["s1"]


def test_suite_json_root_round_trips(tmp_path, patch_attach_nativeness):
    """A monolithic root is persisted via a whole-root save in its format."""
    gap_id = sorted(judge_factor_ids("hi"))[0]
    path = make_hi_results(
        tmp_path,
        [hi_sim("s1", "t1", checks=hi_factor_checks(gap_factor=gap_id))],
        format="json",
    )
    run_suite(_config([path], only=frozenset({SuiteFamily.NATIVENESS})))
    assert patch_attach_nativeness == ["s1"]
    assert "fresh-judge" in path.read_text()
    assert not (path.parent / "simulations").exists()  # format preserved


# --- trial filter ----------------------------------------------------------------


def test_suite_defaults_to_trial_zero_only(tmp_path, patch_attach_nativeness):
    gap_id = sorted(judge_factor_ids("hi"))[0]
    t0 = hi_sim("t0-sim", "t1", checks=hi_factor_checks(gap_factor=gap_id))
    t1 = hi_sim("t1-sim", "t1", checks=hi_factor_checks(gap_factor=gap_id), trial=1)
    path = make_hi_results(tmp_path, [t0, t1], format="dir", num_trials=2)

    records = run_suite(_config([path], only=frozenset({SuiteFamily.NATIVENESS})))

    assert patch_attach_nativeness == ["t0-sim"]
    nativeness = records[0].families["nativeness"]
    assert nativeness.touches.judged == ["t0-sim"]
    # unselected trial: not judged at all
    assert nativeness.touches.skipped == ["t1-sim"]
    assert nativeness.invocation.trials == [0]


def test_suite_trials_all_judges_every_trial(tmp_path, patch_attach_nativeness):
    gap_id = sorted(judge_factor_ids("hi"))[0]
    t0 = hi_sim("t0-sim", "t1", checks=hi_factor_checks(gap_factor=gap_id))
    t1 = hi_sim("t1-sim", "t1", checks=hi_factor_checks(gap_factor=gap_id), trial=1)
    path = make_hi_results(tmp_path, [t0, t1], format="dir", num_trials=2)

    records = run_suite(
        _config([path], only=frozenset({SuiteFamily.NATIVENESS}), trials=None)
    )

    assert sorted(patch_attach_nativeness) == ["t0-sim", "t1-sim"]
    assert records[0].families["nativeness"].touches.judged == ["t0-sim", "t1-sim"]
    assert records[0].families["nativeness"].invocation.trials is None


# --- provenance ------------------------------------------------------------------


def test_suite_provenance_record_shape(
    tmp_path, patch_attach_nativeness, patch_attach_quality, patch_attach_delivery
):
    gap_id = sorted(judge_factor_ids("hi"))[0]
    path = make_hi_results(
        tmp_path,
        [hi_sim("s1", "t1", checks=hi_factor_checks(gap_factor=gap_id))],
        format="dir",
    )

    (record,) = run_suite(_config([path]))

    out = provenance_path(path)
    assert out == path / "judge_suite_provenance.json"
    stored = SuiteProvenance.model_validate(json.loads(out.read_text()))
    assert stored == record
    assert stored.schema_version == "judge-suite-provenance-v2"
    assert stored.results_path == str(path)
    assert set(stored.families) == {"nativeness", "quality", "delivery"}
    nativeness = stored.families["nativeness"]
    assert nativeness.judge_prompt_version == NATIVENESS_JUDGE_PROMPT_VERSION
    quality = stored.families["quality"]
    assert quality.judge_prompt_version == QUALITY_JUDGE_PROMPT_VERSION
    assert quality.rubric_version == QUALITY_RUBRIC_VERSION
    assert quality.metrics_version == QUALITY_METRICS_VERSION
    delivery = stored.families["delivery"]
    assert delivery.judge_prompt_version == DELIVERY_JUDGE_PROMPT_VERSION
    # Every family carries the invocation that ran it.
    for family in stored.families.values():
        assert family.invocation.created_at == stored.updated_at
        assert family.invocation.rejudge is False
        assert family.invocation.trials == [0]
        assert family.invocation.concurrency == 1
    # Every family accounted for the one sim.
    assert nativeness.touches.judged == ["s1"]
    assert quality.touches.judged == ["s1"]
    assert delivery.touches.skipped == ["s1"]  # no ticks: no audio to judge


def test_suite_provenance_merges_disjoint_only_invocations(
    tmp_path, patch_attach_nativeness, patch_attach_quality, patch_attach_delivery
):
    """A later --only invocation merges into the stored record instead of
    overwriting it: both invocations' families survive, newest per family."""
    gap_id = sorted(judge_factor_ids("hi"))[0]
    sim = hi_sim("s1", "t1", checks=hi_factor_checks(gap_factor=gap_id))
    sim.ticks = [Tick(tick_id=0, timestamp="2026-01-01T00:00:00")]
    path = make_hi_results(tmp_path, [sim], format="dir")

    (first,) = run_suite(
        _config([path], only=frozenset({SuiteFamily.NATIVENESS, SuiteFamily.QUALITY}))
    )
    (second,) = run_suite(
        _config([path], only=frozenset({SuiteFamily.DELIVERY}), trials=None)
    )

    stored = SuiteProvenance.model_validate(
        json.loads(provenance_path(path).read_text())
    )
    assert stored == second
    assert set(stored.families) == {"nativeness", "quality", "delivery"}
    # The text families' entries are the FIRST invocation's, untouched.
    assert stored.families["nativeness"] == first.families["nativeness"]
    assert stored.families["quality"] == first.families["quality"]
    assert stored.families["nativeness"].invocation.trials == [0]
    # The delivery entry carries the SECOND invocation's metadata.
    delivery = stored.families["delivery"]
    assert delivery.invocation.created_at == stored.updated_at
    assert delivery.invocation.created_at != first.updated_at
    assert delivery.invocation.trials is None
    assert delivery.touches.judged == ["s1"]


def test_suite_provenance_rerun_replaces_a_family_entry(
    tmp_path, patch_attach_nativeness
):
    """Re-running a family keeps exactly one entry for it — the newest."""
    path = make_hi_results(tmp_path, [hi_sim("s1", "t1")], format="dir")
    only = frozenset({SuiteFamily.NATIVENESS})

    (first,) = run_suite(_config([path], only=only, rejudge=True))
    (second,) = run_suite(_config([path], only=only, rejudge=True))

    stored = SuiteProvenance.model_validate(
        json.loads(provenance_path(path).read_text())
    )
    assert set(stored.families) == {"nativeness"}
    assert stored.families["nativeness"] == second.families["nativeness"]
    assert (
        stored.families["nativeness"].invocation.created_at
        != first.families["nativeness"].invocation.created_at
    )


def test_suite_provenance_upgrades_v1_record_on_merge(tmp_path, patch_attach_delivery):
    """A stored v1 record (flat invocation fields, as written before the
    merge-on-write fix) is upgraded on read: its families survive a later
    partial invocation, each carrying the v1 top-level invocation metadata."""
    sim = hi_sim("s1", "t1")
    sim.ticks = [Tick(tick_id=0, timestamp="2026-01-01T00:00:00")]
    path = make_hi_results(tmp_path, [sim], format="dir")
    v1 = {
        "schema_version": "judge-suite-provenance-v1",
        "created_at": "2026-08-27T00:00:00+00:00",
        "results_path": str(path),
        "concurrency": 10,
        "text_processes": 8,
        "audio_processes": 5,
        "rejudge": False,
        "only": ["nativeness", "quality"],
        "trials": [0],
        "families": {
            "nativeness": {
                "judge_model": "stored-judge",
                "judge_prompt_version": "nativeness-judge-v0",
                "touches": {"judged": ["s1"]},
            },
        },
    }
    provenance_path(path).write_text(json.dumps(v1))

    run_suite(_config([path], only=frozenset({SuiteFamily.DELIVERY})))

    stored = SuiteProvenance.model_validate(
        json.loads(provenance_path(path).read_text())
    )
    assert set(stored.families) == {"nativeness", "delivery"}
    nativeness = stored.families["nativeness"]
    assert nativeness.judge_model == "stored-judge"
    assert nativeness.judge_prompt_version == "nativeness-judge-v0"
    assert nativeness.touches.judged == ["s1"]
    assert nativeness.invocation.created_at == "2026-08-27T00:00:00+00:00"
    assert nativeness.invocation.concurrency == 10
    assert nativeness.invocation.trials == [0]
    assert stored.families["delivery"].touches.judged == ["s1"]


def test_suite_provenance_unreadable_record_never_aborts_the_write(
    tmp_path, patch_attach_nativeness
):
    """A corrupt stored record is logged and dropped — the current
    invocation's provenance still lands."""
    path = make_hi_results(tmp_path, [hi_sim("s1", "t1")], format="dir")
    provenance_path(path).write_text("not json {")

    (record,) = run_suite(
        _config([path], only=frozenset({SuiteFamily.NATIVENESS}), rejudge=True)
    )

    stored = SuiteProvenance.model_validate(
        json.loads(provenance_path(path).read_text())
    )
    assert stored == record
    assert set(stored.families) == {"nativeness"}
