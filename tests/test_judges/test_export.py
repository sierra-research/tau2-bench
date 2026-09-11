# Copyright Sierra
"""judges.export: verdict records, completeness guard, bounded streaming map,
reuse gating / skip policy of the judged-sim stream."""

import json

import pytest

from tau2.data_model.message import AssistantMessage, Tick, ToolCall, UserMessage
from tau2.data_model.simulation import (
    JudgeOutcome,
    NativenessFactorCheck,
    NativenessInfo,
    SimulationRun,
)
from tau2.evaluator.evaluator_communicate import FullDuplexCommunicateEvaluator
from tau2.judges.export import (
    JudgeStreamStats,
    NativenessVerdict,
    UnregisteredLanguageError,
    bounded_map,
    has_complete_nativeness_verdicts,
    iter_judged_sims_detailed,
    judge_factor_ids,
    nativeness_verdicts,
    require_registered_language,
    write_verdicts_jsonl,
)
from tau2.judges.nativeness.factors import judge_factors_for
from tau2.metrics.interaction_quality import agent_utterance_tick_spans
from test_judges.conftest import hi_factor_checks, hi_sim, make_hi_results, make_task

PASS = JudgeOutcome.PASS
FAIL = JudgeOutcome.FAIL


def _sim(checks, text="आपका कोड JMO1MG है", judge_prompt_version=None):
    from tau2.judges.nativeness.harness import NATIVENESS_RUBRIC_VERSION
    from tau2.judges.nativeness.judge import NATIVENESS_JUDGE_PROMPT_VERSION

    return SimulationRun(
        id="s1",
        task_id="t1",
        start_time="x",
        end_time="y",
        duration=1.0,
        termination_reason="agent_stop",
        messages=[AssistantMessage(role="assistant", content=text)],
        nativeness_info=NativenessInfo(
            score=None,
            factor_checks=checks,
            language="hi",
            script="deva",
            rubric_version=NATIVENESS_RUBRIC_VERSION,
            judge_prompt_version=judge_prompt_version
            or NATIVENESS_JUDGE_PROMPT_VERSION,
        ),
    )


def _check(fid, cat, sev, outcome, evidence=None, quote=None):
    return NativenessFactorCheck(
        id=fid,
        category=cat,
        severity=sev,
        outcome=outcome,
        evidence=evidence,
        quote=quote,
    )


# --- bounded_map -------------------------------------------------------------


def test_bounded_map_processes_all_items():
    # Completion order may vary; every item must be processed exactly once.
    out = sorted(bounded_map(lambda x: x * 2, range(25), concurrency=5))
    assert out == [x * 2 for x in range(25)]


def test_bounded_map_concurrency_one():
    assert list(bounded_map(lambda x: x, [1, 2, 3], concurrency=1)) == [1, 2, 3]


# --- completeness guard (drives --reuse-existing) -----------------------------


def test_has_complete_nativeness_verdicts_guards_reuse():
    hi_judge_ids = [
        f.id for f in judge_factors_for("hi") if f.type == "judge" and f.enabled
    ]
    assert hi_judge_ids  # sanity: hi defines judge factors

    # All judge factors present & non-DEFERRED -> complete (safe to reuse).
    complete = _sim([_check(fid, "register", 3, PASS) for fid in hi_judge_ids])
    assert has_complete_nativeness_verdicts(complete) is True

    # No judge rows at all, but the language defines judge factors -> NOT complete
    # (the vacuous-truth hole: empty judge_checks must not count as judged).
    assert has_complete_nativeness_verdicts(_sim([])) is False

    # A judge factor recorded DEFERRED (judge was off) -> NOT complete.
    deferred = _sim(
        [_check(fid, "register", 3, JudgeOutcome.DEFERRED) for fid in hi_judge_ids]
    )
    assert has_complete_nativeness_verdicts(deferred) is False

    # A judge factor recorded ERROR (call failed, no verdict to reuse) -> NOT
    # complete, so gap-filling can heal it instead of treating it as done.
    errored = _sim(
        [_check(hi_judge_ids[0], "register", 3, JudgeOutcome.ERROR, "api down")]
        + [_check(fid, "register", 3, PASS) for fid in hi_judge_ids[1:]]
    )
    assert has_complete_nativeness_verdicts(errored) is False

    # No nativeness_info at all -> NOT complete.
    bare = _sim([])
    bare.nativeness_info = None
    assert has_complete_nativeness_verdicts(bare) is False

    # Complete verdicts from an OLDER judge prompt -> stale, NOT reusable.
    stale = _sim(
        [_check(fid, "register", 3, PASS) for fid in hi_judge_ids],
        judge_prompt_version="v1-ancient",
    )
    assert has_complete_nativeness_verdicts(stale) is False

    # v25 predates interruption metadata at the hybrid-precheck boundary; its
    # deterministic short-circuits must not be silently reused under v26.
    stale_rubric = _sim([_check(fid, "register", 3, PASS) for fid in hi_judge_ids])
    stale_rubric.nativeness_info.rubric_version = "nativeness-rubric-v25"
    assert has_complete_nativeness_verdicts(stale_rubric) is False

    # All-NO_OPPORTUNITY (judge never ran: nothing to judge) -> reusable
    # regardless of version: no prompt was involved, so none can be stale,
    # and re-judging would rebuild the same rows forever (bugbot on #962).
    no_opp = _sim(
        [
            _check(fid, "register", 1, JudgeOutcome.NO_OPPORTUNITY)
            for fid in hi_judge_ids
        ]
    )
    no_opp.nativeness_info.judge_prompt_version = None
    assert has_complete_nativeness_verdicts(no_opp) is True


def test_adopt_stored_verdicts_copies_current_version_by_sim_id(tmp_path):
    """A corpus whose sims were freshly judged elsewhere (same sim ids, e.g. a
    materialized recall subset) inherits those current-version verdicts via
    adopt_stored_verdicts instead of re-paying the judge; stale sims without a
    source counterpart stay stale, and a second pass is a no-op."""
    import fixtures_runs

    from tau2.data_model.simulation import Results
    from tau2.judges.export import adopt_stored_verdicts
    from tau2.judges.nativeness.judge import NATIVENESS_JUDGE_PROMPT_VERSION

    def sim(sim_id, *, version, judge_model):
        return fixtures_runs.hi_sim(
            sim_id,
            "t1",
            checks=hi_factor_checks(),
            judge_model=judge_model,
            judge_prompt_version=version,
        )

    dst = make_hi_results(
        tmp_path,
        [
            sim("s1", version="v1-ancient", judge_model="stale-judge"),
            sim("s2", version="v1-ancient", judge_model="stale-judge"),
        ],
        name="dst_run",
    )
    src_root = tmp_path / "recall_tree"
    src_root.mkdir()
    make_hi_results(
        src_root,
        [sim("s1", version=NATIVENESS_JUDGE_PROMPT_VERSION, judge_model="fresh")],
        name="cell_a",
    )

    stats = adopt_stored_verdicts(src_root, dst)
    assert stats.dest_sims == 2 and stats.matched == 1
    assert stats.nativeness_adopted == 1 and stats.delivery_adopted == 0

    by_id = {s.id: s for s in Results.load(dst).simulations}
    assert (
        by_id["s1"].nativeness_info.judge_prompt_version
        == NATIVENESS_JUDGE_PROMPT_VERSION
    )
    assert by_id["s1"].nativeness_info.judge_model == "fresh"
    assert by_id["s2"].nativeness_info.judge_prompt_version == "v1-ancient"

    again = adopt_stored_verdicts(src_root, dst)
    assert again.nativeness_adopted == 0 and again.delivery_adopted == 0


def test_judge_factor_ids_excludes_disabled(monkeypatch):
    """A disabled judge factor must NOT be in the expected set: the harness
    skips it and never writes a check, so requiring it would keep
    has_complete_nativeness_verdicts permanently False and make --reuse-existing
    re-invoke the judge forever (bugbot #342)."""
    from types import SimpleNamespace

    import tau2.judges.export as export

    factors = [
        SimpleNamespace(id="on_factor", type="judge", enabled=True),
        SimpleNamespace(id="off_factor", type="judge", enabled=False),
        SimpleNamespace(id="det_factor", type="deterministic", enabled=True),
    ]
    monkeypatch.setattr(export, "judge_factors_for", lambda language: factors)
    assert judge_factor_ids("xx") == {"on_factor"}


# --- verdict records ----------------------------------------------------------


def test_nativeness_verdicts_one_per_defined_factor():
    hi_judge_ids = sorted(judge_factor_ids("hi"))
    fid = hi_judge_ids[0]
    sim = _sim([_check(fid, "register", 3, FAIL, "drifted to तुम", "तुम कहाँ")])
    records = nativeness_verdicts(sim)
    # One record per judge factor the language DEFINES, not per stored check —
    # completeness gaps stay visible as outcome=None.
    assert [r.factor_id for r in records] == hi_judge_ids
    by_id = {r.factor_id: r for r in records}
    assert by_id[fid].outcome == FAIL
    assert by_id[fid].reasoning == "drifted to तुम"
    assert by_id[fid].quote == "तुम कहाँ"
    assert by_id[fid].transcript.startswith("agent: ")
    for other in hi_judge_ids[1:]:
        assert by_id[other].outcome is None  # no stored verdict, never invented


def test_nativeness_verdicts_empty_without_info():
    sim = _sim([])
    sim.nativeness_info = None
    assert nativeness_verdicts(sim) == []


def test_write_verdicts_jsonl_roundtrip(tmp_path):
    hi_judge_ids = sorted(judge_factor_ids("hi"))
    sim = _sim([_check(hi_judge_ids[0], "register", 3, PASS)])
    records = nativeness_verdicts(sim)
    path = tmp_path / "out" / "nativeness_verdicts.jsonl"
    n = write_verdicts_jsonl(records, path)
    assert n == len(records)
    lines = path.read_text().splitlines()
    assert len(lines) == n
    parsed = [NativenessVerdict.model_validate(json.loads(line)) for line in lines]
    assert parsed == records


# --- unregistered-language guard ---------------------------------------------


def _unregistered_sim():
    """A sim whose stored verdicts reference judge factors of a language with
    no registered pack (the missing-data-dir hazard)."""
    sim = _sim([_check("register_formality", "register", 3, PASS)])
    sim.nativeness_info.language = "xx"
    return sim


def test_require_registered_language_fails_loudly_for_unknown_pack():
    with pytest.raises(UnregisteredLanguageError) as exc_info:
        require_registered_language(_unregistered_sim())
    msg = str(exc_info.value)
    assert "'xx'" in msg  # names the language
    assert "multilingual" in msg  # names the pack data dir


def test_require_registered_language_passes_for_registered_and_na_sims():
    require_registered_language(_sim([]))  # hi is registered
    bare = _sim([])
    bare.nativeness_info = None
    require_registered_language(bare)  # no nativeness info -> nothing to guard


def test_iter_judged_sims_aborts_on_unregistered_language(tmp_path):
    """The missing-pack config error must abort the stream (before the reuse
    decision, which would otherwise be vacuously 'complete' and export empty
    verdicts) — never be swallowed as a per-sim skip."""
    path = make_hi_results(
        tmp_path,
        [hi_sim("s1", "t1", language="xx")],
        tasks=[make_task("t1")],
    )
    with pytest.raises(UnregisteredLanguageError):
        list(iter_judged_sims_detailed(path, reuse_existing=True, concurrency=1))
    # Same through the concurrent bounded_map path.
    with pytest.raises(UnregisteredLanguageError):
        list(iter_judged_sims_detailed(path, reuse_existing=True, concurrency=4))


# --- iter_judged_sims_detailed: reuse gating / skip policy ---------------------


def _patch_attach(monkeypatch, behavior):
    import tau2.judges.attach as attach_mod

    monkeypatch.setattr(attach_mod, "attach_nativeness", behavior)


def test_iter_judged_sims_reuse_gating(tmp_path, monkeypatch):
    """Complete sims are reused (judge never invoked); gappy sims are judged;
    reuse_existing=False forces a full re-judge. Stats count each bucket."""
    gap_id = sorted(judge_factor_ids("hi"))[0]
    sims = [
        hi_sim("complete", "t1"),
        hi_sim("gappy", "t2", checks=hi_factor_checks(gap_factor=gap_id)),
    ]
    path = make_hi_results(tmp_path, sims)
    judged_ids: list[str] = []

    def attach(
        sim, task, settings=None, agent_provider=None, agent_voice=None, domain=None
    ):
        judged_ids.append(sim.id)
        sim.nativeness_info.factor_checks = hi_factor_checks()

    _patch_attach(monkeypatch, attach)

    stats = JudgeStreamStats()
    out = list(iter_judged_sims_detailed(path, concurrency=1, stats=stats))
    assert judged_ids == ["gappy"]
    assert {(j.sim.id, j.was_judged) for j in out} == {
        ("complete", False),
        ("gappy", True),
    }
    assert (stats.judged, stats.reused, stats.skipped) == (1, 1, 0)

    # Forced full re-judge.
    judged_ids.clear()
    stats2 = JudgeStreamStats()
    list(
        iter_judged_sims_detailed(
            path, reuse_existing=False, concurrency=1, stats=stats2
        )
    )
    assert sorted(judged_ids) == ["complete", "gappy"]
    assert (stats2.judged, stats2.reused) == (2, 0)


def test_iter_judged_sims_missing_task_is_skipped(tmp_path, monkeypatch):
    gap_id = sorted(judge_factor_ids("hi"))[0]
    path = make_hi_results(
        tmp_path,
        [hi_sim("orphan", "missing-task", checks=hi_factor_checks(gap_factor=gap_id))],
        tasks=[make_task("t1")],
    )
    _patch_attach(
        monkeypatch, lambda *a, **k: pytest.fail("must not judge a task-less sim")
    )
    stats = JudgeStreamStats()
    out = list(iter_judged_sims_detailed(path, concurrency=1, stats=stats))
    assert out == []
    assert stats.skipped == 1 and stats.skipped_sim_ids == ["orphan"]


def test_iter_judged_sims_judge_failure_skips_not_aborts(tmp_path, monkeypatch):
    """A per-sim judge crash is logged + counted + skipped; the rest of the
    batch still streams (through the concurrent bounded_map path too)."""
    gap_id = sorted(judge_factor_ids("hi"))[0]
    sims = [
        hi_sim("boom", "t1", checks=hi_factor_checks(gap_factor=gap_id)),
        hi_sim("fine", "t2"),
    ]
    path = make_hi_results(tmp_path, sims)

    def attach(sim, task, settings=None, agent_provider=None, agent_voice=None):
        raise RuntimeError("judge exploded")

    _patch_attach(monkeypatch, attach)
    for concurrency in (1, 4):
        stats = JudgeStreamStats()
        out = list(
            iter_judged_sims_detailed(path, concurrency=concurrency, stats=stats)
        )
        assert [j.sim.id for j in out] == ["fine"]  # reused sim survives
        assert stats.skipped == 1 and stats.skipped_sim_ids == ["boom"]


def test_bounded_map_reraises_worker_exception():
    """bounded_map itself re-raises (per-sim error policy lives in judge_one);
    a raising fn must not be silently dropped."""

    def boom(x):
        raise ValueError(f"bad {x}")

    with pytest.raises(ValueError):
        list(bounded_map(boom, [1, 2, 3], concurrency=2))


# --- agent_utterance_tick_spans ---------------------------------------------


def _speech_tick(tick_id, content, utterance_ids=None):
    return Tick(
        tick_id=tick_id,
        timestamp="2026-01-01T00:00:00",
        agent_chunk=AssistantMessage(
            role="assistant", content=content, utterance_ids=utterance_ids
        ),
    )


def test_agent_utterance_tick_spans_align_with_evaluator_grouping():
    """The span helper must group EXACTLY like the delivery judge's source of
    truth (``ticks_to_message_history``), or seed anchors point at the wrong
    utterance. Covers: id-overlap merging across ticks, a tool tick in the
    middle, and id-less chunks staying separate."""
    ticks = [
        _speech_tick(0, "Hel", ["u1"]),
        _speech_tick(1, "lo ", ["u1", "u2"]),
        _speech_tick(2, "world", ["u2"]),
        Tick(
            tick_id=3,
            timestamp="2026-01-01T00:00:00",
            agent_tool_calls=[ToolCall(id="tc1", name="lookup", arguments={"q": "x"})],
        ),
        _speech_tick(4, "Bye", ["u3"]),
        _speech_tick(5, "hmm"),
        Tick(
            tick_id=6,
            timestamp="2026-01-01T00:00:00",
            user_chunk=UserMessage(role="user", content="ok"),
        ),
    ]
    sim = _sim(hi_factor_checks())
    sim.ticks = ticks

    spans = agent_utterance_tick_spans(sim)
    messages = FullDuplexCommunicateEvaluator.ticks_to_message_history(ticks)

    assert len(spans) == len(messages) == 3
    assert spans == [(0, 2), (4, 4), (5, 5)]

    # Per-utterance text parity (whitespace-insensitive: the merge may insert
    # utterance-aware spacing the raw tick concat doesn't have).
    def squash(text):
        return "".join(text.split())

    by_tick = {t.tick_id: t.agent_chunk.content for t in ticks if t.agent_chunk}
    for (start, end), message in zip(spans, messages):
        joined = "".join(by_tick[tid] for tid in sorted(by_tick) if start <= tid <= end)
        assert squash(joined) == squash(message.content)


def test_agent_utterance_tick_spans_empty_without_ticks():
    sim = _sim(hi_factor_checks())
    assert agent_utterance_tick_spans(sim) == []


def test_verdict_records_carry_bucket_and_disposition():
    """Exported nativeness verdicts are stamped with the calibration bucket."""
    records = nativeness_verdicts(_sim(hi_factor_checks()))
    by_id = {r.factor_id: r for r in records}
    assert by_id["natural_word_choice"].bucket == "phrasing"
    assert by_id["natural_word_choice"].disposition == "scored"
    assert by_id["gender_agreement"].bucket == "grammar"
    # Honorific agreement is a scored grammar factor.
    if "honorific_agreement" in by_id:
        assert by_id["honorific_agreement"].bucket == "grammar"
        assert by_id["honorific_agreement"].disposition == "scored"


def test_bucket_rollup_merges_sibling_factors_per_call():
    """A call failing one constituent fails the bucket exactly once; the
    interaction-metric factors never join a bucket."""
    from tau2.judges.export import (
        DeliveryVerdict,
        QualityVerdict,
        bucket_rollup,
        render_bucket_rollup,
    )

    def qv(sim_id, factor_id, outcome, bucket, disposition="scored"):
        return QualityVerdict(
            sim_id=sim_id,
            task_id="t",
            factor_id=factor_id,
            category="workflow",
            severity=2,
            evaluator="llm",
            outcome=outcome,
            rubric_version="r",
            metrics_version="m",
            bucket=bucket,
            disposition=disposition,
        )

    quality = [
        # call a: repetition fails, tool call passes -> redundancy FAIL once.
        qv("a", "unnecessary_repetition", FAIL, "redundancy"),
        qv("a", "unnecessary_tool_call", PASS, "redundancy"),
        # call b: both pass -> redundancy PASS.
        qv("b", "unnecessary_repetition", PASS, "redundancy"),
        qv("b", "unnecessary_tool_call", PASS, "redundancy"),
        # deterministic interaction factor: unbucketed, never rolled up.
        qv("a", "monologue", FAIL, None, disposition="shadow"),
    ]
    from tau2.data_model.simulation import DeliveryFactorCheck, DeliveryFinding

    def finding(axis):
        return DeliveryFinding(axis=axis, category="mispronunciation", severity=2)

    delivery = [
        # call a: a fidelity-axis finding fails faithfulness.
        DeliveryVerdict(
            sim_id="a",
            task_id="t",
            utterance_idx=0,
            outcome=FAIL,
            severity=2,
            findings=[finding("fidelity")],
        ),
        DeliveryVerdict(
            sim_id="a", task_id="t", utterance_idx=1, outcome=PASS, severity=0
        ),
        # call b: intonation is retired from buckets — an intonation-only
        # finding flips the stored utterance outcome to FAIL but must NOT
        # count against faithfulness.
        DeliveryVerdict(
            sim_id="b",
            task_id="t",
            utterance_idx=0,
            outcome=FAIL,
            severity=1,
            findings=[finding("intonation")],
        ),
        # call c: a failing tone_meaning_flip pack factor check IS a
        # faithfulness constituent even without an axis finding.
        DeliveryVerdict(
            sim_id="c",
            task_id="t",
            utterance_idx=0,
            outcome=FAIL,
            severity=2,
            factor_checks=[
                DeliveryFactorCheck(
                    id="tone_meaning_flip",
                    category="tone_meaning",
                    severity=3,
                    outcome=FAIL,
                )
            ],
        ),
    ]
    rows = {r.bucket_id: r for r in bucket_rollup([], quality, delivery)}

    redundancy = rows["redundancy"]
    assert redundancy.group == "efficiency"
    assert (redundancy.calls_scored, redundancy.calls_failed) == (2, 1)
    assert redundancy.fail_rate == 0.5

    # a fails (fidelity finding), b passes (intonation excluded), c fails
    # (tone_meaning_flip factor check).
    faithfulness = rows["faithfulness"]
    assert (faithfulness.calls_scored, faithfulness.calls_failed) == (3, 2)

    # Unbucketed monologue contributed to no row.
    assert all("monologue" not in r.factor_ids for r in rows.values())

    rendered = render_bucket_rollup(rows.values())
    assert "efficiency" in rendered and "redundancy" in rendered
    assert "unnecessary_repetition + unnecessary_tool_call" in rendered
