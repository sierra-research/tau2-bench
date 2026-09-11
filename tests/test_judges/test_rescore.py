# Copyright Sierra
"""``tau2 evaluate-trajs`` routes through judges.export: reuse of complete
stored verdicts, gap-filling, and persistence back in the original format."""

from tau2.data_model.simulation import Results, Score
from tau2.judges.export import judge_factor_ids
from tau2.scripts.evaluate_trajectories import rescore_results
from test_judges.conftest import hi_factor_checks, hi_sim, make_hi_results


def _patch_attach(monkeypatch, behavior):
    import tau2.judges.attach as attach_mod

    monkeypatch.setattr(attach_mod, "attach_nativeness", behavior)


def test_rescore_reuses_complete_and_heals_gaps(tmp_path, monkeypatch):
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

    results, stats = rescore_results(path, scores={Score.NATIVENESS})
    assert judged_ids == ["gappy"]
    assert (stats.judged, stats.reused) == (1, 1)

    # Healed verdicts were persisted back in place, in dir format.
    assert (path / "results.json").exists()
    stored = {s.id: s for s in Results.load(path).simulations}
    gappy_checks = {c.id: c for c in stored["gappy"].nativeness_info.factor_checks}
    assert gappy_checks[gap_id].outcome.value == "pass"
    assert len(results.simulations) == 2


def test_delivery_only_rescore_never_touches_nativeness(tmp_path, monkeypatch):
    """--scores delivery must not re-judge (or pay for) incomplete nativeness
    verdicts: the unselected axis streams through untouched."""
    gap_id = sorted(judge_factor_ids("hi"))[0]
    sims = [hi_sim("gappy", "t1", checks=hi_factor_checks(gap_factor=gap_id))]
    path = make_hi_results(tmp_path, sims)
    judged_ids: list[str] = []

    def attach(
        sim, task, settings=None, agent_provider=None, agent_voice=None, domain=None
    ):
        judged_ids.append(sim.id)

    _patch_attach(monkeypatch, attach)

    _, stats = rescore_results(path, scores={Score.DELIVERY})
    assert judged_ids == []
    assert stats.judged == 0

    stored = {s.id: s for s in Results.load(path).simulations}
    gappy_checks = {c.id: c for c in stored["gappy"].nativeness_info.factor_checks}
    assert gappy_checks[gap_id].outcome.value != "pass"


def test_delivery_only_rescore_skips_pack_guard(tmp_path, monkeypatch):
    """An unregistered nativeness language must not abort a delivery-only
    rescore: nativeness is never rebuilt, and delivery falls back to its
    universal rubric."""
    sims = [hi_sim("orphan", "t1", language="xx")]
    path = make_hi_results(tmp_path, sims)
    _patch_attach(monkeypatch, lambda *a, **k: None)

    _, stats = rescore_results(path, scores={Score.DELIVERY})
    assert stats.judged == 0
    assert Results.load(path).simulations[0].id == "orphan"


def test_rescore_output_redirect_preserves_input(tmp_path, monkeypatch):
    path = make_hi_results(tmp_path, [hi_sim("complete", "t1")], format="json")
    _patch_attach(monkeypatch, lambda *a, **k: None)
    out = tmp_path / "updated_run.json"
    rescore_results(path, scores={Score.NATIVENESS}, output=out)
    assert out.exists()
    assert Results.load(out).simulations[0].id == "complete"
