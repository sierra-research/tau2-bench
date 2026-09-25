"""D11 — filter generated trajectories to keep only reward==1 AND policy-legal.

tau2 scores outcome (reward) but NOT path-legality. This loads a tau2 Results
file, keeps simulations that (a) succeeded (reward==1, tau2's own definition)
and (b) pass the policy validator, and writes a filtered, still-tau2-compliant
Results file plus a summary. The kept set is the SFT-positive pool (PLAN_v2 D1+D11).

The legality validator is user-provided. Until `retail_policy_validator.py` is
on the path, legality is reported as "skipped" and only the reward filter applies
(loudly flagged). Adapt `_legality()` to your validator's actual API if needed.

Run:  uv run python synth/filter_legal.py path/to/results.json [--out filtered.json]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tau2.data_model.simulation import Results, SimulationRun
from tau2.metrics.agent_metrics import is_successful


def _legality(sim: SimulationRun) -> tuple[str, object]:
    """Return ('legal'|'illegal'|'skipped', detail). Adapter over the user's
    validator — edit the call below to match its real signature if it differs."""
    try:
        import retail_policy_validator as V  # type: ignore
    except Exception:  # noqa: BLE001 - validator not yet provided
        return "skipped", None

    messages = list(sim.get_messages())
    # Expected (per Policy-validator-algorithm.md): CHECK(messages, env) ->
    # {violations: [...], per_step_labels: [...]}. We only need the verdict here.
    if hasattr(V, "check_messages"):
        res = V.check_messages(messages)
    elif hasattr(V, "CHECK"):
        res = V.CHECK(messages, None)
    else:
        return "skipped", "no check_messages/CHECK entrypoint found"
    violations = res.get("violations") if isinstance(res, dict) else res
    return ("legal", None) if not violations else ("illegal", violations)


def filter_results(results: Results) -> tuple[Results, dict]:
    kept, stats = [], Counter()
    legality_active = False
    for sim in results.simulations:
        stats["total"] += 1
        reward = sim.reward_info.reward if sim.reward_info else None
        success = reward is not None and is_successful(reward)
        if not success:
            stats["reward_fail"] += 1
            continue
        stats["reward_pass"] += 1
        status, _ = _legality(sim)
        if status != "skipped":
            legality_active = True
        if status == "illegal":
            stats["illegal"] += 1
            continue
        kept.append(sim)
        stats["kept"] += 1

    kept_task_ids = {s.task_id for s in kept}
    filtered = results.model_copy(update={
        "simulations": kept,
        "tasks": [t for t in results.tasks if t.id in kept_task_ids],
    })
    summary = {
        "legality_active": legality_active,
        "counts": dict(stats),
        "tasks_with_kept_path": len(kept_task_ids),
        "kept_per_task": dict(Counter(s.task_id for s in kept)),
    }
    return filtered, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", help="tau2 Results JSON (or dir).")
    ap.add_argument("--out", default=None, help="Output filtered Results path.")
    args = ap.parse_args()

    results = Results.load(Path(args.results))
    filtered, summary = filter_results(results)

    out = Path(args.out) if args.out else Path(args.results).with_name("results_legal.json")
    filtered.save(out, format="json")

    c = summary["counts"]
    print(f"total={c.get('total',0)}  reward==1={c.get('reward_pass',0)}  "
          f"illegal={c.get('illegal',0)}  kept(r=1 & legal)={c.get('kept',0)}")
    print(f"tasks with >=1 kept path: {summary['tasks_with_kept_path']}")
    if not summary["legality_active"]:
        print("WARNING: legality gate SKIPPED (retail_policy_validator.py not found) — "
              "kept set is reward==1 ONLY, not yet legality-filtered.")
    print(f"wrote filtered Results → {out}")


if __name__ == "__main__":
    main()
