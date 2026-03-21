#!/usr/bin/env python3
"""
τ²-Bench Results Analyzer — Comprehensive Metrics for gpt-oss-120b
===================================================================
Computes an extended set of metrics from any tau2-bench simulation JSON file,
beyond the single pass-rate reported by default.

Metric Groups
─────────────
  1. Reliability & Consistency     — stability across multiple trials
  2. Interaction Efficiency         — conversation length, cost, latency
  3. Policy & Tool Governance       — compliance with domain rules
  4. Tool Call Accuracy             — precision/recall vs ground truth actions
  5. First Call Resolution (FCR)    — success on the first try
  6. State-Based Metrics            — DB correctness, partial success,
                                      environment damage rate

Usage
─────
    python analyze_results.py                          # auto-detect latest file
    python analyze_results.py path/to/results.json
    python analyze_results.py --domain airline --all   # all files for a domain
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

# ── Path helpers ──────────────────────────────────────────────────────────────
# This script lives at scripts/analyze_results.py inside the tau2-bench repo.
REPO_ROOT  = Path(__file__).parent.parent   # tau2-bench/
SIM_DIR    = REPO_ROOT / "data" / "simulations"


def latest_file(domain: Optional[str] = None) -> Path:
    pattern = f"gpt_oss_120b_{domain}.json" if domain else "gpt_oss_120b_*.json"
    files = sorted(SIM_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        sys.exit(f"❌  No simulation files found in {SIM_DIR}\n    Run:  python run_demo.py")
    return files[0]


# ── Data loading ──────────────────────────────────────────────────────────────

def load(path: Path) -> dict:
    """Load tau2-bench Results JSON."""
    raw = json.loads(path.read_text())
    # Build task lookup
    tasks = {t["id"]: t for t in raw.get("tasks", [])}
    sims  = raw.get("simulations", [])
    info  = raw.get("info", {})
    return {"tasks": tasks, "sims": sims, "info": info, "path": path}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reward(sim) -> float:
    ri = sim.get("reward_info") or {}
    return ri.get("reward", 0.0) or 0.0

def _msgs(sim) -> list:
    return sim.get("messages", [])

def _term(sim) -> str:
    return sim.get("termination_reason", "unknown")

def _action_checks(sim) -> list:
    ri = sim.get("reward_info") or {}
    return ri.get("action_checks") or []

def _nl_checks(sim) -> list:
    ri = sim.get("reward_info") or {}
    return ri.get("nl_assertions") or []

def _comm_checks(sim) -> list:
    ri = sim.get("reward_info") or {}
    return ri.get("communicate_checks") or []

def _env_checks(sim) -> list:
    ri = sim.get("reward_info") or {}
    return ri.get("env_assertions") or []

def _db_check(sim) -> Optional[dict]:
    ri = sim.get("reward_info") or {}
    return ri.get("db_check")

def _all_tool_calls(sim) -> list:
    """Return all tool calls made by the assistant across the conversation."""
    calls = []
    for msg in _msgs(sim):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                calls.append(tc)
    return calls

def _tool_errors(sim) -> int:
    """Count ToolMessages where error=True."""
    return sum(
        1 for msg in _msgs(sim)
        if msg.get("role") == "tool" and msg.get("error", False)
    )

def _pct(num: float, den: float) -> float:
    return (num / den * 100) if den > 0 else 0.0

def _avg(lst: list) -> float:
    return mean(lst) if lst else 0.0

def _std(lst: list) -> float:
    return stdev(lst) if len(lst) >= 2 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# METRIC GROUP 1 — Reliability & Consistency
# ══════════════════════════════════════════════════════════════════════════════

def reliability_metrics(data: dict) -> dict:
    """
    Computed across multiple trials of the same task.
    Requires num_trials > 1 for consistency metrics to be meaningful.
    """
    sims   = data["sims"]
    trials = defaultdict(list)   # task_id → [reward, ...]

    for sim in sims:
        trials[sim["task_id"]].append(_reward(sim))

    rewards_all  = [_reward(s) for s in sims]
    pass_at_1    = _pct(sum(1 for r in rewards_all if r >= 1.0), len(rewards_all))

    # pass@k: P(at least one trial passes)
    pass_at_k    = _pct(
        sum(1 for rs in trials.values() if any(r >= 1.0 for r in rs)),
        len(trials)
    )

    # Consistency: tasks where ALL trials pass
    all_pass     = sum(1 for rs in trials.values() if all(r >= 1.0 for r in rs))
    # Flakiness: tasks with mixed results across trials
    flaky        = sum(1 for rs in trials.values()
                       if len(set(r >= 1.0 for r in rs)) > 1)

    # Crash rate: agent_error or user_error terminations
    crashes      = sum(1 for s in sims if _term(s) in ("agent_error", "user_error"))

    # Timeout rate: max_steps reached
    timeouts     = sum(1 for s in sims if _term(s) == "max_steps")

    # Per-task reward variance
    variances    = [_std(rs) for rs in trials.values() if len(rs) >= 2]

    term_counts: dict[str, int] = defaultdict(int)
    for s in sims:
        term_counts[_term(s)] += 1

    return {
        "pass_at_1_rate_%"          : round(pass_at_1, 1),
        "pass_at_k_rate_%"          : round(pass_at_k, 1),
        "all_trials_pass_rate_%"    : round(_pct(all_pass, len(trials)), 1),
        "flakiness_rate_%"          : round(_pct(flaky, len(trials)), 1),
        "crash_rate_%"              : round(_pct(crashes, len(sims)), 1),
        "timeout_rate_%"            : round(_pct(timeouts, len(sims)), 1),
        "avg_reward_std_per_task"   : round(_avg(variances), 3),
        "termination_breakdown"     : dict(sorted(term_counts.items())),
    }


# ══════════════════════════════════════════════════════════════════════════════
# METRIC GROUP 2 — Interaction Efficiency
# ══════════════════════════════════════════════════════════════════════════════

def efficiency_metrics(data: dict) -> dict:
    sims = data["sims"]
    if not sims:
        return {}

    # Turn counting: count unique turn_idx values per simulation
    def turn_count(sim):
        idxs = [m.get("turn_idx") for m in _msgs(sim) if m.get("turn_idx") is not None]
        return max(idxs) + 1 if idxs else len(_msgs(sim))

    def agent_msg_count(sim):
        return sum(1 for m in _msgs(sim) if m.get("role") == "assistant" and m.get("content"))

    def user_msg_count(sim):
        return sum(1 for m in _msgs(sim) if m.get("role") == "user" and m.get("content"))

    def avg_agent_msg_length(sim):
        texts = [m["content"] for m in _msgs(sim)
                 if m.get("role") == "assistant" and m.get("content")]
        return _avg([len(t) for t in texts]) if texts else 0

    turns            = [turn_count(s)            for s in sims]
    agent_msgs       = [agent_msg_count(s)        for s in sims]
    user_msgs        = [user_msg_count(s)         for s in sims]
    tool_calls_per   = [len(_all_tool_calls(s))   for s in sims]
    durations        = [s.get("duration", 0) or 0 for s in sims]
    agent_costs      = [s.get("agent_cost") or 0  for s in sims]
    user_costs       = [s.get("user_cost")  or 0  for s in sims]
    agent_lens       = [avg_agent_msg_length(s)   for s in sims]

    # Efficiency for SUCCESSFUL runs only
    success_sims     = [s for s in sims if _reward(s) >= 1.0]
    succ_turns       = [turn_count(s) for s in success_sims]
    succ_tool_calls  = [len(_all_tool_calls(s)) for s in success_sims]

    return {
        "avg_turns_per_sim"             : round(_avg(turns), 1),
        "avg_agent_messages_per_sim"    : round(_avg(agent_msgs), 1),
        "avg_user_messages_per_sim"     : round(_avg(user_msgs), 1),
        "avg_tool_calls_per_sim"        : round(_avg(tool_calls_per), 1),
        "avg_duration_seconds"          : round(_avg(durations), 1),
        "avg_agent_cost_usd"            : round(_avg(agent_costs), 5),
        "avg_user_cost_usd"             : round(_avg(user_costs), 5),
        "total_cost_usd"                : round(sum(agent_costs) + sum(user_costs), 4),
        "avg_agent_response_length_chars": round(_avg(agent_lens), 0),
        # Efficiency on successful tasks
        "success_avg_turns"             : round(_avg(succ_turns), 1),
        "success_avg_tool_calls"        : round(_avg(succ_tool_calls), 1),
        # Overhead: extra turns in failed vs successful runs (n/a if no failures)
        "fail_vs_success_turn_delta"    : (
            round(
                _avg([turn_count(s) for s in sims if _reward(s) < 1.0]) - _avg(succ_turns), 1
            )
            if any(_reward(s) < 1.0 for s in sims) and success_sims
            else "n/a"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# METRIC GROUP 3 — Policy & Tool Governance Compliance
# ══════════════════════════════════════════════════════════════════════════════

def governance_metrics(data: dict) -> dict:
    sims = data["sims"]
    if not sims:
        return {}

    total_tool_calls  = sum(len(_all_tool_calls(s)) for s in sims)
    total_tool_errors = sum(_tool_errors(s) for s in sims)
    runs_with_errors  = sum(1 for s in sims if _tool_errors(s) > 0)
    too_many_errors   = sum(1 for s in sims if _term(s) == "too_many_errors")

    # Human transfer: look for transfer_to_human_agents tool calls
    def has_transfer(sim):
        return any(
            "transfer" in tc.get("name", "").lower()
            for tc in _all_tool_calls(sim)
        )
    transfers = sum(1 for s in sims if has_transfer(s))

    # NL-assertion policy violations: assertions about policy compliance that failed
    policy_keywords = ["policy", "confirm", "must", "should not", "cannot", "only", "rule"]
    def policy_violation(sim):
        for chk in _nl_checks(sim):
            text = chk.get("nl_assertion", "").lower()
            if not chk.get("met", True) and any(kw in text for kw in policy_keywords):
                return True
        return False
    policy_violations = sum(1 for s in sims if policy_violation(s))

    # Communication compliance: % communicate_checks passed
    all_comm = [chk for s in sims for chk in _comm_checks(s)]
    comm_met = sum(1 for c in all_comm if c.get("met", False))

    # NL assertion overall compliance
    all_nl   = [chk for s in sims for chk in _nl_checks(s)]
    nl_met   = sum(1 for c in all_nl if c.get("met", False))

    return {
        "tool_error_rate_%"             : round(_pct(total_tool_errors, total_tool_calls), 1),
        "runs_with_tool_errors_%"       : round(_pct(runs_with_errors, len(sims)), 1),
        "runs_terminated_too_many_errors_%": round(_pct(too_many_errors, len(sims)), 1),
        "human_transfer_rate_%"         : round(_pct(transfers, len(sims)), 1),
        "policy_violation_rate_%"       : round(_pct(policy_violations, len(sims)), 1),
        "nl_assertion_compliance_%"     : round(_pct(nl_met, len(all_nl)), 1) if all_nl else "n/a",
        "communication_compliance_%"    : round(_pct(comm_met, len(all_comm)), 1) if all_comm else "n/a",
    }


# ══════════════════════════════════════════════════════════════════════════════
# METRIC GROUP 4 — Tool Call Accuracy (vs Ground Truth)
# ══════════════════════════════════════════════════════════════════════════════

def tool_accuracy_metrics(data: dict) -> dict:
    sims  = data["sims"]
    tasks = data["tasks"]
    if not sims:
        return {}

    name_hits, name_total       = 0, 0
    exact_hits, exact_total     = 0, 0
    seq_match_count             = 0
    extra_calls_list            = []
    missed_calls_list           = []

    for sim in sims:
        task = tasks.get(sim["task_id"], {})
        ec   = task.get("evaluation_criteria", {}) or {}
        gt_actions = [a for a in (ec.get("actions") or []) if a.get("requestor", "assistant") == "assistant"]

        if not gt_actions:
            continue

        actual_calls = _all_tool_calls(sim)
        gt_names     = [a["name"] for a in gt_actions]
        actual_names = [tc["name"] for tc in actual_calls]

        # Name-level accuracy: how many GT names appear in actual calls
        for gt_name in gt_names:
            name_total += 1
            if gt_name in actual_names:
                name_hits += 1

        # Exact action match (using action_checks if available)
        ac = _action_checks(sim)
        if ac:
            for chk in ac:
                exact_total += 1
                if chk.get("action_match", False):
                    exact_hits += 1
        else:
            # Fallback: name-only match
            exact_total += len(gt_names)
            exact_hits  += sum(1 for n in gt_names if n in actual_names)

        # Sequence match: full ordered list of names matches
        if gt_names == actual_names[:len(gt_names)]:
            seq_match_count += 1

        # Extra / missed calls
        extra_calls_list.append(max(0, len(actual_calls) - len(gt_actions)))
        missed_calls_list.append(max(0, len(gt_actions) - len(actual_calls)))

    tasks_with_actions = sum(
        1 for sim in sims
        if (tasks.get(sim["task_id"], {}).get("evaluation_criteria") or {}).get("actions")
    )

    return {
        "tool_name_recall_%"            : round(_pct(name_hits, name_total), 1),
        "exact_action_match_%"          : round(_pct(exact_hits, exact_total), 1),
        "action_sequence_match_%"       : round(_pct(seq_match_count, tasks_with_actions), 1),
        "avg_extra_tool_calls"          : round(_avg(extra_calls_list), 2),
        "avg_missed_tool_calls"         : round(_avg(missed_calls_list), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# METRIC GROUP 5 — First Call Resolution (FCR)
# ══════════════════════════════════════════════════════════════════════════════

def fcr_metrics(data: dict) -> dict:
    sims  = data["sims"]
    if not sims:
        return {}

    # Group by task
    by_task = defaultdict(list)
    for sim in sims:
        by_task[sim["task_id"]].append(sim)

    # Sort each task's trials by trial index
    for tid in by_task:
        by_task[tid].sort(key=lambda s: s.get("trial") or 0)

    fcr_count    = 0
    trials_to_pass = []
    never_passes   = 0
    num_tasks      = len(by_task)

    for tid, task_sims in by_task.items():
        passed = False
        for i, sim in enumerate(task_sims):
            if _reward(sim) >= 1.0:
                if i == 0:
                    fcr_count += 1
                trials_to_pass.append(i + 1)
                passed = True
                break
        if not passed:
            never_passes += 1

    return {
        "first_call_resolution_%"       : round(_pct(fcr_count, num_tasks), 1),
        "ever_passes_rate_%"            : round(_pct(num_tasks - never_passes, num_tasks), 1),
        "never_passes_rate_%"           : round(_pct(never_passes, num_tasks), 1),
        "avg_trials_to_first_pass"      : round(_avg(trials_to_pass), 2) if trials_to_pass else "n/a",
    }


# ══════════════════════════════════════════════════════════════════════════════
# METRIC GROUP 6 — State-Based Metrics
# ══════════════════════════════════════════════════════════════════════════════

def state_metrics(data: dict) -> dict:
    sims = data["sims"]
    if not sims:
        return {}

    rewards      = [_reward(s) for s in sims]
    full_success = sum(1 for r in rewards if r >= 1.0)
    partial      = sum(1 for r in rewards if 0.0 < r < 1.0)
    full_fail    = sum(1 for r in rewards if r == 0.0)

    # DB correctness
    db_checks    = [_db_check(s) for s in sims if _db_check(s) is not None]
    db_match     = sum(1 for d in db_checks if d.get("db_match", False))
    db_rewards   = [d.get("db_reward", 0.0) or 0.0 for d in db_checks]

    # Environment damage: DB state was modified incorrectly (db_match=False)
    # AND the sim didn't time out (i.e., agent actually tried something)
    def env_damaged(sim):
        db = _db_check(sim)
        if db is None:
            return False
        return (
            not db.get("db_match", True) and
            _term(sim) not in ("max_steps", "too_many_errors") and
            len(_all_tool_calls(sim)) > 0
        )
    damage_count = sum(1 for s in sims if env_damaged(s))

    # Env assertions
    all_env  = [chk for s in sims for chk in _env_checks(s)]
    env_met  = sum(1 for c in all_env if c.get("met", False))

    # Reward distribution buckets
    buckets = {"0.0": 0, "0.0–0.5": 0, "0.5–1.0": 0, "1.0": 0}
    for r in rewards:
        if r == 0.0:
            buckets["0.0"] += 1
        elif r < 0.5:
            buckets["0.0–0.5"] += 1
        elif r < 1.0:
            buckets["0.5–1.0"] += 1
        else:
            buckets["1.0"] += 1

    return {
        "full_success_rate_%"           : round(_pct(full_success, len(sims)), 1),
        "partial_success_rate_%"        : round(_pct(partial, len(sims)), 1),
        "full_failure_rate_%"           : round(_pct(full_fail, len(sims)), 1),
        "avg_reward"                    : round(_avg(rewards), 3),
        "db_state_match_%"              : round(_pct(db_match, len(db_checks)), 1) if db_checks else "n/a",
        "avg_db_reward"                 : round(_avg(db_rewards), 3) if db_rewards else "n/a",
        "environment_damage_rate_%"     : round(_pct(damage_count, len(sims)), 1),
        "env_assertion_pass_%"          : round(_pct(env_met, len(all_env)), 1) if all_env else "n/a",
        "reward_distribution"           : buckets,
    }


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

GROUPS = [
    ("1. Reliability & Consistency",      reliability_metrics),
    ("2. Interaction Efficiency",          efficiency_metrics),
    ("3. Policy & Tool Governance",        governance_metrics),
    ("4. Tool Call Accuracy",              tool_accuracy_metrics),
    ("5. First Call Resolution (FCR)",     fcr_metrics),
    ("6. State-Based Metrics",             state_metrics),
]


def print_report(data: dict):
    info  = data["info"]
    sims  = data["sims"]
    path  = data["path"]

    print()
    print("═" * 72)
    print("  τ²-Bench Comprehensive Metrics Report")
    print("═" * 72)
    print(f"  File       : {path.name}")
    print(f"  Domain     : {info.get('environment_info', {}).get('domain_name', '?')}")
    print(f"  Model      : {info.get('agent_info', {}).get('llm', '?')}")
    print(f"  Simulations: {len(sims)}")
    print(f"  Num Trials : {info.get('num_trials', '?')}")
    print("═" * 72)

    if not sims:
        print("\n  ⚠️   No completed simulations in this file yet.")
        print("       Run:  python run_demo.py  (and let it complete)\n")
        return

    for title, fn in GROUPS:
        metrics = fn(data)
        if not metrics:
            continue
        print(f"\n  {'─'*68}")
        print(f"  {title}")
        print(f"  {'─'*68}")
        for key, val in metrics.items():
            label = key.replace("_", " ").rstrip()
            if isinstance(val, dict):
                print(f"  {'  '+label:<42}: {val}")
            elif isinstance(val, float):
                print(f"  {'  '+label:<42}: {val}")
            else:
                print(f"  {'  '+label:<42}: {val}")

    print()
    print("═" * 72)
    print("  💡  Tips:")
    print("       • Run with --num-trials 4 for robust reliability metrics")
    print("       • Partial success (0 < reward < 1) indicates incomplete tasks")
    print("       • Environment damage = agent modified DB incorrectly")
    print("       • FCR ≥ 70% is considered production-ready for customer service")
    print("═" * 72 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="τ²-Bench comprehensive metrics analyzer")
    p.add_argument(
        "file",
        nargs="?",
        help="Path to simulation JSON file. Defaults to latest gpt_oss_120b_*.json",
    )
    p.add_argument(
        "--domain",
        choices=["airline", "retail", "telecom"],
        help="Filter to latest file for a specific domain",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Analyze all gpt_oss_120b_*.json files found",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.file:
        files = [Path(args.file)]
    elif args.all:
        files = sorted(SIM_DIR.glob("gpt_oss_120b_*.json"))
        if not files:
            sys.exit(f"❌  No files found in {SIM_DIR}")
    else:
        files = [latest_file(args.domain)]

    for f in files:
        data = load(f)
        print_report(data)


if __name__ == "__main__":
    main()
