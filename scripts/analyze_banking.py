#!/usr/bin/env python3
"""Read-only consolidated analysis of banking_knowledge eval result dirs.

Usage: python scripts/analyze_banking.py <results_dir> [<results_dir> ...]
Each results_dir must contain results.json + simulations/*.json (+ artifacts/).

Reports per dir: official DB reward (avg, pass^1), termination reasons, DB-match
counts, retrieval tool-call evidence (shell/KB_search/grep), sandbox-error count,
and audio-artifact sanity. Prints a final cross-config table.
"""
import json
import os
import sys
from collections import Counter

RETRIEVAL_TOOLS = {"shell", "KB_search", "grep"}


def scan_tools(sim):
    """Walk ticks (voice) AND messages (text) for tool calls/results.
    Return (call_names Counter, n_results, sandbox_error_count)."""
    calls = Counter()
    n_results = 0
    sandbox_errors = 0
    roots = [sim.get("ticks") or [], sim.get("messages") or []]

    def walk(o):
        nonlocal n_results, sandbox_errors
        if isinstance(o, dict):
            for k, v in o.items():
                kl = k.lower()
                if kl in ("tool_calls", "agent_tool_calls", "user_tool_calls") and v:
                    for tc in (v if isinstance(v, list) else [v]):
                        if isinstance(tc, dict):
                            name = tc.get("name") or (tc.get("function") or {}).get("name")
                            if name:
                                calls[name] += 1
                if kl in ("tool_results", "agent_tool_results", "user_tool_results") and v:
                    for tr in (v if isinstance(v, list) else [v]):
                        if isinstance(tr, dict):
                            n_results += 1
                            c = str(tr.get("content") or tr.get("result") or tr.get("output") or "")
                            if "Sandbox dependencies are not available" in c or "Required: ripgrep" in c:
                                sandbox_errors += 1
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    for r in roots:
        walk(r)
    return calls, n_results, sandbox_errors


def analyze(results_dir):
    rj = os.path.join(results_dir, "results.json")
    if not os.path.exists(rj):
        return None
    data = json.load(open(rj))
    idx = data.get("simulation_index", [])
    sim_dir = os.path.join(results_dir, "simulations")
    art_dir = os.path.join(results_dir, "artifacts")

    rewards, terms = [], Counter()
    db_match = Counter()
    retr_call_sims = 0
    total_retr_calls = Counter()
    sandbox_err_sims = 0
    audio_ok = 0
    audio_total = 0
    log_errors = 0

    # Load sims from dir format (simulations/*.json) or monolithic results.json.
    sims = []
    if os.path.isdir(sim_dir):
        for f in os.listdir(sim_dir):
            try:
                sims.append(json.load(open(os.path.join(sim_dir, f))))
            except Exception:
                pass
    if not sims:
        sims = data.get("simulations") or []

    for sim in sims:
        ri = sim.get("reward_info") or {}
        r = ri.get("reward")
        if r is not None:
            rewards.append(r)
        terms[sim.get("termination_reason")] += 1
        dbc = ri.get("db_check") or {}
        db_match[bool(dbc.get("db_match"))] += 1
        calls, _, sb_err = scan_tools(sim)
        retr = sum(v for k, v in calls.items() if k in RETRIEVAL_TOOLS)
        if retr > 0:
            retr_call_sims += 1
        for k, v in calls.items():
            if k in RETRIEVAL_TOOLS:
                total_retr_calls[k] += v
        if sb_err > 0:
            sandbox_err_sims += 1

    # audio + log scan via artifacts
    if os.path.isdir(art_dir):
        for root, _, files in os.walk(art_dir):
            for fn in files:
                p = os.path.join(root, fn)
                if fn == "both.wav":
                    audio_total += 1
                    if os.path.getsize(p) > 10000:
                        audio_ok += 1
                if fn == "task.log":
                    try:
                        txt = open(p, errors="ignore").read()
                        if "Traceback" in txt or "ERROR" in txt:
                            log_errors += 1
                    except Exception:
                        pass

    n = len(rewards)
    avg = sum(rewards) / n if n else float("nan")
    passrate = sum(1 for r in rewards if r >= 1.0) / n if n else float("nan")
    return {
        "dir": results_dir,
        "n": n,
        "avg_reward": avg,
        "pass1": passrate,
        "db_match_true": db_match[True],
        "db_match_false": db_match[False],
        "terms": dict(terms),
        "retr_call_sims": retr_call_sims,
        "total_retr_calls": dict(total_retr_calls),
        "sandbox_err_sims": sandbox_err_sims,
        "audio_ok": audio_ok,
        "audio_total": audio_total,
        "log_errors": log_errors,
    }


def main():
    dirs = sys.argv[1:]
    rows = []
    for d in dirs:
        res = analyze(d)
        if res is None:
            print(f"[skip] no results.json in {d}")
            continue
        rows.append(res)
        print(f"\n=== {d} ===")
        print(f"  sims:            {res['n']}")
        print(f"  avg DB reward:   {res['avg_reward']:.3f}")
        print(f"  pass^1:          {res['pass1']:.3f}")
        print(f"  db_match:        ✓{res['db_match_true']} / ✗{res['db_match_false']}")
        print(f"  termination:     {res['terms']}")
        print(f"  retrieval calls: {res['total_retr_calls']} across {res['retr_call_sims']} sims")
        print(f"  sandbox errors:  {res['sandbox_err_sims']} sims")
        print(f"  audio both.wav:  {res['audio_ok']}/{res['audio_total']} present & >10KB")
        print(f"  task.log errors: {res['log_errors']}")

    if rows:
        print("\n\n===== CROSS-CONFIG SUMMARY (official DB reward) =====")
        print(f"{'config dir':<48}{'n':>4}{'avgReward':>11}{'pass^1':>9}{'retrSims':>10}")
        for r in rows:
            print(f"{r['dir']:<48}{r['n']:>4}{r['avg_reward']:>11.3f}{r['pass1']:>9.3f}{r['retr_call_sims']:>10}")


if __name__ == "__main__":
    main()
