#!/usr/bin/env python3
"""Inspect a tau2 results.json: per-rollout reward, trajectory shape, token usage.

Reusable across smoke runs and full batches. Reads the raw JSON (no tau2 import
needed) so it works on any saved Results file.

Usage:
  uv run python scripts/inspect_rollouts.py data/simulations/<save_to>/results.json
  uv run python scripts/inspect_rollouts.py <path> --trajectory   # also print the turns
"""
import argparse
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def usage_tokens(sim):
    """Sum prompt/completion tokens across assistant messages (best-effort)."""
    pin = pout = 0
    for m in sim.get("messages") or []:
        u = m.get("usage") or {}
        pin += u.get("prompt_tokens") or 0
        pout += u.get("completion_tokens") or 0
    return pin, pout


def short(s, n=90):
    s = (s or "").replace("\n", " ").strip()
    return s[:n] + ("…" if len(s) > n else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--trajectory", action="store_true", help="print each turn")
    a = ap.parse_args()

    data = load(a.path)
    sims = data.get("simulations") or []
    info = data.get("info") or {}
    print(f"file: {a.path}")
    print(f"sims: {len(sims)}  agent={info.get('agent_info',{}).get('llm','?')}  "
          f"user={info.get('user_info',{}).get('llm','?')}")
    print("-" * 70)

    rewards = []
    tot_in = tot_out = 0
    for s in sims:
        ri = s.get("reward_info") or {}
        r = ri.get("reward")
        rewards.append(r if r is not None else 0.0)
        pin, pout = usage_tokens(s)
        tot_in += pin
        tot_out += pout
        msgs = s.get("messages") or []
        n_tool = sum(1 for m in msgs if m.get("tool_calls"))
        print(f"task={s.get('task_id')}  reward={r}  breakdown={ri.get('reward_breakdown')}")
        print(f"   msgs={len(msgs)} tool_calls={n_tool} term={s.get('termination_reason')} "
              f"tokens={pin}->{pout}")
        if a.trajectory:
            for m in msgs:
                role = m.get("role")
                if m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        fn = (tc.get("function") or {}).get("name") or tc.get("name")
                        args = (tc.get("function") or {}).get("arguments") or tc.get("arguments")
                        print(f"     {role} → {fn}({short(str(args),60)})")
                elif role == "tool":
                    print(f"     tool ← {short(str(m.get('content')),70)}")
                elif m.get("content"):
                    print(f"     {role}: {short(m.get('content'))}")

    n = len(rewards) or 1
    print("-" * 70)
    print(f"avg reward: {sum(rewards)/n:.3f}   pass@1: {sum(1 for r in rewards if r>=0.999)}/{len(rewards)}")
    print(f"total tokens: {tot_in} in / {tot_out} out")
    # OpenRouter MiniMax M3 list price (approx, update if it changes): $0.30/M in, $1.20/M out
    est = tot_in/1e6*0.30 + tot_out/1e6*1.20
    print(f"est. M3 cost @ $0.30/$1.20 per M tok: ${est:.4f}  (~${est/n:.4f}/rollout)")


if __name__ == "__main__":
    main()
