#!/usr/bin/env python3
"""Pull the τ³ banking eval rollouts from Supabase and print an analysis digest:
per-model pass@1, the per-task reward matrix (where models diverge), and compact
trajectory excerpts. Used to write the findings report. Reads from Supabase, not
local results.json (those lived in the ephemeral Modal containers).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rollouts_supabase import connect

MODELS = {"m3": "MiniMax M3", "opus48": "Claude Opus 4.8", "opus47": "Opus 4.7 (control)"}
COLS = ["task_id", "reward", "retrieval_config", "num_messages", "num_tool_calls",
        "tokens_in", "tokens_out", "trajectory"]


def short(s, n=80):
    s = (s or "").replace("\n", " ").strip()
    return s[:n] + ("…" if len(s) > n else "")


def fetch(cur, suffix):
    cur.execute(f"select {','.join(COLS)} from tau2_rollouts_{suffix} order by task_id")
    return [dict(zip(COLS, r)) for r in cur.fetchall()]


def digest(traj, limit=24):
    out = []
    for m in (traj or []):
        role = m.get("role")
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = (tc.get("function") or {}).get("name") or tc.get("name")
                ar = (tc.get("function") or {}).get("arguments") or tc.get("arguments")
                out.append(f"    {role} -> {fn}({short(str(ar),46)})")
        elif role == "tool":
            out.append(f"    tool <- {short(str(m.get('content')),66)}")
        elif m.get("content"):
            out.append(f"    {role}: {short(m.get('content'))}")
    return out[:limit]


con = connect(); cur = con.cursor()
data = {s: fetch(cur, s) for s in MODELS}

print("=== SUMMARY (10 banking tasks, AllTools, gpt-5.2 user-sim, seed 300) ===")
for s, name in MODELS.items():
    rows = data[s]; n = len(rows) or 1
    p = sum(1 for r in rows if (r["reward"] or 0) >= 0.999)
    avg = sum((r["reward"] or 0) for r in rows) / n
    tin = sum((r["tokens_in"] or 0) for r in rows) // n
    tout = sum((r["tokens_out"] or 0) for r in rows) // n
    cfg = rows[0]["retrieval_config"] if rows else "?"
    print(f"  {name:22s} pass@1={p}/{len(rows)} ({p/n*100:.0f}%)  avg={avg:.3f}  "
          f"cfg={cfg}  tok≈{tin}/{tout}")

print("\n=== PER-TASK REWARD MATRIX (1=pass, 0=fail) ===")
tasks = sorted({r["task_id"] for s in MODELS for r in data[s]})
print(f"  {'task':22s} {'M3':>4} {'O4.8':>5} {'O4.7':>5}")
for t in tasks:
    def rw(s):
        m = [r for r in data[s] if r["task_id"] == t]
        return f"{m[0]['reward']:.0f}" if m and m[0]["reward"] is not None else "-"
    print(f"  {short(t,22):22s} {rw('m3'):>4} {rw('opus48'):>5} {rw('opus47'):>5}")

# trajectory excerpts: one task where M3 passed, one where all failed
def by_task(s, t):
    m = [r for r in data[s] if r["task_id"] == t]; return m[0] if m else None

m3_win = next((t for t in tasks if (by_task("m3", t) or {}).get("reward") == 1.0
               and (by_task("opus48", t) or {}).get("reward") == 0.0), None)
all_fail = next((t for t in tasks if all((by_task(s, t) or {}).get("reward") == 0.0 for s in MODELS)), None)
for label, t in [("M3 PASSED, Opus 4.8 FAILED", m3_win), ("ALL FAILED", all_fail)]:
    if not t:
        continue
    print(f"\n=== TRAJECTORY: task {t}  ({label}) ===")
    for s, name in MODELS.items():
        r = by_task(s, t)
        if not r:
            continue
        print(f"  --- {name} (reward {r['reward']}, {r['num_messages']} msgs, {r['num_tool_calls']} tool calls) ---")
        for line in digest(r["trajectory"]):
            print(line)
