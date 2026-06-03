#!/usr/bin/env python3
"""Create per-model rollout tables in Supabase and push tau2 results.json into them.

One table per evaluated model (tau2_rollouts_<suffix>), one row per SimulationRun
(the full trajectory + the DB-state reward). Same Supabase project / connection
pattern as the agent-supervisor board tooling.

Usage:
  uv run --with psycopg2-binary python scripts/rollouts_supabase.py create --suffix m3
  uv run --with psycopg2-binary python scripts/rollouts_supabase.py push \
      --suffix m3 --results data/simulations/<run>/results.json --retrieval-config alltools
  uv run --with psycopg2-binary python scripts/rollouts_supabase.py stats --suffix m3
"""
import argparse
import json
import os
import pathlib
import psycopg2
from psycopg2.extras import Json

ENV = pathlib.Path(os.environ.get(
    "FLEETOS_BUS_ENV", "/Users/lilyzhang/Documents/lily-memory/GeniusTeam/genius-builder/.env"))
REF = "uneopomdwrxalfgrnflt"


def _env(name):
    # env var first (Modal secrets / CI), then the local .env file
    v = os.environ.get(name)
    if v:
        return v
    try:
        for line in ENV.read_text().splitlines():
            s = line.strip()
            if s.startswith(name + "="):
                return s.split("=", 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        return None
    return None


def connect():
    pw = _env("SOFAGENIUS_SUPABASE_DB_PASSWORD") or _env("SUPABASE_DB_PASSWORD")
    return psycopg2.connect(
        host="aws-0-us-west-2.pooler.supabase.com", port=6543,
        user=f"postgres.{REF}", dbname="postgres", password=pw,
        sslmode="require", connect_timeout=15,
    )


DDL = """
create table if not exists tau2_rollouts_{suffix} (
  id uuid primary key default gen_random_uuid(),
  sim_id text,
  task_id text not null,
  domain text not null,
  agent_llm text,
  user_llm text,
  retrieval_config text,
  trial int,
  seed int,
  reward double precision,
  reward_breakdown jsonb,
  termination_reason text,
  num_messages int,
  num_tool_calls int,
  tokens_in int,
  tokens_out int,
  trajectory jsonb,
  duration double precision,
  created_at timestamptz not null default now()
);
create index if not exists idx_{suffix}_task on tau2_rollouts_{suffix}(task_id);
"""


def _usage(sim):
    pin = pout = 0
    for m in sim.get("messages") or []:
        u = m.get("usage") or {}
        pin += u.get("prompt_tokens") or 0
        pout += u.get("completion_tokens") or 0
    return pin, pout


def cmd_create(a):
    con = connect(); con.autocommit = True
    con.cursor().execute(DDL.format(suffix=a.suffix))
    print(f"table tau2_rollouts_{a.suffix} ready")


def cmd_push(a):
    data = json.load(open(a.results))
    sims = data.get("simulations") or []
    info = data.get("info") or {}
    agent_llm = (info.get("agent_info") or {}).get("llm")
    user_llm = (info.get("user_info") or {}).get("llm")
    domain = (info.get("environment_info") or {}).get("domain") or a.domain
    con = connect(); con.autocommit = True; cur = con.cursor()
    n = 0
    for s in sims:
        ri = s.get("reward_info") or {}
        msgs = s.get("messages") or []
        pin, pout = _usage(s)
        cur.execute(
            f"""insert into tau2_rollouts_{a.suffix}
            (sim_id,task_id,domain,agent_llm,user_llm,retrieval_config,trial,seed,
             reward,reward_breakdown,termination_reason,num_messages,num_tool_calls,
             tokens_in,tokens_out,trajectory,duration)
            values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (s.get("id"), s.get("task_id"), domain, agent_llm, user_llm,
             a.retrieval_config, s.get("trial"), s.get("seed"),
             ri.get("reward"), Json(ri.get("reward_breakdown")),
             s.get("termination_reason"), len(msgs),
             sum(1 for m in msgs if m.get("tool_calls")),
             pin, pout, Json(msgs), s.get("duration")),
        )
        n += 1
    print(f"pushed {n} rollouts -> tau2_rollouts_{a.suffix}")


def cmd_stats(a):
    con = connect(); cur = con.cursor()
    cur.execute(f"select count(*), avg(reward), "
                f"count(*) filter (where reward>=0.999) from tau2_rollouts_{a.suffix};")
    n, avg, passed = cur.fetchone()
    print(f"tau2_rollouts_{a.suffix}: {n} rollouts, avg reward "
          f"{avg if avg is None else round(avg,3)}, pass@1 {passed}/{n}")


def cmd_clean(a):
    """Delete rows: reward-null (failed) rows and/or rows whose retrieval_config
    is not the one to keep. Scoped to the named table only."""
    conds, params = [], []
    if a.null_only:
        conds.append("reward is null")
    if a.keep_config:
        conds.append("retrieval_config is distinct from %s")
        params.append(a.keep_config)
    if not conds:
        print("nothing to clean (pass --null-only and/or --keep-config)"); return
    con = connect(); con.autocommit = True; cur = con.cursor()
    cur.execute(f"delete from tau2_rollouts_{a.suffix} where " + " or ".join(conds), params)
    print(f"cleaned tau2_rollouts_{a.suffix}: deleted {cur.rowcount} rows")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create"); c.add_argument("--suffix", required=True); c.set_defaults(fn=cmd_create)
    cl = sub.add_parser("clean")
    cl.add_argument("--suffix", required=True)
    cl.add_argument("--null-only", action="store_true", help="delete rows with null reward")
    cl.add_argument("--keep-config", default=None, help="delete rows whose retrieval_config != this")
    cl.set_defaults(fn=cmd_clean)
    pu = sub.add_parser("push")
    pu.add_argument("--suffix", required=True)
    pu.add_argument("--results", required=True)
    pu.add_argument("--retrieval-config", default=None)
    pu.add_argument("--domain", default=None)
    pu.set_defaults(fn=cmd_push)
    st = sub.add_parser("stats"); st.add_argument("--suffix", required=True); st.set_defaults(fn=cmd_stats)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
