#!/usr/bin/env python3
"""Append-only bug log for the tau3 GLM-5 leaderboard-replication effort.

Every debugging attempt is one row, so the whole investigation is recoverable:
what config we ran, what number we got vs the board, the hypothesis, the finding,
and the resulting status. Backed by Supabase (same DB creds as the supervisor board).

  uv run --with psycopg2-binary python scripts/log_replication_bug.py add \
      --commit <sha> --our 0.21 --board 0.0979 --status investigating \
      --hypothesis "avg-reward != strict pass@1" --finding "..." --config '{"k":"v"}'
  uv run --with psycopg2-binary python scripts/log_replication_bug.py list
"""
import argparse, json, os, pathlib, psycopg2

ENV = pathlib.Path(os.environ.get(
    "FLEETOS_ENV", "/Users/lilyzhang/Documents/lily-memory/GeniusTeam/genius-builder/.env"))
REF = os.environ.get("SUPABASE_REF", "uneopomdwrxalfgrnflt")
TABLE = "tau3_replication_log"


def _env(name):
    try:
        for line in ENV.read_text().splitlines():
            s = line.strip()
            if s.startswith(name + "="):
                return s.split("=", 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return None


def connect():
    pw = _env("SOFAGENIUS_SUPABASE_DB_PASSWORD") or os.environ.get("SUPABASE_DB_PASSWORD")
    return psycopg2.connect(
        host="aws-0-us-west-2.pooler.supabase.com", port=6543,
        user=f"postgres.{REF}", dbname="postgres", password=pw,
        sslmode="require", connect_timeout=12)


DDL = f"""
create table if not exists {TABLE} (
  id uuid primary key default gen_random_uuid(),
  attempt_no serial,
  commit_sha text,
  config jsonb,
  our_metric numeric,
  board_metric numeric,
  hypothesis text,
  finding text,
  status text default 'investigating',  -- investigating | fixed | stuck
  notes text,
  created_at timestamptz not null default now()
);
"""


def cmd_add(a, cur):
    cur.execute(
        f"insert into {TABLE} (commit_sha, config, our_metric, board_metric, hypothesis, finding, status, notes) "
        "values (%s,%s,%s,%s,%s,%s,%s,%s) returning attempt_no;",
        (a.commit, json.dumps(json.loads(a.config)) if a.config else None,
         a.our, a.board, a.hypothesis, a.finding, a.status, a.notes))
    print(f"logged attempt #{cur.fetchone()[0]} (status={a.status})")


def cmd_list(a, cur):
    cur.execute(f"select attempt_no, our_metric, board_metric, status, hypothesis, finding, created_at "
                f"from {TABLE} order by attempt_no;")
    for r in cur.fetchall():
        print(f"#{r[0]} our={r[1]} board={r[2]} [{r[3]}] {r[4] or ''} -> {r[5] or ''}  ({r[6]:%Y-%m-%d %H:%M})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("add")
    for f in ("commit", "config", "hypothesis", "finding", "notes"):
        s.add_argument(f"--{f}", default=None)
    s.add_argument("--our", type=float, default=None)
    s.add_argument("--board", type=float, default=0.0979)
    s.add_argument("--status", default="investigating")
    s.set_defaults(fn=cmd_add)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    a = p.parse_args()
    c = connect(); c.autocommit = True; cur = c.cursor()
    cur.execute(DDL)
    a.fn(a, cur)
