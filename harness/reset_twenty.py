#!/usr/bin/env python3
"""Reset the Twenty CRM state before a v2 trial.

The MCP API is tried first. Twenty soft deletes, so if the records are still
visible to find_many_* afterwards, the script falls back to the hard SQL reset.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from twenty_mcp import MCPClient, _read_dotenv, get_credentials

DB_URL_KEYS = ("TWENTY_DB_URL", "DATABASE_URL", "PG_DATABASE_URL")


def get_db_url(repo: Path | None) -> str | None:
    dotenv = _read_dotenv(repo)
    for key in DB_URL_KEYS:
        value = os.environ.get(key) or dotenv.get(key)
        if value:
            return value
    return None


def reset_sql(repo: Path | None, db_url: str | None = None) -> int:
    url = db_url or get_db_url(repo)
    psql_cmd = os.environ.get("TWENTY_PSQL_CMD")
    if not url and not psql_cmd:
        print(
            "No Postgres URL found. Set TWENTY_DB_URL, DATABASE_URL or "
            "PG_DATABASE_URL, pass --db-url, or set TWENTY_PSQL_CMD."
        )
        return 1

    sql_file = Path(__file__).with_name("reset_twenty.sql")
    sql = sql_file.read_text()
    target = url.rsplit("@", 1)[-1] if url else "TWENTY_PSQL_CMD"

    if psql_cmd:
        print("Running hard SQL reset via TWENTY_PSQL_CMD...")
        result = subprocess.run(
            shlex.split(psql_cmd), input=sql, text=True
        )
        if result.returncode != 0:
            print(f"psql command exited with code {result.returncode}")
        return result.returncode

    psql = shutil.which("psql")
    if psql:
        print(f"Running hard SQL reset against {target}...")
        result = subprocess.run(
            [psql, url, "-v", "ON_ERROR_STOP=1"],
            input=sql,
            text=True,
        )
        if result.returncode != 0:
            print(f"psql exited with code {result.returncode}")
        return result.returncode

    try:
        import psycopg2
    except ImportError:
        psycopg2 = None
    if psycopg2 is not None:
        print(f"Running hard SQL reset against {target} with psycopg2...")
        with psycopg2.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        return 0

    try:
        import psycopg
    except ImportError:
        psycopg = None
    if psycopg is not None:
        print(f"Running hard SQL reset against {target} with psycopg...")
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        return 0

    print("psql not found and no Python Postgres driver is installed.")
    print("Install postgresql-client, or run: pip install psycopg2-binary")
    print(
        "Or set TWENTY_PSQL_CMD, for example: "
        "docker exec -i <db-container> psql -U postgres -d default"
    )
    return 1


def reset(
    repo: Path | None,
    url: str | None,
    token: str | None,
    db_url: str | None = None,
    known_before: dict[str, int] | None = None,
) -> int:
    mcp_url, mcp_token = get_credentials(repo, url, token)
    client = MCPClient(mcp_url, mcp_token)

    if known_before is not None:
        before = known_before
    else:
        try:
            before = client.clean_counts()
        except Exception as exc:
            print("MCP before-count failed:", exc)
            before = None

    deleted: dict[str, int] = {}
    after: dict[str, int] | None = None
    try:
        deleted = client.reset()
        after = client.clean_counts()
    except Exception as exc:
        print("MCP reset failed:", exc)

    print("before:", before)
    print("deleted:", deleted)

    if after is not None:
        print("after:", after)
        dirty = {k: v for k, v in after.items() if v != 0}
        if not dirty:
            print("OK: clean state.")
            return 0
        print("MCP reset left records:", dirty)
    else:
        print("MCP reset did not complete; falling back to SQL.")

    if reset_sql(repo, db_url) != 0:
        print("Hard SQL reset failed or no database URL is configured.")
        return 1

    try:
        after = client.clean_counts()
    except Exception as exc:
        print("MCP after-count failed after SQL reset:", exc)
        return 0

    print("after SQL reset:", after)
    dirty = {k: v for k, v in after.items() if v != 0}
    if dirty:
        print("WARNING: records remain after SQL reset:", dirty)
        print("Restart Twenty if it caches records, then rerun.")
        return 1
    print("OK: clean state.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()
    raise SystemExit(
        reset(args.repo, args.url, args.token, args.db_url)
    )


if __name__ == "__main__":
    main()
