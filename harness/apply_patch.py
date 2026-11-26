#!/usr/bin/env python3
"""Apply the v2 CRM verifier and strict-MCP patch to a tau2-bench-copy checkout.

Usage:
    python apply_patch.py --repo /path/to/23-tau2-bench-copy
    python apply_patch.py --repo ... --dry-run

The patch is idempotent: re-running it does nothing if the marker is present.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "    # --- v2 strict verifiers ---"

VERIFIERS = '''    # --- v2 strict verifiers ---

    @staticmethod
    def _v2_name(record: dict) -> tuple[str, str]:
        name = record.get("name") or {}
        if not isinstance(name, dict):
            name = {}
        first = record.get("firstName") or name.get("firstName") or ""
        last = record.get("lastName") or name.get("lastName") or ""
        return (first or "").strip().lower(), (last or "").strip().lower()

    def _v2_records(self, tool_name: str, arguments: dict | None = None) -> list[dict]:
        try:
            result = self._call_mcp(tool_name, arguments or {})
            return self._extract_find_many_records(result)
        except Exception:
            return []

    def _v2_people(self, first_name=None, last_name=None) -> list[dict]:
        people = self._v2_records("find_many_people", {"limit": 200})
        if first_name:
            needle = str(first_name).strip().lower()
            people = [p for p in people if self._v2_name(p)[0] == needle]
        if last_name:
            needle = str(last_name).strip().lower()
            people = [p for p in people if self._v2_name(p)[1] == needle]
        return people

    def _v2_companies(self, company_name=None) -> list[dict]:
        companies = self._v2_records("find_many_companies", {"limit": 200})
        if company_name:
            needle = str(company_name).strip().lower()
            companies = [
                c for c in companies
                if (c.get("name") or "").strip().lower() == needle
            ]
        return companies

    def _v2_opportunities(self) -> list[dict]:
        return self._v2_records("find_many_opportunities", {"limit": 200})

    def _v2_tasks(self, title_contains=None) -> list[dict]:
        tasks = self._v2_records("find_many_tasks", {"limit": 200})
        if title_contains:
            needle = str(title_contains).strip().lower()
            tasks = [
                t for t in tasks
                if needle in (t.get("title") or t.get("subject") or "").strip().lower()
            ]
        return tasks

    def verify_person_named_company(
        self, first_name: str, last_name: str, company_name: str
    ) -> bool:
        people = self._v2_people(first_name, last_name)
        if len(people) != 1:
            return False
        company_id = people[0].get("companyId")
        if isinstance(company_id, dict):
            company_id = company_id.get("id")
        if not company_id:
            company = people[0].get("company") or {}
            if isinstance(company, dict):
                company_id = company.get("id")
        if not company_id:
            return False
        companies = self._v2_companies(company_name)
        return any(c.get("id") == company_id for c in companies)

    def verify_opportunity_named(
        self, first_name: str, last_name: str, name_contains: str
    ) -> bool:
        people = self._v2_people(first_name, last_name)
        if len(people) != 1:
            return False
        opportunities = self._find_opportunities_in_twenty(people[0]["id"])
        needle = (name_contains or "").strip().lower()
        return any(
            needle in (o.get("name") or "").strip().lower()
            for o in opportunities
        )

    def verify_task_named(self, title_contains: str) -> bool:
        return len(self._v2_tasks(title_contains)) >= 1

    def verify_people_total(self, count: int) -> bool:
        return len(self._v2_people()) == count

    def verify_opportunities_total(self, count: int) -> bool:
        return len(self._v2_opportunities()) == count

    def verify_tasks_total(self, count: int) -> bool:
        return len(self._v2_tasks()) == count

    def verify_person_count(self, first_name: str, last_name: str, count: int) -> bool:
        return len(self._v2_people(first_name, last_name)) == count

    def verify_opportunity_count(
        self, first_name: str, last_name: str, count: int
    ) -> bool:
        people = self._v2_people(first_name, last_name)
        if len(people) != 1:
            return False
        return len(self._find_opportunities_in_twenty(people[0]["id"])) == count

    def verify_clean_state(self) -> bool:
        return (
            len(self._v2_people()) == 0
            and len(self._v2_companies()) == 0
            and len(self._v2_opportunities()) == 0
            and len(self._v2_tasks()) == 0
        )

'''

REPLACEMENTS = [
    (
        "                # Fallback to local ID if MCP fails\n"
        "                person_id = self._get_new_person_id()\n",
        "                raise RuntimeError(\n"
        "                    \"MCP create_one_person failed: \"\n"
        "                    f\"{mcp_result.get('message') or mcp_result}\"\n"
        "                )\n",
    ),
    (
        "            else:\n"
        "                opportunity_id = self._get_new_opportunity_id()\n",
        "            else:\n"
        "                raise RuntimeError(\n"
        "                    \"MCP create_one_opportunity failed: \"\n"
        "                    f\"{mcp_result.get('message') or mcp_result}\"\n"
        "                )\n",
    ),
    (
        "            else:\n"
        "                task_id = self._get_new_task_id()\n",
        "            else:\n"
        "                raise RuntimeError(\n"
        "                    \"MCP create_one_task failed: \"\n"
        "                    f\"{mcp_result.get('message') or mcp_result}\"\n"
        "                )\n",
    ),
    (
        "        except Exception:\n"
        "            # MCP update failed, but local update succeeded\n"
        "            pass\n",
        "        except Exception as exc:\n"
        "            raise RuntimeError(f\"MCP update failed: {exc}\") from exc\n",
    ),
]


def apply(repo: Path, dry_run: bool) -> int:
    tools = repo / "src" / "tau2" / "domains" / "crm" / "tools.py"
    if not tools.exists():
        print(f"ERROR: not found: {tools}", file=sys.stderr)
        return 2

    text = tools.read_text()
    changed = []

    if MARKER in text:
        print("Verifier block already present; skipping insertion.")
    else:
        anchor = "    def _get_person(self, person_id: str) -> Person:\n"
        if anchor not in text:
            print("ERROR: anchor _get_person not found in tools.py", file=sys.stderr)
            return 2
        text = text.replace(anchor, VERIFIERS + anchor, 1)
        changed.append("inserted verifier block")

    for old, new in REPLACEMENTS:
        count = text.count(old)
        if count == 0:
            continue
        if count > 1:
            print(f"WARNING: replacement matched {count} times; applying all.")
        text = text.replace(old, new)
        changed.append(f"replaced fallback ({count}x)")

    if not changed:
        print("No changes needed.")
        return 0

    print("Changes:")
    for item in changed:
        print(f"  - {item}")

    if dry_run:
        print("Dry run: no files written.")
        return 0

    backup = tools.with_suffix(".py.bak-v2")
    if not backup.exists():
        shutil.copy2(tools, backup)
        print(f"Backup: {backup}")
    tools.write_text(text)
    print(f"Wrote: {tools}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(apply(args.repo, args.dry_run))


if __name__ == "__main__":
    main()
