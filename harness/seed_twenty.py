#!/usr/bin/env python3
"""Seed the Twenty CRM fixtures for one v2 task and build matching local state.

The local CRMDB mirror must use the same UUIDs as Twenty, otherwise the agent's
read tools (which read the mirror) and the verifiers (which read Twenty) diverge.
This script creates the records in Twenty, captures the returned UUIDs, and
writes both representations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twenty_mcp import MCPClient, get_credentials


def seed_task(client: MCPClient, fixture: dict) -> dict:
    agent_data: dict[str, dict] = {
        "companies": {},
        "people": {},
        "opportunities": {},
        "tasks": {},
    }

    company_ids: dict[str, str] = {}
    for company in fixture.get("companies", []):
        name = company["name"]
        company_id = client.create_company(name)
        company_ids[name] = company_id
        agent_data["companies"][company_id] = {
            "id": company_id,
            "name": name,
            "industry": company.get("industry"),
            "annual_revenue": None,
            "employee_count": None,
        }

    person_ids: dict[tuple[str, str], str] = {}
    for person in fixture.get("people", []):
        first = person["first_name"]
        last = person["last_name"]
        company_name = person.get("company")
        company_id = company_ids.get(company_name) if company_name else None
        person_id = client.create_person(
            first,
            last,
            company_id=company_id,
            job_title=person.get("job_title"),
        )
        person_ids[(first, last)] = person_id
        agent_data["people"][person_id] = {
            "id": person_id,
            "first_name": first,
            "last_name": last,
            "email": person.get("email"),
            "phone": person.get("phone"),
            "job_title": person.get("job_title"),
            "company_id": company_id,
            "lead_score": 0,
            "relationship_stage": "prospect",
            "trust_level": 50,
            "commitment_level": 0,
        }

    for opportunity in fixture.get("opportunities", []):
        person_key = tuple(opportunity.get("person", []))
        person_id = person_ids.get(person_key)
        if not person_id:
            raise RuntimeError(f"Opportunity person not found: {person_key}")
        company_name = opportunity.get("company")
        company_id = company_ids.get(company_name) if company_name else None
        stage = opportunity.get("stage", "NEW")
        opportunity_id = client.create_opportunity(
            opportunity["name"],
            person_id,
            company_id=company_id,
            stage=stage,
        )
        agent_data["opportunities"][opportunity_id] = {
            "id": opportunity_id,
            "name": opportunity["name"],
            "person_id": person_id,
            "company_id": company_id,
            "stage": stage,
            "value": opportunity.get("value"),
            "probability": 0.1,
            "expected_close_date": None,
        }

    for task in fixture.get("tasks", []):
        title = task["title"]
        task_id = client.create_task(title)
        person_key = tuple(task.get("person", []))
        person_id = person_ids.get(person_key) if person_key else None
        company_name = task.get("company")
        company_id = company_ids.get(company_name) if company_name else None
        agent_data["tasks"][task_id] = {
            "id": task_id,
            "subject": title,
            "person_id": person_id,
            "company_id": company_id,
            "status": "open",
            "priority": None,
        }

    return agent_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    mcp_url, mcp_token = get_credentials(args.repo, args.url, args.token)
    client = MCPClient(mcp_url, mcp_token)

    fixtures = json.loads(args.fixtures.read_text())
    fixture = fixtures[str(args.task_id)]
    agent_data = seed_task(client, fixture)

    text = json.dumps(agent_data, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
