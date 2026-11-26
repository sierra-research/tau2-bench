#!/usr/bin/env python3
"""Generate the v2 CRM task set and fixtures.

Input:  the original tau2 CRM tasks.json (v1), default /tmp/crm_tasks.json
Output: tasks_v2.json, fixtures_v2.json, task_map.md

The v2 design removes the LLM judge from the reward, gives every task a
distinct field-level target state, and seeds fixtures for the existing-customer
scenarios.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def assertion(func: str, **arguments) -> dict:
    return {"env_type": "assistant", "func_name": func, "arguments": arguments}


SCENARIOS = {
    "0": {
        "paper_label": "T1 Lead creation and qualification",
        "short": "Skeptical owner, new lead",
        "instructions": (
            "You are Sarah Chen, CEO of TechStart Inc. You are evaluating CRM "
            "options and you are skeptical because you have been burned before. "
            "Ask detailed questions about implementation, support and evidence. "
            "As part of the conversation, ask the agent to create a lead record "
            "for you (Sarah Chen) linked to the existing TechStart Inc company, "
            "and to create a CRM opportunity named 'TechStart CRM Evaluation'."
        ),
        "fixtures": {
            "companies": [{"name": "TechStart Inc", "industry": "SaaS"}],
            "people": [],
            "opportunities": [],
            "tasks": [],
        },
        "assertions": [
            assertion("verify_person_named_company", first_name="Sarah", last_name="Chen", company_name="TechStart Inc"),
            assertion("verify_opportunity_named", first_name="Sarah", last_name="Chen", name_contains="TechStart CRM Evaluation"),
            assertion("verify_people_total", count=1),
            assertion("verify_opportunities_total", count=1),
        ],
    },
    "1": {
        "paper_label": "T2 Customer trust building",
        "short": "Stubborn buyer, trust task",
        "instructions": (
            "You are Mark Reynolds, a stubborn decision maker at Enterprise Corp. "
            "You were burned by a vendor before and you do not trust salespeople. "
            "Ask for specific evidence and do not commit. As part of the "
            "conversation, ask the agent to create a lead record for you (Mark "
            "Reynolds) linked to the existing Enterprise Corp company, and to "
            "create a follow-up task titled 'Trust-building follow-up'."
        ),
        "fixtures": {
            "companies": [{"name": "Enterprise Corp", "industry": "Manufacturing"}],
            "people": [],
            "opportunities": [],
            "tasks": [],
        },
        "assertions": [
            assertion("verify_person_named_company", first_name="Mark", last_name="Reynolds", company_name="Enterprise Corp"),
            assertion("verify_task_named", title_contains="Trust-building follow-up"),
            assertion("verify_people_total", count=1),
            assertion("verify_tasks_total", count=1),
            assertion("verify_opportunities_total", count=0),
        ],
    },
    "2": {
        "paper_label": "T3 Customer relationship development",
        "short": "Analytical manager, pilot",
        "instructions": (
            "You are Priya Sharma, an analytical evaluation manager at Growth Inc. "
            "You test vendors on technical competence and reliability. Ask detailed "
            "technical questions. As part of the conversation, ask the agent to "
            "create a lead record for you (Priya Sharma) linked to the existing "
            "Growth Inc company, and to create a pilot opportunity named 'Growth "
            "Pilot' at the screening stage."
        ),
        "fixtures": {
            "companies": [{"name": "Growth Inc", "industry": "Fintech"}],
            "people": [],
            "opportunities": [],
            "tasks": [],
        },
        "assertions": [
            assertion("verify_person_named_company", first_name="Priya", last_name="Sharma", company_name="Growth Inc"),
            assertion("verify_opportunity_named", first_name="Priya", last_name="Sharma", name_contains="Growth Pilot"),
            assertion("verify_opportunity_stage", first_name="Priya", last_name="Sharma", stage="SCREENING"),
            assertion("verify_people_total", count=1),
            assertion("verify_opportunities_total", count=1),
        ],
    },
    "3": {
        "paper_label": "T4 Evidence and trial",
        "short": "Evidence-seeking buyer, trial",
        "instructions": (
            "You are David Kim, an evidence-seeking buyer at ScaleUp Inc. You want "
            "case studies, references and measurable results before you trust a "
            "vendor. As part of the conversation, ask the agent to create a lead "
            "record for you (David Kim) linked to the existing ScaleUp Inc company, "
            "and to create a trial opportunity named 'ScaleUp Trial'."
        ),
        "fixtures": {
            "companies": [{"name": "ScaleUp Inc", "industry": "Logistics"}],
            "people": [],
            "opportunities": [],
            "tasks": [],
        },
        "assertions": [
            assertion("verify_person_named_company", first_name="David", last_name="Kim", company_name="ScaleUp Inc"),
            assertion("verify_opportunity_named", first_name="David", last_name="Kim", name_contains="ScaleUp Trial"),
            assertion("verify_people_total", count=1),
            assertion("verify_opportunities_total", count=1),
        ],
    },
    "4": {
        "paper_label": "T5 Follow-up and commitment management",
        "short": "Cautious tester, small pilot + follow-up",
        "instructions": (
            "You are Emma Rodriguez, a cautious commitment tester at Midway Corp. "
            "You test reliability with small commitments before you commit to "
            "anything larger. As part of the conversation, ask the agent to create "
            "a lead record for you (Emma Rodriguez) linked to the existing Midway "
            "Corp company, to create a small pilot opportunity named 'Midway Small "
            "Pilot', and to create a follow-up task titled 'Follow up in 7 days'."
        ),
        "fixtures": {
            "companies": [{"name": "Midway Corp", "industry": "Retail"}],
            "people": [],
            "opportunities": [],
            "tasks": [],
        },
        "assertions": [
            assertion("verify_person_named_company", first_name="Emma", last_name="Rodriguez", company_name="Midway Corp"),
            assertion("verify_opportunity_named", first_name="Emma", last_name="Rodriguez", name_contains="Midway Small Pilot"),
            assertion("verify_task_named", title_contains="Follow up in 7 days"),
            assertion("verify_people_total", count=1),
            assertion("verify_opportunities_total", count=1),
            assertion("verify_tasks_total", count=1),
        ],
    },
    "5": {
        "paper_label": "T6 Sales opportunity management",
        "short": "Strategic buyer, partnership",
        "instructions": (
            "You are James Wilson, a strategic decision maker at FutureTech. You "
            "care about long-term value, scalability and partnership. As part of the "
            "conversation, ask the agent to create a lead record for you (James "
            "Wilson) linked to the existing FutureTech company, and to create a "
            "strategic opportunity named 'FutureTech Strategic Partnership' at the "
            "meeting stage."
        ),
        "fixtures": {
            "companies": [{"name": "FutureTech", "industry": "Technology"}],
            "people": [],
            "opportunities": [],
            "tasks": [],
        },
        "assertions": [
            assertion("verify_person_named_company", first_name="James", last_name="Wilson", company_name="FutureTech"),
            assertion("verify_opportunity_named", first_name="James", last_name="Wilson", name_contains="FutureTech Strategic Partnership"),
            assertion("verify_opportunity_stage", first_name="James", last_name="Wilson", stage="MEETING"),
            assertion("verify_people_total", count=1),
            assertion("verify_opportunities_total", count=1),
        ],
    },
    "6": {
        "paper_label": "T7 Customer complaint management",
        "short": "Challenged customer, stage change + follow-up",
        "instructions": (
            "You are Maria Garcia, an existing customer at Innovate Inc. You have an "
            "onboarding problem and you are frustrated. Ask the agent to move your "
            "existing opportunity 'Innovate Onboarding' to the proposal stage and to "
            "create a follow-up task titled 'Complaint follow-up'. Do not ask for a "
            "new lead record; your record already exists."
        ),
        "fixtures": {
            "companies": [{"name": "Innovate Inc", "industry": "Healthcare"}],
            "people": [
                {"first_name": "Maria", "last_name": "Garcia", "company": "Innovate Inc", "job_title": "COO"}
            ],
            "opportunities": [
                {"name": "Innovate Onboarding", "person": ["Maria", "Garcia"], "company": "Innovate Inc", "stage": "NEW"}
            ],
            "tasks": [],
        },
        "assertions": [
            assertion("verify_person_exists", first_name="Maria", last_name="Garcia"),
            assertion("verify_person_named_company", first_name="Maria", last_name="Garcia", company_name="Innovate Inc"),
            assertion("verify_opportunity_stage", first_name="Maria", last_name="Garcia", stage="PROPOSAL"),
            assertion("verify_task_named", title_contains="Complaint follow-up"),
            assertion("verify_people_total", count=1),
            assertion("verify_opportunities_total", count=1),
            assertion("verify_tasks_total", count=1),
        ],
    },
    "7": {
        "paper_label": "T8 Advocacy and referral",
        "short": "Advocate, referral opportunity",
        "instructions": (
            "You are Alex Thompson, an existing advocate customer at Success LLC. "
            "You are happy with the service and you are willing to refer another "
            "company. Ask the agent to create a referral opportunity named 'Success "
            "LLC Referral'. Do not ask for a new lead record; your record already "
            "exists."
        ),
        "fixtures": {
            "companies": [{"name": "Success LLC", "industry": "Consulting"}],
            "people": [
                {"first_name": "Alex", "last_name": "Thompson", "company": "Success LLC", "job_title": "Founder"}
            ],
            "opportunities": [],
            "tasks": [],
        },
        "assertions": [
            assertion("verify_person_exists", first_name="Alex", last_name="Thompson"),
            assertion("verify_person_named_company", first_name="Alex", last_name="Thompson", company_name="Success LLC"),
            assertion("verify_opportunity_named", first_name="Alex", last_name="Thompson", name_contains="Success LLC Referral"),
            assertion("verify_people_total", count=1),
            assertion("verify_opportunities_total", count=1),
        ],
    },
    "8": {
        "paper_label": "T9 Customer retention and renewal",
        "short": "Loyal customer, renewal opportunity",
        "instructions": (
            "You are Rachel Brown, a loyal customer at Enterprise Co. You want more "
            "value and you are discussing a renewal. Ask the agent to create a "
            "renewal opportunity named 'Enterprise Co Renewal'. Do not ask for a new "
            "lead record; your record already exists."
        ),
        "fixtures": {
            "companies": [{"name": "Enterprise Co", "industry": "Media"}],
            "people": [
                {"first_name": "Rachel", "last_name": "Brown", "company": "Enterprise Co", "job_title": "VP Operations"}
            ],
            "opportunities": [
                {"name": "Enterprise Co Subscription", "person": ["Rachel", "Brown"], "company": "Enterprise Co", "stage": "CUSTOMER"}
            ],
            "tasks": [],
        },
        "assertions": [
            assertion("verify_person_exists", first_name="Rachel", last_name="Brown"),
            assertion("verify_person_named_company", first_name="Rachel", last_name="Brown", company_name="Enterprise Co"),
            assertion("verify_opportunity_named", first_name="Rachel", last_name="Brown", name_contains="Enterprise Co Renewal"),
            assertion("verify_people_total", count=1),
            assertion("verify_opportunities_total", count=2),
        ],
    },
    "9": {
        "paper_label": "T10 Customer relationship recovery",
        "short": "At-risk customer, recovery opportunity + task",
        "instructions": (
            "You are Chris Martinez, an at-risk customer at Transition Corp. You "
            "have had several bad experiences and you are considering leaving. Ask "
            "the agent to create a churn-risk recovery opportunity named 'Transition "
            "Corp Recovery' and a task titled 'Recovery follow-up'. Do not ask for a "
            "new lead record; your record already exists."
        ),
        "fixtures": {
            "companies": [{"name": "Transition Corp", "industry": "Energy"}],
            "people": [
                {"first_name": "Chris", "last_name": "Martinez", "company": "Transition Corp", "job_title": "Director"}
            ],
            "opportunities": [
                {"name": "Transition Corp Subscription", "person": ["Chris", "Martinez"], "company": "Transition Corp", "stage": "SCREENING"}
            ],
            "tasks": [],
        },
        "assertions": [
            assertion("verify_person_exists", first_name="Chris", last_name="Martinez"),
            assertion("verify_person_named_company", first_name="Chris", last_name="Martinez", company_name="Transition Corp"),
            assertion("verify_opportunity_named", first_name="Chris", last_name="Martinez", name_contains="Transition Corp Recovery"),
            assertion("verify_task_named", title_contains="Recovery follow-up"),
            assertion("verify_people_total", count=1),
            assertion("verify_opportunities_total", count=2),
            assertion("verify_tasks_total", count=1),
        ],
    },
}


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/crm_tasks.json")
    original = json.loads(src.read_text())

    tasks = []
    fixtures = {}
    mapping = []

    for task in original:
        task_id = str(task["id"])
        scenario = SCENARIOS[task_id]
        new_task = json.loads(json.dumps(task))
        new_task["id"] = task_id
        new_task["description"]["purpose"] = (
            f"{scenario['short']}. {new_task['description'].get('purpose', '')}"
        ).strip()
        new_task["user_scenario"]["instructions"]["task_instructions"] = scenario["instructions"]
        new_task["initial_state"] = None
        new_task["evaluation_criteria"] = {
            "actions": [],
            "communicate_info": [],
            "nl_assertions": [],
            "reward_basis": ["ENV_ASSERTION"],
            "env_assertions": scenario["assertions"],
        }
        tasks.append(new_task)
        fixtures[task_id] = scenario["fixtures"]
        mapping.append(
            {
                "task_id": task_id,
                "paper_label": scenario["paper_label"],
                "short": scenario["short"],
                "fixtures": scenario["fixtures"],
                "assertions": scenario["assertions"],
            }
        )

    (HERE / "tasks_v2.json").write_text(json.dumps(tasks, indent=1) + "\n")
    (HERE / "fixtures_v2.json").write_text(json.dumps(fixtures, indent=1) + "\n")

    lines = [
        "# Task map v2",
        "",
        "The old Table 5 rows for 'Customer information retrieval' and 'CRM data",
        "maintenance' are removed because the domain contains no such tasks. The",
        "remaining rows are relabelled to the actual personas.",
        "",
        "| v2 ID | Paper label | Scenario | Fixtures |",
        "|---|---|---|---|",
    ]
    for row in mapping:
        lines.append(
            f"| {row['task_id']} | {row['paper_label']} | {row['short']} | "
            f"{len(row['fixtures']['companies'])} companies, "
            f"{len(row['fixtures']['people'])} people, "
            f"{len(row['fixtures']['opportunities'])} opportunities |"
        )
    (HERE / "task_map.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote {HERE/'tasks_v2.json'} ({len(tasks)} tasks)")
    print(f"Wrote {HERE/'fixtures_v2.json'}")
    print(f"Wrote {HERE/'task_map.md'}")


if __name__ == "__main__":
    main()
