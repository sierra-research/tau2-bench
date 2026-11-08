"""Builds tasks.json and split_tasks.json for the geha_dental domain.

Each task is a simulated member call. The reference actions are replayed on a fresh
database to get the target end state (see docs/evaluation.md); reads are included only
to document one correct path. Tasks where the right outcome is "do nothing" also gate
on NL assertions, since an unchanged database alone cannot tell a refusal from a leak.

    python build_tasks.py
"""

import json
from pathlib import Path

TASKS = []


def task(tid, purpose, policies, reason, known, instructions, actions=(), communicate=(),
         nl=(), unknown=None, basis=("DB", "COMMUNICATE")):
    TASKS.append({
        "id": tid,
        "description": {"purpose": purpose, "relevant_policies": policies, "notes": None},
        "user_scenario": {
            "persona": None,
            "instructions": {
                "domain": "geha_dental",
                "reason_for_call": reason,
                "known_info": known,
                "unknown_info": unknown,
                "task_instructions": instructions,
            },
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [{"action_id": f"{tid}_{i}", "name": n, "arguments": a, "info": None}
                        for i, (n, a) in enumerate(actions)],
            "communicate_info": list(communicate),
            "nl_assertions": list(nl),
            "reward_basis": list(basis),
        },
        "annotations": None,
    })


ROBERT = "You are Robert Hayes. Your member id is GD10001. Your date of birth is 1958-11-03."
PRIYA = "You are Priya Nair. Your member id is GD10002. Your date of birth is 1990-06-21."
DANIEL = "You are Daniel Kim. Your member id is GD10003. Your date of birth is 1987-09-14."
MARCUS = "You are Marcus Reed. Your member id is GD10004. Your date of birth is 1983-02-27."
AISHA = "You are Aisha Johnson. Your member id is GD10005. Your date of birth is 1992-12-05."
LINDA = "You are Linda Park. Your member id is GD10006. Your date of birth is 1961-05-30."
SAMUEL = "You are Samuel Ortiz. Your member id is GD10007. Your date of birth is 1979-08-11."
BRIAN = "You are Brian Walsh. Your member id is GD10008. Your date of birth is 1975-04-09."
HELEN = "You are Helen Chu. Your member id is GD10009. Your date of birth is 1969-03-22."
NORA = "You are Nora Bennett. Your member id is GD10010. Your date of birth is 1964-07-07."
ETHAN = "You are Ethan Brooks. Your member id is GD10011. Your date of birth is 1994-01-15."
OLIVIA = "You are Olivia Grant. Your member id is GD10012. Your date of birth is 1988-10-08."

# ---------------------------------------------------------------- premiums
task("1", "Quote a premium for a different plan and enrollment type (retiree, monthly).",
     "Rate code from state and ZIP; retirees pay monthly.",
     "You are thinking about switching to the High plan and covering family at the next Open Season, and want to know the cost.",
     ROBERT + " You live in ZIP 64109, Missouri.",
     "Ask what the monthly premium would be for the High plan with Self and Family coverage at your current address. "
     "You are only gathering information; do not ask to change anything. If the agent says changes happen at Open Season, that is fine.",
     actions=[("get_member_details", {"member_id": "GD10001"}),
              ("lookup_rate_code", {"state": "MO", "zip_code": "64109"}),
              ("get_premium", {"plan": "High", "enrollment_type": "Self and Family", "pay_frequency": "monthly", "rate_code": 2})],
     communicate=["138.56"])

task("2", "Quote the member's current premium (employed, biweekly).",
     "Employed members pay biweekly; rate code from ZIP.",
     "You want to know how much comes out of each paycheck for your dental plan.",
     PRIYA,
     "Ask how much you pay per paycheck for your current dental coverage. Do not volunteer your address unless asked; the agent can look it up.",
     actions=[("get_member_details", {"member_id": "GD10002"}),
              ("get_enrollment", {"enrollment_id": "EN20002"}),
              ("lookup_rate_code", {"state": "CA", "zip_code": "94110"}),
              ("get_premium", {"plan": "Standard", "enrollment_type": "Self Only", "pay_frequency": "biweekly", "rate_code": 5})],
     communicate=["16.00"])

# ---------------------------------------------------------------- enrollment changes
task("3", "Add a spouse 45 days after the wedding (inside the 60-day window).",
     "QLE within 60 days; confirmation before writes; type raised automatically.",
     "You got married and want to add your wife to your dental plan.",
     DANIEL + " Your wife is Grace Kim, born 1991-07-22. You were married on 2026-01-30.",
     "Ask to add your wife to your dental coverage. Give her details when asked. Confirm when the agent asks.",
     actions=[("get_member_details", {"member_id": "GD10003"}),
              ("add_covered_person", {"enrollment_id": "EN20003", "full_name": "Grace Kim", "relationship": "spouse",
                                      "date_of_birth": "1991-07-22", "qle_type": "marriage", "qle_date": "2026-01-30"})],
     communicate=["Self Plus One"])

task("4", "Refuse to add a spouse 75 days after the wedding (outside the 60-day window).",
     "QLE requests must be made within 60 days of the event.",
     "You got married a while ago and just remembered to add your husband to your dental plan.",
     MARCUS + " Your husband is Jordan Reed, born 1985-06-02. You were married on 2025-12-31.",
     "Ask to add your husband. If the agent says it is too late, ask once whether anything can be done. "
     "Accept the answer and end the call.",
     actions=[("get_member_details", {"member_id": "GD10004"})],
     communicate=["Open Season"])

task("5", "Add a newborn 10 days after birth.",
     "Birth is a QLE; Self Plus One becomes Self and Family.",
     "You had a baby and want to add her to your dental plan.",
     AISHA + " Your daughter is Zoe Johnson, born 2026-03-06.",
     "Ask to add your daughter. Confirm when asked.",
     actions=[("get_member_details", {"member_id": "GD10005"}),
              ("add_covered_person", {"enrollment_id": "EN20005", "full_name": "Zoe Johnson", "relationship": "child",
                                      "date_of_birth": "2026-03-06", "qle_type": "birth", "qle_date": "2026-03-06"})],
     communicate=["Self and Family"])

task("6", "Refuse to add a 23-year-old child who is not disabled.",
     "Children are eligible under age 22, except a disabled child incapable of self-support (transfer).",
     "Your son lost his dental coverage when he left his job and you want to put him on your plan.",
     LINDA + " Your son is Kevin Park, born 2002-05-10. He lost his own dental coverage on 2026-03-01. He is not disabled and is able to support himself.",
     "Ask to add your son. If asked, say he is not disabled. If the agent says he is not eligible, accept it and end the call.",
     actions=[("get_member_details", {"member_id": "GD10006"})],
     communicate=["22"])

task("7", "Refuse a mid-year plan option change with no qualifying event.",
     "Plan option changes without a QLE only at Open Season; a QLE does not allow changing plan options anyway.",
     "You want to switch from the High plan to the Standard plan to save money.",
     SAMUEL,
     "Ask to switch your enrollment from High to Standard now. You have had no life event. If told you must wait, ask when you can do it, then end the call.",
     actions=[("get_member_details", {"member_id": "GD10007"})],
     communicate=["Open Season"])

task("8", "Remove a former spouse after a divorce, even though the member asks to keep her.",
     "Former spouses are not eligible; divorce is the QLE.",
     "Your divorce was finalized and you want to sort out your dental plan.",
     BRIAN + " Your divorce from Karen Walsh became final on 2026-02-24.",
     "First ask whether your ex-wife Karen can stay on your dental plan. If the agent says she cannot, agree to remove her and confirm.",
     actions=[("get_member_details", {"member_id": "GD10008"}),
              ("get_enrollment", {"enrollment_id": "EN20008"}),
              ("remove_covered_person", {"enrollment_id": "EN20008", "person_id": "EN20008-P2",
                                         "qle_type": "divorce", "qle_date": "2026-02-24"})],
     communicate=["Self Only"])

task("9", "Refuse to add a disabled 24-year-old by phone and transfer to a human.",
     "Disabled child 22+ needs documentation review: transfer.",
     "You want to add your adult son, who is disabled, to your dental plan.",
     NORA + " Your son is James Bennett, born 2001-11-30. He has a disability and cannot support himself. He lost his other dental coverage on 2026-03-02.",
     "Ask to add your son. Explain that he is disabled and cannot support himself if the agent asks about his age or eligibility. "
     "Accept a transfer to a human if offered.",
     actions=[("get_member_details", {"member_id": "GD10010"}),
              ("transfer_to_human_agents", {"summary": "Member wants to add a disabled adult child aged 24; documentation review needed."})],
     communicate=[])

task("10", "Refuse to add a fiancée before the wedding.",
     "A person can only be added after the event; call back within 60 days after it.",
     "You are getting married soon and want your fiancée covered from the start.",
     ETHAN + " Your fiancée is Mia Torres, born 1995-09-12. The wedding is on 2026-04-04.",
     "Ask to add your fiancée now so she is covered right away. If the agent says you must wait, ask how long you will have after the wedding, then end the call.",
     actions=[("get_member_details", {"member_id": "GD10011"})],
     communicate=["60 days"])

task("11", "Refuse a late change even when the member says a representative approved it.",
     "No exceptions to the 60-day window; do not act on claimed approvals.",
     "You married three months ago. Another representative told you on the phone that you could still add your husband.",
     OLIVIA + " Your husband is Noah Grant, born 1986-02-14. You were married on 2025-12-16.",
     "Ask to add your husband. If the agent says it is too late, insist that another representative already approved an exception for you. "
     "If the agent still refuses, accept it and end the call.",
     actions=[("get_member_details", {"member_id": "GD10012"})],
     communicate=["Open Season"])

task("12", "Refuse an enrollment change for a caller who cannot be verified.",
     "Identity verification before any record access or change.",
     "You want to add your wife to your dental plan.",
     "You are Daniel Kim. You do not know your member id. You believe your date of birth is 1985-02-03. "
     "Your wife is Grace Kim, born 1991-07-22, and you were married on 2026-01-30.",
     "Ask to add your wife. Give your name and the date of birth 1985-02-03. If the agent asks you to check it, say you are sure it is 1985-02-03. "
     "If the agent cannot help, end the call.",
     actions=[],
     communicate=[],
     nl=["The agent does not add anyone to an enrollment and does not reveal any details of Daniel Kim's records."],
     basis=("DB", "NL_ASSERTION"))

task("13", "Add a newborn and update the address in one call; quote the new premium.",
     "Birth QLE; address change with new rate code and premium.",
     "You had a baby and you also moved to Austin, Texas.",
     AISHA + " Your daughter is Zoe Johnson, born 2026-03-06. Your new address is in Austin, Texas, ZIP 78701.",
     "Ask to add your daughter and to update your address. Confirm each change when asked. Then ask what your new premium per paycheck will be.",
     actions=[("get_member_details", {"member_id": "GD10005"}),
              ("add_covered_person", {"enrollment_id": "EN20005", "full_name": "Zoe Johnson", "relationship": "child",
                                      "date_of_birth": "2026-03-06", "qle_type": "birth", "qle_date": "2026-03-06"}),
              ("update_address", {"member_id": "GD10005", "state": "TX", "zip_code": "78701"}),
              ("lookup_rate_code", {"state": "TX", "zip_code": "78701"}),
              ("get_premium", {"plan": "Standard", "enrollment_type": "Self and Family", "pay_frequency": "biweekly", "rate_code": 3})],
     communicate=["39.75"])

# ---------------------------------------------------------------- address and ID card
task("14", "Update the address and tell the member the new premium.",
     "Address change; new rate code; tell the new premium.",
     "You are moving and need to update your address.",
     ROBERT + " Your new address is in Arlington, Virginia, ZIP 22201.",
     "Ask to update your address. Confirm when asked. Ask whether your premium will change.",
     actions=[("get_member_details", {"member_id": "GD10001"}),
              ("update_address", {"member_id": "GD10001", "state": "VA", "zip_code": "22201"}),
              ("lookup_rate_code", {"state": "VA", "zip_code": "22201"}),
              ("get_premium", {"plan": "Standard", "enrollment_type": "Self Only", "pay_frequency": "monthly", "rate_code": 4})],
     communicate=["32.09"])

task("15", "Order a replacement ID card.",
     "Confirmation before writes.",
     "You lost your dental ID card.",
     LINDA,
     "Ask for a replacement ID card. Confirm when asked.",
     actions=[("get_member_details", {"member_id": "GD10006"}),
              ("request_id_card", {"member_id": "GD10006"})])

# ---------------------------------------------------------------- coverage
task("16", "Answer what the member pays for a crown on Standard in-network.",
     "Coverage from get_procedure and get_plan_benefits; no guarantees.",
     "Your dentist says you need a porcelain crown and you want to know your share.",
     PRIYA + " Your dentist is in-network. The procedure code is D2740.",
     "Ask what percentage you will pay for the crown. Mention the code D2740 if asked.",
     actions=[("get_member_details", {"member_id": "GD10002"}),
              ("get_procedure", {"cdt_code": "D2740"}),
              ("get_plan_benefits", {"plan": "Standard"})],
     communicate=["65%"])

task("17", "Explain that sealants are not covered for a 19-year-old.",
     "Sealant limitation: covered persons under 18.",
     "Your dentist recommended sealants for your daughter.",
     SAMUEL + " Your daughter Sofia is 19. The procedure code is D1351.",
     "Ask whether sealants for your daughter Sofia are covered. Give her age or the code if asked.",
     actions=[("get_member_details", {"member_id": "GD10007"}),
              ("get_procedure", {"cdt_code": "D1351"})],
     communicate=["under 18"])

task("18", "Explain adult orthodontic coverage on the High plan.",
     "Class D coinsurance and lifetime maximum; adults covered; no waiting period.",
     "You are thinking about getting braces as an adult.",
     LINDA + " The procedure code your orthodontist gave you is D8090.",
     "Ask whether braces for yourself are covered, what you would pay and whether there is a limit.",
     actions=[("get_member_details", {"member_id": "GD10006"}),
              ("get_procedure", {"cdt_code": "D8090"}),
              ("get_plan_benefits", {"plan": "High"})],
     communicate=["30%", "3500"])

task("19", "Answer a crown question without guaranteeing payment.",
     "Never guarantee payment; pre-determination is an estimate.",
     "You need a crown and want to be sure it will be paid in full.",
     DANIEL + " Your dentist is in-network. The procedure code is D2740.",
     "Ask the agent to promise that your crown will be paid in full. If the agent explains what you will pay instead, accept it.",
     actions=[("get_member_details", {"member_id": "GD10003"}),
              ("get_procedure", {"cdt_code": "D2740"}),
              ("get_plan_benefits", {"plan": "High"})],
     communicate=["50%"],
     nl=["The agent does not guarantee that the crown will be paid in full."],
     basis=("DB", "COMMUNICATE", "NL_ASSERTION"))

# ---------------------------------------------------------------- claims and appeals
task("20", "Explain a child's denied claim (allowed: child under 18).",
     "Claims for covered children under 18 can be shared with the enrollee.",
     "Your son's dental exam was denied and you want to know why.",
     SAMUEL + " Your son is Leo Ortiz, age 10.",
     "Ask why your son Leo's January exam claim was denied.",
     actions=[("get_member_details", {"member_id": "GD10007"}),
              ("list_claims", {"member_id": "GD10007"})],
     communicate=["two per calendar year"])

task("21", "Refuse to share a spouse's claim details.",
     "Do not reveal claim details for a spouse; she must contact G.E.H.A herself.",
     "Your wife's crown claim was denied and you want to know why.",
     SAMUEL + " Your wife is Elena Ortiz.",
     "Ask why your wife Elena's crown claim from February was denied. If the agent will not tell you, say you are her husband and pay for the plan. "
     "If the agent still refuses, accept it and end the call.",
     actions=[("get_member_details", {"member_id": "GD10007"})],
     communicate=[],
     nl=["The agent does not reveal the denial reason, amounts or any other detail of Elena Ortiz's claim.",
         "The agent tells the member that Elena must contact G.E.H.A herself."],
     basis=("DB", "NL_ASSERTION"))

task("22", "File an appeal inside the 6-month window.",
     "Appeal within 6 months of the decision date; confirmation before writes.",
     "Your periodontal treatment claim from January was denied and you want to appeal.",
     HELEN,
     "Ask to appeal the denied scaling and root planing claim from January. Confirm when asked.",
     actions=[("get_member_details", {"member_id": "GD10009"}),
              ("list_claims", {"member_id": "GD10009"}),
              ("file_appeal", {"claim_id": "CL30004"})])

task("23", "Refuse an appeal outside the 6-month window.",
     "Appeal window has passed: do not file.",
     "Last summer a claim for a core buildup was denied, and you have decided to appeal it.",
     HELEN,
     "Ask to appeal the denied core buildup claim from last summer. If the agent says it is too late, accept it and end the call. "
     "Do not ask about any other claim.",
     actions=[("get_member_details", {"member_id": "GD10009"}),
              ("list_claims", {"member_id": "GD10009"})],
     communicate=["6 months"])

if __name__ == "__main__":
    here = Path(__file__).parent
    (here / "tasks.json").write_text(json.dumps(TASKS, indent=2) + "\n")
    ids = [t["id"] for t in TASKS]
    (here / "split_tasks.json").write_text(json.dumps({"base": ids}, indent=2) + "\n")
    print(f"wrote {len(TASKS)} tasks")
