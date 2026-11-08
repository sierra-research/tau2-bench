"""Builds db.json for the geha_dental domain.

Plan tables (rate codes, premiums, coinsurance, procedure limits) come from the public
2026 G.E.H.A dental benefits guide (pp. 5, 10-11) and the 2026 plan brochure (Sections 4-5).
Members, enrollments and claims are synthetic.

    python build_db.py
"""

import json
from pathlib import Path

R = lambda *pairs: [[a, b] for a, b in pairs]  # zip3 ranges

# guide p. 10: state, zip3 ranges (empty = rest of state / entire state), rate code
RATE_RULES = [
    ("AK", [], 5), ("AL", [], 1), ("AR", [], 1),
    ("AZ", R(("850", "853"), ("864", "864")), 3), ("AZ", [], 2),
    ("CA", R(("900", "908"), ("910", "928"), ("930", "931"), ("933", "935"), ("939", "952"),
             ("954", "954"), ("956", "959")), 5), ("CA", [], 4),
    ("CO", [], 4), ("CT", R(("060", "063")), 4), ("CT", R(("064", "069")), 5),
    ("DC", [], 4), ("DE", [], 3),
    ("FL", R(("329", "334"), ("349", "349")), 3), ("FL", [], 2),
    ("GA", R(("300", "303"), ("305", "306"), ("311", "311"), ("399", "399")), 3), ("GA", [], 2),
    ("GU", [], 1), ("HI", [], 3), ("IA", [], 1), ("ID", [], 2),
    ("IL", R(("600", "609"), ("613", "613")), 3), ("IL", R(("620", "620"), ("622", "622")), 2), ("IL", [], 1),
    ("IN", R(("460", "462"), ("470", "470"), ("472", "473")), 2), ("IN", R(("463", "464")), 3), ("IN", [], 1),
    ("KS", R(("660", "662"), ("666", "666")), 2), ("KS", [], 1),
    ("KY", R(("410", "410")), 2), ("KY", [], 1), ("LA", [], 2),
    ("MA", R(("012", "012")), 2), ("MA", R(("010", "011"), ("013", "027"), ("055", "055")), 4),
    ("MD", R(("205", "212"), ("214", "214"), ("216", "217")), 4), ("MD", R(("219", "219")), 3), ("MD", [], 2),
    ("ME", R(("039", "042")), 4), ("ME", [], 3),
    ("MI", R(("480", "485")), 3), ("MI", [], 2),
    ("MN", R(("550", "551"), ("553", "555"), ("563", "563")), 3), ("MN", [], 2),
    ("MO", R(("726", "726")), 1), ("MO", [], 2),
    ("MS", [], 1), ("MT", [], 2), ("NC", [], 2), ("ND", [], 1), ("NE", [], 1), ("NH", [], 4),
    ("NJ", R(("080", "084")), 3), ("NJ", R(("070", "079"), ("085", "089")), 5), ("NM", [], 3),
    ("NV", R(("897", "897")), 5), ("NV", [], 3),
    ("NY", R(("005", "005"), ("100", "119"), ("124", "126")), 5), ("NY", R(("063", "063")), 4),
    ("NY", R(("120", "123"), ("128", "128"), ("140", "143")), 2), ("NY", [], 1),
    ("OH", R(("430", "433"), ("437", "437"), ("440", "443"), ("446", "447"), ("450", "455"), ("459", "459")), 2),
    ("OH", [], 1), ("OK", [], 2), ("OR", [], 3),
    ("PA", R(("172", "174")), 4), ("PA", R(("180", "181"), ("183", "183")), 5), ("PA", R(("189", "196")), 3), ("PA", [], 1),
    ("PR", [], 1), ("RI", [], 4), ("SC", [], 2), ("SD", [], 2), ("TN", [], 2),
    ("TX", R(("739", "739"), ("750", "754"), ("760", "762"), ("770", "770"), ("772", "775"), ("780", "782")), 2),
    ("TX", R(("733", "733"), ("786", "787")), 3), ("TX", [], 1),
    ("UT", [], 2), ("VA", R(("201", "201"), ("205", "205"), ("220", "227")), 4), ("VA", [], 2),
    ("VI", [], 1), ("VT", [], 2), ("WA", R(("980", "985")), 5), ("WA", R(("986", "986")), 3), ("WA", [], 4),
    ("WI", R(("540", "540")), 3), ("WI", [], 2), ("WV", R(("254", "254")), 4), ("WV", [], 1),
    ("WY", R(("834", "834")), 2), ("WY", [], 1), ("INTL", [], 5),
]

# guide p. 11: rate codes 1..5
PREMIUMS = {
    "Standard": {
        "biweekly": {"Self Only": [10.82, 12.11, 13.27, 14.81, 16.00],
                     "Self Plus One": [21.61, 24.22, 26.48, 29.60, 32.00],
                     "Self and Family": [32.41, 36.24, 39.75, 44.39, 48.00]},
        "monthly": {"Self Only": [23.44, 26.24, 28.75, 32.09, 34.67],
                    "Self Plus One": [46.82, 52.48, 57.37, 64.13, 69.33],
                    "Self and Family": [70.22, 78.52, 86.13, 96.18, 104.00]},
    },
    "High": {
        "biweekly": {"Self Only": [18.97, 21.32, 23.26, 26.05, 28.23],
                     "Self Plus One": [37.92, 42.62, 46.53, 52.08, 56.46],
                     "Self and Family": [56.88, 63.95, 69.79, 78.13, 84.63]},
        "monthly": {"Self Only": [41.10, 46.19, 50.40, 56.44, 61.17],
                    "Self Plus One": [82.16, 92.34, 100.82, 112.84, 122.33],
                    "Self and Family": [123.24, 138.56, 151.21, 169.28, 183.37]},
    },
}

BENEFITS = {
    "High": {
        "plan": "High",
        "coinsurance_member_pays": {
            "A": {"in": "0%", "out": "0% plus any amount over the plan allowance"},
            "B": {"in": "20%", "out": "20% plus any amount over the plan allowance"},
            "C": {"in": "50%", "out": "50% plus any amount over the plan allowance"},
            "D": {"in": "30%", "out": "30% plus any amount over the plan allowance"},
        },
        "annual_maximum": {"in": "unlimited", "out": "unlimited"},
        "orthodontic_lifetime_maximum": {"in": "$3,500", "out": "$3,500"},
        "deductible": {"in": "none", "out": "none"},
        "notes": ["No waiting periods for any service.", "Orthodontics covers children and adults.",
                  "Implants limited to $2,500 per person per year.",
                  "Three adult cleanings per calendar year."],
    },
    "Standard": {
        "plan": "Standard",
        "coinsurance_member_pays": {
            "A": {"in": "0%", "out": "25% plus any amount over the plan allowance"},
            "B": {"in": "45%", "out": "50% plus any amount over the plan allowance"},
            "C": {"in": "65%", "out": "70% plus any amount over the plan allowance"},
            "D": {"in": "50%", "out": "50% plus any amount over the plan allowance"},
        },
        "annual_maximum": {"in": "$2,500", "out": "$2,000"},
        "orthodontic_lifetime_maximum": {"in": "$2,500", "out": "$1,500"},
        "deductible": {"in": "none", "out": "$75 per person per calendar year (Classes A, B, C)"},
        "notes": ["No waiting periods for any service.", "Orthodontics covers children and adults.",
                  "Implants limited to $2,500 per person per year in-network, $2,000 out-of-network.",
                  "Two cleanings per calendar year."],
    },
}

# brochure Section 5 (subset)
PROCEDURES = [
    ("D0120", "Periodic oral evaluation", "A", "Oral evaluations limited to two per calendar year."),
    ("D0210", "Full mouth radiographic images", "A", "Full mouth and panoramic images combined: once every 3 calendar years."),
    ("D0274", "Bitewings, four images", "A", "Twice per calendar year age 22 and younger; once per calendar year age 23 and older."),
    ("D1110", "Prophylaxis (cleaning), adult", "A", "Standard: twice per calendar year. High: three per calendar year."),
    ("D1120", "Prophylaxis (cleaning), child", "A", "Twice per calendar year."),
    ("D1206", "Topical fluoride varnish", "A", "Covered persons under age 22, twice per calendar year."),
    ("D1351", "Sealant, per tooth", "A", "Covered persons under 18, unrestored permanent molars only, one per tooth every 3 calendar years."),
    ("D2391", "Resin-based composite filling, one surface, posterior", "B", "One restoration per tooth surface every 2 calendar years."),
    ("D2930", "Prefabricated stainless steel crown, primary tooth", "B", "One per tooth per lifetime, covered persons under 15."),
    ("D2740", "Crown, porcelain/ceramic", "C", "Only when needed for extensive decay or fracture; replacement only after 5 years. A lower-cost alternative benefit may apply."),
    ("D2950", "Core buildup", "C", "Only with evidence of insufficient retention for a crown."),
    ("D3330", "Root canal, molar", "C", "Once per tooth per lifetime; retreatment only 12 months after the prior root canal."),
    ("D4341", "Periodontal scaling and root planing, four or more teeth per quadrant", "B", "Once per quadrant every 2 calendar years."),
    ("D4910", "Periodontal maintenance", "B", "Only after active periodontal treatment; with cleanings, four combined per calendar year."),
    ("D6010", "Surgical placement of implant body", "C", "Implant services limited to $2,500 per person per year ($2,000 Standard out-of-network); replacement after 5 years."),
    ("D7140", "Extraction, erupted tooth", "B", "None beyond plan maximums."),
    ("D8080", "Comprehensive orthodontic treatment, adolescent", "D", "Only to treat malocclusion; not for mild-moderate crowding; not covered if the appliance was placed before coverage began."),
    ("D8090", "Comprehensive orthodontic treatment, adult", "D", "Only to treat malocclusion; not for mild-moderate crowding; not covered if the appliance was placed before coverage began."),
    ("D9944", "Occlusal guard, hard, full arch", "C", "Once per calendar year, age 13 and older, for bruxism (grinding); not for TMJ."),
]


def person(pid, name, rel, dob):
    return {"person_id": pid, "full_name": name, "relationship": rel, "date_of_birth": dob}


# member id, name, dob, status, state, zip, enrollment id, plan, effective date, [covered persons]
MEMBERS = [
    ("GD10001", "Robert Hayes", "1958-11-03", "retired", "MO", "64109", "EN20001", "Standard", "2022-01-01", []),
    ("GD10002", "Priya Nair", "1990-06-21", "employed", "CA", "94110", "EN20002", "Standard", "2025-01-01", []),
    ("GD10003", "Daniel Kim", "1987-09-14", "employed", "VA", "22201", "EN20003", "High", "2024-01-01", []),
    ("GD10004", "Marcus Reed", "1983-02-27", "employed", "MD", "20852", "EN20004", "High", "2023-01-01", []),
    ("GD10005", "Aisha Johnson", "1992-12-05", "employed", "KS", "66044", "EN20005", "Standard", "2025-01-01",
     [("spouse", "Tom Johnson", "1990-03-18")]),
    ("GD10006", "Linda Park", "1961-05-30", "retired", "TX", "75201", "EN20006", "High", "2021-01-01", []),
    ("GD10007", "Samuel Ortiz", "1979-08-11", "employed", "FL", "33130", "EN20007", "High", "2020-01-01",
     [("spouse", "Elena Ortiz", "1981-01-25"), ("child", "Sofia Ortiz", "2006-10-02"), ("child", "Leo Ortiz", "2015-07-19")]),
    ("GD10008", "Brian Walsh", "1975-04-09", "employed", "OH", "43215", "EN20008", "High", "2019-01-01",
     [("spouse", "Karen Walsh", "1976-12-14")]),
    ("GD10009", "Helen Chu", "1969-03-22", "retired", "WA", "98101", "EN20009", "Standard", "2018-01-01", []),
    ("GD10010", "Nora Bennett", "1964-07-07", "retired", "NC", "27601", "EN20010", "High", "2017-01-01", []),
    ("GD10011", "Ethan Brooks", "1994-01-15", "employed", "CO", "80202", "EN20011", "Standard", "2026-01-01", []),
    ("GD10012", "Olivia Grant", "1988-10-08", "employed", "GA", "30303", "EN20012", "High", "2024-01-01", []),
]

TYPES = ["Self Only", "Self Plus One", "Self and Family"]

# claim id, member, patient person id, service date, code, network, billed, paid, status, decision, reason
CLAIMS = [
    ("CL30001", "GD10007", "EN20007-P2", "2026-02-10", "D2740", "in", 1363.0, 0.0, "denied", "2026-02-24",
     "Crown not payable: no evidence the tooth could not be restored with a filling."),
    ("CL30002", "GD10007", "EN20007-P4", "2026-01-12", "D0120", "in", 65.0, 0.0, "denied", "2026-01-26",
     "Frequency limit: oral evaluations are limited to two per calendar year."),
    ("CL30003", "GD10007", "EN20007-P1", "2026-01-05", "D1110", "in", 110.0, 110.0, "paid", "2026-01-15", None),
    ("CL30004", "GD10009", "EN20009-P1", "2026-01-08", "D4341", "in", 260.0, 0.0, "denied", "2026-01-20",
     "Frequency limit: scaling and root planing is limited to once per quadrant every 2 calendar years."),
    ("CL30005", "GD10009", "EN20009-P1", "2025-06-30", "D2950", "out", 340.0, 0.0, "denied", "2025-07-10",
     "Core buildup not payable: no evidence of insufficient retention for a crown."),
    ("CL30006", "GD10003", "EN20003-P1", "2026-02-02", "D2391", "in", 210.0, 168.0, "paid", "2026-02-12", None),
]


def build():
    members, enrollments = {}, {}
    for mid, name, dob, status, state, zip_code, eid, plan, eff, deps in MEMBERS:
        persons = [person(f"{eid}-P1", name, "self", dob)]
        for i, (rel, dname, ddob) in enumerate(deps, start=2):
            persons.append(person(f"{eid}-P{i}", dname, rel, ddob))
        members[mid] = {"member_id": mid, "full_name": name, "date_of_birth": dob, "status": status,
                        "state": state, "zip_code": zip_code, "enrollment_id": eid, "id_card_requests": []}
        enrollments[eid] = {"enrollment_id": eid, "member_id": mid, "plan": plan,
                            "enrollment_type": TYPES[min(len(persons), 3) - 1],
                            "covered_persons": persons, "effective_date": eff, "changes": []}
    claims = {}
    for cid, mid, pid, dos, code, net, billed, paid, status, dec, reason in CLAIMS:
        claims[cid] = {"claim_id": cid, "member_id": mid, "patient_person_id": pid, "date_of_service": dos,
                       "cdt_code": code, "network": net, "billed": billed, "paid": paid, "status": status,
                       "decision_date": dec, "denial_reason": reason, "appeal_status": None,
                       "appeal_filed_on": None}
    return {
        "members": members,
        "enrollments": enrollments,
        "claims": claims,
        "procedures": {c: {"cdt_code": c, "description": d, "benefit_class": k, "limitations": lim}
                       for c, d, k, lim in PROCEDURES},
        "rate_code_rules": [{"state": s, "zip3_ranges": r, "rate_code": code} for s, r, code in RATE_RULES],
        "premiums": PREMIUMS,
        "benefits": BENEFITS,
    }


if __name__ == "__main__":
    out = Path(__file__).with_name("db.json")
    out.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {out}")
