import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
from pathlib import Path
from typing import List

from tau2.domains.claims.data_model import (
     Name, Address, IdentityDocument, PaymentMethod, ClaimPayment,
    LossDetails, ClaimDocument, FraudAssessment, BeneficiaryDispute,
    AssignmentDetails, Assignments, Claim, Policy,Claimant
)

# Input & output files
input_file = "R:/Reading Materials/MOOC LLMs Fall 2025/db_v1.json"
output_file = "claims_pydantic.json"


# Load JSON data
with open(input_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

claims_list: List[Claim] = []

for claim_id, c in raw_data.items():
    # -----------------------------
    # Policyholder
    # -----------------------------
    ph = c.get("policyholder", {})
    ph_name_data = ph.get("name", {})
    first_name = ph_name_data.get("first_name") or ""  # if None, use empty string
    last_name = ph_name_data.get("last_name") or ""    # if None, use empty string
    ph_name = Name(first_name=first_name, last_name=last_name)

    ph_addr = Address(**ph.get("address", {
        "line1": "", "city": "", "state": "", "country": "", "postal_code": ""
    }))

    payment_methods = [
        PaymentMethod(**pm) for pm in ph.get("payment_methods", [])
    ]
    policyholder_dob = ph.get("dob") or ""  # default to empty string if None
    policy_obj = Policy(
        policy_id=c.get("claim_id"),  # using claim_id as policy_id if missing
        insurance_type=c.get("insurance_type", "property"),
        policyholder_name=ph_name,
        policyholder_dob=policyholder_dob,
        policyholder_address=ph_addr,
        policyholder_contact=ph.get("contact", ""),
        effective_date=ph.get("effective_date", "2024-01-01"),
        expiry_date=ph.get("expiry_date", "2025-01-01"),
        status=ph.get("policy_status", "active"),
        premium_amount=ph.get("premium_amount", 0.0),
        currency=ph.get("currency", "USD"),
        payment_methods=payment_methods,
        coverages=[]  # add if you have coverage info
    )

    # -----------------------------
    # Claimant
    # -----------------------------
    
    cl = c.get("claimant", {})
    cl_name = Name(**cl.get("name", {"first_name": "", "last_name": ""}))
    cl_addr = Address(**cl.get("address", {
        "line1": "", "city": "", "state": "", "country": "", "postal_code": ""
    }))

    identity_docs = [
        IdentityDocument(**doc) for doc in cl.get("identity_documents", [])
    ]

    claimant_obj = Claimant(
        role=cl.get("role", "policyholder"),
        name=cl_name,
        dob=cl.get("dob"),
        address=cl_addr,
        contact=cl.get("contact", ""),
        identity_documents=identity_docs,
        relationship_to_policyholder=cl.get("relationship_to_policyholder")
    )

    # -----------------------------
    # Additional claimants
    # -----------------------------
    add_claimants = []
    for ac in c.get("additional_claimants", []):
        ac_name = Name(**ac.get("name", {"first_name": "", "last_name": ""}))
        ac_addr = Address(**ac.get("address", {
            "line1": "", "city": "", "state": "", "country": "", "postal_code": ""
        }))
        ac_docs = [IdentityDocument(**doc) for doc in ac.get("identity_documents", [])]
        add_claimants.append(
            Claimant(
                role=ac.get("role", "third_party"),
                name=ac_name,
                dob=ac.get("dob"),
                address=ac_addr,
                contact=ac.get("contact", ""),
                identity_documents=ac_docs,
                relationship_to_policyholder=ac.get("relationship_to_policyholder")
            )
        )

    # -----------------------------
    # Loss details
    # -----------------------------
    loss_obj = LossDetails(**c.get("loss_details", {
        "date_of_loss": "2024-01-01",
        "cause_of_loss": "unknown",
        "description": "No description",
        "location": "unknown",
        "estimated_loss_amount": 0.0,
        "currency": "USD"
    }))

    # -----------------------------
    # Fraud assessment
    # -----------------------------
    fraud_data = c.get("fraud_assessment", {})

    fraud_flag_raw = fraud_data.get("fraud_flag", "none")
    # Map invalid or null values to 'none'
    if fraud_flag_raw not in {"none", "suspected", "confirmed"}:
        fraud_flag = "none"
    else:
        fraud_flag = fraud_flag_raw

    # Ensure indicators is a list
    indicators = fraud_data.get("indicators") or []

    notes = fraud_data.get("notes")  # can be None

    fraud_obj = FraudAssessment(
        fraud_flag=fraud_flag,
        indicators=indicators,
        notes=notes
    )


# -----------------------------
# Documents
# -----------------------------
    allowed_doc_types = {
        "claim_form","medical_report","police_report","invoice","death_certificate",
        "repair_estimate","repair_quotes","identity_proof","fire_report","doctor_signed_form","other"
    }

    documents = []
    for doc in c.get("documents", []):
        doc_type = doc.get("document_type", "other")
        
        if doc_type not in allowed_doc_types:
            mapped_type = "other"
        else:
            mapped_type = doc_type
        
        documents.append(
            ClaimDocument(
                document_id=doc.get("document_id", ""),
                document_type=mapped_type,
                real_document_type=doc_type,  # keep the real type
                received=doc.get("received", False),
                verified=doc.get("verified", False),
                notes=doc.get("notes")
            )
        )

    # -----------------------------
    # Payments
    # -----------------------------
    payments = [ClaimPayment(**p) for p in c.get("payments", [])]

    # -----------------------------
    # Assignments (optional)
    # -----------------------------
    from tau2.domains.claims.data_model import AssignmentDetails, Assignments, Name, AssignmentRole, AssignmentTeam

    def parse_assignments(assgn: dict) -> Assignments:
        if not assgn:
            return Assignments(
                taken_to_human_agent=False,
                current_assignment=None,
                assignment_history=[]
            )

        taken_to_human_agent = assgn.get("taken_to_human_agent", False)

        # 🔑 unwrap the inner assignments object
        current = assgn.get("assignments")
        if not current:
            return Assignments(
                taken_to_human_agent=taken_to_human_agent,
                current_assignment=None,
                assignment_history=[]
            )

        # Handler team
        handler_team = current.get("handler_team", "General Claims Team")
        if handler_team not in AssignmentTeam.__args__:
            handler_team = "General Claims Team"

        # Claims adjuster
        ca = current.get("claims_adjuster", {})
        ca_name_str = ca.get("name", "")
        ca_role = ca.get("role", "Junior Claims Analyst")

        ca_first, ca_last = (ca_name_str.strip().split(" ", 1) + [""])[:2]
        claims_adjuster = Name(first_name=ca_first, last_name=ca_last)

        if ca_role not in AssignmentRole.__args__:
            ca_role = "Junior Claims Analyst"

        # Assessor
        asg = current.get("assessor", {})
        as_name_str = asg.get("name", "")
        as_role = asg.get("role", "Loss Assessor")

        as_first, as_last = (as_name_str.strip().split(" ", 1) + [""])[:2]
        assessor = Name(first_name=as_first, last_name=as_last)

        if as_role not in AssignmentRole.__args__:
            as_role = "Loss Assessor"

        # Assigned date
        assigned_date = current.get("assigned_date", "")

        current_assignment = AssignmentDetails(
            handler_team=handler_team,
            claims_adjuster=claims_adjuster,
            adjuster_role=ca_role,
            assessor=assessor,
            assessor_role=as_role,
            assigned_date=assigned_date
        )

        return Assignments(
            taken_to_human_agent=taken_to_human_agent,
            current_assignment=current_assignment,
            assignment_history=[]
        )

    # -----------------------------
    # Build Claim object
    # -----------------------------
    # Get claim_status, strip whitespace, fallback to 'new' if empty
    claim_status = (c.get("claim_status") or "").strip()
    if not claim_status:
        claim_status = "new"



    audit_trail = c.get("audit_trail") or {}

    claim_obj = Claim(
        claim_id=claim_id,
        policy=policy_obj,
        claimant=claimant_obj,
        additional_claimants=add_claimants if add_claimants else None,
        loss_details=loss_obj,
        claim_status=claim_status,
        reported_date=c.get("reported_date", "2024-01-01"),
        documents=documents,
        fraud_assessment=fraud_obj,
        beneficiary_dispute=None,
        payments=payments,
        subrogation_possible=c.get("subrogation_possible", False),
        subrogation_notes=c.get("subrogation_notes"),
        audit_trail=audit_trail,
        assignments=parse_assignments(c.get("assignments", {})),
        recoveries=[]
    )


    claims_list.append(claim_obj)

# Optional: print summary
print(f"Loaded {len(claims_list)} claims successfully.")

# claims_dict = {
#     claim.claim_id: {k: v for k, v in claim.model_dump().items() if k != "claim_id"}
#     for claim in claims_list
# }

claims_dict = {
    claim.claim_id: claim.model_dump()  # keep claim_id inside
    for claim in claims_list
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(claims_dict, f, ensure_ascii=False, indent=4)

print(f"Saved {len(claims_dict)} claims to {output_file}")
