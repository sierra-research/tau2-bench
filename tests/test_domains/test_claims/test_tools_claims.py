import os
import sys


import pytest
from tau2.domains.claims.data_model import InsuranceClaimsDB, Claim, ClaimPayment
from tau2.environment.environment import Environment
from tau2.data_model.message import ToolCall
from tau2.domains.claims.environment import get_environment
from tau2.domains.claims.tools import ClaimsTools


@pytest.fixture
def claims_db() -> InsuranceClaimsDB:
    return InsuranceClaimsDB(
        claims={
            "MT-ET-2024-00002": Claim(**{
                "claim_id": "MT-ET-2024-00002",
                "policy": {
                    "policy_id": "MT-ET-2024-00002",
                    "insurance_type": "motor",
                    "policyholder_name": {
                        "first_name": "James",
                        "last_name": "Cole"
                    },
                    "policyholder_dob": "1965-12-08",
                    "policyholder_address": {
                        "line1": "57202 Carlson Trail Suite 798",
                        "line2": None,
                        "city": "Charlesmouth",
                        "state": "North Dakota",
                        "country": "Ethiopia",
                        "postal_code": "29331"
                    },
                    "policyholder_contact": "(381)782-6886",
                    "effective_date": "2024-01-01",
                    "expiry_date": "2025-01-01",
                    "status": "active",
                    "premium_amount": 3579.93,
                    "currency": "EBT",
                    "payment_methods": [
                        {
                            "method_id": "PM_1",
                            "method_type": "credit_card",
                            "masked_reference": "****3938",
                            "verified": False
                        },
                        {
                            "method_id": "PM_2",
                            "method_type": "credit_card",
                            "masked_reference": "****4394",
                            "verified": True
                        }
                    ],
                    "coverages": []
                },
                "claimant": {
                    "role": "beneficiary",
                    "name": {
                        "first_name": "Tonya",
                        "last_name": "Jones"
                    },
                    "dob": "1970-01-14",
                    "address": {
                        "line1": "555 Robinson Terrace",
                        "line2": None,
                        "city": "Barajastown",
                        "state": "Delaware",
                        "country": "Ethiopia",
                        "postal_code": "88180"
                    },
                    "contact": "(673)299-3056x164",
                    "identity_documents": [
                        {
                            "doc_type": "national_id",
                            "doc_number": "85Pm887704",
                            "verified": False
                        }
                    ],
                    "relationship_to_policyholder": "beneficiary"
                },
                "additional_claimants": None,
                "loss_details": {
                    "date_of_loss": "2024-07-13",
                    "cause_of_loss": "fire damage",
                    "description": "Reported insured loss event",
                    "location": "Willieville",
                    "estimated_loss_amount": 1107639.37,
                    "currency": "EBT"
                },
                "claim_status": "under_review",
                "reported_date": "2024-07-25",
                "documents": [
                    {
                        "document_id": "DOC_1",
                        "document_type": "other",
                        "real_document_type": "motor_accident_claim_form",
                        "received": True,
                        "verified": False,
                        "notes": None
                    },
                    {
                        "document_id": "DOC_2",
                        "document_type": "other",
                        "real_document_type": "police_abstract",
                        "received": True,
                        "verified": False,
                        "notes": None
                    },
                    {
                        "document_id": "DOC_3",
                        "document_type": "other",
                        "real_document_type": "driver_license",
                        "received": True,
                        "verified": False,
                        "notes": None
                    },
                    {
                        "document_id": "DOC_4",
                        "document_type": "other",
                        "real_document_type": "logbook",
                        "received": True,
                        "verified": False,
                        "notes": None
                    },
                    {
                        "document_id": "DOC_5",
                        "document_type": "repair_estimate",
                        "real_document_type": "repair_estimate",
                        "received": True,
                        "verified": False,
                        "notes": None
                    }
                ],
                "fraud_assessment": {
                    "fraud_flag": "none",
                    "indicators": [],
                    "notes": None
                },
                "beneficiary_dispute": None,
                "payments": [],
                "subrogation_possible": True,
                "subrogation_notes": None,
                "audit_trail": {
                    "2024-03-24T20:31:59": "Claim created",
                    "2024-05-03T20:37:57": "Document motor_accident_claim_form received",
                    "2024-06-18T13:58:31": "Document repair_estimate received",
                    "2024-12-03T10:48:04": "Document motor_accident_claim_form verified",
                    "2024-12-12T19:40:52": "Document police_abstract received",
                    "2024-12-18T02:51:19": "Claim status updated to under_review"
                },
                "assignments": {
                    "taken_to_human_agent": True,
                    "current_assignment": {
                        "handler_team": "Recoveries",
                        "claims_adjuster": {
                            "first_name": "Nancy",
                            "last_name": "Wilson"
                        },
                        "adjuster_role": "Senior Claims Analyst",
                        "assessor": {
                            "first_name": "Douglas",
                            "last_name": "Baker"
                        },
                        "assessor_role": "Loss Assessor",
                        "assigned_date": "2024-01-24T02:42:24"
                    },
                    "assignment_history": []
                },
                "recoveries": []
                    
            }),
            "HOME-ZA-2024-00001": Claim(**{
                "claim_id": "HOME-ZA-2024-00001",
                "policy": {
                    "policy_id": "HOME-ZA-2024-00001",
                    "insurance_type": "property",
                    "policyholder_name": {
                        "first_name": "Danielle",
                        "last_name": "Johnson"
                    },
                    "policyholder_dob": "1969-11-12",
                    "policyholder_address": {
                        "line1": "908 Jennifer Squares",
                        "line2": None,
                        "city": "Robinsonshire",
                        "state": "Louisiana",
                        "country": "Zambia",
                        "postal_code": "01352"
                    },
                    "policyholder_contact": "001-626-254-2351x16155",
                    "effective_date": "2024-01-01",
                    "expiry_date": "2025-01-01",
                    "status": "active",
                    "premium_amount": 1751.86,
                    "currency": "EBT",
                    "payment_methods": [
                        {
                            "method_id": "PM_1",
                            "method_type": "bank_transfer",
                            "masked_reference": "****7924",
                            "verified": True
                        }
                    ],
                    "coverages": []
                },
                "claimant": {
                    "role": "third_party",
                    "name": {
                        "first_name": "Joshua",
                        "last_name": "Walker"
                    },
                    "dob": "2004-08-23",
                    "address": {
                        "line1": "161 Calderon River Suite 931",
                        "line2": None,
                        "city": "Lake Jeremyport",
                        "state": "Colorado",
                        "country": "Zambia",
                        "postal_code": "31013"
                    },
                    "contact": "664-375-2553",
                    "identity_documents": [
                        {
                            "doc_type": "national_id",
                            "doc_number": "92OS832764",
                            "verified": True
                        }
                    ],
                    "relationship_to_policyholder": "third_party"
                },
                "additional_claimants": [
                    {
                        "role": "third_party",
                        "name": {
                            "first_name": "Angela",
                            "last_name": "Cohen"
                        },
                        "dob": "2007-04-09",
                        "address": {
                            "line1": "05641 Robin Port",
                            "line2": None,
                            "city": "Jasonfort",
                            "state": "Montana",
                            "country": "Ethiopia",
                            "postal_code": "84760"
                        },
                        "contact": "724.523.8849x696",
                        "identity_documents": [],
                        "relationship_to_policyholder": "relative"
                    },
                    {
                        "role": "third_party",
                        "name": {
                            "first_name": "Wendy",
                            "last_name": "Taylor"
                        },
                        "dob": "1988-07-27",
                        "address": {
                            "line1": "12269 Paul Ranch",
                            "line2": None,
                            "city": "Riceside",
                            "state": "Alabama",
                            "country": "Ethiopia",
                            "postal_code": "89667"
                        },
                        "contact": "001-418-345-1462x7048",
                        "identity_documents": [],
                        "relationship_to_policyholder": "relative"
                    }
                ],
                "loss_details": {
                    "date_of_loss": "2024-05-20",
                    "cause_of_loss": "fire",
                    "description": "Reported insured loss event",
                    "location": "Meaganhaven",
                    "estimated_loss_amount": 450305.42,
                    "currency": "ZMW"
                },
                "claim_status": "awaiting_documents",
                "reported_date": "2024-05-27",
                "documents": [
                    {
                        "document_id": "DOC_1",
                        "document_type": "claim_form",
                        "real_document_type": "claim_form",
                        "received": True,
                        "verified": True,
                        "notes": None
                    },
                    {
                        "document_id": "DOC_2",
                        "document_type": "fire_report", 
                        "real_document_type": "fire_report",
                        "received": True,
                        "verified": True,
                        "notes": None
                    },
                    {
                        "document_id": "DOC_3",
                        "document_type": "repair_quotes",
                        "real_document_type": "repair_quotes",
                        "received": True,
                        "verified": True,
                        "notes": None
                    },
                    {
                        "document_id": "DOC_4",
                        "document_type": "other",
                        "real_document_type": "proof_of_ownership",
                        "received": True,
                        "verified": True,
                        "notes": None
                    },
                    {
                        "document_id": "DOC_5",
                        "document_type": "other",
                        "real_document_type": "photos",
                        "received": True,
                        "verified": True,
                        "notes": None
                    }
                ],
                "fraud_assessment": {
                    "fraud_flag": "none",  # Fixed from "None"
                    "indicators": [],
                    "notes": None
                },
                "beneficiary_dispute": None,
                "payments": [],
                "subrogation_possible": True,
                "subrogation_notes": None,
                "audit_trail": {
                    "2024-02-10T20:05:57": "Document proof_of_ownership received",
                    "2024-02-19T13:04:12": "Claim created",
                    "2024-06-25T21:58:51": "Document claim_form received",
                    "2024-07-10T18:59:14": "Document proof_of_ownership verified",
                    "2024-08-23T19:31:24": "Document repair_quotes received",
                    "2024-08-31T23:33:46": "Document claim_form verified",
                    "2024-09-24T09:18:19": "Document repair_quotes verified",
                    "2024-11-12T13:10:00": "Document photos received",
                    "2024-12-18T20:10:03": "Document photos verified",
                    "2024-12-26T02:31:56": "Claim status updated to awaiting_documents"
                },
                "assignments": {
                    "taken_to_human_agent": True,
                    "current_assignment": {
                        "handler_team": "Recoveries",
                        "claims_adjuster": {
                            "first_name": "Megan",
                            "last_name": "Mcclain"
                        },
                        "adjuster_role": "Senior Claims Analyst",
                        "assessor": {
                            "first_name": "Javier",
                            "last_name": "Johnson"
                        },
                        "assessor_role": "Loss Assessor",
                        "assigned_date": "2024-08-17T21:27:16"
                    }
                }
            })
        }
    )


# @pytest.fixture
# def claims_environment(claims_db: InsuranceClaimsDB) -> Environment:
#     toolkit = ClaimsTools(claims_db)
#     return get_environment(toolkit)

# claim.audit_trail[self._now()] = message #uncomment this on tools.py when running tests to update the audit trail

@pytest.fixture
def env(claims_db: InsuranceClaimsDB) -> Environment:
    return get_environment(claims_db)

def get_claim(env, claim_id):
    return env.tools._get_claim(claim_id)


def audit_contains(claim, text):
    return any(text.lower() in v.lower() for v in claim.audit_trail.values())


def test_create_claim_duplicate(env: Environment):
    call = ToolCall(
        id="1",
        name="create_claim",
        arguments={"claim_id": "MT-ET-2024-00002"},
    )

    response = env.get_response(call)

    assert response.error is False         
    assert response.content 


def test_submit_claim(env: Environment):
    call = ToolCall(
        id="2",
        name="submit_claim",
        arguments={"claim_id": "MT-ET-2024-00002"},
    )

    response = env.get_response(call)

    assert response.error is True
    assert "only new claims" in response.content.lower()


def test_request_documents_updates_status(env: Environment):
    call = ToolCall(
        id="3",
        name="request_documents",
        arguments={
            "claim_id": "MT-ET-2024-00002",
            "document_types": ["repair_estimate"],
            "note": "Needed for review",
        },
    )

    response = env.get_response(call)

    assert response.error is False
    claim = get_claim(env, "MT-ET-2024-00002")
    assert claim.claim_status == "awaiting_documents"
    assert audit_contains(claim, "document")


def test_verify_document(env: Environment):
    claim = get_claim(env, "MT-ET-2024-00002")
    doc = next(d for d in claim.documents if d.document_id == "DOC_5")

    assert not doc.verified

    call = ToolCall(
        id="4",
        name="verify_documents",
        arguments={
            "claim_id": "MT-ET-2024-00002",
            "document_id": "DOC_5",
            "verified": True,
            "notes": "Verified",
        },
    )

    response = env.get_response(call)

    assert response.error is False

    doc = next(d for d in claim.documents if d.document_id == "DOC_5")
    assert doc.verified


def test_approve_claim(env: Environment):
    call = ToolCall(
        id="5",
        name="approve_claim",
        arguments={
            "claim_id": "MT-ET-2024-00002",
            "notes": "All docs verified",
        },
    )

    response = env.get_response(call)

    assert response.error is False

    claim = get_claim(env, "MT-ET-2024-00002")
    assert claim.claim_status == "approved"
    assert audit_contains(claim, "approved")


def test_payment_and_settlement_flow(env: Environment):
    # Ensure proper order: approve → pay → settle
    env.get_response(ToolCall(
        id="6",
        name="approve_claim",
        arguments={"claim_id": "MT-ET-2024-00002", "notes": "OK"},
    ))

    pay = ToolCall(
        id="7",
        name="make_payment",
        arguments={
            "claim_id": "MT-ET-2024-00002",
            "amount": 1107639.37,
            "currency": "EBT",
            "method": "bank_transfer",
            "reason": "Approved payment",
        },
    )

    settle = ToolCall(
        id="8",
        name="settle_claim",
        arguments={
            "claim_id": "MT-ET-2024-00002",
            "notes": "Settled successfully",
        },
    )

    assert not env.get_response(pay).error
    assert not env.get_response(settle).error

    claim = get_claim(env, "MT-ET-2024-00002")
    


# -----------------------------
# RECOVERY FLOW
# -----------------------------

def test_recovery_flow(env: Environment):
    claim_id = "HOME-ZA-2024-00001"
    assert not env.get_response(ToolCall(
        id="9",
        name="identify_subrogation",
        arguments={"claim_id": claim_id, "notes": "Possible"},
    )).error

    assert not env.get_response(ToolCall(
        id="10",
        name="initiate_recovery",
        arguments={
            "claim_id": claim_id,
            "third_party_name": "Third Party",
            "recovery_amount": 200,
            "currency": "USD",
            "notes": "Recover funds",
        },
    )).error

    claim = get_claim(env, claim_id)
    recovery = claim.recoveries[-1]

    assert recovery.status == "in_progress"

    # Step 3: Update status
    assert not env.get_response(ToolCall(
        id="11",
        name="update_recovery_status",
        arguments={
            "claim_id": claim_id,
            "recovery_id": recovery.recovery_id,
            "status": "recovered",
            "notes": "Done",
        },
    )).error

    assert recovery.status == "recovered"


def test_reject_claim(env: Environment):
    call = ToolCall(
        id="12",
        name="reject_claim",
        arguments={
            "claim_id": "HOME-ZA-2024-00001",
            "reason": "Invalid claim",

        },
    )

    response = env.get_response(call)

    assert response.error is False

    claim = get_claim(env, "HOME-ZA-2024-00001")
    assert claim.claim_status == "rejected"
    assert claim.assignments.taken_to_human_agent

if __name__ == "__main__":
    pass
