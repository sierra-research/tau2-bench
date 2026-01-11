import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

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
                "insurance_type": "motor",
                "policyholder": {
                "name": {
                    "first_name": "James",
                    "last_name": "Cole"
                },
                "dob": "1965-12-08",
                "address": {
                    "line1": "57202 Carlson Trail Suite 798",
                    "city": "Charlesmouth",
                    "state": "North Dakota",
                    "country": "Ethiopia",
                    "postal_code": "29331"
                },
                "contact": "(381)782-6886",
                "policy_status": "active",
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
                ]
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
                "additional_claimants": [],
                "loss_details": {
                "date_of_loss": "2024-07-13",
                "cause_of_loss": "fire damage",
                "description": "Reported insured loss event",
                "location": "Willieville",
                "estimated_loss_amount": 1107639.37,
                "currency": "EBT"
                },
                "claim_status": "new",
                "reported_date": "2024-07-25",
                "documents": [
                {
                    "document_id": "DOC_1",
                    "document_type": "motor_accident_claim_form",
                    "received": True,
                    "verified": True,
                },
                {
                    "document_id": "DOC_2",
                    "document_type": "police_abstract",
                    "received": True,
                    "verified": True,
                },
                {
                    "document_id": "DOC_3",
                    "document_type": "driver_license",
                    "received": True,
                    "verified": True,
                },
                {
                    "document_id": "DOC_4",
                    "document_type": "logbook",
                    "received": True,
                    "verified": True,
                },
                {
                    "document_id": "DOC_5",
                    "document_type": "repair_estimate",
                    "received": False,
                    "verified": False
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
                "audit_trail": {
                "2024-03-24T20:31:59": "Claim created",
                "2024-05-03T20:37:57": "Document motor_accident_claim_form received",
                "2024-12-03T10:48:04": "Document motor_accident_claim_form verified",
                "2024-12-12T19:40:52": "Document police_abstract received",
                "2024-12-18T02:51:19": "Claim status updated to new"
                },
                "assignments": {
                "taken_to_human_agent": False,
                "assignments": None
                }
            }),
            "HOME-ZA-2024-00001": Claim(**{
                "claim_id": "HOME-ZA-2024-00001",
                "insurance_type": "property",
                "policyholder": {
                "name": {
                    "first_name": "Danielle",
                    "last_name": "Johnson"
                },
                "dob": "1969-11-12",
                "address": {
                    "line1": "908 Jennifer Squares",
                    "city": "Robinsonshire",
                    "state": "Louisiana",
                    "country": "Zambia",
                    "postal_code": "01352"
                },
                "contact": "001-626-254-2351x16155",
                "policy_status": "active",
                "premium_amount": 1751.86,
                "currency": "EBT",
                "payment_methods": [
                    {
                    "method_id": "PM_1",
                    "method_type": "bank_transfer",
                    "masked_reference": "****7924",
                    "verified": True
                    }
                ]
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
                "cause_of_loss": "water leak",
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
                    "received": True,
                    "verified": True
                },
                {
                    "document_id": "DOC_2",
                    "document_type": "fire_report",
                    "received": True,
                    "verified": True
                },
                {
                    "document_id": "DOC_3",
                    "document_type": "repair_quotes",
                    "received": True,
                    "verified": True
                },
                {
                    "document_id": "DOC_4",
                    "document_type": "proof_of_ownership",
                    "received": True,
                    "verified": True
                },
                {
                    "document_id": "DOC_5",
                    "document_type": "photos",
                    "received": True,
                    "verified": True
                }
                ],
                "fraud_assessment": {
                "fraud_flag": "None",
                "indicators": [],
                "notes": None
                },
                "beneficiary_dispute": None,
                "payments": [],
                "subrogation_possible": True,
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
                "assignments": {
                    "handler_team": "Recoveries",
                    "claims_adjuster": {
                    "name": "Megan Mcclain",
                    "role": "Senior Claims Analyst"
                    },
                    "assessor": {
                    "name": "Javier Johnson",
                    "role": "Loss Assessor"
                    },
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


@pytest.fixture
def claims_environment(claims_db: InsuranceClaimsDB) -> Environment:
    environment = get_environment(claims_db)
    return environment

#Tool calls
@pytest.fixture
def create_claim_call() -> ToolCall:
    return ToolCall(
        id="1", 
        name="create_claim", 
        arguments={"claim_id": "MT-ET-2024-00002"})

@pytest.fixture
def submit_claim_call() -> ToolCall:
    return ToolCall(
        id="1", 
        name="submit_claim", 
        arguments={"claim_id": "MT-ET-2024-00002"})


@pytest.fixture
def request_documents_call() -> ToolCall:
    return ToolCall(
        id="2",
        name="request_documents",
        arguments={
            "claim_id": "MT-ET-2024-00002",
            "document_types": ["repair_estimate"],
            "note": "Needed for review"
        }
    )
@pytest.fixture
def approve_claim_call() -> ToolCall:
    return ToolCall(
        id="4",
        name="approve_claim",
        arguments={"claim_id": "MT-ET-2024-00002", "notes": "All docs verified"}
    )


@pytest.fixture
def reject_claim_call() -> ToolCall:
    return ToolCall(
        id="5",
        name="reject_claim",
        arguments={"claim_id": "HOME-ZA-2024-00001", "reason": "Invalid claim"}
    )

@pytest.fixture
def verify_document_call() -> ToolCall:
    return ToolCall(
        id="3",
        name="verify_documents",
        arguments={
            "claim_id": "MT-ET-2024-00002",
            "document_id": "DOC_5",
            "verified": True,
            "notes": "Verified by system"
        }
    )

@pytest.fixture
def identify_subrogation_call() -> ToolCall:
    return ToolCall(
        id="6",
        name="identify_subrogation",
        arguments={
            "claim_id": "HOME-ZA-2024-00001",
            "notes": "Possible subrogation"
        }
    )


@pytest.fixture
def initiate_recovery_call() -> ToolCall:
    return ToolCall(
        id="7",
        name="initiate_recovery",
        arguments={
            "claim_id": "HOME-ZA-2024-00001",
            "third_party_name": "Third Party",
            "recovery_amount": 200,
            "currency": "USD",
            "notes": "Recover funds"
        }
    )
@pytest.fixture
def update_recovery_status_call() -> ToolCall:
    return ToolCall(
        id="8",
        name="update_recovery_status",
        arguments={
            "claim_id": "HOME-ZA-2024-00001",
            "recovery_id": "RCV_1",
            "status": "recovered",
            "notes": "Funds recovered"
        }
    )

@pytest.fixture
def settle_claim_call() -> ToolCall:
    return ToolCall(
        id="9",
        name="settle_claim",
        arguments={
            "claim_id": "MT-ET-2024-00002",
            "notes": "Settled successfully"
        }
    )

@pytest.fixture
def make_payment_call() -> ToolCall:
    return ToolCall(
        id="10",
        name="_make_payment",
        arguments={
            "claim_id": "MT-ET-2024-00002",
            "amount": 1107639.37,
            "currency": "EBT",
            "method": "bank_transfer",
            "reason": "Approved payment"
        }
    )

def test_create_claim(claims_environment: Environment, create_claim_call: ToolCall):
    response = claims_environment.get_response(create_claim_call)
    assert not response.error
    assert "Claim already exists" in response.error.message

def test_submit_claim(claims_environment: Environment, submit_claim_call: ToolCall):
    response = claims_environment.get_response(submit_claim_call)
    assert not response.error
    claim = claims_environment.tools._get_claim("MT-ET-2024-00002")
    assert "Claim submitted" in claim.audit_trail.values() 

def test_request_documents(claims_environment: Environment, request_documents_call: ToolCall):
    response = claims_environment.get_response(request_documents_call)
    assert not response.error
    claim = claims_environment.tools._get_claim("MT-ET-2024-00002")
    assert any(d.document_type == "repair_estimate" for d in claim.documents)
    assert claim.claim_status == "awaiting_documents"

def test_verify_document(claims_environment: Environment, verify_document_call: ToolCall):
    claim = claims_environment.tools._get_claim("MT-ET-2024-00002")
    doc = next(d for d in claim.documents if d.document_id == "DOC_5")
    assert not doc.verified
    response = claims_environment.get_response(verify_document_call)
    assert not response.error
    doc = next(d for d in claim.documents if d.document_id == "DOC_5")
    assert doc.verified

def test_approve_claim(claims_environment: Environment, approve_claim_call: ToolCall):
    response = claims_environment.get_response(approve_claim_call)
    assert not response.error
    claim = claims_environment.tools._get_claim("MT-ET-2024-00002")
    assert claim.claim_status == "approved"
    assert "All docs verified" in list(claim.audit_trail.values())[-1]


def test_settle_claim(claims_environment: Environment, make_payment_call: ToolCall, settle_claim_call: ToolCall):
    response = claims_environment.get_response(make_payment_call)
    assert not response.error
    response = claims_environment.get_response(settle_claim_call)
    assert not response.error
    claim = claims_environment.tools._get_claim("MT-ET-2024-00002")
    assert claim.claim_status == "settled"

#scenario 2
def test_get_claims(claims_environment: Environment):
    claims = claims_environment.tools.get_claims()
    assert isinstance(claims, list)
    assert any(c.claim_id == "HOME-ZA-2024-00001" for c in claims)

def test_initiate_recovery(claims_environment: Environment, identify_subrogation_call: ToolCall, initiate_recovery_call: ToolCall):
    response = claims_environment.get_response(identify_subrogation_call)
    assert not response.error
    response = claims_environment.get_response(initiate_recovery_call)
    assert not response.error
    claim = claims_environment.tools._get_claim("HOME-ZA-2024-00001")
    recovery = claim.recoveries[-1]
    assert recovery.third_party_name == "Third Party"
    assert recovery.status == "in_progress"

def test_update_recovery_status(claims_environment: Environment, update_recovery_status_call: ToolCall):
    response = claims_environment.get_response(update_recovery_status_call)
    assert not response.error
    claim = claims_environment.tools._get_claim("HOME-ZA-2024-00001")
    recovery = next(r for r in claim.recoveries if r.recovery_id == "RCV_1")
    assert recovery.status == "recovered"

def test_reject_claim(claims_environment: Environment, reject_claim_call: ToolCall):
    response = claims_environment.get_response(reject_claim_call)
    assert not response.error
    claim = claims_environment.tools._get_claim("HOME-ZA-2024-00001")
    assert claim.claim_status == "rejected"
    assert claim.assignments.taken_to_human_agent 

if __name__ == "__main__":
    pass
    # test_create_claim()
    # test_get_claims()
    # test_initiate_recovery()
    # test_update_recovery_status()
    # test_reject_claim()
    # test_submit_claim()
    # test_request_documents()
    # test_verify_document()
    # test_approve_claim()
    # test_settle_claim()