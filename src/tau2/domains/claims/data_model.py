import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from tau2.environment.db import DB
from tau2.domains.claims.utils import CLAIMS_DB_PATH
from pydantic import RootModel


InsuranceType = Literal[
    "motor", "health_outpatient", "health_inpatient", "health_last_expense",
    "property", "travel", "life"
]

PolicyStatus = Literal["active", "lapsed", "expired", "cancelled"]

ClaimStatus = Literal[
    "new","under_review","awaiting_documents","approved","reviewed",
    "rejected","partially_settled","settled","closed"
]

FraudFlag = Literal["none","suspected","confirmed"]

ClaimantRole = Literal["policyholder","beneficiary","third_party"]

PaymentType = Literal["credit_card","bank_transfer","mobile_money","cheque"]

DocumentType = Literal[
    "claim_form","medical_report","police_report","invoice","death_certificate",
    "repair_estimate","identity_proof","fire_report","doctor_signed_form","repair_quotes",
    "other"
]

AssignmentRole = Literal["Junior Claims Analyst","Senior Claims Analyst","Loss Assessor","Medical Assessor"]

AssignmentTeam = Literal["Motor Team","Health Team","Life Team","Property Team","Travel Team","Recoveries","General Claims Team"]

RecoveryStatus = Literal["not_applicable", "identified", "in_progress","recovered","unrecoverable"]
   

class Name(BaseModel):
    first_name: str
    last_name: str

class Address(BaseModel):
    line1: str
    line2: Optional[str] = None
    city: str
    state: str
    country: str
    postal_code: str

class IdentityDocument(BaseModel):
    doc_type: Literal["national_id","passport","driving_license", "birth_certificate"]
    doc_number: str
    verified: bool

class PaymentMethod(BaseModel):
    method_id: str
    method_type: PaymentType
    masked_reference: str
    verified: bool

class ClaimPayment(BaseModel):
    payment_id: str
    amount: float
    currency: str
    payment_date: str
    payment_method: PaymentType
    recipient_name: str
    reason: str

class Coverage(BaseModel):
    coverage_name: str
    sum_insured: float
    deductible: Optional[float]
    limit_per_event: Optional[float]

class Policy(BaseModel):
    policy_id: str
    insurance_type: InsuranceType
    policyholder_name: Name
    policyholder_dob: str
    policyholder_address: Address
    policyholder_contact: str
    effective_date: str
    expiry_date: str
    status: PolicyStatus
    premium_amount: float
    currency: str
    payment_methods: List[PaymentMethod]
    coverages: List[Coverage]

class Claimant(BaseModel):
    role: ClaimantRole
    name: Name
    dob: Optional[str]
    address: Address
    contact: str
    identity_documents: List[IdentityDocument]
    relationship_to_policyholder: Optional[str]

class LossDetails(BaseModel):
    date_of_loss: str
    cause_of_loss: str
    description: str
    location: str
    estimated_loss_amount: float
    currency: str

class ClaimDocument(BaseModel):
    document_id: str
    document_type: DocumentType
    real_document_type: Optional[str] = None
    received: bool
    verified: bool
    notes: Optional[str]

class FraudAssessment(BaseModel):
    fraud_flag: FraudFlag
    indicators: List[str]
    notes: Optional[str]

class BeneficiaryDispute(BaseModel):
    dispute_exists: bool
    disputed_by: Optional[str]
    reason: Optional[str]
    status: Optional[Literal["open","resolved"]]

class AssignmentDetails(BaseModel):
    handler_team: AssignmentTeam
    claims_adjuster: Name
    adjuster_role: AssignmentRole
    assessor: Name
    assessor_role: AssignmentRole
    assigned_date: str


class Assignments(BaseModel):
    taken_to_human_agent: bool
    current_assignment: Optional[AssignmentDetails] = None
    assignment_history: List[AssignmentDetails] = Field(default_factory=list)

class Recovery(BaseModel):
    recovery_id: str
    third_party_name: str
    recovery_amount: float
    currency: str
    status: RecoveryStatus
    initiated_date: str
    recovered_date: Optional[str]
    notes: Optional[str]

class Claim(BaseModel):
    claim_id: str
    policy: Policy
    claimant: Claimant
    additional_claimants: Optional[List[Claimant]] = None
    loss_details: LossDetails
    claim_status: ClaimStatus
    reported_date: str
    documents: List[ClaimDocument]
    fraud_assessment: FraudAssessment
    beneficiary_dispute: Optional[BeneficiaryDispute]
    payments: List[ClaimPayment]
    subrogation_possible: bool
    subrogation_notes: Optional[str]
    audit_trail: Dict[str,str]
    assignments: Optional[Assignments]
    # recoveries: Optional[List[Recovery]] = []
    recoveries: List[Recovery] = Field(default_factory=list)


# class InsuranceClaimsDB(DB):
#     claims: Dict[str, Claim]

#     def get_statistics(self) -> Dict[str, Any]:
#         return {
#             "num_claims": len(self.claims),
#             "by_status": {
#                 status: sum(1 for c in self.claims.values() if c.claim_status == status)
#                 for status in [
#                     "new",
#                     "under_review",
#                     "approved",
#                     "rejected",
#                     "settled",
#                     "closed",
#                 ]
#             },
#             "fraud_cases": sum(
#                 1 for c in self.claims.values()
#                 if c.fraud_assessment.fraud_flag != "none"
#             ),
#             "third_party_claims": sum(
#                 1 for c in self.claims.values()
#                 if c.claimant.role == "third_party"
#             ),
#         }


class InsuranceClaimsDB(RootModel[Dict[str, Claim]]):
    # property for Tau2 compatibility
    @property
    def claims(self) -> Dict[str, Claim]:
        return self.root

    def get_statistics(self) -> Dict[str, Any]:
        claims = self.root  # still use root internally
        return {
            "num_claims": len(claims),
            "by_status": {
                status: sum(1 for c in claims.values() if c.claim_status == status)
                for status in ["new", "under_review", "approved", "rejected", "settled", "closed"]
            },
            "fraud_cases": sum(1 for c in claims.values() if c.fraud_assessment.fraud_flag != "none"),
            "third_party_claims": sum(1 for c in claims.values() if c.claimant.role == "third_party"),
        }


def get_db(db_path: str):
    return InsuranceClaimsDB.model_validate_json(open(db_path, "r", encoding="utf-8").read())

if __name__ == "__main__":
    db = get_db(CLAIMS_DB_PATH)
    print(db.get_statistics())

