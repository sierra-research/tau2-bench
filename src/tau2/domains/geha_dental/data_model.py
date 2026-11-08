from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from tau2.domains.geha_dental.utils import GEHA_DENTAL_DB_PATH
from tau2.environment.db import DB

Plan = Literal["High", "Standard"]
EnrollmentType = Literal["Self Only", "Self Plus One", "Self and Family"]
PayFrequency = Literal["biweekly", "monthly"]
Relationship = Literal["self", "spouse", "child", "stepchild"]
MemberStatus = Literal["employed", "retired"]
BenefitClass = Literal["A", "B", "C", "D"]


class CoveredPerson(BaseModel):
    person_id: str = Field(description="Unique id of the covered person")
    full_name: str
    relationship: Relationship
    date_of_birth: str = Field(description="YYYY-MM-DD")


class EnrollmentChange(BaseModel):
    action: Literal["add_person", "remove_person"]
    person_id: str
    qle_type: str
    qle_date: str
    requested_on: str
    effective_date: str


class Enrollment(BaseModel):
    enrollment_id: str
    member_id: str = Field(description="The enrollee (subscriber)")
    plan: Plan
    enrollment_type: EnrollmentType
    covered_persons: List[CoveredPerson]
    effective_date: str
    changes: List[EnrollmentChange] = Field(default_factory=list)


class Member(BaseModel):
    member_id: str
    full_name: str
    date_of_birth: str = Field(description="YYYY-MM-DD")
    status: MemberStatus = Field(
        description="employed members pay biweekly; retired members pay monthly"
    )
    state: str = Field(description="Two-letter state code, or INTL")
    zip_code: str
    enrollment_id: str
    id_card_requests: List[str] = Field(
        default_factory=list, description="Dates replacement ID cards were requested"
    )


class Claim(BaseModel):
    claim_id: str
    member_id: str = Field(description="The enrollee whose enrollment the claim is under")
    patient_person_id: str
    date_of_service: str
    cdt_code: str
    network: Literal["in", "out"]
    billed: float
    paid: float
    status: Literal["paid", "denied"]
    decision_date: str
    denial_reason: Optional[str] = None
    appeal_status: Optional[Literal["filed"]] = None
    appeal_filed_on: Optional[str] = None


class Procedure(BaseModel):
    cdt_code: str
    description: str
    benefit_class: BenefitClass
    limitations: str


class RateCodeRule(BaseModel):
    state: str
    zip3_ranges: List[List[str]] = Field(
        description="Inclusive [start, end] ranges of the first three ZIP digits; empty = rest of state"
    )
    rate_code: int


class PlanBenefits(BaseModel):
    plan: Plan
    coinsurance_member_pays: Dict[str, Dict[str, str]] = Field(
        description="class -> network -> what the member pays"
    )
    annual_maximum: Dict[str, str]
    orthodontic_lifetime_maximum: Dict[str, str]
    deductible: Dict[str, str]
    notes: List[str]


class GehaDentalDB(DB):
    """Synthetic member, enrollment and claim data plus 2026 plan tables."""

    members: Dict[str, Member]
    enrollments: Dict[str, Enrollment]
    claims: Dict[str, Claim]
    procedures: Dict[str, Procedure]
    rate_code_rules: List[RateCodeRule]
    premiums: Dict[str, Dict[str, Dict[str, List[float]]]] = Field(
        description="plan -> pay frequency -> enrollment type -> premium for rate codes 1..5"
    )
    benefits: Dict[str, PlanBenefits]


def get_db() -> GehaDentalDB:
    return GehaDentalDB.load(GEHA_DENTAL_DB_PATH)
