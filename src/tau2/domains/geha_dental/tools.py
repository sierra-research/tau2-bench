"""Tools for the G.E.H.A dental member-service domain.

Tools check data integrity only (ids exist, dates are well formed). Policy rules, such as
qualifying-life-event windows, eligibility and identity checks, are the agent's job: that is
what the benchmark measures.
"""

import re
from typing import List, Optional

from tau2.domains.geha_dental.data_model import (
    Claim,
    CoveredPerson,
    Enrollment,
    EnrollmentChange,
    EnrollmentType,
    GehaDentalDB,
    Member,
    PayFrequency,
    Plan,
    PlanBenefits,
    Procedure,
    Relationship,
)
from tau2.domains.geha_dental.utils import NEXT_EFFECTIVE_DATE, TODAY
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TYPES: List[str] = ["Self Only", "Self Plus One", "Self and Family"]


def _check_date(value: str, field: str) -> str:
    value = value.strip()
    if not _DATE.match(value):
        raise ValueError(f"{field} must be in YYYY-MM-DD format, got {value!r}")
    return value


def _clean_name(name: str) -> str:
    return " ".join(part.capitalize() for part in name.split())


def _type_for(count: int) -> EnrollmentType:
    return _TYPES[min(count, 3) - 1]  # 1 person, 2 people, 3+ people


class GehaDentalTools(ToolKitBase):
    """All the tools for the G.E.H.A dental member-service domain."""

    db: GehaDentalDB

    def __init__(self, db: GehaDentalDB) -> None:
        super().__init__(db)

    # ------------------------------------------------------------------ helpers
    def _member(self, member_id: str) -> Member:
        if member_id not in self.db.members:
            raise ValueError(f"Member {member_id} not found")
        return self.db.members[member_id]

    def _enrollment(self, enrollment_id: str) -> Enrollment:
        if enrollment_id not in self.db.enrollments:
            raise ValueError(f"Enrollment {enrollment_id} not found")
        return self.db.enrollments[enrollment_id]

    def _claim(self, claim_id: str) -> Claim:
        if claim_id not in self.db.claims:
            raise ValueError(f"Claim {claim_id} not found")
        return self.db.claims[claim_id]

    # ------------------------------------------------------------------ read
    @is_tool(ToolType.READ)
    def find_member(self, full_name: str, date_of_birth: str) -> str:
        """
        Find a member's id from their full name and date of birth. Both must match the record.

        Args:
            full_name: The member's full name, e.g. 'Daniel Kim'.
            date_of_birth: The member's date of birth in YYYY-MM-DD format.

        Returns:
            The member id.

        Raises:
            ValueError: If no member matches both the name and the date of birth.
        """
        dob = _check_date(date_of_birth, "date_of_birth")
        name = _clean_name(full_name).lower()
        for member in self.db.members.values():
            if member.full_name.lower() == name and member.date_of_birth == dob:
                return member.member_id
        raise ValueError("No member matches that name and date of birth")

    @is_tool(ToolType.READ)
    def get_member_details(self, member_id: str) -> Member:
        """
        Get a member's record: name, date of birth, employment status, address state and ZIP,
        and enrollment id.

        Args:
            member_id: The member id, e.g. 'GD10001'.

        Returns:
            The member record.

        Raises:
            ValueError: If the member is not found.
        """
        return self._member(member_id)

    @is_tool(ToolType.READ)
    def get_enrollment(self, enrollment_id: str) -> Enrollment:
        """
        Get an enrollment: plan option, enrollment type, covered persons and change history.

        Args:
            enrollment_id: The enrollment id, e.g. 'EN20001'.

        Returns:
            The enrollment.

        Raises:
            ValueError: If the enrollment is not found.
        """
        return self._enrollment(enrollment_id)

    @is_tool(ToolType.READ)
    def list_claims(self, member_id: str) -> List[Claim]:
        """
        List all claims under a member's enrollment, for every covered person.

        Args:
            member_id: The enrollee's member id.

        Returns:
            The claims.

        Raises:
            ValueError: If the member is not found.
        """
        self._member(member_id)
        return [c for c in self.db.claims.values() if c.member_id == member_id]

    @is_tool(ToolType.READ)
    def get_claim(self, claim_id: str) -> Claim:
        """
        Get one claim: patient, procedure code, amounts, status, decision date and any denial
        reason.

        Args:
            claim_id: The claim id, e.g. 'CL30001'.

        Returns:
            The claim.

        Raises:
            ValueError: If the claim is not found.
        """
        return self._claim(claim_id)

    @is_tool(ToolType.READ)
    def lookup_rate_code(self, state: str, zip_code: str) -> int:
        """
        Find the premium rate code (1 to 5) for an address, from the state and the first three
        digits of the ZIP code. Use state 'INTL' for members living outside the United States.

        Args:
            state: Two-letter state code, or 'INTL'.
            zip_code: The ZIP code (at least the first three digits).

        Returns:
            The rate code.

        Raises:
            ValueError: If the state is not in the rate table.
        """
        state = state.strip().upper()
        zip3 = zip_code.strip()[:3]
        rules = [r for r in self.db.rate_code_rules if r.state == state]
        if not rules:
            raise ValueError(f"State {state} not found in the rate table")
        for rule in rules:
            for start, end in rule.zip3_ranges:
                if start <= zip3 <= end:
                    return rule.rate_code
        for rule in rules:
            if not rule.zip3_ranges:
                return rule.rate_code
        raise ValueError(f"No rate code for {state} {zip_code}")

    @is_tool(ToolType.READ)
    def get_premium(
        self,
        plan: Plan,
        enrollment_type: EnrollmentType,
        pay_frequency: PayFrequency,
        rate_code: int,
    ) -> float:
        """
        Get the 2026 premium. Employed members pay biweekly; retired members pay monthly.

        Args:
            plan: 'High' or 'Standard'.
            enrollment_type: 'Self Only', 'Self Plus One' or 'Self and Family'.
            pay_frequency: 'biweekly' (employed) or 'monthly' (retired).
            rate_code: The rate code, 1 to 5.

        Returns:
            The premium in US dollars per pay period (biweekly) or per month (monthly).

        Raises:
            ValueError: If an argument is not valid.
        """
        if not 1 <= int(rate_code) <= 5:
            raise ValueError("rate_code must be between 1 and 5")
        return self.db.premiums[plan][pay_frequency][enrollment_type][int(rate_code) - 1]

    @is_tool(ToolType.READ)
    def get_procedure(self, cdt_code: str) -> Procedure:
        """
        Get a dental procedure's benefit class (A basic, B intermediate, C major, D orthodontic)
        and its limitations, by CDT code.

        Args:
            cdt_code: The CDT code, e.g. 'D2740'.

        Returns:
            The procedure.

        Raises:
            ValueError: If the code is not covered or not in the procedure list.
        """
        code = cdt_code.strip().upper()
        if code not in self.db.procedures:
            raise ValueError(f"Procedure {code} not found in the covered procedure list")
        return self.db.procedures[code]

    @is_tool(ToolType.READ)
    def get_plan_benefits(self, plan: Plan) -> PlanBenefits:
        """
        Get a plan option's coinsurance by benefit class and network, maximums and deductible.

        Args:
            plan: 'High' or 'Standard'.

        Returns:
            The plan's benefits.
        """
        return self.db.benefits[plan]

    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """
        Calculate the result of a mathematical expression.

        Args:
            expression: The expression, using numbers, operators (+, -, *, /), parentheses and
                spaces.

        Returns:
            The result, rounded to 2 decimal places.

        Raises:
            ValueError: If the expression contains other characters.
        """
        if not all(ch in "0123456789+-*/(). " for ch in expression):
            raise ValueError("Invalid characters in expression")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))

    # ------------------------------------------------------------------ write
    @is_tool(ToolType.WRITE)
    def add_covered_person(
        self,
        enrollment_id: str,
        full_name: str,
        relationship: Relationship,
        date_of_birth: str,
        qle_type: str,
        qle_date: str,
    ) -> Enrollment:
        """
        Add a person to an enrollment because of a qualifying life event. The enrollment type
        is raised automatically (Self Only -> Self Plus One -> Self and Family). The change
        takes effect on the first day of the next month.

        Args:
            enrollment_id: The enrollment id.
            full_name: The person's full name.
            relationship: 'spouse', 'child' or 'stepchild'.
            date_of_birth: The person's date of birth, YYYY-MM-DD.
            qle_type: The qualifying life event: 'marriage', 'birth', 'adoption' or
                'loss_of_other_dental_coverage'.
            qle_date: The date of the event, YYYY-MM-DD.

        Returns:
            The updated enrollment.

        Raises:
            ValueError: If the enrollment is not found or an argument is not valid.
        """
        enrollment = self._enrollment(enrollment_id)
        if relationship == "self":
            raise ValueError("relationship must be spouse, child or stepchild")
        qle_type = qle_type.strip().lower()
        if qle_type not in {"marriage", "birth", "adoption", "loss_of_other_dental_coverage"}:
            raise ValueError(f"Unknown qle_type {qle_type!r} for adding a person")
        person = CoveredPerson(
            person_id=f"{enrollment_id}-P{len(enrollment.covered_persons) + len(enrollment.changes) + 1}",
            full_name=_clean_name(full_name),
            relationship=relationship,
            date_of_birth=_check_date(date_of_birth, "date_of_birth"),
        )
        enrollment.covered_persons.append(person)
        enrollment.enrollment_type = _type_for(len(enrollment.covered_persons))
        enrollment.changes.append(
            EnrollmentChange(
                action="add_person",
                person_id=person.person_id,
                qle_type=qle_type,
                qle_date=_check_date(qle_date, "qle_date"),
                requested_on=TODAY,
                effective_date=NEXT_EFFECTIVE_DATE,
            )
        )
        return enrollment

    @is_tool(ToolType.WRITE)
    def remove_covered_person(
        self, enrollment_id: str, person_id: str, qle_type: str, qle_date: str
    ) -> Enrollment:
        """
        Remove a person from an enrollment because of a qualifying life event. The enrollment
        type is lowered automatically. The change takes effect on the first day of the next
        month.

        Args:
            enrollment_id: The enrollment id.
            person_id: The covered person's id (from get_enrollment).
            qle_type: The qualifying life event: 'divorce', 'death' or 'other_coverage'.
            qle_date: The date of the event, YYYY-MM-DD.

        Returns:
            The updated enrollment.

        Raises:
            ValueError: If the enrollment or person is not found, the person is the enrollee,
                or an argument is not valid.
        """
        enrollment = self._enrollment(enrollment_id)
        match = [p for p in enrollment.covered_persons if p.person_id == person_id]
        if not match:
            raise ValueError(f"Person {person_id} is not covered under {enrollment_id}")
        if match[0].relationship == "self":
            raise ValueError("The enrollee cannot be removed with this tool")
        qle_type = qle_type.strip().lower()
        if qle_type not in {"divorce", "death", "other_coverage"}:
            raise ValueError(f"Unknown qle_type {qle_type!r} for removing a person")
        enrollment.covered_persons = [p for p in enrollment.covered_persons if p.person_id != person_id]
        enrollment.enrollment_type = _type_for(len(enrollment.covered_persons))
        enrollment.changes.append(
            EnrollmentChange(
                action="remove_person",
                person_id=person_id,
                qle_type=qle_type,
                qle_date=_check_date(qle_date, "qle_date"),
                requested_on=TODAY,
                effective_date=NEXT_EFFECTIVE_DATE,
            )
        )
        return enrollment

    @is_tool(ToolType.WRITE)
    def update_address(self, member_id: str, state: str, zip_code: str) -> Member:
        """
        Update a member's address state and ZIP code. The premium rate code may change.

        Args:
            member_id: The member id.
            state: Two-letter state code, or 'INTL'.
            zip_code: The 5-digit ZIP code.

        Returns:
            The updated member record.

        Raises:
            ValueError: If the member is not found or the ZIP code is not 5 digits.
        """
        member = self._member(member_id)
        zip_code = zip_code.strip()
        if not re.match(r"^\d{5}$", zip_code):
            raise ValueError("zip_code must be 5 digits")
        member.state = state.strip().upper()
        member.zip_code = zip_code
        return member

    @is_tool(ToolType.WRITE)
    def file_appeal(self, claim_id: str) -> Claim:
        """
        File a formal appeal of a denied claim on the member's behalf.

        Args:
            claim_id: The id of the denied claim.

        Returns:
            The updated claim.

        Raises:
            ValueError: If the claim is not found, is not denied, or already has an appeal.
        """
        claim = self._claim(claim_id)
        if claim.status != "denied":
            raise ValueError("Only denied claims can be appealed")
        if claim.appeal_status is not None:
            raise ValueError("An appeal has already been filed for this claim")
        claim.appeal_status = "filed"
        claim.appeal_filed_on = TODAY
        return claim

    @is_tool(ToolType.WRITE)
    def request_id_card(self, member_id: str) -> str:
        """
        Order a replacement ID card, mailed to the member's address on file.

        Args:
            member_id: The member id.

        Returns:
            A confirmation message.

        Raises:
            ValueError: If the member is not found.
        """
        member = self._member(member_id)
        member.id_card_requests.append(TODAY)
        return "Replacement ID card ordered"

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the member to a human agent, with a summary of the member's issue.
        Only transfer if
         - the member explicitly asks for a human agent, or
         - the policy says the request needs a human, or
         - given the policy and the available tools, you cannot solve the member's issue.

        Args:
            summary: A summary of the member's issue.

        Returns:
            A message indicating the member has been transferred to a human agent.
        """
        return "Transfer successful"
