"""User tools for the hospitality domain."""

from typing import Any, Dict, List, Optional

from tau2.domains.hospitality.user_data_model import (
    CustomerContext,
    CustomerMood,
    CustomerType,
    HospitalityUserDB,
    ReceivedSMS,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class HospitalityUserTools(ToolKitBase):
    """Tools available to the user/customer in the hospitality domain."""

    db: HospitalityUserDB

    def __init__(self, db: HospitalityUserDB) -> None:
        """Initialize user tools with a database instance."""
        super().__init__(db)

    # ============== Setup Methods (not tools, used for initialization) ==============

    def set_user_info(self, name: str, phone: str) -> str:
        """Set the user's basic information."""
        self.db.context.name = name
        self.db.context.phone = phone
        return f"User info set: {name}, {phone}"

    def set_customer_type(self, customer_type: str) -> str:
        """Set the customer type/persona."""
        self.db.context.customer_type = CustomerType(customer_type)
        return f"Customer type set to: {customer_type}"

    def set_mood(self, mood: str) -> str:
        """Set the customer's mood."""
        self.db.context.mood = CustomerMood(mood)
        return f"Mood set to: {mood}"

    def set_party_info(self, party_size: int, num_kids: int = 0) -> str:
        """Set party information."""
        self.db.context.party_size = party_size
        self.db.context.num_kids = num_kids
        self.db.context.has_kids = num_kids > 0
        return f"Party: {party_size} people, {num_kids} kids"

    def set_special_occasion(
        self,
        is_birthday: bool = False,
        is_anniversary: bool = False,
        is_business_meal: bool = False,
    ) -> str:
        """Set special occasion flags."""
        self.db.context.is_birthday = is_birthday
        self.db.context.is_anniversary = is_anniversary
        self.db.context.is_business_meal = is_business_meal
        occasions = []
        if is_birthday:
            occasions.append("birthday")
        if is_anniversary:
            occasions.append("anniversary")
        if is_business_meal:
            occasions.append("business meal")
        return f"Special occasion: {', '.join(occasions) if occasions else 'none'}"

    def set_allergies(self, allergies: List[str]) -> str:
        """Set customer allergies."""
        self.db.context.allergies = allergies
        return f"Allergies set: {', '.join(allergies)}"

    def set_membership(self, tier: str, points: int, visit_count: int = 0) -> str:
        """Set membership information."""
        self.db.context.membership_tier = tier
        self.db.context.points_balance = points
        self.db.context.previous_visit_count = visit_count
        return f"Membership: {tier}, {points} points, {visit_count} visits"

    def add_received_sms(
        self,
        date: str,
        content: str,
        promo_code: Optional[str] = None,
        missing_terms: Optional[str] = None,
    ) -> str:
        """Add an SMS that the customer received."""
        sms = ReceivedSMS(
            date=date,
            content=content,
            promo_code=promo_code,
            missing_terms=missing_terms,
        )
        self.db.received_sms.append(sms)
        return "SMS added"

    def set_expectations(
        self,
        wants_compensation: bool = False,
        minimum_acceptable_discount: Optional[float] = None,
        will_accept_alternative: bool = True,
        will_threaten_review: bool = False,
    ) -> str:
        """Set customer expectations."""
        self.db.expectations.wants_compensation = wants_compensation
        self.db.expectations.minimum_acceptable_discount = minimum_acceptable_discount
        self.db.expectations.will_accept_alternative = will_accept_alternative
        self.db.expectations.will_threaten_review = will_threaten_review
        return "Expectations set"

    # ============== READ Tools (available to user during simulation) ==============

    @is_tool(ToolType.READ)
    def check_received_sms(self) -> str:
        """
        Check the promotional SMS messages you have received from the restaurant.

        Returns:
            List of SMS messages with their content.
        """
        if not self.db.received_sms:
            return "You have no promotional SMS messages."

        messages = []
        for sms in self.db.received_sms:
            messages.append(f"[{sms.date}] {sms.content}")

        return "Your SMS messages:\n" + "\n".join(messages)

    @is_tool(ToolType.READ)
    def check_my_membership(self) -> str:
        """
        Check your membership status and points balance.

        Returns:
            Membership tier and points information.
        """
        ctx = self.db.context
        if not ctx.membership_tier:
            return "You don't have a membership account yet."

        return (
            f"Membership Tier: {ctx.membership_tier}\n"
            f"Points Balance: {ctx.points_balance}\n"
            f"Visit Count: {ctx.previous_visit_count}"
        )

    @is_tool(ToolType.READ)
    def check_my_allergies(self) -> str:
        """
        Check your recorded allergies.

        Returns:
            List of your allergies.
        """
        allergies = self.db.context.allergies
        if not allergies:
            return "You have no recorded allergies."
        return f"Your allergies: {', '.join(allergies)}"

    @is_tool(ToolType.READ)
    def check_party_info(self) -> str:
        """
        Check information about your party.

        Returns:
            Party size and composition.
        """
        ctx = self.db.context
        info = [f"Party size: {ctx.party_size}"]
        if ctx.has_kids:
            info.append(f"Number of kids: {ctx.num_kids}")
        if ctx.is_birthday:
            info.append("Celebrating a birthday")
        if ctx.is_anniversary:
            info.append("Celebrating an anniversary")
        if ctx.is_business_meal:
            info.append("Business meal")
        return "\n".join(info)

    @is_tool(ToolType.READ)
    def check_current_satisfaction(self) -> str:
        """
        Check your current satisfaction level with the service.

        Returns:
            Current satisfaction score and issues reported.
        """
        issues = self.db.issues_reported
        compensation = self.db.compensation_received
        score = self.db.satisfaction_score

        result = f"Current satisfaction: {score}/10"
        if issues:
            result += f"\nIssues reported: {', '.join(issues)}"
        if compensation:
            result += f"\nCompensation received: {', '.join(compensation)}"
        return result

    # ============== WRITE Tools ==============

    @is_tool(ToolType.WRITE)
    def report_issue(self, issue: str) -> str:
        """
        Report an issue with the service.

        Args:
            issue: Description of the issue.

        Returns:
            Confirmation that issue was recorded.
        """
        self.db.issues_reported.append(issue)
        # Lower satisfaction when issue is reported
        self.db.satisfaction_score = max(1, self.db.satisfaction_score - 2)
        return f"Issue recorded: {issue}"

    @is_tool(ToolType.WRITE)
    def acknowledge_compensation(self, compensation: str) -> str:
        """
        Acknowledge compensation received from the restaurant.

        Args:
            compensation: Description of compensation received.

        Returns:
            Confirmation and updated satisfaction.
        """
        self.db.compensation_received.append(compensation)
        # Increase satisfaction when compensation is given
        self.db.satisfaction_score = min(10, self.db.satisfaction_score + 1)
        return f"Compensation acknowledged: {compensation}. Satisfaction: {self.db.satisfaction_score}/10"

    @is_tool(ToolType.WRITE)
    def update_satisfaction(self, change: int, reason: str) -> str:
        """
        Update satisfaction score based on service experience.

        Args:
            change: Amount to change satisfaction (-5 to +5).
            reason: Reason for the change.

        Returns:
            New satisfaction score.
        """
        change = max(-5, min(5, change))
        self.db.satisfaction_score = max(
            1, min(10, self.db.satisfaction_score + change)
        )
        return (
            f"Satisfaction updated to {self.db.satisfaction_score}/10. Reason: {reason}"
        )

    # ============== Assertion Methods (for deterministic evaluation) ==============

    def assert_satisfaction_level(self, min_score: int) -> bool:
        """Assert that satisfaction is at or above a minimum level."""
        return self.db.satisfaction_score >= min_score

    def assert_issue_resolved(self, issue_keyword: str) -> bool:
        """Assert that an issue was addressed with compensation."""
        for issue in self.db.issues_reported:
            if issue_keyword.lower() in issue.lower():
                # Check if any compensation was given
                return len(self.db.compensation_received) > 0
        return True  # Issue wasn't reported, so nothing to resolve

    def assert_no_bad_review_threat(self) -> bool:
        """Assert that customer is not threatening a bad review."""
        return (
            not self.db.expectations.will_threaten_review
            or self.db.satisfaction_score >= 4
        )

    def assert_allergy_acknowledged(self) -> bool:
        """Assert that allergies were properly acknowledged."""
        return True  # Checked via ACTION or NL_ASSERTION

    def assert_compensation_received(self) -> bool:
        """Assert that some form of compensation was received."""
        return len(self.db.compensation_received) > 0

    def assert_no_compensation_received(self) -> bool:
        """Assert that NO compensation was received (for cases where it shouldn't be given)."""
        return len(self.db.compensation_received) == 0

    def assert_special_occasion_acknowledged(self) -> bool:
        """Assert that the special occasion (birthday/anniversary) was acknowledged."""
        # This is typically checked via NL_ASSERTION for conversation content
        return True

    def assert_customer_is_regular(self) -> bool:
        """Assert that customer is recognized as a regular based on visit count."""
        return self.db.context.previous_visit_count >= 5

    def assert_allergy_is_gluten(self) -> bool:
        """Assert that customer has gluten allergy (for gluten-specific test cases)."""
        allergies = [a.lower() for a in self.db.context.allergies]
        return "gluten" in allergies or "wheat" in allergies or "celiac" in allergies

    def assert_has_kids(self) -> bool:
        """Assert that customer has kids in their party."""
        return self.db.context.has_kids

    def assert_is_birthday(self) -> bool:
        """Assert that it's a birthday celebration."""
        return self.db.context.is_birthday

    def assert_is_business_meal(self) -> bool:
        """Assert that it's a business meal."""
        return self.db.context.is_business_meal

    def assert_membership_tier(self, expected_tier: str) -> bool:
        """Assert that customer is at a specific membership tier."""
        return self.db.context.membership_tier == expected_tier

    def assert_sms_evidence_exists(self) -> bool:
        """Assert that customer has SMS evidence (for SMS dispute cases)."""
        return len(self.db.received_sms) > 0

    def assert_party_size(self, min_size: int) -> bool:
        """Assert that party size is at least the specified minimum."""
        return self.db.context.party_size >= min_size

    def assert_mood(self, expected_mood: str) -> bool:
        """Assert customer mood matches expected."""
        return self.db.context.mood.value == expected_mood


if __name__ == "__main__":
    from tau2.domains.hospitality.user_data_model import get_default_user_db

    db = get_default_user_db()
    tools = HospitalityUserTools(db)
    print(tools.get_statistics())
