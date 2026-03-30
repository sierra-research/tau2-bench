"""Toolkit for the vacation rental domain."""

import ast
import operator
from datetime import datetime

from tau2.domains.vacation_rental.data_model import (
    GuestHistory,
    HostDecision,
    HostProfile,
    Issue,
    Listing,
    Reservation,
    User,
    VacationRentalDB,
)
from tau2.domains.vacation_rental.utils import CURRENT_TIME
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

VALID_ISSUE_TYPES = [
    "property_condition",
    "cleanliness",
    "amenity_malfunction",
    "not_as_described",
    "rule_violation",
    "safety_concern",
    "cancellation_dispute",
]
VALID_SEVERITIES = ["minor", "moderate", "major", "critical"]


class VacationRentalTools(ToolKitBase):
    """All the tools for the vacation rental domain."""

    db: VacationRentalDB

    def __init__(self, db: VacationRentalDB) -> None:
        super().__init__(db)

    def _get_current_time(self) -> datetime:
        return datetime.fromisoformat(CURRENT_TIME)

    def _get_user(self, user_id: str) -> User:
        if user_id not in self.db.users:
            raise ValueError(f"User {user_id} not found")
        return self.db.users[user_id]

    def _get_reservation(self, reservation_id: str) -> Reservation:
        if reservation_id not in self.db.reservations:
            raise ValueError(f"Reservation {reservation_id} not found")
        return self.db.reservations[reservation_id]

    def _get_listing(self, listing_id: str) -> Listing:
        if listing_id not in self.db.listings:
            raise ValueError(f"Listing {listing_id} not found")
        return self.db.listings[listing_id]

    def _calculate_nights(self, check_in_date: str, check_out_date: str) -> int:
        check_in = datetime.strptime(check_in_date, "%Y-%m-%d")
        check_out = datetime.strptime(check_out_date, "%Y-%m-%d")
        return (check_out - check_in).days

    def _calculate_days_until_checkin(self, check_in_date: str) -> float:
        current = self._get_current_time()
        check_in = datetime.strptime(check_in_date, "%Y-%m-%d")
        delta = check_in - current
        return delta.total_seconds() / (24 * 3600)

    def _calculate_hours_since_booking(self, created_at: str) -> float:
        current = self._get_current_time()
        created = datetime.fromisoformat(created_at)
        delta = current - created
        return delta.total_seconds() / 3600

    def _calculate_refund_amount(
        self,
        reservation: Reservation,
        listing: Listing,
        cancelled_by: str = "guest",
    ) -> float:
        """
        Calculate the refund amount based on cancellation policy and timing.

        Args:
            reservation: The reservation being cancelled
            listing: The listing associated with the reservation
            cancelled_by: Who initiated the cancellation ('guest' or 'host')

        Returns:
            The refund amount in USD
        """
        total_amount = reservation.amount_paid
        nightly_rate = listing.nightly_rate
        num_nights = self._calculate_nights(
            reservation.check_in_date, reservation.check_out_date
        )

        # Host cancellation always results in full refund
        if cancelled_by == "host":
            return total_amount

        days_until_checkin = self._calculate_days_until_checkin(
            reservation.check_in_date
        )
        hours_since_booking = self._calculate_hours_since_booking(
            reservation.created_at
        )

        # Free cancellation period: within 24 hours of booking AND 7+ days before check-in
        if hours_since_booking < 24 and days_until_checkin >= 7:
            return total_amount

        policy = listing.cancellation_policy

        if policy == "flexible":
            # 24 hours or more: full refund
            # Less than 24 hours: first night non-refundable
            if days_until_checkin >= 1:
                return total_amount
            else:
                return total_amount - nightly_rate

        elif policy == "moderate":
            # 5 days or more: full refund
            # Less than 5 days: first night non-refundable, 50% of remaining nights refunded
            if days_until_checkin >= 5:
                return total_amount
            else:
                remaining_nights = num_nights - 1
                refund = remaining_nights * nightly_rate * 0.5
                return round(refund, 2)

        elif policy == "firm":
            # 30 days or more: full refund
            # 7-29 days: 50% refund
            # Less than 7 days: no refund
            if days_until_checkin >= 30:
                return total_amount
            elif days_until_checkin >= 7:
                return round(total_amount * 0.5, 2)
            else:
                return 0.0

        elif policy == "strict":
            # 7 days or more: 50% refund
            # Less than 7 days: no refund
            if days_until_checkin >= 7:
                return round(total_amount * 0.5, 2)
            else:
                return 0.0

        return 0.0

    @is_tool(ToolType.READ)
    def get_current_time(self) -> str:
        """
        Get the current date and time.

        Use this to calculate timing for cancellation policies, such as:
        - Days until check-in date
        - Hours since booking was created

        Returns:
            The current datetime in ISO format (e.g., '2024-12-15T10:30:00').
        """
        return CURRENT_TIME

    def _safe_eval_expr(self, node: ast.expr) -> float:
        """Safely evaluate an AST expression node containing only arithmetic operations."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.BinOp):
            ops: dict[type, operator] = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
            }
            op_func = ops.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            left = self._safe_eval_expr(node.left)
            right = self._safe_eval_expr(node.right)
            return op_func(left, right)
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -self._safe_eval_expr(node.operand)
            elif isinstance(node.op, ast.UAdd):
                return self._safe_eval_expr(node.operand)
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    @is_tool(ToolType.READ)
    def get_cancellation_policy_rules(self) -> dict:
        """
        Get the cancellation policy rules for all policy types.

        Use this to understand how refunds are calculated based on timing
        and policy type. The agent must apply these rules to calculate
        the correct refund amount.

        Returns:
            A dictionary mapping policy names to their refund rules.
        """
        return {
            "flexible": {
                "full_refund_condition": "24+ hours before check-in",
                "partial_refund": "Less than 24 hours: first night non-refundable, "
                "remainder refunded",
            },
            "moderate": {
                "full_refund_condition": "5+ days before check-in",
                "partial_refund": "Less than 5 days: first night non-refundable, "
                "50% of remaining nights refunded",
            },
            "firm": {
                "full_refund_condition": "30+ days before check-in",
                "partial_refund": "7-29 days: 50% refund. Less than 7 days: no refund",
            },
            "strict": {
                "full_refund_condition": "None (no full refund available)",
                "partial_refund": "7+ days before check-in: 50% refund. "
                "Less than 7 days: no refund",
            },
            "grace_period": {
                "description": "Free cancellation within 24 hours of booking "
                "if check-in is 7+ days away (applies to all policies)",
            },
            "host_cancellation": {
                "description": "If cancelled by host, guest always receives full refund",
            },
        }

    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """
        Calculate the result of a mathematical expression.

        Args:
            expression: The mathematical expression to calculate, such as
                       '4 * 200 * 0.5' or '1500 - 200'. Supports +, -, *, /,
                       parentheses, and decimal numbers.

        Returns:
            The result of the mathematical expression, rounded to 2 decimal places.

        Raises:
            ValueError: If the expression contains invalid characters or is malformed.
        """
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        try:
            tree = ast.parse(expression, mode="eval")
            result = self._safe_eval_expr(tree.body)
            return str(round(result, 2))
        except SyntaxError as e:
            raise ValueError(f"Invalid expression: {e}") from e

    @is_tool(ToolType.READ)
    def get_user_details(self, user_id: str) -> User:
        """
        Get the details of a user, including their reservations.

        Args:
            user_id: The user ID, such as 'frieda_schmidt_1234'.

        Returns:
            The user details including name, email, phone, payment methods, and reservation IDs.

        Raises:
            ValueError: If the user is not found.
        """
        return self._get_user(user_id)

    @is_tool(ToolType.READ)
    def get_reservation_details(self, reservation_id: str) -> Reservation:
        """
        Get the details of a reservation.

        Args:
            reservation_id: The reservation ID, such as 'RES001'.

        Returns:
            The reservation details including dates, amounts, status, and associated listing.

        Raises:
            ValueError: If the reservation is not found.
        """
        return self._get_reservation(reservation_id)

    @is_tool(ToolType.READ)
    def get_listing_details(self, listing_id: str) -> Listing:
        """
        Get the details of a listing, including its cancellation policy.

        Args:
            listing_id: The listing ID, such as 'LST001'.

        Returns:
            The listing details including title, address, nightly rate, and cancellation policy.

        Raises:
            ValueError: If the listing is not found.
        """
        return self._get_listing(listing_id)

    @is_tool(ToolType.WRITE)
    def cancel_reservation(
        self,
        reservation_id: str,
        expected_refund_amount: float,
        cancelled_by: str = "guest",
    ) -> dict:
        """
        Cancel a reservation. The agent must calculate and provide the expected
        refund amount based on the listing's cancellation policy and timing.

        Before calling this function, the agent should:
        1. Get reservation details to find check-in date, amount paid, and listing ID
        2. Get listing details to find the cancellation policy and nightly rate
        3. Get current time to calculate days until check-in
        4. Apply the cancellation policy rules to calculate the refund
        5. Use the calculate tool if needed for arithmetic

        A reservation can only be cancelled if its status is 'confirmed' and
        the check-in date has not passed.

        The refund is processed to the original payment method.

        Args:
            reservation_id: The reservation ID, such as 'RES001'.
            expected_refund_amount: The refund amount calculated by the agent based
                                   on the cancellation policy and timing. Must match
                                   the correct refund amount within $0.01.
            cancelled_by: Who initiated the cancellation: 'guest' or 'host'. Defaults to 'guest'.

        Returns:
            A dictionary containing:
            - success: Whether the cancellation was successful
            - reservation_id: The cancelled reservation ID
            - refund_amount: The refund amount in USD
            - message: A description of the cancellation result

        Raises:
            ValueError: If the reservation is not found.
            ValueError: If the reservation is not in 'confirmed' status.
            ValueError: If the check-in date has already passed.
            ValueError: If cancelled_by is not 'guest' or 'host'.
            ValueError: If expected_refund_amount does not match the correct calculation.
        """
        expected_refund_amount = float(expected_refund_amount)

        if cancelled_by not in ("guest", "host"):
            raise ValueError("cancelled_by must be 'guest' or 'host'")

        reservation = self._get_reservation(reservation_id)

        # Validate reservation can be cancelled
        if reservation.status != "confirmed":
            raise ValueError(
                f"Reservation {reservation_id} cannot be cancelled. "
                f"Current status: {reservation.status}"
            )

        # Check if check-in date has passed
        current_time = self._get_current_time()
        check_in = datetime.strptime(reservation.check_in_date, "%Y-%m-%d")
        if current_time.date() > check_in.date():
            raise ValueError(
                f"Cannot cancel reservation {reservation_id}. "
                f"Check-in date {reservation.check_in_date} has already passed."
            )

        # Get listing to determine cancellation policy
        listing = self._get_listing(reservation.listing_id)

        # Calculate the correct refund amount
        actual_refund_amount = self._calculate_refund_amount(
            reservation, listing, cancelled_by
        )

        # Validate agent's calculation matches (within $0.01 tolerance)
        if abs(expected_refund_amount - actual_refund_amount) > 0.01:
            raise ValueError(
                f"Refund calculation mismatch. Please review the listing's "
                f"cancellation policy ('{listing.cancellation_policy}') and timing. "
                f"Use get_cancellation_policy_rules() to understand the refund rules, "
                f"and get_current_time() to calculate days until check-in."
            )

        # Update reservation status
        reservation.status = "cancelled"
        reservation.refund_amount = actual_refund_amount
        reservation.cancelled_by = cancelled_by

        return {
            "success": True,
            "reservation_id": reservation_id,
            "refund_amount": actual_refund_amount,
            "message": f"Reservation {reservation_id} has been cancelled. "
            f"A refund of ${actual_refund_amount:.2f} will be processed to the original payment method.",
        }

    @is_tool(ToolType.WRITE)
    def process_refund(
        self,
        reservation_id: str,
        payment_method_id: str,
        amount: float,
    ) -> dict:
        """
        Process a refund for a cancelled reservation to a specific payment method.

        This is used when the original payment method is no longer valid and
        a refund needs to be sent to a different payment method.

        Args:
            reservation_id: The reservation ID, such as 'RES001'.
            payment_method_id: The payment method ID to refund to, such as 'credit_card_1001'.
            amount: The refund amount in USD.

        Returns:
            A dictionary containing:
            - success: Whether the refund was processed successfully
            - reservation_id: The reservation ID
            - payment_method_id: The payment method that received the refund
            - amount: The refund amount
            - message: A description of the refund result

        Raises:
            ValueError: If the reservation is not found.
            ValueError: If the reservation is not in 'cancelled' status.
            ValueError: If the payment method is not found for the user.
            ValueError: If the amount exceeds the amount paid.
        """
        amount = float(amount)
        reservation = self._get_reservation(reservation_id)

        if reservation.status != "cancelled":
            raise ValueError(
                f"Cannot process refund for reservation {reservation_id}. "
                f"Reservation must be cancelled first. Current status: {reservation.status}"
            )

        if amount > reservation.amount_paid:
            raise ValueError(
                f"Refund amount ${amount:.2f} exceeds amount paid ${reservation.amount_paid:.2f}"
            )

        # Verify payment method exists for the user
        user = self._get_user(reservation.guest_user_id)
        if payment_method_id not in user.payment_methods:
            raise ValueError(
                f"Payment method {payment_method_id} not found for user {reservation.guest_user_id}"
            )

        return {
            "success": True,
            "reservation_id": reservation_id,
            "payment_method_id": payment_method_id,
            "amount": amount,
            "message": f"Refund of ${amount:.2f} has been processed to payment method {payment_method_id}. "
            f"Please allow 10-15 business days for the refund to appear.",
        }

    # === Host Consideration Tools ===

    @is_tool(ToolType.READ)
    def get_host_profile(self, host_user_id: str) -> HostProfile:
        """
        Get the host's profile including their preferences, philosophy, and flexibility settings.

        Use this to understand how a host approaches their business and what factors
        might influence their decisions on exceptions or disputes.

        Args:
            host_user_id: The user ID of the host (e.g., 'host_ibrahim_alzahrani_1111')

        Returns:
            The host's profile including philosophy, flexibility settings, hard limits,
            soft spots, and deal breakers.

        Raises:
            ValueError: If the host profile is not found.
        """
        if host_user_id not in self.db.host_profiles:
            raise ValueError(f"Host profile for {host_user_id} not found")
        return self.db.host_profiles[host_user_id]

    @is_tool(ToolType.READ)
    def get_guest_history(self, guest_user_id: str) -> GuestHistory:
        """
        Get a guest's stay history.

        Use this to identify repeat guests and understand their track record
        before making decisions that might be influenced by guest loyalty.

        Args:
            guest_user_id: The user ID of the guest

        Returns:
            Guest history including total stays, stays by host, issues reported,
            and cancellation count.

        Raises:
            ValueError: If the guest history is not found.
        """
        if guest_user_id not in self.db.guest_history:
            raise ValueError(f"Guest history for {guest_user_id} not found")
        return self.db.guest_history[guest_user_id]

    @is_tool(ToolType.READ)
    def get_issue_details(self, issue_id: str) -> Issue:
        """
        Get the details of a reported issue.

        Args:
            issue_id: The issue ID (e.g., 'ISS_RES002_001')

        Returns:
            The issue details including type, severity, evidence status, and resolution.

        Raises:
            ValueError: If the issue is not found.
        """
        if issue_id not in self.db.issues:
            raise ValueError(f"Issue {issue_id} not found")
        return self.db.issues[issue_id]

    def _get_evidence_recommendation(self, issue: Issue) -> str:
        if issue.evidence_status == "validated":
            return (
                "Evidence supports claim. Consider significant compensation."
                if issue.severity in ("major", "critical")
                else "Evidence supports minor claim. Consider partial compensation or service credit."
            )
        if issue.evidence_status == "invalidated":
            return "Evidence does not support claim. Apply standard policy."
        if issue.evidence_status == "inconclusive":
            return "Evidence is inconclusive. Consider partial resolution or escalate to host."
        return "Awaiting evidence validation."

    @is_tool(ToolType.READ)
    def validate_issue_evidence(self, issue_id: str) -> dict:
        """
        Validate the evidence submitted for an issue.

        This checks pre-populated validation results for the issue's evidence.
        Use this after an issue is reported to determine if the claim is supported.

        Args:
            issue_id: The issue ID to validate evidence for

        Returns:
            A dictionary containing:
            - evidence_status: 'validated', 'invalidated', 'inconclusive', or 'pending'
            - validation_result: Details of what the evidence showed
            - recommendation: Suggested action based on evidence

        Raises:
            ValueError: If the issue is not found.
        """
        if issue_id not in self.db.issues:
            raise ValueError(f"Issue {issue_id} not found")

        issue = self.db.issues[issue_id]

        if not issue.evidence_submitted:
            return {
                "evidence_status": "pending",
                "validation_result": "No evidence has been submitted for this issue.",
                "recommendation": "Request evidence from the guest before proceeding.",
            }

        return {
            "evidence_status": issue.evidence_status,
            "validation_result": issue.validation_result,
            "recommendation": self._get_evidence_recommendation(issue),
        }

    @is_tool(ToolType.READ)
    def request_host_decision(
        self,
        host_user_id: str,
        situation_type: str,
        guest_context: str | None = None,
    ) -> HostDecision:
        """
        Request the host's decision for a specific situation.

        This looks up pre-computed host decisions based on their profile and the
        situation type. Use this when a guest request goes beyond standard policy
        and host input is needed.

        Args:
            host_user_id: The host's user ID
            situation_type: Type of situation ('cancellation_exception', 'partial_refund',
                           'issue_compensation', 'early_checkin_request', etc.)
            guest_context: Optional context about the guest (e.g., 'repeat_guest',
                          'has_documentation', 'medical_emergency_no_documentation')

        Returns:
            The host's decision. If no specific host preference exists for this
            situation, returns a default decision with decision='defer_to_policy',
            indicating standard platform policy should be applied.
        """
        # Look for matching decision, preferring specific guest_context over default
        for context in (guest_context, None):
            for decision in self.db.host_decisions.values():
                if (
                    decision.host_user_id == host_user_id
                    and decision.situation_type == situation_type
                    and decision.guest_context == context
                ):
                    return decision

        # Return defer_to_policy if no decision found
        return HostDecision(
            decision_id=f"HD_{host_user_id}_{situation_type}_default",
            host_user_id=host_user_id,
            situation_type=situation_type,
            guest_context=guest_context,
            decision="defer_to_policy",
            reasoning="No specific host preference for this situation. Apply standard platform policy.",
            conditions=[],
        )

    @is_tool(ToolType.WRITE)
    def submit_issue_report(
        self,
        reservation_id: str,
        issue_type: str,
        description: str,
        severity: str,
        evidence_submitted: bool = False,
    ) -> Issue:
        """
        Submit an issue report for a reservation.

        Use this when a guest reports a problem with their stay. The issue will
        be created and can then be validated and resolved.

        Args:
            reservation_id: The reservation ID
            issue_type: Type of issue ('property_condition', 'cleanliness',
                       'amenity_malfunction', 'not_as_described', 'rule_violation',
                       'safety_concern', 'cancellation_dispute')
            description: Description of the problem
            severity: Severity level ('minor', 'moderate', 'major', 'critical')
            evidence_submitted: Whether the guest has submitted evidence (photos, etc.)

        Returns:
            The created issue with its ID and initial status.

        Raises:
            ValueError: If the reservation is not found or issue_type/severity is invalid.
        """
        reservation = self._get_reservation(reservation_id)

        if issue_type not in VALID_ISSUE_TYPES:
            raise ValueError(f"Invalid issue_type. Must be one of: {VALID_ISSUE_TYPES}")

        if severity not in VALID_SEVERITIES:
            raise ValueError(f"Invalid severity. Must be one of: {VALID_SEVERITIES}")

        # Generate issue ID
        existing_issues = [
            i for i in self.db.issues.keys() if i.startswith(f"ISS_{reservation_id}")
        ]
        issue_num = len(existing_issues) + 1
        issue_id = f"ISS_{reservation_id}_{issue_num:03d}"

        issue = Issue(
            issue_id=issue_id,
            reservation_id=reservation_id,
            guest_user_id=reservation.guest_user_id,
            reported_by="guest",
            issue_type=issue_type,
            description=description,
            severity=severity,
            evidence_submitted=evidence_submitted,
            evidence_status="pending" if evidence_submitted else None,
            status="open",
            created_at=self._get_current_time().isoformat(),
        )

        self.db.issues[issue_id] = issue
        return issue

    @is_tool(ToolType.WRITE)
    def process_goodwill_refund(
        self, reservation_id: str, amount: float, justification: str
    ) -> dict:
        """
        Process a goodwill refund beyond the standard cancellation policy.

        Use this when a host has approved additional compensation beyond what
        the cancellation policy provides. This is separate from the standard
        cancel_reservation refund.

        Args:
            reservation_id: The reservation ID
            amount: The goodwill refund amount in USD
            justification: Reason for the goodwill refund (e.g., 'Host-approved
                          exception for medical emergency')

        Returns:
            A dictionary containing success status and refund details.

        Raises:
            ValueError: If the reservation is not found or amount exceeds limits.
        """
        amount = float(amount)
        reservation = self._get_reservation(reservation_id)
        listing = self._get_listing(reservation.listing_id)

        # Get host profile to check max goodwill percentage
        host_profile = self.db.host_profiles.get(listing.host_user_id)
        if host_profile:
            max_goodwill_pct = host_profile.flexibility_settings.max_goodwill_refund_pct
        else:
            # Default cap when host profile is not available
            max_goodwill_pct = 15

        max_goodwill = reservation.amount_paid * (max_goodwill_pct / 100)
        if amount > max_goodwill:
            raise ValueError(
                f"Goodwill refund amount ${amount:.2f} exceeds maximum "
                f"goodwill limit of ${max_goodwill:.2f} ({max_goodwill_pct}%)"
            )

        return {
            "success": True,
            "reservation_id": reservation_id,
            "goodwill_amount": amount,
            "justification": justification,
            "message": f"Goodwill refund of ${amount:.2f} has been processed. "
            f"Reason: {justification}",
        }

    @is_tool(ToolType.WRITE)
    def apply_service_credit(self, user_id: str, amount: float, reason: str) -> dict:
        """
        Apply a service credit to a user's account for future bookings.

        Use this as an alternative to refunds, especially for minor issues
        or when the host prefers to offer future value instead of cash refunds.

        Args:
            user_id: The user ID to credit
            amount: The credit amount in USD
            reason: Reason for the credit

        Returns:
            A dictionary containing success status and credit details.

        Raises:
            ValueError: If the user is not found.
        """
        amount = float(amount)
        user = self._get_user(user_id)

        return {
            "success": True,
            "user_id": user_id,
            "credit_amount": amount,
            "reason": reason,
            "message": f"Service credit of ${amount:.2f} has been applied to "
            f"{user.name.first_name}'s account. This can be used on future bookings.",
        }

    @is_tool(ToolType.WRITE)
    def add_reservation_note(self, reservation_id: str, note: str) -> dict:
        """
        Add a note to a reservation for documentation purposes.

        Use this to document decisions, exceptions granted, or other important
        information about the reservation.

        Args:
            reservation_id: The reservation ID
            note: The note to add

        Returns:
            A dictionary confirming the note was added.

        Raises:
            ValueError: If the reservation is not found.
        """
        self._get_reservation(reservation_id)

        return {
            "success": True,
            "reservation_id": reservation_id,
            "note": note,
            "timestamp": self._get_current_time().isoformat(),
            "message": f"Note added to reservation {reservation_id}.",
        }

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent, with a summary of the user's issue.

        Only transfer if:
        - The user explicitly asks for a human agent
        - Given the policy and the available tools, you cannot solve the user's issue

        Args:
            summary: A summary of the user's issue.

        Returns:
            A message indicating the user has been transferred to a human agent.
        """
        return "Transfer successful"


if __name__ == "__main__":
    from tau2.domains.vacation_rental.utils import VACATION_RENTAL_DB_PATH

    tools = VacationRentalTools(VacationRentalDB.load(VACATION_RENTAL_DB_PATH))
    print(tools.get_statistics())
