"""Toolkit for the vacation rental domain."""

from datetime import datetime, timedelta

from tau2.domains.vacation_rental.data_model import (
    Listing,
    Reservation,
    User,
    VacationRentalDB,
)
from tau2.domains.vacation_rental.utils import CURRENT_TIME
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class VacationRentalTools(ToolKitBase):
    """All the tools for the vacation rental domain."""

    db: VacationRentalDB

    def __init__(self, db: VacationRentalDB) -> None:
        super().__init__(db)

    def _get_current_time(self) -> datetime:
        """Get the current time for the domain."""
        return datetime.fromisoformat(CURRENT_TIME)

    def _get_user(self, user_id: str) -> User:
        """Get user from database."""
        if user_id not in self.db.users:
            raise ValueError(f"User {user_id} not found")
        return self.db.users[user_id]

    def _get_reservation(self, reservation_id: str) -> Reservation:
        """Get reservation from database."""
        if reservation_id not in self.db.reservations:
            raise ValueError(f"Reservation {reservation_id} not found")
        return self.db.reservations[reservation_id]

    def _get_listing(self, listing_id: str) -> Listing:
        """Get listing from database."""
        if listing_id not in self.db.listings:
            raise ValueError(f"Listing {listing_id} not found")
        return self.db.listings[listing_id]

    def _calculate_nights(self, check_in_date: str, check_out_date: str) -> int:
        """Calculate the number of nights between check-in and check-out."""
        check_in = datetime.strptime(check_in_date, "%Y-%m-%d")
        check_out = datetime.strptime(check_out_date, "%Y-%m-%d")
        return (check_out - check_in).days

    def _calculate_days_until_checkin(self, check_in_date: str) -> float:
        """Calculate days until check-in from current time."""
        current = self._get_current_time()
        check_in = datetime.strptime(check_in_date, "%Y-%m-%d")
        delta = check_in - current
        return delta.total_seconds() / (24 * 3600)

    def _calculate_hours_since_booking(self, created_at: str) -> float:
        """Calculate hours since booking was created."""
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
            ValueError: If the expression contains invalid characters.
        """
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))

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
