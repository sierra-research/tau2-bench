"""User tools for the hospitality domain.

The user simulator plays a hotel guest. These tools model what a real guest
can do on their own: dig up a booking confirmation from their inbox, and
complete online check-in. They operate on the same database as the agent
tools, so user actions are immediately visible to the agent.
"""

import re
from datetime import date

from tau2.domains.hospitality.data_model import HospitalityDB
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class HospitalityUserTools(ToolKitBase):
    """Tools available to the guest (user simulator)."""

    db: HospitalityDB

    def __init__(self, db: HospitalityDB) -> None:
        super().__init__(db)

    def _today(self) -> date:
        return date.fromisoformat(self.db.hotel.current_time.split(" ")[0])

    @is_tool(ToolType.READ)
    def check_booking_confirmation_email(self, email: str) -> list:
        """
        Check your inbox for booking confirmation emails from the hotel.
        Returns the confirmation number, dates and status of each booking made
        with this email address.

        Args:
            email: Your email address.

        Raises:
            ValueError: If there are no booking confirmations for this email.
        """
        guest = None
        for candidate in self.db.guests.values():
            if candidate.email.lower() == email.lower():
                guest = candidate
                break
        if guest is None:
            raise ValueError(f"No booking confirmations found for {email}.")
        confirmations = [
            {
                "confirmation_number": reservation.reservation_id,
                "room_type": self.db.room_types[reservation.room_type_id].name,
                "check_in": reservation.check_in,
                "check_out": reservation.check_out,
                "status": reservation.status,
            }
            for reservation in self.db.reservations.values()
            if reservation.guest_id == guest.guest_id
        ]
        if not confirmations:
            raise ValueError(f"No booking confirmations found for {email}.")
        return confirmations

    @is_tool(ToolType.WRITE)
    def submit_online_checkin(
        self, confirmation_number: str, last_name: str, arrival_time: str
    ) -> str:
        """
        Complete online check-in for a reservation through the hotel website.
        Online check-in opens 2 days before arrival.

        Args:
            confirmation_number: Confirmation number of the reservation,
                e.g. HV-1001.
            last_name: Last name on the reservation.
            arrival_time: Planned arrival time, format HH:MM.

        Raises:
            ValueError: If the reservation is not found or not confirmed, the
                last name does not match, online check-in is not open yet, the
                arrival time format is invalid, or check-in was already
                completed.
        """
        if confirmation_number not in self.db.reservations:
            raise ValueError(f"Reservation {confirmation_number} not found.")
        reservation = self.db.reservations[confirmation_number]
        if reservation.status != "confirmed":
            raise ValueError(
                f"Reservation {confirmation_number} is {reservation.status}, "
                f"online check-in is only available for confirmed reservations."
            )
        guest = self.db.guests[reservation.guest_id]
        if last_name.lower() not in guest.name.lower():
            raise ValueError("Last name does not match the reservation.")
        if reservation.online_checkin_completed:
            raise ValueError(
                f"Online check-in for {confirmation_number} is already completed."
            )
        days_until_check_in = (
            date.fromisoformat(reservation.check_in) - self._today()
        ).days
        if days_until_check_in > 2:
            raise ValueError(
                "Online check-in opens 2 days before arrival. "
                f"Check-in date is {reservation.check_in}."
            )
        if not re.fullmatch(r"\d{1,2}:\d{2}", arrival_time):
            raise ValueError(
                f"Invalid arrival_time '{arrival_time}'. Expected format HH:MM."
            )
        reservation.online_checkin_completed = True
        return (
            f"Online check-in completed for {confirmation_number}. "
            f"See you around {arrival_time}!"
        )
