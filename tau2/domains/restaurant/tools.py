from typing import List, Dict
import json

from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.domains.restaurant.data_model import (
    Dish,
    LineEntry,
    Patron,
    PatronLocation,
    PaymentInstrument,
    RestaurantDB,
    ServiceTicket,
    TableInfo,
    TicketCharge,
    Plate,
)


class RestaurantTools(ToolKitBase):
    """Frequently-used tools for the restaurant domain."""

    db: RestaurantDB

    def __init__(self, db: RestaurantDB) -> None:
        super().__init__(db)

    # -----------------------
    # Internal helper methods
    # -----------------------
    def _get_ticket(self, ticket_ref: str) -> ServiceTicket:
        """Get a service ticket from the database."""
        if ticket_ref not in self.db.service_tickets:
            raise ValueError("Ticket not found")
        return self.db.service_tickets[ticket_ref]

    def _get_patron(self, guest_ref: str) -> Patron:
        """Get a patron from the database."""
        if guest_ref not in self.db.patron_registry:
            raise ValueError("Patron not found")
        return self.db.patron_registry[guest_ref]

    def _get_dish(self, dish_ref: str) -> Dish:
        """Get a dish from the database."""
        if dish_ref not in self.db.menu_board:
            raise ValueError("Dish not found")
        return self.db.menu_board[dish_ref]

    def _get_plate(self, dish_ref: str, plate_ref: str) -> Plate:
        """Get a plate selection from a dish."""
        dish = self._get_dish(dish_ref)
        if plate_ref not in dish.selections:
            raise ValueError("Plate not found")
        return dish.selections[plate_ref]

    def _get_instrument(self, guest_ref: str, instrument_ref: str) -> PaymentInstrument:
        """Get a saved payment instrument for a patron."""
        patron = self._get_patron(guest_ref)
        if instrument_ref not in patron.saved_instruments:
            raise ValueError("Payment instrument not found")
        return patron.saved_instruments[instrument_ref]

    def _is_placed_ticket(self, ticket: ServiceTicket) -> bool:
        """Check if the ticket is in 'placed' state (modifiable)."""
        return ticket.state == "placed"

    def _find_dish_by_plate(self, plate_ref: str) -> Dish:
        """Find the dish that contains the given plate_ref."""
        for dish in self.db.menu_board.values():
            if plate_ref in dish.selections:
                return dish
        raise ValueError("Plate not found in any dish")

    # -------------
    # Generic tools
    # -------------
    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """
        Calculate the result of a mathematical expression.

        Args:
            expression: The mathematical expression to calculate, such as '2 + 2'.
                        The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.

        Returns:
            The result of the mathematical expression (rounded to 2 decimals).

        Raises:
            ValueError: If the expression is invalid or contains invalid characters.
        """
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent, with a summary of the user's issue.
        Only transfer if
         - the user explicitly asks for a human agent, or
         - given the policy and the available tools, you cannot solve the user's issue.

        Args:
            summary: A concise summary of the user's issue.

        Returns:
            A message indicating the user has been transferred to a human agent.
        """
        return "Transfer successful"

    # ----------
    # Read tools
    # ----------
    @is_tool(ToolType.READ)
    def get_ticket_details(self, ticket_ref: str) -> ServiceTicket:
        """Get the current state and details of a service ticket.

        Args:
            ticket_ref: The ticket reference, such as 'TCKT-00001'.

        Returns:
            ServiceTicket: The ticket details.

        Raises:
            ValueError: If the ticket is not found.
        """
        return self._get_ticket(ticket_ref)

    @is_tool(ToolType.READ)
    def get_dish_details(self, dish_ref: str) -> Dish:
        """Get the menu details of a dish.

        Args:
            dish_ref: The dish reference, such as 'DISH-0001'.

        Returns:
            Dish: The dish details.

        Raises:
            ValueError: If the dish is not found.
        """
        return self._get_dish(dish_ref)

    @is_tool(ToolType.READ)
    def get_patron_details(self, guest_ref: str) -> Patron:
        """Get the details of a patron, including their saved instruments and ticket history.

        Args:
            guest_ref: The patron reference, such as 'GUEST-0001'.

        Returns:
            Patron: The patron details.

        Raises:
            ValueError: If the patron is not found.
        """
        return self._get_patron(guest_ref)

    @is_tool(ToolType.READ)
    def list_all_dishes(self) -> str:
        """List the title and dish_ref of all dishes.

        Returns:
            str: A JSON string mapping dish titles to their dish_refs, sorted alphabetically by title.
        """
        mapping = {dish.title: dish.dish_ref for dish in self.db.menu_board.values()}
        return json.dumps(mapping, sort_keys=True)

    @is_tool(ToolType.READ)
    def find_patron_ref_by_email(self, email: str) -> str:
        """Find patron reference by contact email.

        Args:
            email: The email of the patron, such as 'guest@example.com'.

        Returns:
            str: The guest_ref if found.

        Raises:
            ValueError: If the patron is not found.
        """
        for guest_ref, patron in self.db.patron_registry.items():
            if patron.contact_email.lower() == email.lower():
                return guest_ref
        raise ValueError("Patron not found")

    @is_tool(ToolType.READ)
    def find_patron_ref_by_name_zip(self, given: str, family: str, postal_code: str) -> str:
        """Find patron reference by name and postal code. Use when email is unknown or not found.

        Args:
            given: The given (first) name, such as 'John'.
            family: The family (last) name, such as 'Doe'.
            postal_code: The postal code, such as '12345'.

        Returns:
            str: The guest_ref if found.

        Raises:
            ValueError: If the patron is not found.
        """
        for guest_ref, patron in self.db.patron_registry.items():
            if (
                patron.identity.given.lower() == given.lower()
                and patron.identity.family.lower() == family.lower()
                and patron.location.postal_code == postal_code
            ):
                return guest_ref
        raise ValueError("Patron not found")

    # -----------
    # Write tools
    # -----------
    @is_tool(ToolType.WRITE)
    def cancel_placed_ticket(self, ticket_ref: str, reason: str) -> ServiceTicket:
        """Cancel a placed ticket. If the ticket has already started preparation or is delivered,
        it cannot be cancelled. The agent must explain the cancellation details and ask for explicit
        user confirmation (yes/no) to proceed. Upon cancellation, refund entries will be recorded
        for any prior payments.

        Args:
            ticket_ref: The ticket reference, such as 'TCKT-00001'.
            reason: The reason for cancellation (e.g., 'no longer needed', 'ordered by mistake').

        Returns:
            ServiceTicket: The ticket details after the cancellation.

        Raises:
            ValueError: If the ticket is not in 'placed' state.
        """
        ticket = self._get_ticket(ticket_ref)
        if ticket.state != "placed":
            raise ValueError("Only tickets in 'placed' state can be cancelled")

        # Record refunds for any 'payment' charges
        refunds: List[TicketCharge] = []
        for ch in ticket.charges:
            if ch.kind == "payment":
                refunds.append(
                    TicketCharge(kind="refund", total=ch.total, instrument_ref=ch.instrument_ref)
                )

        ticket.charges.extend(refunds)
        ticket.state = "cancelled"
        # Note: ServiceTicket has no explicit cancel_reason field; reason can be kept externally/logged.
        return ticket

    @is_tool(ToolType.WRITE)
    def modify_placed_ticket_dropoff(
        self,
        ticket_ref: str,
        line_one: str,
        line_two: str,
        municipality: str,
        province: str,
        nation: str,
        postal_code: str,
    ) -> ServiceTicket:
        """Modify the dropoff address of a placed delivery ticket.
        The agent must explain the update and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            ticket_ref: The ticket reference, such as 'TCKT-00001'.
            line_one: Primary address line.
            line_two: Secondary address line.
            municipality: City or municipality.
            province: State or province.
            nation: Country name.
            postal_code: Postal or ZIP code.

        Returns:
            ServiceTicket: The ticket details after the modification.

        Raises:
            ValueError: If the ticket is not 'placed' or not a delivery ticket.
        """
        ticket = self._get_ticket(ticket_ref)
        if not self._is_placed_ticket(ticket):
            raise ValueError("Non-placed ticket cannot be modified")
        if ticket.service_mode != "delivery":
            raise ValueError("Only delivery tickets have a dropoff address")

        ticket.dropoff = PatronLocation(
            line_one=line_one,
            line_two=line_two,
            municipality=municipality,
            province=province,
            nation=nation,
            postal_code=postal_code,
        )
        return ticket

    @is_tool(ToolType.WRITE)
    def modify_placed_ticket_table(
        self,
        ticket_ref: str,
        zone: str,
        table_no: str,
        seat_count: int,
    ) -> ServiceTicket:
        """Modify the table information for a placed dine-in ticket.
        The agent must explain the update and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            ticket_ref: The ticket reference, such as 'TCKT-00001'.
            zone: Dining area or zone identifier.
            table_no: Table number.
            seat_count: Number of seats at the table.

        Returns:
            ServiceTicket: The ticket details after the modification.

        Raises:
            ValueError: If the ticket is not 'placed' or not a dine-in ticket.
        """
        ticket = self._get_ticket(ticket_ref)
        if not self._is_placed_ticket(ticket):
            raise ValueError("Non-placed ticket cannot be modified")
        if ticket.service_mode != "dine_in":
            raise ValueError("Only dine-in tickets have table information")

        ticket.table_info = TableInfo(zone=zone, table_no=table_no, seat_count=seat_count)
        return ticket

    @is_tool(ToolType.WRITE)
    def modify_placed_ticket_items(
        self,
        ticket_ref: str,
        plate_refs: List[str],
        new_plate_refs: List[str],
        instrument_ref: str,
    ) -> ServiceTicket:
        """Modify plate selections in a placed ticket to new plate selections of the same dish.
        This function can only be applied to tickets in 'placed' state. The agent needs to explain the
        modification and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            ticket_ref: The ticket reference, such as 'TCKT-00001'.
            plate_refs: The current plate_refs to be modified; duplicates allowed.
            new_plate_refs: The new plate_refs to replace with; must match positional count and be same dish.
            instrument_ref: The payment instrument to charge/refund the price difference.

        Returns:
            ServiceTicket: The ticket details after the modification.

        Raises:
            ValueError: If the ticket is not 'placed'.
            ValueError: If the plate_refs do not exist in the ticket (considering duplicates).
            ValueError: If new plates do not exist, are not served today, or do not belong to the same dish as the originals.
            ValueError: If the number of items to be modified does not match.
            ValueError: If the payment instrument does not exist for the patron.
        """
        ticket = self._get_ticket(ticket_ref)
        if not self._is_placed_ticket(ticket):
            raise ValueError("Non-placed ticket cannot be modified")

        # Validate instrument exists for the patron
        self._get_instrument(ticket.guest_ref, instrument_ref)

        # Check counts of existing plates in ticket
        ticket_plate_refs = [entry.plate_ref for entry in ticket.line_entries]
        for pr in plate_refs:
            if plate_refs.count(pr) > ticket_plate_refs.count(pr):
                raise ValueError(f"Plate {pr} not found in the ticket the requested number of times")

        if len(plate_refs) != len(new_plate_refs):
            raise ValueError("The number of plates to be modified should match")

        # Compute price differences and validate new plates
        diff_total = 0.0
        # Keep track of which line entries have been modified for duplicates
        used_indices: Dict[str, int] = {}  # plate_ref -> how many times consumed
        modifications: List[Dict] = []  # store (index, new_plate, new_cost)

        for old_pr, new_pr in zip(plate_refs, new_plate_refs):
            # Find the line entry index to replace for old_pr, considering duplicates
            start_idx = 0
            if old_pr in used_indices:
                # skip already used occurrences
                occurrences = used_indices[old_pr]
            else:
                occurrences = 0

            count_seen = 0
            target_idx = None
            for idx, le in enumerate(ticket.line_entries):
                if le.plate_ref == old_pr:
                    if count_seen == occurrences:
                        target_idx = idx
                        break
                    count_seen += 1

            if target_idx is None:
                raise ValueError(f"Plate {old_pr} not found")

            used_indices[old_pr] = occurrences + 1
            old_entry = ticket.line_entries[target_idx]

            # Validate new plate exists and belongs to the same dish as the old one
            old_dish = self._find_dish_by_plate(old_pr)
            try:
                new_dish = self._find_dish_by_plate(new_pr)
            except ValueError:
                raise ValueError(f"New plate {new_pr} not found")
            if old_dish.dish_ref != new_dish.dish_ref:
                raise ValueError("New plate must be from the same dish as the original")

            new_plate = new_dish.selections[new_pr]
            if not new_plate.served_today:
                raise ValueError(f"New plate {new_pr} is not available today")

            old_cost = old_entry.cost
            new_cost = new_plate.cost
            diff_total += (new_cost - old_cost)

            modifications.append({"index": target_idx, "new_plate": new_plate})

        # Record the payment/refund for the difference
        diff_total = round(diff_total, 2)
        if diff_total != 0:
            ticket.charges.append(
                TicketCharge(
                    kind="payment" if diff_total > 0 else "refund",
                    total=abs(diff_total),
                    instrument_ref=instrument_ref,
                )
            )

        # Apply the modifications
        for mod in modifications:
            idx = mod["index"]
            plate: Plate = mod["new_plate"]
            ticket.line_entries[idx].plate_ref = plate.plate_ref
            ticket.line_entries[idx].cost = plate.cost
            # Keep dish_ref and label as-is; label could be kept consistent with the dish.

        # Mark ticket as placed (items modified). We retain 'placed' state but indicate via label
        # or by relying on ticket history externally. No explicit field exists to flag modifications.
        return ticket

    @is_tool(ToolType.WRITE)
    def modify_placed_ticket_instrument(
        self,
        ticket_ref: str,
        instrument_ref: str,
    ) -> ServiceTicket:
        """Modify the payment instrument on a placed ticket. The agent needs to explain the change
        and ask for explicit user confirmation (yes/no) to proceed.

        Constraints:
        - Only tickets in 'placed' state are eligible.
        - There must be exactly one 'payment' charge on the ticket.
        - The new instrument must be different from the current one.

        Args:
            ticket_ref: The ticket reference, such as 'TCKT-00001'.
            instrument_ref: The new instrument reference to use for the payment.

        Returns:
            ServiceTicket: The ticket details after the modification.

        Raises:
            ValueError: If the ticket is not 'placed'.
            ValueError: If there is not exactly one 'payment' charge.
            ValueError: If the new instrument is the same as the current one.
            ValueError: If the instrument does not exist for the patron.
        """
        ticket = self._get_ticket(ticket_ref)
        if not self._is_placed_ticket(ticket):
            raise ValueError("Non-placed ticket cannot be modified")

        # Validate new instrument exists
        self._get_instrument(ticket.guest_ref, instrument_ref)

        payment_charges = [ch for ch in ticket.charges if ch.kind == "payment"]
        if len(payment_charges) != 1:
            raise ValueError("There should be exactly one payment charge for a placed ticket")

        current_payment = payment_charges[0]
        if current_payment.instrument_ref == instrument_ref:
            raise ValueError("The new instrument should be different from the current one")

        amount = current_payment.total

        # Append a new payment with the new instrument, and a refund to the old instrument
        ticket.charges.extend(
            [
                TicketCharge(kind="payment", total=amount, instrument_ref=instrument_ref),
                TicketCharge(kind="refund", total=amount, instrument_ref=current_payment.instrument_ref),
            ]
        )
        return ticket

    @is_tool(ToolType.WRITE)
    def add_tip_to_ticket(
        self,
        ticket_ref: str,
        tip_total: float,
        instrument_ref: str,
    ) -> ServiceTicket:
        """Add a tip to a ticket. The agent must explain the tip addition and ask for explicit
        user confirmation (yes/no) to proceed.

        Notes:
        - Tip can be added as long as the ticket is not cancelled.
        - The tip is charged to the specified instrument.

        Args:
            ticket_ref: The ticket reference, such as 'TCKT-00001'.
            tip_total: The tip amount to add (must be non-negative).
            instrument_ref: The instrument reference to charge the tip.

        Returns:
            ServiceTicket: The ticket details after adding the tip.

        Raises:
            ValueError: If the ticket is cancelled.
            ValueError: If the tip_total is negative.
            ValueError: If the instrument does not exist for the patron.
        """
        ticket = self._get_ticket(ticket_ref)
        if ticket.state == "cancelled":
            raise ValueError("Cannot add tip to a cancelled ticket")
        if tip_total < 0:
            raise ValueError("Tip amount must be non-negative")

        # Validate instrument
        self._get_instrument(ticket.guest_ref, instrument_ref)

        ticket.charges.append(
            TicketCharge(kind="tip", total=round(float(tip_total), 2), instrument_ref=instrument_ref)
        )
        return ticket
