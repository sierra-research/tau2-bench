from __future__ import annotations

import json
from typing import List

from tau2.domains.ecommerce.data_model import (
    Account,
    CatalogueGroup,
    ECommerceDB,
    FundingSource,
    LedgerEntry,
    Offering,
    Sale,
    Shipment,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class ECommerceTools(ToolKitBase):
    """All the tools for the e-commerce domain, following RetailTools patterns."""

    db: ECommerceDB

    def __init__(self, db: ECommerceDB) -> None:
        super().__init__(db)

    # --------
    # Helpers
    # --------
    def _get_sale(self, sale_ref: str) -> Sale:
        """Get a sale from the database."""
        if sale_ref not in self.db.sales:
            raise ValueError("Sale not found")
        return self.db.sales[sale_ref]

    def _get_account(self, account_key: str) -> Account:
        """Get an account from the database."""
        if account_key not in self.db.accounts:
            raise ValueError("Account not found")
        return self.db.accounts[account_key]

    def _get_catalogue_group(self, group_ref: str) -> CatalogueGroup:
        """Get a catalogue group from the database."""
        if group_ref not in self.db.catalogue:
            raise ValueError("Catalogue group not found")
        return self.db.catalogue[group_ref]

    def _get_offering(self, group_ref: str, unit_sku: str) -> Offering:
        """Get an offering (SKU) in a catalogue group."""
        group = self._get_catalogue_group(group_ref)
        if unit_sku not in group.offerings:
            raise ValueError("Offering not found")
        return group.offerings[unit_sku]

    def _get_funding_source(self, account_key: str, instrument_id: str) -> FundingSource:
        """Get a funding source for an account."""
        account = self._get_account(account_key)
        if instrument_id not in account.funding_sources:
            raise ValueError("Funding source not found")
        return account.funding_sources[instrument_id]

    def _is_pending_sale(self, sale: Sale) -> bool:
        """Non-strict pending check for a sale."""
        return "pending" in sale.state.lower()

    # --------------
    # Generic tools
    # --------------
    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """
        Calculate the result of a mathematical expression.

        Args:
            expression: The mathematical expression to calculate, such as '2 + 2'. The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.

        Returns:
            The result of the mathematical expression.

        Raises:
            ValueError: If the expression is invalid.
        """
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent, with a summary of the user's issue.
        Only transfer if
         - the user explicitly asks for a human agent
         - given the policy and the available tools, you cannot solve the user's issue.

        Args:
            summary: A summary of the user's issue.

        Returns:
            A message indicating the user has been transferred to a human agent.
        """
        return "Transfer successful"

    # ------------
    # Read tools
    # ------------
    @is_tool(ToolType.READ)
    def find_account_key_by_email(self, email: str) -> str:
        """Find account key by email.

        Args:
            email: The contact email of the account, such as 'something@example.com'.

        Returns:
            str: The account_key if found.

        Raises:
            ValueError: If the account is not found.
        """
        for account_key, account in self.db.accounts.items():
            if account.contact_email.lower() == email.lower():
                return account_key
        raise ValueError("Account not found")

    @is_tool(ToolType.READ)
    def find_account_key_by_name_zip(self, given: str, family: str, postal_code: str) -> str:
        """Find account key by first name, last name, and postal code.
        Use this only if email search fails or the user cannot remember the email.

        Args:
            given: First name, such as 'John'.
            family: Last name, such as 'Doe'.
            postal_code: The postal/ZIP code, such as '12345'.

        Returns:
            str: The account_key if found.

        Raises:
            ValueError: If the account is not found.
        """
        for account_key, account in self.db.accounts.items():
            if (
                account.person.given.lower() == given.lower()
                and account.person.family.lower() == family.lower()
                and account.location.postal_code == postal_code
            ):
                return account_key
        raise ValueError("Account not found")

    @is_tool(ToolType.READ)
    def get_sale_details(self, sale_ref: str) -> Sale:
        """Get the status and details of a sale (order).

        Args:
            sale_ref: The sale reference, such as 'S0000001'.

        Returns:
            Sale: The sale details.

        Raises:
            ValueError: If the sale is not found.
        """
        return self._get_sale(sale_ref)

    @is_tool(ToolType.READ)
    def get_catalogue_group_details(self, group_ref: str) -> CatalogueGroup:
        """Get details of a catalogue group and its offerings.

        Args:
            group_ref: The group reference, such as 'G00001'.

        Returns:
            CatalogueGroup: The group details.

        Raises:
            ValueError: If the group is not found.
        """
        return self._get_catalogue_group(group_ref)

    @is_tool(ToolType.READ)
    def get_account_details(self, account_key: str) -> Account:
        """Get the details of an account.

        Args:
            account_key: The account key, such as 'acct_123'.

        Returns:
            Account: The account details.

        Raises:
            ValueError: If the account is not found.
        """
        return self._get_account(account_key)

    @is_tool(ToolType.READ)
    def list_all_catalogue_groups(self) -> str:
        """List the title and group_ref of all catalogue groups.

        Returns:
            str: A JSON string mapping group titles to their group_refs, sorted alphabetically by title.
        """
        group_map = {group.title: group.group_ref for group in self.db.catalogue.values()}
        return json.dumps(group_map, sort_keys=True)

    @is_tool(ToolType.READ)
    def get_db_statistics(self) -> str:
        """Get basic statistics of the e-commerce database.

        Returns:
            str: A JSON string of database statistics.
        """
        stats = self.db.get_statistics()
        return json.dumps(stats, sort_keys=True)

    # -------------
    # Write tools
    # -------------
    @is_tool(ToolType.WRITE)
    def modify_account_location(
        self,
        account_key: str,
        line1: str,
        line2: str,
        municipality: str,
        region: str,
        nation: str,
        postal_code: str,
    ) -> Account:
        """Modify the default address of an account. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            account_key: The account key, such as 'acct_123'.
            line1: Primary address line.
            line2: Secondary address line (or '').
            municipality: City or municipality.
            region: State or region.
            nation: Country.
            postal_code: Postal or ZIP code.

        Returns:
            Account: The account after modification.

        Raises:
            ValueError: If the account is not found.
        """
        account = self._get_account(account_key)
        account.location.line1 = line1
        account.location.line2 = line2
        account.location.municipality = municipality
        account.location.region = region
        account.location.nation = nation
        account.location.postal_code = postal_code
        return account

    @is_tool(ToolType.WRITE)
    def modify_pending_sale_delivery(
        self,
        sale_ref: str,
        line1: str,
        line2: str,
        municipality: str,
        region: str,
        nation: str,
        postal_code: str,
    ) -> Sale:
        """Modify the delivery address of a pending sale. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            sale_ref: The sale reference, such as 'S0000001'.
            line1: Primary address line.
            line2: Secondary address line (or '').
            municipality: City or municipality.
            region: State or region.
            nation: Country.
            postal_code: Postal or ZIP code.

        Returns:
            Sale: The sale after the modification.

        Raises:
            ValueError: If the sale is not pending.
        """
        sale = self._get_sale(sale_ref)
        if not self._is_pending_sale(sale):
            raise ValueError("Non-pending sale cannot be modified")

        sale.delivery.line1 = line1
        sale.delivery.line2 = line2
        sale.delivery.municipality = municipality
        sale.delivery.region = region
        sale.delivery.nation = nation
        sale.delivery.postal_code = postal_code
        return sale

    @is_tool(ToolType.WRITE)
    def modify_pending_sale_payment(
        self,
        sale_ref: str,
        instrument_id: str,
    ) -> Sale:
        """Modify the funding instrument of a pending sale. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            sale_ref: The sale reference, such as 'S0000001'.
            instrument_id: The funding instrument id (e.g., 'gift_card_0000000' or 'credit_card_0000000').

        Returns:
            Sale: The sale after the modification.

        Raises:
            ValueError: If the sale is not pending.
            ValueError: If the funding source does not exist.
            ValueError: If the ledger has more than one payment.
            ValueError: If the new instrument is the same as the current one.
        """
        sale = self._get_sale(sale_ref)
        if not self._is_pending_sale(sale):
            raise ValueError("Non-pending sale cannot be modified")

        # Validate funding source exists
        self._get_funding_source(sale.account_key, instrument_id)

        # Must have exactly one payment in the ledger
        payments = [e for e in sale.ledger if e.entry_kind == "payment"]
        if len(payments) != 1:
            raise ValueError("There should be exactly one payment for a pending sale")

        current_payment = payments[0]
        if current_payment.instrument_id == instrument_id:
            raise ValueError("The new funding instrument should be different from the current one")

        amount = current_payment.value

        # Append a new payment with the new instrument and a refund for the old one
        sale.ledger.extend(
            [
                LedgerEntry(entry_kind="payment", value=amount, instrument_id=instrument_id),
                LedgerEntry(entry_kind="refund", value=amount, instrument_id=current_payment.instrument_id),
            ]
        )

        return sale

    @is_tool(ToolType.WRITE)
    def cancel_pending_sale(self, sale_ref: str, reason: str) -> Sale:
        """Cancel a pending sale. If the sale is already processed or delivered, it cannot be cancelled.
        The agent needs to explain the cancellation detail and ask for explicit user confirmation (yes/no) to proceed.
        If the user confirms, the sale state will be changed to 'cancelled' and a refund will be recorded for each payment entry.
        If the original funding instrument origin is 'gift_card', the refund is credited immediately; otherwise refunds take 5-7 business days to process.

        Args:
            sale_ref: The sale reference, such as 'S0000001'.
            reason: The reason for cancellation, either 'no longer needed' or 'ordered by mistake'.

        Returns:
            Sale: The sale after the cancellation.

        Raises:
            ValueError: If the sale is not pending.
            ValueError: If the reason is invalid.
        """
        sale = self._get_sale(sale_ref)
        if sale.state != "pending":
            raise ValueError("Non-pending sale cannot be cancelled")

        if reason not in {"no longer needed", "ordered by mistake"}:
            raise ValueError("Invalid reason")

        # Issue refunds for all payment entries
        for entry in sale.ledger:
            if entry.entry_kind == "payment":
                sale.ledger.append(
                    LedgerEntry(entry_kind="refund", value=entry.value, instrument_id=entry.instrument_id)
                )

        sale.state = "cancelled"
        return sale

    @is_tool(ToolType.WRITE)
    def return_delivered_sale_lines(
        self,
        sale_ref: str,
        unit_skus: List[str],
        instrument_id: str,
    ) -> Sale:
        """Request a return of specific items in a delivered sale.
        The sale state will be changed to 'return requested'.
        The agent needs to explain the return detail and ask for explicit user confirmation (yes/no) to proceed.
        The user will receive follow-up email with return instructions.

        Args:
            sale_ref: The sale reference, such as 'S0000001'.
            unit_skus: The unit SKUs to be returned. Duplicates allowed to represent quantities.
            instrument_id: The funding instrument to receive the refund. Must be either the original payment instrument or a gift card.

        Returns:
            Sale: The sale after requesting the return.

        Raises:
            ValueError: If the sale is not delivered.
            ValueError: If the instrument is not the original payment instrument or a gift card.
            ValueError: If the items to be returned do not exist in sufficient quantity.
        """
        sale = self._get_sale(sale_ref)
        if sale.state != "delivered":
            raise ValueError("Non-delivered sale cannot be returned")

        # Validate instrument: original payment instrument or a gift card
        self._get_funding_source(sale.account_key, instrument_id)
        first_payment = next((e for e in sale.ledger if e.entry_kind == "payment"), None)
        if first_payment is None:
            raise ValueError("Original payment not found")
        # Check instrument either original or gift card
        fs = self._get_funding_source(sale.account_key, instrument_id)
        if fs.origin != "gift_card" and instrument_id != first_payment.instrument_id:
            raise ValueError("Funding instrument should be the original payment instrument or a gift card")

        # Validate items exist (respecting duplicates)
        all_skus = [line.unit_sku for line in sale.lines]
        for sku in unit_skus:
            if unit_skus.count(sku) > all_skus.count(sku):
                raise ValueError("Some item not found")

        sale.state = "return requested"
        return sale

    @is_tool(ToolType.WRITE)
    def exchange_delivered_sale_lines(
        self,
        sale_ref: str,
        unit_skus_old: List[str],
        unit_skus_new: List[str],
        instrument_id: str,
    ) -> Sale:
        """Request an exchange of items in a delivered sale to new offerings within the same catalogue group.
        For a delivered sale, return or exchange can be only done once by the agent.
        The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            sale_ref: The sale reference, such as 'S0000001'.
            unit_skus_old: The unit SKUs to be exchanged. Duplicates allowed to represent quantities.
            unit_skus_new: The new unit SKUs to exchange for, aligned by position with unit_skus_old.
                           Each new SKU must belong to the same catalogue group as the old line and be in stock.
            instrument_id: The funding instrument to pay or receive refund for the price difference.

        Returns:
            Sale: The sale details after requesting the exchange.

        Raises:
            ValueError: If the sale is not delivered.
            ValueError: If the items to be exchanged do not exist in sufficient quantity.
            ValueError: If the number of items does not match.
            ValueError: If the new items are not in the same catalogue group or are out of stock.
            ValueError: If the funding instrument does not exist.
        """
        sale = self._get_sale(sale_ref)
        if sale.state != "delivered":
            raise ValueError("Non-delivered sale cannot be exchanged")

        if len(unit_skus_old) != len(unit_skus_new):
            raise ValueError("The number of items to be exchanged should match")

        # Validate old SKUs exist with sufficient quantity
        all_skus_in_sale = [line.unit_sku for line in sale.lines]
        for sku in unit_skus_old:
            if unit_skus_old.count(sku) > all_skus_in_sale.count(sku):
                raise ValueError(f"Number of {sku} not found")

        # Validate funding instrument exists
        self._get_funding_source(sale.account_key, instrument_id)

        # Compute price difference and validate new SKUs match group and are in stock
        diff_price = 0.0
        for old_sku, new_sku in zip(unit_skus_old, unit_skus_new):
            # Find the corresponding sale line for old_sku
            line = next((ln for ln in sale.lines if ln.unit_sku == old_sku), None)
            if line is None:
                raise ValueError(f"Item {old_sku} not found")

            # New SKU must be in the same catalogue group as the line
            group_ref = line.catalog_ref
            new_offering = self._get_offering(group_ref, new_sku)
            if not new_offering.in_stock:
                raise ValueError(f"New item {new_sku} not found or not in stock")

            old_price = line.unit_price
            new_price = new_offering.unit_price
            diff_price += new_price - old_price

        diff_price = round(diff_price, 2)

        # For delivered exchanges, we only record the request and price difference handling will be processed offline.
        # We do not mutate lines or ledger here; this mirrors the "exchange requested" flow in retail tools.
        sale.state = "exchange requested"
        return sale
