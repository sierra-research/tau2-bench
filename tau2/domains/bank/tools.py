from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional, Tuple, Literal

from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.domains.bank.data_model import (
    Account,
    AccountStatus,
    AccountType,
    BankDB,
    Beneficiary,
    BeneficiaryStatus,
    BeneficiaryType,
    BeneficiaryVerification,
    Card,
    CardStatus,
    Client,
    Loan,
    LoanRepaymentHistoryEntry,
    Transaction,
    TransactionDirection,
    TransactionMethod,
    TransactionStatus,
    TransactionType,
)


class BankTools(ToolKitBase):
    """
    Frequently-used tools for the bank domain.

    Notes:
    - For WRITE tools that change state (e.g., transfers, freezing cards), the agent
      should explain the details and ask for explicit user confirmation (yes/no) before calling.
    - All monetary values are treated as floats in account currency. Available balance
      is checked where applicable and updated alongside current balance.
    """

    db: BankDB

    def __init__(self, db: BankDB) -> None:
        super().__init__(db)

    # -------------------------
    # Internal helpers
    # -------------------------

    def _get_client(self, client_id: str) -> Client:
        if client_id not in self.db.clients:
            raise ValueError("Client not found")
        return self.db.clients[client_id]

    def _get_account_by_id(self, account_id: str) -> Tuple[str, Account]:
        for cid, client in self.db.clients.items():
            if account_id in client.accounts:
                return cid, client.accounts[account_id]
        raise ValueError("Account not found")

    def _get_account(self, client_id: str, account_id: str) -> Account:
        client = self._get_client(client_id)
        if account_id not in client.accounts:
            raise ValueError("Account not found for this client")
        return client.accounts[account_id]

    def _get_card_by_id(self, card_id: str) -> Tuple[str, Card]:
        for cid, client in self.db.clients.items():
            if card_id in client.cards:
                return cid, client.cards[card_id]
        raise ValueError("Card not found")

    def _get_loan(self, loan_id: str) -> Loan:
        if loan_id not in self.db.loans:
            raise ValueError("Loan not found")
        return self.db.loans[loan_id]

    def _get_beneficiary(self, beneficiary_id: str) -> Beneficiary:
        if beneficiary_id not in self.db.beneficiaries:
            raise ValueError("Beneficiary not found")
        return self.db.beneficiaries[beneficiary_id]

    def _now(self) -> str:
        # Assuming ISO-like timestamps in DB; using ISO format for consistency
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def _generate_transaction_id(self) -> str:
        # Simple generator; assumes single writer; acceptable for demo
        return f"tx_{len(self.db.transactions) + 1:07d}"

    def _assert_account_active(self, account: Account) -> None:
        if account.status != "active":
            raise ValueError("Account is not active")

    def _assert_card_active_or_blocked(self, card: Card) -> None:
        if card.status not in ("active", "blocked"):
            raise ValueError("Card not in a manageable state")

    def _assert_beneficiary_owned_by_client(self, client_id: str, beneficiary_id: str) -> None:
        client = self._get_client(client_id)
        if beneficiary_id not in client.beneficiary_ids:
            raise ValueError("Beneficiary not owned by this client")

    # -------------------------
    # Generic utilities
    # -------------------------

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

    # -------------------------
    # READ tools
    # -------------------------

    @is_tool(ToolType.READ)
    def find_client_id_by_email(self, email: str) -> str:
        """
        Find client id by email.

        Args:
            email: The client's email, e.g., 'someone@example.com'.

        Returns:
            str: The client id if found.

        Raises:
            ValueError: If the client is not found.
        """
        for cid, client in self.db.clients.items():
            if client.contact.email.lower() == email.lower():
                return cid
        raise ValueError("Client not found")

    @is_tool(ToolType.READ)
    def get_client_details(self, client_id: str) -> Client:
        """
        Get the details of a client, including accounts, cards, loans, and beneficiaries.

        Args:
            client_id: The client id, such as 'client_0001'.

        Returns:
            Client: The client details.

        Raises:
            ValueError: If the client is not found.
        """
        return self._get_client(client_id)

    @is_tool(ToolType.READ)
    def get_account_details(self, account_id: str) -> Account:
        """
        Get the details of an account by account id.

        Args:
            account_id: The account id, such as 'acc_0001'.

        Returns:
            Account: The account details.

        Raises:
            ValueError: If the account is not found.
        """
        _, account = self._get_account_by_id(account_id)
        return account

    @is_tool(ToolType.READ)
    def get_card_details(self, card_id: str) -> Card:
        """
        Get the details of a card by card id.

        Args:
            card_id: The card id, such as 'card_0001'.

        Returns:
            Card: The card details.

        Raises:
            ValueError: If the card is not found.
        """
        _, card = self._get_card_by_id(card_id)
        return card

    @is_tool(ToolType.READ)
    def get_loan_details(self, loan_id: str) -> Loan:
        """
        Get the details of a loan by loan id.

        Args:
            loan_id: The loan id, such as 'loan_0001'.

        Returns:
            Loan: The loan details.

        Raises:
            ValueError: If the loan is not found.
        """
        return self._get_loan(loan_id)

    @is_tool(ToolType.READ)
    def get_beneficiary_details(self, beneficiary_id: str) -> Beneficiary:
        """
        Get the details of a beneficiary by beneficiary id.

        Args:
            beneficiary_id: The beneficiary id, such as 'bnf_0001'.

        Returns:
            Beneficiary: The beneficiary details.

        Raises:
            ValueError: If the beneficiary is not found.
        """
        return self._get_beneficiary(beneficiary_id)

    @is_tool(ToolType.READ)
    def list_client_accounts(self, client_id: str) -> str:
        """
        List all accounts for a client as a JSON string mapping account_id to a summary.

        Args:
            client_id: The client id.

        Returns:
            str: A JSON string mapping account_id to a summary {type, status, masked}.
        """
        client = self._get_client(client_id)
        summary = {
            acc_id: {
                "type": acc.type,
                "status": acc.status,
                "masked": acc.account_number_masked,
                "currency": acc.currency,
            }
            for acc_id, acc in client.accounts.items()
        }
        return json.dumps(summary, sort_keys=True)

    @is_tool(ToolType.READ)
    def list_client_beneficiaries(self, client_id: str) -> str:
        """
        List all beneficiaries for a client as a JSON mapping beneficiary_id to a display name.

        Args:
            client_id: The client id.

        Returns:
            str: A JSON string mapping beneficiary_id to display name or business name.
        """
        client = self._get_client(client_id)
        result = {}
        for bid in client.beneficiary_ids:
            if bid in self.db.beneficiaries:
                b = self.db.beneficiaries[bid]
                display = (
                    b.name.display_name
                    or b.name.business_name
                    or " ".join(filter(None, [b.name.first_name, b.name.last_name]))  # type: ignore
                    or "Unnamed Beneficiary"
                )
                result[bid] = display
        return json.dumps(result, sort_keys=True)

    @is_tool(ToolType.READ)
    def get_recent_transactions(self, account_id: str, limit: int = 10) -> List[Transaction]:
        """
        Get recent transactions for an account.

        Args:
            account_id: The account id.
            limit: Max number of transactions to return, default 10.

        Returns:
            List[Transaction]: The most recent transactions sorted by timestamp desc.

        Raises:
            ValueError: If the account is not found.
        """
        _, _ = self._get_account_by_id(account_id)  # validate existence
        txs = [
            tx
            for tx in self.db.transactions.values()
            if tx.account_id == account_id and tx.status in ("posted", "pending")
        ]
        txs.sort(key=lambda t: t.timestamp, reverse=True)
        return txs[: max(0, limit)]

    @is_tool(ToolType.READ)
    def search_transactions(
        self,
        client_id: str,
        account_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        transaction_type: Optional[str] = None,
        status: Optional[str] = None,
        merchant_name_contains: Optional[str] = None,
    ) -> List[Transaction]:
        """
        Search transactions for a client account with optional filters.

        Args:
            client_id: The client id.
            account_id: The account id owned by the client.
            start_date: ISO timestamp lower bound (inclusive).
            end_date: ISO timestamp upper bound (inclusive).
            min_amount: Minimum amount filter.
            max_amount: Maximum amount filter.
            transaction_type: Transaction type filter.
            status: Transaction status filter.
            merchant_name_contains: Case-insensitive substring to match merchant name.

        Returns:
            List[Transaction]: Matching transactions sorted by timestamp desc.

        Raises:
            ValueError: If the client or account is not found or not owned by client.
        """
        account = self._get_account(client_id, account_id)
        _ = account  # ensure ownership

        def in_range(ts: str) -> bool:
            if start_date and ts < start_date:
                return False
            if end_date and ts > end_date:
                return False
            return True

        def amount_ok(a: float) -> bool:
            if min_amount is not None and a < min_amount:
                return False
            if max_amount is not None and a > max_amount:
                return False
            return True

        m_sub = merchant_name_contains.lower() if merchant_name_contains else None

        results = []
        for tx in self.db.transactions.values():
            if tx.account_id != account_id:
                continue
            if not in_range(tx.timestamp):
                continue
            if not amount_ok(tx.amount):
                continue
            if transaction_type and tx.type != transaction_type:
                continue
            if status and tx.status != status:
                continue
            if m_sub:
                mname = tx.merchant.name.lower() if (tx.merchant and tx.merchant.name) else ""
                if mname.find(m_sub) == -1:
                    continue
            results.append(tx)

        results.sort(key=lambda t: t.timestamp, reverse=True)
        return results

    # -------------------------
    # WRITE tools
    # -------------------------

    @is_tool(ToolType.WRITE)
    def initiate_internal_transfer(
        self,
        client_id: str,
        from_account_id: str,
        to_account_id: str,
        amount: float,
        description: Optional[str] = None,
    ) -> List[Transaction]:
        """
        Initiate an internal transfer between two accounts owned by the same client.
        The agent needs to explain details and ask for explicit user confirmation (yes/no) before calling.

        Rules:
          - Both accounts must be active and owned by the same client.
          - Cross-currency internal transfers are not supported.
          - From-account type 'credit' is not allowed (no cash advance via this tool).
          - Sufficient available balance is required in the from-account.

        Args:
            client_id: The client id.
            from_account_id: The source account id.
            to_account_id: The destination account id.
            amount: Transfer amount (> 0).
            description: Optional memo/description.

        Returns:
            List[Transaction]: Two posted transactions (debit for source, credit for destination).

        Raises:
            ValueError: On invalid inputs or constraints violation.
        """
        if amount <= 0:
            raise ValueError("Amount must be greater than zero")

        client = self._get_client(client_id)
        if from_account_id not in client.accounts or to_account_id not in client.accounts:
            raise ValueError("Both accounts must belong to the same client")

        from_acc = client.accounts[from_account_id]
        to_acc = client.accounts[to_account_id]

        self._assert_account_active(from_acc)
        self._assert_account_active(to_acc)

        if from_acc.type == "credit":
            raise ValueError("Transfers from a credit account are not supported")

        if from_acc.currency != to_acc.currency:
            raise ValueError("Cross-currency internal transfers are not supported")

        if from_acc.balance.available < amount:
            raise ValueError("Insufficient available balance")

        ts_now = self._now()
        currency = from_acc.currency

        # Update balances
        from_acc.balance.available = round(from_acc.balance.available - amount, 2)
        from_acc.balance.current = round(from_acc.balance.current - amount, 2)
        to_acc.balance.available = round(to_acc.balance.available + amount, 2)
        to_acc.balance.current = round(to_acc.balance.current + amount, 2)

        # Create linked transactions
        tx_id_out = self._generate_transaction_id()
        tx_out = Transaction(
            transaction_id=tx_id_out,
            client_id=client_id,
            account_id=from_account_id,
            timestamp=ts_now,
            type="transfer",
            direction="debit",
            amount=round(amount, 2),
            currency=currency,
            description=description or f"Internal transfer to {to_account_id}",
            method="Internal",
            status="posted",
            related_transaction_id=None,
            balance_after=from_acc.balance.current,
        )

        tx_id_in = self._generate_transaction_id()
        tx_in = Transaction(
            transaction_id=tx_id_in,
            client_id=client_id,
            account_id=to_account_id,
            timestamp=ts_now,
            type="transfer",
            direction="credit",
            amount=round(amount, 2),
            currency=currency,
            description=description or f"Internal transfer from {from_account_id}",
            method="Internal",
            status="posted",
            related_transaction_id=tx_id_out,
            balance_after=to_acc.balance.current,
        )

        # Link reverse relation
        tx_out.related_transaction_id = tx_id_in

        # Save
        self.db.transactions[tx_id_out] = tx_out
        self.db.transactions[tx_id_in] = tx_in

        return [tx_out, tx_in]

    @is_tool(ToolType.WRITE)
    def add_beneficiary(
        self,
        client_id: str,
        beneficiary_id: str,
        beneficiary_type: str,
        display_name: Optional[str],
        first_name: Optional[str],
        last_name: Optional[str],
        business_name: Optional[str],
        bank_name: str,
        account_number_masked: str,
        routing_number: Optional[str],
        iban: Optional[str],
        swift_bic: Optional[str],
        address1: str,
        address2: str,
        city: str,
        state: str,
        zip: str,
        country: str,
        allowed_from_account_ids: List[str],
        per_transfer_limit: float,
        daily_limit: float,
        verification_method: str = "document",
    ) -> Beneficiary:
        """
        Add a beneficiary for a client. The agent needs to explain and ask for explicit user confirmation (yes/no).

        Args: Various beneficiary details (names, bank details, address, limits).
        Returns:
            Beneficiary: The newly created beneficiary.

        Raises:
            ValueError: If client not found or invalid inputs.
        """
        client = self._get_client(client_id)

        # Validate allowed accounts belong to client
        for acc_id in allowed_from_account_ids:
            if acc_id not in client.accounts:
                raise ValueError(f"Allowed account {acc_id} does not belong to client")

        # Construct beneficiary model
        from tau2.domains.bank.data_model import (
            BeneficiaryName,
            BankDetails,
            BankAddress,
            TransferLimits,
            BeneficiaryVerification,
        )

        name = BeneficiaryName(
            display_name=display_name,
            first_name=first_name,
            last_name=last_name,
            business_name=business_name,
        )
        bank_details = BankDetails(
            bank_name=bank_name,
            account_number_masked=account_number_masked,
            routing_number=routing_number,
            iban=iban,
            swift_bic=swift_bic,
        )
        address = BankAddress(
            address1=address1,
            address2=address2,
            city=city,
            state=state,
            zip=zip,
            country=country,
        )
        limits = TransferLimits(
            per_transfer_limit=per_transfer_limit,
            daily_limit=daily_limit,
        )
        verification = BeneficiaryVerification(
            status="pending",
            method=verification_method,  # type: ignore
            verified_at=None,
        )

        beneficiary = Beneficiary(
            beneficiary_id=beneficiary_id,
            client_id=client_id,
            name=name,
            type=beneficiary_type,
            bank_details=bank_details,
            address=address,
            allowed_from_account_ids=allowed_from_account_ids,
            transfer_limits=limits,
            verification=verification,
            status="active",
            created_at=self._now(),
            notes=None,
        )

        self.db.beneficiaries[beneficiary_id] = beneficiary
        if beneficiary_id not in client.beneficiary_ids:
            client.beneficiary_ids.append(beneficiary_id)

        return beneficiary

    @is_tool(ToolType.WRITE)
    def verify_beneficiary(
        self,
        client_id: str,
        beneficiary_id: str,
        method: str = "document",
    ) -> Beneficiary:
        """
        Mark a beneficiary as verified. The agent should confirm with the user before proceeding.

        Args:
            client_id: The client id.
            beneficiary_id: The beneficiary id.
            method: Verification method label.

        Returns:
            Beneficiary: The beneficiary after verification update.

        Raises:
            ValueError: If not owned by client or not found.
        """
        self._assert_beneficiary_owned_by_client(client_id, beneficiary_id)
        b = self._get_beneficiary(beneficiary_id)
        b.verification.status = "verified"
        b.verification.method = method  # type: ignore
        b.verification.verified_at = self._now()
        return b

    @is_tool(ToolType.WRITE)
    def initiate_transfer_to_beneficiary(
        self,
        client_id: str,
        from_account_id: str,
        beneficiary_id: str,
        amount: float,
        method: str = "ACH",
        description: Optional[str] = None,
    ) -> Transaction:
        """
        Initiate a transfer to a verified beneficiary. The agent needs to explain and ask for explicit user confirmation (yes/no).

        Rules:
          - Beneficiary must be owned by the client, active, and verified.
          - From account must be active and included in beneficiary.allowed_from_account_ids.
          - Sufficient available balance required.
          - Amount must not exceed per_transfer_limit (daily limit not enforced in this tool).

        Args:
            client_id: The client id.
            from_account_id: Source account id.
            beneficiary_id: Beneficiary id.
            amount: Transfer amount (> 0).
            method: One of "ACH", "Wire" (default ACH).
            description: Optional memo.

        Returns:
            Transaction: A posted debit transaction from the source account.

        Raises:
            ValueError: On invalid inputs or constraints violation.
        """
        if amount <= 0:
            raise ValueError("Amount must be greater than zero")

        client = self._get_client(client_id)
        if from_account_id not in client.accounts:
            raise ValueError("Source account must belong to the client")

        acc = client.accounts[from_account_id]
        self._assert_account_active(acc)

        self._assert_beneficiary_owned_by_client(client_id, beneficiary_id)
        b = self._get_beneficiary(beneficiary_id)

        if b.status != "active":
            raise ValueError("Beneficiary is not active")
        if b.verification.status != "verified":
            raise ValueError("Beneficiary must be verified before transfers")
        if from_account_id not in b.allowed_from_account_ids:
            raise ValueError("This account is not authorized for transfers to the beneficiary")

        if amount > b.transfer_limits.per_transfer_limit:
            raise ValueError("Amount exceeds per-transfer limit for beneficiary")

        if acc.balance.available < amount:
            raise ValueError("Insufficient available balance")

        # Update balances
        acc.balance.available = round(acc.balance.available - amount, 2)
        acc.balance.current = round(acc.balance.current - amount, 2)

        # Create transaction (debit from source)
        tx_id = self._generate_transaction_id()
        tx = Transaction(
            transaction_id=tx_id,
            client_id=client_id,
            account_id=from_account_id,
            timestamp=self._now(),
            type="transfer",
            direction="debit",
            amount=round(amount, 2),
            currency=acc.currency,
            description=description
            or f"Transfer to beneficiary {b.name.display_name or b.beneficiary_id}",
            method=method,
            status="posted",
            related_transaction_id=None,
            balance_after=acc.balance.current,
        )
        self.db.transactions[tx_id] = tx
        return tx

    @is_tool(ToolType.WRITE)
    def freeze_card(self, card_id: str, reason: str) -> Card:
        """
        Block (freeze) a card. The agent needs to explain and ask for explicit user confirmation (yes/no).

        Args:
            card_id: The card id.
            reason: Reason for freezing/blocking the card.

        Returns:
            Card: The card details after update.

        Raises:
            ValueError: If card not found or not manageable.
        """
        _, card = self._get_card_by_id(card_id)
        self._assert_card_active_or_blocked(card)
        if card.status == "blocked":
            return card  # already blocked
        card.status = "blocked"
        # Reason could be stored in notes if model had it; here it's just acknowledged
        return card

    @is_tool(ToolType.WRITE)
    def unfreeze_card(self, card_id: str) -> Card:
        """
        Unblock (unfreeze) a card back to active. The agent needs to confirm with the user before proceeding.

        Args:
            card_id: The card id.

        Returns:
            Card: The card details after update.

        Raises:
            ValueError: If card not found or not manageable.
        """
        _, card = self._get_card_by_id(card_id)
        self._assert_card_active_or_blocked(card)
        if card.status == "active":
            return card  # already active
        card.status = "active"
        return card

    @is_tool(ToolType.WRITE)
    def freeze_account(self, account_id: str, reason: str) -> Account:
        """
        Freeze an account (set status to 'frozen'). The agent should get explicit user confirmation (yes/no).

        Args:
            account_id: The account id.
            reason: Reason for freezing.

        Returns:
            Account: The account after update.

        Raises:
            ValueError: If account not found.
        """
        _, acc = self._get_account_by_id(account_id)
        if acc.status == "closed":
            raise ValueError("Account is closed and cannot be frozen")
        acc.status = "frozen"
        return acc

    @is_tool(ToolType.WRITE)
    def unfreeze_account(self, account_id: str) -> Account:
        """
        Unfreeze an account (set status to 'active'). The agent should get explicit user confirmation (yes/no).

        Args:
            account_id: The account id.

        Returns:
            Account: The account after update.

        Raises:
            ValueError: If account not found.
        """
        _, acc = self._get_account_by_id(account_id)
        if acc.status == "closed":
            raise ValueError("Account is closed and cannot be unfrozen")
        acc.status = "active"
        return acc

    @is_tool(ToolType.WRITE)
    def make_loan_payment(
        self,
        client_id: str,
        loan_id: str,
        from_account_id: str,
        amount: float,
        method: str = "Internal",
        description: Optional[str] = None,
    ) -> Transaction:
        """
        Make a loan payment from a client-owned account. The agent needs to explain and ask for explicit user confirmation (yes/no).

        Simplified rules:
          - Loan must belong to the client.
          - From account must belong to the client, be active, and have sufficient available balance.
          - Payment posts immediately; allocation is simplified (principal only).

        Args:
            client_id: The client id.
            loan_id: The loan id.
            from_account_id: Source account id.
            amount: Payment amount (> 0).
            method: Transaction method ("Internal", "ACH", "Wire", etc.).
            description: Optional memo.

        Returns:
            Transaction: The posted debit transaction from the source account.

        Raises:
            ValueError: On invalid inputs or constraints violation.
        """
        if amount <= 0:
            raise ValueError("Amount must be greater than zero")

        client = self._get_client(client_id)
        if from_account_id not in client.accounts:
            raise ValueError("Source account must belong to the client")

        loan = self._get_loan(loan_id)
        if loan.client_id != client_id:
            raise ValueError("Loan does not belong to the client")

        acc = client.accounts[from_account_id]
        self._assert_account_active(acc)

        if acc.balance.available < amount:
            raise ValueError("Insufficient available balance")

        # Update balances
        acc.balance.available = round(acc.balance.available - amount, 2)
        acc.balance.current = round(acc.balance.current - amount, 2)

        # Record transaction
        tx_id = self._generate_transaction_id()
        tx = Transaction(
            transaction_id=tx_id,
            client_id=client_id,
            account_id=from_account_id,
            timestamp=self._now(),
            type="payment",
            direction="debit",
            amount=round(amount, 2),
            currency=acc.currency,
            description=description or f"Loan payment {loan_id}",
            method=method,
            status="posted",
            related_transaction_id=None,
            balance_after=acc.balance.current,
        )
        self.db.transactions[tx_id] = tx

        # Update loan repayment history (simplified allocation)
        loan.repayment_history.append(
            LoanRepaymentHistoryEntry(
                transaction_id=tx_id,
                posted_at=tx.timestamp,
                amount=tx.amount,
                currency=tx.currency,
                method=tx.method,
                principal_component=tx.amount,
                interest_component=0.0,
                fees_component=0.0,
            )
        )

        return tx
