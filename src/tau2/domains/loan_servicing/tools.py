"""Toolkit for the loan servicing domain."""

import re
import unicodedata
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import List

from tau2.domains.loan_servicing.data_model import (
    Autopay,
    Borrower,
    DocumentRequest,
    DocumentType,
    HardshipEnrollment,
    HardshipPlan,
    LateFee,
    Loan,
    LoanServicingDB,
    Payment,
    PaymentAllocation,
    PayoffQuote,
    Transfer,
    TransferReason,
)
from tau2.domains.loan_servicing.utils import LOAN_SERVICING_TODAY
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

CLOSED_STATUSES = ("paid_off", "charged_off")
PLAN_MONTHS = {"deferral_1": 1, "deferral_2": 2, "reduced_payment_3": 3}


def _money(value: float) -> float:
    """Round to cents, half up."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _fold(text: str) -> str:
    """Lowercase and strip accents, for lookups."""
    decomposed = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    return date(d.year + m // 12, m % 12 + 1, d.day)


class LoanServicingTools(ToolKitBase):
    """All the tools for the loan servicing domain."""

    db: LoanServicingDB

    def __init__(self, db: LoanServicingDB) -> None:
        super().__init__(db)

    # ---------- helpers ----------

    def _today(self) -> date:
        return date.fromisoformat(LOAN_SERVICING_TODAY)

    def _parse_date(self, value: str, name: str) -> date:
        try:
            return date.fromisoformat(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid {name} '{value}': use the format YYYY-MM-DD")

    def _get_borrower(self, borrower_id: str) -> Borrower:
        if borrower_id not in self.db.borrowers:
            raise ValueError(f"Borrower {borrower_id} not found")
        return self.db.borrowers[borrower_id]

    def _get_loan(self, loan_id: str) -> Loan:
        if loan_id not in self.db.loans:
            raise ValueError(f"Loan {loan_id} not found")
        return self.db.loans[loan_id]

    def _loan_method_ids(self, loan: Loan) -> set[str]:
        return {
            account.method_id
            for borrower_id in loan.borrower_ids
            for account in self.db.borrowers[borrower_id].bank_accounts
        }

    def _next_id(self, prefix: str, existing: List[str], start: int) -> str:
        numbers = [int(x.split("-")[1]) for x in existing if x.startswith(prefix)]
        return f"{prefix}-{max(numbers) + 1 if numbers else start}"

    def _open_fees_total(self, loan: Loan) -> float:
        return _money(sum(f.amount for f in loan.fees if f.status == "open"))

    def _payoff(self, loan: Loan, payoff_date: date) -> PayoffQuote:
        days = (payoff_date - self._today()).days
        interest = _money(
            loan.accrued_interest
            + loan.principal_balance * loan.annual_rate / 365 * days
        )
        fees = self._open_fees_total(loan)
        return PayoffQuote(
            loan_id=loan.loan_id,
            payoff_date=payoff_date.isoformat(),
            principal=_money(loan.principal_balance),
            interest=interest,
            fees=fees,
            total=_money(loan.principal_balance + interest + fees),
        )

    # ---------- read tools ----------

    @is_tool(ToolType.READ)
    def find_borrower_by_email(self, email: str) -> str:
        """
        Find a borrower's id by their email address.

        Finding a borrower does not verify their identity.

        Args:
            email: The email address, such as 'sophie.gagnon10@example.com'.

        Returns:
            The borrower id.

        Raises:
            ValueError: If no borrower has this email.
        """
        target = email.strip().lower()
        for borrower in self.db.borrowers.values():
            if borrower.email == target:
                return borrower.borrower_id
        raise ValueError("Borrower not found")

    @is_tool(ToolType.READ)
    def find_borrower_by_phone(self, phone: str) -> str:
        """
        Find a borrower's id by their phone number.

        Finding a borrower does not verify their identity.

        Args:
            phone: The phone number, such as '514-555-0142'.

        Returns:
            The borrower id.

        Raises:
            ValueError: If no borrower has this phone number.
        """
        digits = re.sub(r"\D", "", phone)
        for borrower in self.db.borrowers.values():
            if re.sub(r"\D", "", borrower.phone) == digits[-10:]:
                return borrower.borrower_id
        raise ValueError("Borrower not found")

    @is_tool(ToolType.READ)
    def find_borrower_by_name_dob(
        self, first_name: str, last_name: str, date_of_birth: str
    ) -> str:
        """
        Find a borrower's id by their first name, last name, and date of birth.

        Finding a borrower does not verify their identity.

        Args:
            first_name: The borrower's first name, such as 'Sophie'.
            last_name: The borrower's last name, such as 'Gagnon'.
            date_of_birth: The date of birth in the format 'YYYY-MM-DD'.

        Returns:
            The borrower id.

        Raises:
            ValueError: If no borrower matches.
        """
        for borrower in self.db.borrowers.values():
            if (
                _fold(borrower.first_name) == _fold(first_name)
                and _fold(borrower.last_name) == _fold(last_name)
                and borrower.date_of_birth == date_of_birth.strip()
            ):
                return borrower.borrower_id
        raise ValueError("Borrower not found")

    @is_tool(ToolType.READ)
    def get_borrower_details(self, borrower_id: str) -> Borrower:
        """
        Get the details of a borrower, including their bank accounts and loan ids.

        Args:
            borrower_id: The borrower id, such as 'BF-10001'.

        Returns:
            The borrower details.

        Raises:
            ValueError: If the borrower is not found.
        """
        return self._get_borrower(borrower_id)

    @is_tool(ToolType.READ)
    def get_loan_details(self, loan_id: str) -> Loan:
        """
        Get the details of a loan.

        Includes its status, balances, fees, autopay settings, due date changes,
        and hardship history.

        Args:
            loan_id: The loan id, such as 'LN-20001'.

        Returns:
            The loan details.

        Raises:
            ValueError: If the loan is not found.
        """
        return self._get_loan(loan_id)

    @is_tool(ToolType.READ)
    def list_payments(self, loan_id: str, limit: int = 12) -> List[Payment]:
        """
        List the payments on a loan, most recent first.

        Args:
            loan_id: The loan id, such as 'LN-20001'.
            limit: The maximum number of payments to return. Default is 12.

        Returns:
            The payments, most recent first.

        Raises:
            ValueError: If the loan is not found or the limit is not positive.
        """
        self._get_loan(loan_id)
        if limit < 1:
            raise ValueError("Limit must be positive")
        payments = [p for p in self.db.payments.values() if p.loan_id == loan_id]
        payments.sort(key=lambda p: (p.date, p.payment_id), reverse=True)
        return payments[:limit]

    @is_tool(ToolType.READ)
    def calculate_payoff(self, loan_id: str, payoff_date: str) -> PayoffQuote:
        """
        Calculate the amount needed to pay off a loan on a given date.

        The payoff is the remaining principal, plus interest accrued up to that
        date, plus open late fees.

        Args:
            loan_id: The loan id, such as 'LN-20001'.
            payoff_date: The payoff date in the format 'YYYY-MM-DD'. Must not be
                before today.

        Returns:
            The payoff quote.

        Raises:
            ValueError: If the loan is not found, is paid off, or the date is
                invalid or in the past.
        """
        loan = self._get_loan(loan_id)
        if loan.status == "paid_off":
            raise ValueError(f"Loan {loan_id} is paid off")
        when = self._parse_date(payoff_date, "payoff date")
        if when < self._today():
            raise ValueError("Payoff date cannot be in the past")
        return self._payoff(loan, when)

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

    # ---------- write tools ----------

    @is_tool(ToolType.WRITE)
    def make_payment(
        self, loan_id: str, amount: float, method_id: str, payment_date: str
    ) -> Payment:
        """
        Make a payment on a loan from a bank account on file.

        A payment dated today is posted immediately; a later date is scheduled. A
        posted payment is applied to open late fees (oldest first, each fee only
        if fully covered), then accrued interest, then principal.

        Args:
            loan_id: The loan id, such as 'LN-20001'.
            amount: The amount in dollars, such as 250.00.
            method_id: The bank account id, such as 'BA-40001'. It must belong
                to a borrower on the loan.
            payment_date: The payment date in the format 'YYYY-MM-DD'. Must not
                be before today.

        Returns:
            The payment, including its payment id.

        Raises:
            ValueError: If the loan or method is not found, the amount is not
                positive, has more than two decimals, or exceeds the payoff
                amount, the date is invalid or in the past, or the loan is
                paid off.
        """
        loan = self._get_loan(loan_id)
        if loan.status == "paid_off":
            raise ValueError(f"Loan {loan_id} is paid off")
        if method_id not in self._loan_method_ids(loan):
            raise ValueError(f"Payment method {method_id} not found for this loan")
        amount = float(amount)
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if _money(amount) != amount:
            raise ValueError("Amount must have at most two decimals")
        when = self._parse_date(payment_date, "payment date")
        today = self._today()
        if when < today:
            raise ValueError("Payment date cannot be in the past")
        payoff = self._payoff(loan, when).total
        if amount > payoff:
            raise ValueError(
                f"Amount exceeds the payoff amount of {payoff:.2f} on {when}"
            )

        payment_id = self._next_id("PM", list(self.db.payments), 30001)
        allocation = None
        status = "scheduled"
        if when == today:
            status = "posted"
            remaining = amount
            fees_paid = 0.0
            for fee in sorted(loan.fees, key=lambda f: (f.assessed_date, f.fee_id)):
                if fee.status == "open" and remaining >= fee.amount:
                    fee.status = "paid"
                    fees_paid = _money(fees_paid + fee.amount)
                    remaining = _money(remaining - fee.amount)
            interest = _money(min(remaining, loan.accrued_interest))
            remaining = _money(remaining - interest)
            principal = _money(min(remaining, loan.principal_balance))
            loan.accrued_interest = _money(loan.accrued_interest - interest)
            loan.principal_balance = _money(loan.principal_balance - principal)
            loan.past_due_amount = _money(
                max(0.0, loan.past_due_amount - interest - principal)
            )
            if loan.past_due_amount == 0 and loan.status in (
                "past_due_30",
                "past_due_60",
            ):
                loan.status = "current"
            if (
                loan.principal_balance == 0
                and loan.accrued_interest == 0
                and self._open_fees_total(loan) == 0
            ):
                loan.status = "paid_off"
                loan.next_due_date = None
                loan.past_due_amount = 0.0
                loan.autopay = Autopay(enabled=False)
            allocation = PaymentAllocation(
                fees=fees_paid, interest=interest, principal=principal
            )
        payment = Payment(
            payment_id=payment_id,
            loan_id=loan_id,
            amount=amount,
            date=when.isoformat(),
            method_id=method_id,
            status=status,
            allocation=allocation,
        )
        self.db.payments[payment_id] = payment
        return payment

    @is_tool(ToolType.WRITE)
    def cancel_scheduled_payment(self, payment_id: str) -> Payment:
        """
        Cancel a scheduled payment.

        Args:
            payment_id: The payment id, such as 'PM-30001'.

        Returns:
            The cancelled payment.

        Raises:
            ValueError: If the payment is not found or is not scheduled.
        """
        if payment_id not in self.db.payments:
            raise ValueError(f"Payment {payment_id} not found")
        payment = self.db.payments[payment_id]
        if payment.status != "scheduled":
            raise ValueError(f"Payment {payment_id} is {payment.status}, not scheduled")
        payment.status = "cancelled"
        return payment

    @is_tool(ToolType.WRITE)
    def enable_autopay(self, loan_id: str, method_id: str, day: int) -> Autopay:
        """
        Enable autopay on a loan, or change its bank account or day.

        The new settings replace the current ones, so to change only one
        setting, pass the current value of the other.

        Args:
            loan_id: The loan id, such as 'LN-20001'.
            method_id: The bank account id, such as 'BA-40001'. It must belong
                to a borrower on the loan.
            day: The day of the month autopay is drafted, between 1 and 28.

        Returns:
            The new autopay settings.

        Raises:
            ValueError: If the loan or method is not found, the day is invalid,
                or the loan is closed.
        """
        loan = self._get_loan(loan_id)
        if loan.status in CLOSED_STATUSES:
            raise ValueError(f"Loan {loan_id} is {loan.status}")
        if method_id not in self._loan_method_ids(loan):
            raise ValueError(f"Payment method {method_id} not found for this loan")
        if not 1 <= day <= 28:
            raise ValueError("Day must be between 1 and 28")
        loan.autopay = Autopay(enabled=True, method_id=method_id, day=day)
        return loan.autopay

    @is_tool(ToolType.WRITE)
    def disable_autopay(self, loan_id: str) -> Autopay:
        """
        Disable autopay on a loan.

        Args:
            loan_id: The loan id, such as 'LN-20001'.

        Returns:
            The new autopay settings.

        Raises:
            ValueError: If the loan is not found or autopay is already off.
        """
        loan = self._get_loan(loan_id)
        if not loan.autopay.enabled:
            raise ValueError(f"Autopay is already disabled on loan {loan_id}")
        loan.autopay = Autopay(enabled=False)
        return loan.autopay

    @is_tool(ToolType.WRITE)
    def change_due_date(self, loan_id: str, new_day: int) -> Loan:
        """
        Change the due day of a loan.

        The next due date moves to the new day in the same month as the current
        next due date.

        Args:
            loan_id: The loan id, such as 'LN-20001'.
            new_day: The new due day of the month, between 1 and 28.

        Returns:
            The updated loan.

        Raises:
            ValueError: If the loan is not found or closed, the day is invalid
                or unchanged, or the new due date would not be after today.
        """
        loan = self._get_loan(loan_id)
        if loan.next_due_date is None:
            raise ValueError(f"Loan {loan_id} is {loan.status}")
        if not 1 <= new_day <= 28:
            raise ValueError("Day must be between 1 and 28")
        if new_day == loan.due_day:
            raise ValueError(f"The due day is already {new_day}")
        next_due = date.fromisoformat(loan.next_due_date).replace(day=new_day)
        if next_due <= self._today():
            raise ValueError("The new due date would not be after today")
        loan.due_day = new_day
        loan.next_due_date = next_due.isoformat()
        loan.due_date_changes.append(LOAN_SERVICING_TODAY)
        return loan

    @is_tool(ToolType.WRITE)
    def waive_late_fee(self, loan_id: str, fee_id: str) -> LateFee:
        """
        Waive an open late fee on a loan.

        Args:
            loan_id: The loan id, such as 'LN-20001'.
            fee_id: The fee id, such as 'FE-50001'.

        Returns:
            The waived fee.

        Raises:
            ValueError: If the loan or fee is not found, or the fee is not open.
        """
        loan = self._get_loan(loan_id)
        for fee in loan.fees:
            if fee.fee_id == fee_id:
                if fee.status != "open":
                    raise ValueError(f"Fee {fee_id} is {fee.status}, not open")
                fee.status = "waived"
                fee.waived_date = LOAN_SERVICING_TODAY
                return fee
        raise ValueError(f"Fee {fee_id} not found on loan {loan_id}")

    @is_tool(ToolType.WRITE)
    def enroll_hardship_plan(self, loan_id: str, plan: HardshipPlan) -> Loan:
        """
        Enroll a loan in a hardship plan starting today.

        The loan becomes in_hardship. deferral_1 and deferral_2 skip one or two
        monthly payments: the past-due amount is cleared and the next due date moves
        forward by one or two months. reduced_payment_3 halves the monthly payment
        for three months.

        Args:
            loan_id: The loan id, such as 'LN-20001'.
            plan: The plan: 'deferral_1', 'deferral_2', or 'reduced_payment_3'.

        Returns:
            The updated loan, including the new hardship enrollment id.

        Raises:
            ValueError: If the loan is not found, is closed, or is already in a
                hardship plan, or the plan is invalid.
        """
        loan = self._get_loan(loan_id)
        if plan not in PLAN_MONTHS:
            raise ValueError(f"Invalid plan {plan}")
        if loan.status in CLOSED_STATUSES or loan.next_due_date is None:
            raise ValueError(f"Loan {loan_id} is {loan.status}")
        if loan.status == "in_hardship":
            raise ValueError(f"Loan {loan_id} is already in a hardship plan")
        existing = [
            h.hardship_id for x in self.db.loans.values() for h in x.hardship_history
        ]
        start = self._today()
        end = _add_months(start, PLAN_MONTHS[plan]) - timedelta(days=1)
        loan.hardship_history.append(
            HardshipEnrollment(
                hardship_id=self._next_id("HP", existing, 70001),
                plan=plan,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
            )
        )
        loan.status = "in_hardship"
        if plan in ("deferral_1", "deferral_2"):
            loan.past_due_amount = 0.0
            loan.next_due_date = _add_months(
                date.fromisoformat(loan.next_due_date), PLAN_MONTHS[plan]
            ).isoformat()
        return loan

    @is_tool(ToolType.WRITE)
    def update_email(self, borrower_id: str, email: str) -> Borrower:
        """
        Update a borrower's email address.

        Args:
            borrower_id: The borrower id, such as 'BF-10001'.
            email: The new email address.

        Returns:
            The updated borrower.

        Raises:
            ValueError: If the borrower is not found or the email is invalid.
        """
        borrower = self._get_borrower(borrower_id)
        new_email = email.strip().lower()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", new_email):
            raise ValueError(f"Invalid email '{email}'")
        borrower.email = new_email
        return borrower

    @is_tool(ToolType.WRITE)
    def update_phone(self, borrower_id: str, phone: str) -> Borrower:
        """
        Update a borrower's phone number.

        Args:
            borrower_id: The borrower id, such as 'BF-10001'.
            phone: The new 10-digit phone number, such as '514-555-0142'.

        Returns:
            The updated borrower.

        Raises:
            ValueError: If the borrower is not found or the phone is invalid.
        """
        borrower = self._get_borrower(borrower_id)
        digits = re.sub(r"\D", "", phone)
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        if len(digits) != 10:
            raise ValueError(f"Invalid phone '{phone}': need 10 digits")
        borrower.phone = f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        return borrower

    @is_tool(ToolType.WRITE)
    def send_document(
        self, loan_id: str, borrower_id: str, doc_type: DocumentType
    ) -> DocumentRequest:
        """
        Send a document about a loan to the requesting borrower's email on file.

        A payoff letter is quoted for 10 days after today.

        Args:
            loan_id: The loan id, such as 'LN-20001'.
            borrower_id: The id of the borrower requesting the document. They
                must be on the loan.
            doc_type: The document: 'statement', 'payoff_letter', or
                'tax_summary'.

        Returns:
            The document request, including its request id.

        Raises:
            ValueError: If the loan or borrower is not found, the borrower is not
                on the loan, or the document type is invalid.
        """
        loan = self._get_loan(loan_id)
        borrower = self._get_borrower(borrower_id)
        if borrower_id not in loan.borrower_ids:
            raise ValueError(f"Borrower {borrower_id} is not on loan {loan_id}")
        if doc_type not in ("statement", "payoff_letter", "tax_summary"):
            raise ValueError(f"Invalid document type {doc_type}")
        request_id = self._next_id("DR", list(self.db.document_requests), 60001)
        request = DocumentRequest(
            request_id=request_id,
            loan_id=loan_id,
            borrower_id=borrower_id,
            doc_type=doc_type,
            sent_to=borrower.email,
        )
        self.db.document_requests[request_id] = request
        return request

    @is_tool(ToolType.WRITE)
    def transfer_to_human_agents(
        self, reason: TransferReason, summary: str
    ) -> Transfer:
        """
        Transfer the borrower to a human agent.

        Give the reason and a summary of the borrower's issue. Only transfer in
        the cases listed in the policy.

        Args:
            reason: The reason: 'dispute', 'legal', 'fraud', 'complaint',
                'failed_verification', 'customer_request', or 'other'.
            summary: A summary of the borrower's issue.

        Returns:
            The transfer, including its transfer id.

        Raises:
            ValueError: If the reason is invalid.
        """
        valid = (
            "dispute",
            "legal",
            "fraud",
            "complaint",
            "failed_verification",
            "customer_request",
            "other",
        )
        if reason not in valid:
            raise ValueError(f"Invalid reason {reason}")
        transfer_id = self._next_id("TR", list(self.db.transfers), 80001)
        transfer = Transfer(transfer_id=transfer_id, reason=reason)
        self.db.transfers[transfer_id] = transfer
        return transfer
