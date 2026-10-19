from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from tau2.domains.loan_servicing.utils import LOAN_SERVICING_DB_PATH
from tau2.environment.db import DB

Language = Literal["en", "fr"]
Product = Literal["personal", "auto"]
LoanStatus = Literal[
    "current",
    "past_due_30",
    "past_due_60",
    "in_hardship",
    "paid_off",
    "charged_off",
]
PaymentStatus = Literal["posted", "scheduled", "cancelled", "returned"]
FeeStatus = Literal["open", "waived", "paid"]
HardshipPlan = Literal["deferral_1", "deferral_2", "reduced_payment_3"]
DocumentType = Literal["statement", "payoff_letter", "tax_summary"]
TransferReason = Literal[
    "dispute",
    "legal",
    "fraud",
    "complaint",
    "failed_verification",
    "customer_request",
    "other",
]


class BankAccount(BaseModel):
    method_id: str = Field(description="Unique identifier of the bank account")
    bank_name: str = Field(description="Name of the bank")
    last4: str = Field(description="Last four digits of the account number")


class Borrower(BaseModel):
    borrower_id: str = Field(description="Unique identifier of the borrower")
    first_name: str = Field(description="Borrower's first name")
    last_name: str = Field(description="Borrower's last name")
    date_of_birth: str = Field(description="Date of birth in the format YYYY-MM-DD")
    postal_code: str = Field(description="Canadian postal code, such as 'H2X 1Y4'")
    email: str = Field(description="Email address on file")
    phone: str = Field(description="Phone number on file, such as '514-555-0142'")
    preferred_language: Language = Field(description="Preferred language")
    authorized_third_parties: List[str] = Field(
        description="Full names of people authorized to receive general information"
    )
    bank_accounts: List[BankAccount] = Field(description="Bank accounts on file")
    loan_ids: List[str] = Field(description="Loans the borrower is on")


class Autopay(BaseModel):
    enabled: bool = Field(description="Whether autopay is enabled")
    method_id: Optional[str] = Field(
        None, description="Bank account used for autopay, if enabled"
    )
    day: Optional[int] = Field(
        None, description="Day of the month autopay is drafted, if enabled"
    )


class LateFee(BaseModel):
    fee_id: str = Field(description="Unique identifier of the fee")
    type: Literal["late_fee"] = Field(description="Type of fee")
    amount: float = Field(description="Fee amount in dollars")
    assessed_date: str = Field(description="Date the fee was assessed")
    status: FeeStatus = Field(description="Status of the fee")
    waived_date: Optional[str] = Field(
        None, description="Date the fee was waived, if waived"
    )


class HardshipEnrollment(BaseModel):
    hardship_id: str = Field(description="Unique identifier of the enrollment")
    plan: HardshipPlan = Field(description="Hardship plan")
    start_date: str = Field(description="First day of the plan")
    end_date: str = Field(description="Last day of the plan")


class Loan(BaseModel):
    loan_id: str = Field(description="Unique identifier of the loan")
    borrower_ids: List[str] = Field(
        description="Borrowers on the loan, primary borrower first"
    )
    product: Product = Field(description="Loan product")
    origination_date: str = Field(description="Date the loan was funded")
    original_principal: float = Field(description="Amount borrowed in dollars")
    annual_rate: float = Field(
        description="Annual interest rate as a decimal, such as 0.0899"
    )
    term_months: int = Field(description="Term in months")
    monthly_payment: float = Field(description="Scheduled monthly payment")
    principal_balance: float = Field(description="Remaining principal")
    accrued_interest: float = Field(description="Interest accrued as of today")
    due_day: int = Field(description="Day of the month payments are due")
    next_due_date: Optional[str] = Field(
        None, description="Next payment due date, if the loan is not closed"
    )
    past_due_amount: float = Field(description="Amount of missed payments")
    status: LoanStatus = Field(description="Status of the loan")
    autopay: Autopay = Field(description="Autopay settings")
    fees: List[LateFee] = Field(description="Late fees on the loan")
    due_date_changes: List[str] = Field(
        description="Dates on which the due day was changed"
    )
    hardship_history: List[HardshipEnrollment] = Field(
        description="Hardship plans on the loan"
    )


class PaymentAllocation(BaseModel):
    fees: float = Field(description="Amount applied to late fees")
    interest: float = Field(description="Amount applied to interest")
    principal: float = Field(description="Amount applied to principal")


class Payment(BaseModel):
    payment_id: str = Field(description="Unique identifier of the payment")
    loan_id: str = Field(description="Loan the payment is for")
    amount: float = Field(description="Payment amount in dollars")
    date: str = Field(description="Payment date")
    method_id: str = Field(description="Bank account used")
    status: PaymentStatus = Field(description="Status of the payment")
    allocation: Optional[PaymentAllocation] = Field(
        None, description="How a posted payment was applied"
    )


class DocumentRequest(BaseModel):
    request_id: str = Field(description="Unique identifier of the request")
    loan_id: str = Field(description="Loan the document is about")
    borrower_id: str = Field(description="Borrower who requested it")
    doc_type: DocumentType = Field(description="Type of document")
    sent_to: str = Field(description="Email address the document was sent to")


class Transfer(BaseModel):
    transfer_id: str = Field(description="Unique identifier of the transfer")
    reason: TransferReason = Field(description="Reason for the transfer")


class PayoffQuote(BaseModel):
    loan_id: str = Field(description="Loan the quote is for")
    payoff_date: str = Field(description="Date the quote is valid for")
    principal: float = Field(description="Remaining principal")
    interest: float = Field(description="Interest accrued up to the payoff date")
    fees: float = Field(description="Open late fees")
    total: float = Field(description="Amount needed to pay off the loan")


class LoanServicingDB(DB):
    """Database of borrowers, loans, payments, document requests, and transfers."""

    borrowers: Dict[str, Borrower] = Field(
        description="Dictionary of all borrowers indexed by borrower ID"
    )
    loans: Dict[str, Loan] = Field(
        description="Dictionary of all loans indexed by loan ID"
    )
    payments: Dict[str, Payment] = Field(
        description="Dictionary of all payments indexed by payment ID"
    )
    document_requests: Dict[str, DocumentRequest] = Field(
        description="Dictionary of all document requests indexed by request ID"
    )
    transfers: Dict[str, Transfer] = Field(
        description="Dictionary of all transfers indexed by transfer ID"
    )

    def get_statistics(self) -> dict[str, Any]:
        """Get the statistics of the database."""
        statuses: dict[str, int] = {}
        for loan in self.loans.values():
            statuses[loan.status] = statuses.get(loan.status, 0) + 1
        return {
            "num_borrowers": len(self.borrowers),
            "num_loans": len(self.loans),
            "num_payments": len(self.payments),
            "loan_statuses": statuses,
        }


def get_db():
    return LoanServicingDB.load(LOAN_SERVICING_DB_PATH)


if __name__ == "__main__":
    db = get_db()
    print(db.get_statistics())
