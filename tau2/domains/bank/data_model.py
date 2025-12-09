from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from tau2.environment.db import DB


# -------------------------
# Bank domain enumerations
# -------------------------

AccountType = Literal["checking", "savings", "credit"]
AccountStatus = Literal["active", "frozen", "closed"]

CardType = Literal["debit", "credit"]
CardStatus = Literal["active", "blocked", "expired"]

KYCStatus = Literal["pending", "verified", "failed"]

TransactionType = Literal[
    "deposit", "withdrawal", "transfer", "payment", "fee", "interest"
]

TransactionDirection = Literal["debit", "credit"]
TransactionMethod = Literal["ACH", "Wire", "Card", "ATM", "Internal"]
TransactionStatus = Literal["pending", "posted", "reversed"]

LoanType = Literal["mortgage", "auto", "personal", "credit_line"]
AmortizationType = Literal["fixed", "interest_only", "balloon"]
LoanStatus = Literal["active", "closed", "delinquent", "default"]
CollateralType = Literal["real_estate", "vehicle", "unsecured", "other"]

PaymentScheduleStatus = Literal["due", "paid", "late", "deferred"]

BeneficiaryType = Literal["individual", "business"]
VerificationStatus = Literal["pending", "verified", "failed"]
VerificationMethod = Literal["micro_deposits", "document", "instant"]
BeneficiaryStatus = Literal["active", "suspended", "revoked"]


# -------------------------
# Shared/basic models
# -------------------------

class BankAddress(BaseModel):
    """Represents a physical address in the bank domain"""
    address1: str = Field(description="Primary address line")
    address2: str = Field(description="Secondary address line")
    city: str = Field(description="City name")
    state: str = Field(description="State or province name")
    zip: str = Field(description="Postal code")
    country: str = Field(description="Country name")


class ClientName(BaseModel):
    """Represents a client's full name"""
    first_name: str = Field(description="Client's first name")
    last_name: str = Field(description="Client's last name")
    middle_name: Optional[str] = Field(
        description="Client's middle name", default=None
    )


class ClientContact(BaseModel):
    """Represents a client's contact information"""
    email: str = Field(description="Client's email address")
    phone: str = Field(description="Client's phone number")


# -------------------------
# Account-related models
# -------------------------

class AccountBalance(BaseModel):
    """Represents account balances"""
    current: float = Field(description="Current ledger balance of the account")
    available: float = Field(description="Available balance that can be used")
    on_hold: float = Field(description="Amount currently on hold")


class AccountFeatures(BaseModel):
    """Represents enabled features for an account"""
    checks_enabled: bool = Field(description="Whether checks are enabled for the account")
    atm_access: bool = Field(description="Whether ATM access is enabled")
    online_banking: bool = Field(description="Whether online banking is enabled")


class Account(BaseModel):
    """Represents a bank account"""
    account_id: str = Field(description="Unique identifier for the account")
    type: AccountType = Field(description="Type of the account")
    currency: str = Field(description="Currency code of the account (e.g., USD)")
    status: AccountStatus = Field(description="Current status of the account")
    account_number_masked: str = Field(
        description="Masked account number (e.g., ****1234)"
    )
    routing_number: str = Field(description="Routing number of the account")
    balance: AccountBalance = Field(description="Balances associated with the account")
    interest_rate_apr: Optional[float] = Field(
        description="Annual percentage rate for interest (if applicable)",
        default=None,
    )
    overdraft_limit: Optional[float] = Field(
        description="Overdraft limit for the account (if applicable)",
        default=None,
    )
    opened_at: str = Field(description="Account opening timestamp")
    features: AccountFeatures = Field(description="Enabled features for the account")


# -------------------------
# Card-related models
# -------------------------

class CardExtraInfo(BaseModel):
    """Additional card details"""
    brand: str = Field(description="Card network brand (e.g., Visa, Mastercard)")
    last_four: str = Field(description="Last four digits of the card number")
    exp_month: str = Field(description="Expiration month of the card")
    exp_year: str = Field(description="Expiration year of the card")


class CardLimits(BaseModel):
    """Represents card limits"""
    daily_atm_limit: float = Field(description="Daily ATM withdrawal limit")
    daily_pos_limit: float = Field(description="Daily POS purchase limit")


class Card(BaseModel):
    """Represents a bank card"""
    card_id: str = Field(description="Unique identifier for the card")
    type: CardType = Field(description="Type of the card")
    linked_account_id: str = Field(description="Account ID linked to the card")
    status: CardStatus = Field(description="Current status of the card")
    issuer: str = Field(description="Card issuer name")
    extra_info: CardExtraInfo = Field(description="Additional card information")
    limits: CardLimits = Field(description="Limits applied to the card")
    pin_set: bool = Field(description="Whether a PIN has been set for this card")


# -------------------------
# KYC models
# -------------------------

class KYCInfo(BaseModel):
    """Represents KYC (Know Your Customer) information for a client"""
    status: KYCStatus = Field(description="Current KYC verification status")
    last_reviewed_at: str = Field(description="Timestamp of last KYC review")
    tax_id_masked: str = Field(description="Masked tax ID (e.g., ****1234)")


# -------------------------
# Transaction-related models
# -------------------------

class MerchantInfo(BaseModel):
    """Represents merchant details for card/payment transactions"""
    name: str = Field(description="Merchant name")
    mcc: str = Field(description="Merchant category code")
    city: str = Field(description="Merchant city")
    country: str = Field(description="Merchant country")


class ExchangeInfo(BaseModel):
    """Represents currency exchange details for a transaction"""
    source_currency: str = Field(description="Original currency code")
    target_currency: str = Field(description="Converted currency code")
    rate: float = Field(description="Exchange rate applied")
    source_amount: float = Field(description="Amount in source currency")
    target_amount: float = Field(description="Amount in target currency")


class TransactionFee(BaseModel):
    """Represents a fee applied to a transaction"""
    type: str = Field(description="Type of fee (e.g., wire_fee, overdraft_fee)")
    amount: float = Field(description="Fee amount")
    currency: str = Field(description="Currency code of the fee")


class TransactionHold(BaseModel):
    """Represents an authorization hold on a transaction"""
    is_hold: bool = Field(description="Whether the transaction is currently on hold")
    release_at: Optional[str] = Field(
        description="Timestamp when the hold will be released",
        default=None,
    )
    reason: Optional[str] = Field(
        description="Reason for the hold",
        default=None,
    )


class Transaction(BaseModel):
    """Represents a banking transaction"""
    transaction_id: str = Field(description="Unique identifier for the transaction")
    client_id: str = Field(description="Client ID associated with the transaction")
    account_id: str = Field(description="Account ID associated with the transaction")
    timestamp: str = Field(description="Transaction timestamp")
    type: TransactionType = Field(description="Type/category of the transaction")
    direction: TransactionDirection = Field(
        description="Direction relative to the account (debit or credit)"
    )
    amount: float = Field(description="Transaction amount")
    currency: str = Field(description="Currency code of the transaction amount")
    description: Optional[str] = Field(
        description="Transaction description or memo",
        default=None,
    )
    method: TransactionMethod = Field(description="Processing method of the transaction")
    status: TransactionStatus = Field(description="Current status of the transaction")
    merchant: Optional[MerchantInfo] = Field(
        description="Merchant details for the transaction (if applicable)",
        default=None,
    )
    related_transaction_id: Optional[str] = Field(
        description="Related transaction ID (e.g., transfer counterpart or reversal)",
        default=None,
    )
    exchange: Optional[ExchangeInfo] = Field(
        description="Exchange details if currency conversion occurred",
        default=None,
    )
    fees: List[TransactionFee] = Field(
        description="List of fees applied to the transaction",
        default_factory=list,
    )
    hold: Optional[TransactionHold] = Field(
        description="Hold details if the transaction amount is on hold",
        default=None,
    )
    balance_after: Optional[float] = Field(
        description="Account balance immediately after this transaction",
        default=None,
    )


# -------------------------
# Loan-related models
# -------------------------

class Collateral(BaseModel):
    """Represents collateral for a loan"""
    type: CollateralType = Field(description="Type of collateral")
    description: str = Field(description="Description of the collateral")
    value_amount: float = Field(description="Estimated value of the collateral")
    value_currency: str = Field(description="Currency code for the collateral value")


class LoanPaymentScheduleEntry(BaseModel):
    """Represents a scheduled payment for a loan"""
    due_date: str = Field(description="Due date for the scheduled payment")
    amount: float = Field(description="Amount due for the scheduled payment")
    currency: str = Field(description="Currency code of the scheduled payment")
    status: PaymentScheduleStatus = Field(description="Status of the scheduled payment")
    transaction_id: Optional[str] = Field(
        description="Transaction ID when paid (if applicable)",
        default=None,
    )


class LoanRepaymentHistoryEntry(BaseModel):
    """Represents a repayment history entry for a loan"""
    transaction_id: str = Field(description="Transaction ID of the repayment")
    posted_at: str = Field(description="Timestamp when the repayment was posted")
    amount: float = Field(description="Total repayment amount")
    currency: str = Field(description="Currency code of the repayment")
    method: TransactionMethod = Field(description="Method used for repayment")
    principal_component: float = Field(description="Amount applied to principal")
    interest_component: float = Field(description="Amount applied to interest")
    fees_component: float = Field(description="Amount applied to fees")


class EscrowAccount(BaseModel):
    """Represents an escrow account linked to a loan"""
    account_id: str = Field(description="Account ID of the escrow account")
    balance: float = Field(description="Current escrow balance")
    currency: str = Field(description="Currency code of the escrow account")


class Loan(BaseModel):
    """Represents a loan"""
    loan_id: str = Field(description="Unique identifier for the loan")
    client_id: str = Field(description="Client ID that owns the loan")
    linked_repayment_account_id: str = Field(
        description="Account ID used for repayments"
    )
    type: LoanType = Field(description="Type of loan")
    principal: float = Field(description="Original principal amount of the loan")
    currency: str = Field(description="Currency code of the loan")
    interest_rate_apr: float = Field(description="Annual percentage rate for the loan")
    amortization: AmortizationType = Field(description="Amortization type of the loan")
    term_months: int = Field(description="Loan term in months")
    origination_date: str = Field(description="Origination date of the loan")
    first_payment_date: str = Field(description="Date of the first payment")
    maturity_date: str = Field(description="Loan maturity date")
    status: LoanStatus = Field(description="Current status of the loan")
    collateral: Optional[Collateral] = Field(
        description="Collateral details for the loan (if applicable)",
        default=None,
    )
    payment_schedule: List[LoanPaymentScheduleEntry] = Field(
        description="Scheduled payments for the loan"
    )
    repayment_history: List[LoanRepaymentHistoryEntry] = Field(
        description="Repayment history entries for the loan"
    )
    escrow: Optional[EscrowAccount] = Field(
        description="Escrow account details (if applicable)",
        default=None,
    )


# -------------------------
# Beneficiary-related models
# -------------------------

class BeneficiaryName(BaseModel):
    """Represents a beneficiary's name information"""
    display_name: Optional[str] = Field(
        description="Display name for the beneficiary", default=None
    )
    first_name: Optional[str] = Field(
        description="First name of the beneficiary", default=None
    )
    last_name: Optional[str] = Field(
        description="Last name of the beneficiary", default=None
    )
    business_name: Optional[str] = Field(
        description="Business name, if beneficiary is a business", default=None
    )


class BankDetails(BaseModel):
    """Represents bank details for a beneficiary"""
    bank_name: str = Field(description="Name of the beneficiary's bank")
    account_number_masked: str = Field(
        description="Masked account number for the beneficiary"
    )
    routing_number: Optional[str] = Field(
        description="Routing number for domestic transfers (if applicable)",
        default=None,
    )
    iban: Optional[str] = Field(
        description="IBAN for international transfers (if applicable)",
        default=None,
    )
    swift_bic: Optional[str] = Field(
        description="SWIFT/BIC code for international transfers (if applicable)",
        default=None,
    )


class TransferLimits(BaseModel):
    """Represents transfer limits for a beneficiary"""
    per_transfer_limit: float = Field(description="Maximum amount per transfer")
    daily_limit: float = Field(description="Maximum total daily transfer limit")


class BeneficiaryVerification(BaseModel):
    """Represents beneficiary verification details"""
    status: VerificationStatus = Field(description="Verification status")
    method: VerificationMethod = Field(description="Verification method used")
    verified_at: Optional[str] = Field(
        description="Timestamp when verification was completed",
        default=None,
    )


class Beneficiary(BaseModel):
    """Represents a beneficiary authorized by a client"""
    beneficiary_id: str = Field(description="Unique identifier for the beneficiary")
    client_id: str = Field(
        description="Client ID who owns/authorized this beneficiary"
    )
    name: BeneficiaryName = Field(description="Name information of the beneficiary")
    type: BeneficiaryType = Field(description="Type of beneficiary")
    bank_details: BankDetails = Field(description="Bank details of the beneficiary")
    address: BankAddress = Field(description="Address of the beneficiary")
    allowed_from_account_ids: List[str] = Field(
        description="List of account IDs from which transfers to this beneficiary are allowed"
    )
    transfer_limits: TransferLimits = Field(
        description="Transfer limits for this beneficiary"
    )
    verification: BeneficiaryVerification = Field(
        description="Verification details for this beneficiary"
    )
    status: BeneficiaryStatus = Field(description="Current status of the beneficiary")
    created_at: str = Field(description="Timestamp when the beneficiary was created")
    notes: Optional[str] = Field(
        description="Additional notes related to the beneficiary",
        default=None,
    )


# -------------------------
# Client model
# -------------------------

class Client(BaseModel):
    """Represents a bank client with their accounts, cards, and related entities"""
    client_id: str = Field(description="Unique identifier for the client")
    name: ClientName = Field(description="Client's full name")
    contact: ClientContact = Field(description="Client's contact information")
    address: BankAddress = Field(description="Client's primary address")
    accounts: Dict[str, Account] = Field(
        description="Dictionary of accounts indexed by account ID"
    )
    cards: Dict[str, Card] = Field(
        description="Dictionary of cards indexed by card ID"
    )
    loan_ids: List[str] = Field(
        description="List of loan IDs associated with this client"
    )
    beneficiary_ids: List[str] = Field(
        description="List of beneficiary IDs authorized by this client"
    )
    created_at: str = Field(description="Timestamp when the client profile was created")
    kyc: KYCInfo = Field(description="KYC information for the client")


# -------------------------
# Bank database
# -------------------------

class BankDB(DB):
    """Database containing all bank-related data including clients, transactions, loans and beneficiaries"""

    clients: Dict[str, Client] = Field(
        description="Dictionary of all clients indexed by client ID"
    )
    transactions: Dict[str, Transaction] = Field(
        description="Dictionary of all transactions indexed by transaction ID"
    )
    loans: Dict[str, Loan] = Field(
        description="Dictionary of all loans indexed by loan ID"
    )
    beneficiaries: Dict[str, Beneficiary] = Field(
        description="Dictionary of all beneficiaries indexed by beneficiary ID"
    )

    def get_statistics(self) -> dict[str, Any]:
        """Get the statistics of the bank database."""
        num_clients = len(self.clients)
        num_transactions = len(self.transactions)
        num_loans = len(self.loans)
        num_beneficiaries = len(self.beneficiaries)
        total_accounts = sum(len(client.accounts) for client in self.clients.values())
        total_cards = sum(len(client.cards) for client in self.clients.values())
        total_current_balance = sum(
            sum(account.balance.current for account in client.accounts.values())
            for client in self.clients.values()
        )
        return {
            "num_clients": num_clients,
            "num_transactions": num_transactions,
            "num_loans": num_loans,
            "num_beneficiaries": num_beneficiaries,
            "total_accounts": total_accounts,
            "total_cards": total_cards,
            "total_current_balance": total_current_balance,
        }
