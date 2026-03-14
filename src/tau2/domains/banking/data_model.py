from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from tau2.environment.db import DB


class Address(BaseModel):
    """Korean address"""

    city: str = Field(description="City name, e.g. 'Seoul'")
    district: str = Field(description="District name, e.g. 'Gangnam-gu'")
    detail: str = Field(description="Detailed address, e.g. '123 Teheran-ro'")
    postal_code: str = Field(description="5-digit postal code")


AccountType = Literal["savings", "deposit", "checking", "loan"]
AccountStatus = Literal["active", "dormant", "closed", "frozen"]
CardType = Literal["debit", "credit", "prepaid"]
ProductCategory = Literal["savings", "deposit", "loan", "card", "subscription", "checking"]
TransactionType = Literal["deposit", "withdrawal", "transfer_in", "transfer_out", "interest", "fee"]


class Customer(BaseModel):
    """Bank customer"""

    customer_id: str = Field(description="Unique customer identifier")
    name: str = Field(description="Customer full name in Korean or English")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number in 010-XXXX-XXXX format")
    date_of_birth: str = Field(description="Date of birth in YYYY-MM-DD format")
    address: Address = Field(description="Customer address")
    annual_income: int = Field(description="Annual income in KRW")
    credit_score: int = Field(description="Credit score, 1-1000 range")
    kyc_verified: bool = Field(description="Whether KYC verification is complete")
    accounts: List[str] = Field(description="List of account IDs owned by this customer")


class Product(BaseModel):
    """Bank product (savings, deposit, loan, card, subscription)"""

    product_id: str = Field(description="Unique product identifier")
    name: str = Field(description="Product name")
    category: ProductCategory = Field(description="Product category")
    interest_rate: float = Field(description="Annual interest rate in percent, e.g. 3.5")
    min_balance: int = Field(description="Minimum balance/amount in KRW")
    max_balance: int = Field(description="Maximum balance/amount in KRW")
    features: List[str] = Field(description="List of product features")
    terms_months: Optional[int] = Field(
        description="Lock-in period in months, null if flexible", default=None
    )
    eligibility_credit_score: Optional[int] = Field(
        description="Minimum credit score required, null if none", default=None
    )
    eligibility_min_income: Optional[int] = Field(
        description="Minimum annual income in KRW, null if none", default=None
    )


class Account(BaseModel):
    """Bank account"""

    account_id: str = Field(description="Unique account identifier")
    customer_id: str = Field(description="Owner customer ID")
    product_id: str = Field(description="Associated product ID")
    account_type: AccountType = Field(description="Type of account")
    status: AccountStatus = Field(description="Current account status")
    balance: int = Field(description="Current balance in KRW")
    currency: str = Field(description="Currency code, always KRW", default="KRW")
    opened_date: str = Field(description="Account opening date in YYYY-MM-DD format")
    maturity_date: Optional[str] = Field(
        description="Maturity date for fixed deposits, null otherwise", default=None
    )


class Transaction(BaseModel):
    """Bank transaction record"""

    transaction_id: str = Field(description="Unique transaction identifier")
    account_id: str = Field(description="Account this transaction belongs to")
    type: TransactionType = Field(description="Transaction type")
    amount: int = Field(description="Transaction amount in KRW")
    balance_after: int = Field(description="Account balance after this transaction")
    counterparty: Optional[str] = Field(
        description="Counterparty account ID or description", default=None
    )
    date: str = Field(description="Transaction datetime in ISO format")
    description: str = Field(description="Transaction description")


class BankingDB(DB):
    """Database for a Korean internet bank domain"""

    customers: Dict[str, Customer] = Field(
        description="Customers indexed by customer_id"
    )
    accounts: Dict[str, Account] = Field(
        description="Accounts indexed by account_id"
    )
    products: Dict[str, Product] = Field(
        description="Products indexed by product_id"
    )
    transactions: Dict[str, Transaction] = Field(
        description="Transactions indexed by transaction_id"
    )

    def get_statistics(self) -> dict[str, Any]:
        return {
            "num_customers": len(self.customers),
            "num_accounts": len(self.accounts),
            "num_products": len(self.products),
            "num_transactions": len(self.transactions),
        }
