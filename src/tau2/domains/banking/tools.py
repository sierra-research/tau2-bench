"""Toolkit for the banking domain."""

import json
from datetime import datetime
from typing import List

from tau2.domains.banking.data_model import (
    Account,
    BankingDB,
    Customer,
    Product,
    Transaction,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class BankingTools(ToolKitBase):
    """All the tools for the banking domain."""

    db: BankingDB

    def __init__(self, db: BankingDB) -> None:
        super().__init__(db)

    def _get_customer(self, customer_id: str) -> Customer:
        """Get the customer from the database.

        Args:
            customer_id: The customer id.

        Returns:
            The customer.

        Raises:
            ValueError: If the customer is not found.
        """
        if customer_id not in self.db.customers:
            raise ValueError("Customer not found")
        return self.db.customers[customer_id]

    def _get_account(self, account_id: str) -> Account:
        """Get the account from the database.

        Args:
            account_id: The account id.

        Returns:
            The account.

        Raises:
            ValueError: If the account is not found.
        """
        if account_id not in self.db.accounts:
            raise ValueError("Account not found")
        return self.db.accounts[account_id]

    def _get_product(self, product_id: str) -> Product:
        """Get the product from the database.

        Args:
            product_id: The product id.

        Returns:
            The product.

        Raises:
            ValueError: If the product is not found.
        """
        if product_id not in self.db.products:
            raise ValueError("Product not found")
        return self.db.products[product_id]

    @is_tool(ToolType.READ)
    def find_customer_by_email(self, email: str) -> str:
        """Find customer id by email address.

        Args:
            email: The email of the customer, such as 'someone@example.com'.

        Returns:
            str: The customer id if found.

        Raises:
            ValueError: If the customer is not found.
        """
        for customer_id, customer in self.db.customers.items():
            if customer.email.lower() == email.lower():
                return customer_id
        raise ValueError("Customer not found")

    @is_tool(ToolType.READ)
    def find_customer_by_phone(self, phone: str) -> str:
        """Find customer id by phone number.

        Args:
            phone: The phone number of the customer, such as '010-1234-5678'.

        Returns:
            str: The customer id if found.

        Raises:
            ValueError: If the customer is not found.
        """
        for customer_id, customer in self.db.customers.items():
            if customer.phone == phone:
                return customer_id
        raise ValueError("Customer not found")

    @is_tool(ToolType.READ)
    def get_customer_info(self, customer_id: str) -> Customer:
        """Get the details of a customer.

        Args:
            customer_id: The customer id.

        Returns:
            Customer: The customer details.

        Raises:
            ValueError: If the customer is not found.
        """
        return self._get_customer(customer_id)

    @is_tool(ToolType.READ)
    def get_account(self, account_id: str) -> Account:
        """Get the details of a bank account.

        Args:
            account_id: The account id.

        Returns:
            Account: The account details.

        Raises:
            ValueError: If the account is not found.
        """
        return self._get_account(account_id)

    @is_tool(ToolType.READ)
    def list_products(self, category: str) -> str:
        """List available bank products, optionally filtered by category.

        Args:
            category: The product category to filter by, such as 'savings', 'deposit', 'loan', 'card', 'subscription', 'checking'. Use 'all' to list all products.

        Returns:
            str: A JSON string of the product list with product_id and name.
        """
        products = {}
        for product in self.db.products.values():
            if category == "all" or product.category == category:
                products[product.product_id] = product.name
        return json.dumps(products, sort_keys=True, ensure_ascii=False)

    @is_tool(ToolType.READ)
    def get_product_details(self, product_id: str) -> Product:
        """Get the details of a bank product.

        Args:
            product_id: The product id.

        Returns:
            Product: The product details.

        Raises:
            ValueError: If the product is not found.
        """
        return self._get_product(product_id)

    @is_tool(ToolType.READ)
    def get_transaction_history(self, account_id: str, limit: int = 20) -> str:
        """Get the transaction history for a bank account.

        Args:
            account_id: The account id.
            limit: Maximum number of transactions to return, default 20.

        Returns:
            str: A JSON string of the transaction list, sorted by date descending.

        Raises:
            ValueError: If the account is not found.
        """
        self._get_account(account_id)
        transactions = [
            tx
            for tx in self.db.transactions.values()
            if tx.account_id == account_id
        ]
        transactions.sort(key=lambda tx: tx.date, reverse=True)
        transactions = transactions[:limit]
        return json.dumps(
            [tx.model_dump() for tx in transactions], ensure_ascii=False
        )

    @is_tool(ToolType.WRITE)
    def open_account(
        self, customer_id: str, product_id: str, initial_deposit: int = 0
    ) -> str:
        """Open a new bank account for a customer. The agent needs to explain the account details
        and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            customer_id: The customer id.
            product_id: The product id for the account.
            initial_deposit: The initial deposit amount in KRW, default 0.

        Returns:
            str: A JSON string with the new account details.

        Raises:
            ValueError: If the customer is not found.
            ValueError: If the product is not found.
            ValueError: If KYC verification is not complete.
            ValueError: If the initial deposit is below the minimum balance.
            ValueError: If the initial deposit exceeds the maximum balance.
        """
        customer = self._get_customer(customer_id)
        product = self._get_product(product_id)

        if not customer.kyc_verified:
            raise ValueError("KYC verification is not complete")

        if initial_deposit < product.min_balance:
            raise ValueError(
                f"Initial deposit {initial_deposit} is below minimum balance {product.min_balance}"
            )

        if initial_deposit > product.max_balance:
            raise ValueError(
                f"Initial deposit {initial_deposit} exceeds maximum balance {product.max_balance}"
            )

        # Generate new account id
        account_id = f"ACC{len(self.db.accounts) + 1:06d}"

        # Map product category to account type
        category_to_type = {
            "savings": "savings",
            "deposit": "deposit",
            "checking": "checking",
            "loan": "loan",
        }
        account_type = category_to_type.get(product.category, "savings")

        account = Account(
            account_id=account_id,
            customer_id=customer_id,
            product_id=product_id,
            account_type=account_type,
            status="active",
            balance=initial_deposit,
            currency="KRW",
            opened_date=datetime.now().strftime("%Y-%m-%d"),
        )

        self.db.accounts[account_id] = account
        customer.accounts.append(account_id)

        return json.dumps(
            {"account_id": account_id, "status": "opened", "balance": initial_deposit},
            ensure_ascii=False,
        )

    @is_tool(ToolType.WRITE)
    def close_account(self, account_id: str, reason: str) -> str:
        """Close a bank account. The agent needs to explain the closure detail
        and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            account_id: The account id.
            reason: The reason for closing the account.

        Returns:
            str: A JSON string confirming the account closure.

        Raises:
            ValueError: If the account is not found.
            ValueError: If the account balance is not zero.
            ValueError: If the account is already closed.
        """
        account = self._get_account(account_id)

        if account.status == "closed":
            raise ValueError("Account is already closed")

        if account.balance != 0:
            raise ValueError("Account balance must be zero before closing")

        account.status = "closed"

        return json.dumps(
            {"account_id": account_id, "status": "closed", "reason": reason},
            ensure_ascii=False,
        )

    @is_tool(ToolType.WRITE)
    def transfer(self, from_account_id: str, to_account_id: str, amount: int) -> str:
        """Transfer money between two accounts. The agent needs to explain the transfer detail
        and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            from_account_id: The source account id.
            to_account_id: The destination account id.
            amount: The transfer amount in KRW.

        Returns:
            str: A JSON string with the transfer result.

        Raises:
            ValueError: If either account is not found.
            ValueError: If either account is not active.
            ValueError: If the source account has insufficient balance.
            ValueError: If the amount is not positive.
        """
        if amount <= 0:
            raise ValueError("Transfer amount must be positive")

        from_account = self._get_account(from_account_id)
        to_account = self._get_account(to_account_id)

        if from_account.status != "active":
            raise ValueError("Source account is not active")

        if to_account.status != "active":
            raise ValueError("Destination account is not active")

        if from_account.balance < amount:
            raise ValueError("Insufficient balance")

        from_account.balance -= amount
        to_account.balance += amount

        # Create transaction records
        now = datetime.now().isoformat()
        tx_out_id = f"TX{len(self.db.transactions) + 1:08d}"
        tx_in_id = f"TX{len(self.db.transactions) + 2:08d}"

        tx_out = Transaction(
            transaction_id=tx_out_id,
            account_id=from_account_id,
            type="transfer_out",
            amount=amount,
            balance_after=from_account.balance,
            counterparty=to_account_id,
            date=now,
            description=f"Transfer to {to_account_id}",
        )
        tx_in = Transaction(
            transaction_id=tx_in_id,
            account_id=to_account_id,
            type="transfer_in",
            amount=amount,
            balance_after=to_account.balance,
            counterparty=from_account_id,
            date=now,
            description=f"Transfer from {from_account_id}",
        )

        self.db.transactions[tx_out_id] = tx_out
        self.db.transactions[tx_in_id] = tx_in

        return json.dumps(
            {
                "status": "completed",
                "from_account_id": from_account_id,
                "to_account_id": to_account_id,
                "amount": amount,
                "from_balance_after": from_account.balance,
                "to_balance_after": to_account.balance,
            },
            ensure_ascii=False,
        )

    @is_tool(ToolType.WRITE)
    def apply_loan(self, customer_id: str, product_id: str, amount: int) -> str:
        """Apply for a loan product. The agent needs to explain the loan details
        and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            customer_id: The customer id.
            product_id: The loan product id.
            amount: The requested loan amount in KRW.

        Returns:
            str: A JSON string with the loan application result.

        Raises:
            ValueError: If the customer is not found.
            ValueError: If the product is not found.
            ValueError: If the product is not a loan product.
            ValueError: If the amount is outside the allowed range.
        """
        customer = self._get_customer(customer_id)
        product = self._get_product(product_id)

        if product.category != "loan":
            raise ValueError("Product is not a loan product")

        if amount < product.min_balance:
            raise ValueError(
                f"Loan amount {amount} is below minimum {product.min_balance}"
            )

        if amount > product.max_balance:
            raise ValueError(
                f"Loan amount {amount} exceeds maximum {product.max_balance}"
            )

        # Generate new account id for the loan
        account_id = f"ACC{len(self.db.accounts) + 1:06d}"

        account = Account(
            account_id=account_id,
            customer_id=customer_id,
            product_id=product_id,
            account_type="loan",
            status="active",
            balance=amount,
            currency="KRW",
            opened_date=datetime.now().strftime("%Y-%m-%d"),
        )

        self.db.accounts[account_id] = account
        customer.accounts.append(account_id)

        return json.dumps(
            {
                "status": "approved",
                "account_id": account_id,
                "amount": amount,
                "interest_rate": product.interest_rate,
                "terms_months": product.terms_months,
            },
            ensure_ascii=False,
        )

    @is_tool(ToolType.WRITE)
    def update_customer_info(self, customer_id: str, field: str, value: str) -> str:
        """Update a customer's information. The agent needs to explain the update detail
        and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            customer_id: The customer id.
            field: The field to update. Must be one of 'email', 'phone', or 'address'.
            value: The new value. For address, provide a JSON string with keys: city, district, detail, postal_code.

        Returns:
            str: A JSON string confirming the update.

        Raises:
            ValueError: If the customer is not found.
            ValueError: If the field is not allowed to be updated.
            ValueError: If the address value is invalid JSON.
        """
        customer = self._get_customer(customer_id)

        allowed_fields = {"email", "phone", "address"}
        if field not in allowed_fields:
            raise ValueError(
                f"Field '{field}' cannot be updated. Allowed fields: {', '.join(allowed_fields)}"
            )

        if field == "address":
            try:
                address_data = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Address value must be a valid JSON string")
            from tau2.domains.banking.data_model import Address

            customer.address = Address(**address_data)
        elif field == "email":
            customer.email = value
        elif field == "phone":
            customer.phone = value

        return json.dumps(
            {"status": "updated", "customer_id": customer_id, "field": field},
            ensure_ascii=False,
        )

    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """Calculate the result of a mathematical expression.

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
        """Transfer the user to a human agent, with a summary of the user's issue.
        Only transfer if the user explicitly asks for a human agent or the issue
        cannot be resolved with available tools.

        Args:
            summary: A summary of the user's issue.

        Returns:
            A message indicating the user has been transferred to a human agent.
        """
        return "Transfer successful"
