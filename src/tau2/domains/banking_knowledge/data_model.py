"""Data models for the knowledge domain."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tau2.environment.db import DB


class Document(BaseModel):
    """A document in the knowledge base."""

    id: str = Field(..., description="The unique identifier of the document")
    title: str = Field(..., description="The title of the document")
    content: str = Field(..., description="The content of the document")


class KnowledgeBase(BaseModel):
    """Knowledge base containing documents for semantic search.

    This is separate from the transactional database and is used for
    document retrieval/search operations.
    """

    documents: Dict[str, Document] = Field(
        default_factory=dict, description="Documents in the knowledge base"
    )

    @classmethod
    def load(cls, documents_dir: str) -> "KnowledgeBase":
        """Load documents from a directory.

        Args:
            documents_dir: Path to directory containing document JSON files

        Returns:
            KnowledgeBase instance with loaded documents
        """
        import json
        from pathlib import Path

        documents = {}
        doc_path = Path(documents_dir)

        if doc_path.exists():
            for file_path in doc_path.glob("*.json"):
                with open(file_path, "r") as f:
                    doc_data = json.load(f)
                    doc = Document(**doc_data)
                    documents[doc.id] = doc

        return cls(documents=documents)

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(doc_id)

    def get_all_documents(self) -> List[Document]:
        """Get all documents."""
        return list(self.documents.values())

    def get_document_texts(self) -> List[str]:
        """Get text content of all documents."""
        return [doc.content for doc in self.documents.values()]

    def get_document_ids(self) -> List[str]:
        """Get all document IDs."""
        return list(self.documents.keys())


# Backward compatibility alias
KnowledgeDB = KnowledgeBase


# =============================================================================
# Transactional Database Models (mirrors db.json structure)
# =============================================================================


class DatabaseTable(BaseModel):
    """A database table with data and optional notes."""

    data: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    notes: str = ""


class TransactionalDB(DB):
    """Transactional database for the knowledge domain.

    This contains all mutable state (users, accounts, referrals, applications)
    and is what gets hashed for DB state comparison during evaluation.
    """

    users: DatabaseTable = Field(default_factory=DatabaseTable)
    accounts: DatabaseTable = Field(default_factory=DatabaseTable)
    # Debit cards linked to checking accounts
    debit_cards: DatabaseTable = Field(default_factory=DatabaseTable)
    referrals: DatabaseTable = Field(default_factory=DatabaseTable)
    credit_card_applications: DatabaseTable = Field(default_factory=DatabaseTable)
    # User discoverable tools tracking: tools given from agent to user
    user_discoverable_tools: DatabaseTable = Field(default_factory=DatabaseTable)
    # User discoverable tool calls: user calls to discoverable tools
    user_discoverable_tool_calls: DatabaseTable = Field(default_factory=DatabaseTable)
    # Verification history: audit log of user identity verifications
    verification_history: DatabaseTable = Field(default_factory=DatabaseTable)
    # Credit card transaction history: record of credit card transactions
    credit_card_transaction_history: DatabaseTable = Field(
        default_factory=DatabaseTable
    )
    # Cash back disputes: user-submitted disputes for incorrect rewards
    cash_back_disputes: DatabaseTable = Field(default_factory=DatabaseTable)
    # Bank account transaction history: record of bank account transactions
    bank_account_transaction_history: DatabaseTable = Field(
        default_factory=DatabaseTable
    )
    # Credit card accounts: user credit card accounts with balances and rewards
    credit_card_accounts: DatabaseTable = Field(default_factory=DatabaseTable)
    # Agent discoverable tools tracking: tools called by the agent
    agent_discoverable_tools: DatabaseTable = Field(default_factory=DatabaseTable)
    # Task configuration: per-instance settings that control handler behavior
    task_config: DatabaseTable = Field(default_factory=DatabaseTable)
    # Human transfer requests: tracks user requests to be transferred to a human agent
    human_transfer_requests: DatabaseTable = Field(default_factory=DatabaseTable)
    # Transaction disputes: formal disputes filed for credit card transactions
    transaction_disputes: DatabaseTable = Field(default_factory=DatabaseTable)
    # Credit card orders: replacement credit card orders
    credit_card_orders: DatabaseTable = Field(default_factory=DatabaseTable)
    # Debit card orders: new debit card orders for checking accounts
    debit_card_orders: DatabaseTable = Field(default_factory=DatabaseTable)
    # Credit card closure reasons: logs why customers want to close their accounts
    credit_card_closure_reasons: DatabaseTable = Field(default_factory=DatabaseTable)
    # Credit card account flags: flags applied to accounts (e.g., annual fee waivers)
    credit_card_account_flags: DatabaseTable = Field(default_factory=DatabaseTable)
    # Credit limit increase requests: tracks CLI requests and their history
    credit_limit_increase_requests: DatabaseTable = Field(default_factory=DatabaseTable)
    # Payment history: tracks payment history for credit card accounts
    payment_history: DatabaseTable = Field(default_factory=DatabaseTable)
    # Debit card disputes: formal disputes filed for debit card transactions (Regulation E)
    debit_card_disputes: DatabaseTable = Field(default_factory=DatabaseTable)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        return {
            "num_users": len(self.users.data),
            "num_accounts": len(self.accounts.data),
            "num_debit_cards": len(self.debit_cards.data),
            "num_referrals": len(self.referrals.data),
            "num_credit_card_applications": len(self.credit_card_applications.data),
            "num_user_discoverable_tools": len(self.user_discoverable_tools.data),
            "num_user_discoverable_tool_calls": len(
                self.user_discoverable_tool_calls.data
            ),
            "num_verification_history": len(self.verification_history.data),
            "num_credit_card_transactions": len(
                self.credit_card_transaction_history.data
            ),
            "num_cash_back_disputes": len(self.cash_back_disputes.data),
            "num_credit_card_accounts": len(self.credit_card_accounts.data),
            "num_agent_discoverable_tools": len(self.agent_discoverable_tools.data),
            "num_human_transfer_requests": len(self.human_transfer_requests.data),
            "num_transaction_disputes": len(self.transaction_disputes.data),
            "num_credit_card_orders": len(self.credit_card_orders.data),
            "num_credit_card_closure_reasons": len(
                self.credit_card_closure_reasons.data
            ),
            "num_credit_card_account_flags": len(self.credit_card_account_flags.data),
            "num_credit_limit_increase_requests": len(
                self.credit_limit_increase_requests.data
            ),
            "num_payment_history": len(self.payment_history.data),
        }
