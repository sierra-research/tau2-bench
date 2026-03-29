"""Utilities for the banking_knowledge domain."""

import hashlib
import json
from datetime import date, datetime
from typing import Any, Dict, Optional

from tau2.utils.utils import DATA_DIR

# Fixed date for the knowledge domain (all scenarios are set at this time)
KNOWLEDGE_FIXED_DATE = date(2025, 11, 14)


def get_today() -> date:
    """Get the fixed 'today' date for the knowledge domain.

    Returns 11/14/2025 - the locked time for all knowledge domain scenarios.
    """
    return KNOWLEDGE_FIXED_DATE


def get_today_str() -> str:
    """Get the fixed 'today' date as a formatted string (MM/DD/YYYY).

    Returns '11/14/2025' - the locked time for all knowledge domain scenarios.
    """
    return KNOWLEDGE_FIXED_DATE.strftime("%m/%d/%Y")


def get_now() -> datetime:
    """Get the fixed 'now' datetime for the knowledge domain.

    Returns 2025-11-14 03:40:00 EST - the locked time for all knowledge domain scenarios.
    """
    return datetime(2025, 11, 14, 3, 40, 0)


KNOWLEDGE_DATA_DIR = DATA_DIR / "tau2" / "domains" / "banking_knowledge"
KNOWLEDGE_DOCUMENTS_DIR = KNOWLEDGE_DATA_DIR / "documents"
KNOWLEDGE_TASK_SET_PATH = KNOWLEDGE_DATA_DIR / "tasks"

# Transactional database path (users, accounts, applications, referrals)
KNOWLEDGE_DB_PATH = KNOWLEDGE_DATA_DIR / "db.json"


# =============================================================================
# Deterministic ID Generation
# =============================================================================


def _deterministic_id(seed_string: str, length: int = 16) -> str:
    """Generate a deterministic hex ID from a seed string.

    This is the core hashing function used by all ID generators.

    Args:
        seed_string: String to hash for deterministic ID generation
        length: Length of the hex ID (default: 16 characters = 8 bytes)

    Returns:
        Deterministic hex string ID
    """
    hash_bytes = hashlib.sha256(seed_string.encode()).digest()
    return hash_bytes[: length // 2].hex()


def generate_transaction_id(
    user_id: str,
    credit_card_type: str,
    merchant_name: str,
    amount: float,
    category: str,
    date: Optional[str] = None,
) -> str:
    """Generate a deterministic transaction ID from transaction details.

    The ID is prefixed with 'txn_' followed by a 12-character hex string.

    Args:
        user_id: The user's ID
        credit_card_type: Type of credit card used
        merchant_name: Name of the merchant
        amount: Transaction amount
        category: Transaction category
        date: Optional transaction date (for additional uniqueness)

    Returns:
        Transaction ID in format 'txn_xxxxxxxxxxxx'
    """
    seed_parts = [
        "transaction",
        user_id,
        credit_card_type,
        merchant_name,
        f"{amount:.2f}",
        category,
    ]
    if date:
        seed_parts.append(date)

    seed = ":".join(seed_parts)
    return f"txn_{_deterministic_id(seed, length=12)}"


def generate_referral_id(
    referrer_id: str,
    referred_account_type: str,
    date: Optional[str] = None,
) -> str:
    """Generate a deterministic referral ID.

    Args:
        referrer_id: The user ID of the person making the referral
        referred_account_type: The account type being referred
        date: Optional date for additional uniqueness

    Returns:
        16-character hex referral ID
    """
    seed_parts = ["referral", referrer_id, referred_account_type]
    if date:
        seed_parts.append(date)

    seed = ":".join(seed_parts)
    return _deterministic_id(seed, length=16)


def generate_application_id(
    card_type: str,
    customer_name: str,
    annual_income: float,
    rho_bank_subscription: bool = False,
) -> str:
    """Generate a deterministic credit card application ID.

    Args:
        card_type: Type of credit card applied for
        customer_name: Full legal name of the applicant
        annual_income: Annual income in USD
        rho_bank_subscription: Whether user has Rho-Bank+ subscription

    Returns:
        16-character hex application ID
    """
    seed = f"credit_card:{card_type}:{customer_name}:{annual_income}:{rho_bank_subscription}"
    return _deterministic_id(seed, length=16)


def generate_verification_id(
    user_id: str,
    time_verified: str,
) -> str:
    """Generate a deterministic verification record ID.

    Args:
        user_id: The verified user's ID
        time_verified: Timestamp of verification

    Returns:
        Verification record ID in format '{user_id}_{sanitized_time}'
    """
    time_suffix = time_verified.replace(" ", "_").replace(":", "").replace("-", "")
    return f"{user_id}_{time_suffix}"


def generate_user_discoverable_tool_id(
    tool_name: str,
) -> str:
    """Generate a deterministic ID for a user discoverable tool instance.

    Only uses the tool name, not arguments, since agents may give tools
    with different argument variants but what matters is the tool was given.

    Args:
        tool_name: Name of the user discoverable tool

    Returns:
        16-character hex ID
    """
    seed = f"user_discoverable_tool:{tool_name}"
    return _deterministic_id(seed, length=16)


def generate_user_discoverable_tool_call_id(
    tool_name: str,
    arguments: Dict[str, Any],
) -> str:
    """Generate a deterministic ID for a user discoverable tool call record.

    Args:
        tool_name: Name of the user discoverable tool
        arguments: Dictionary of arguments given to the tool

    Returns:
        16-character hex ID
    """
    seed = f"user_discoverable_tool_call:{tool_name}:{json.dumps(arguments, sort_keys=True)}"
    return _deterministic_id(seed, length=16)


def generate_dispute_id(
    user_id: str,
    transaction_id: str,
) -> str:
    """Generate a deterministic dispute ID.

    Args:
        user_id: The user's ID
        transaction_id: The transaction being disputed

    Returns:
        Dispute ID in format 'dsp_xxxxxxxxxxxx'
    """
    seed = f"dispute:{user_id}:{transaction_id}"
    return f"dsp_{_deterministic_id(seed, length=12)}"


def generate_referral_link_id(
    user_id: str,
    card_name: str,
) -> str:
    """Generate a deterministic referral link ID.

    Args:
        user_id: The user's ID (referrer)
        card_name: The name of the credit card for the referral

    Returns:
        16-character hex referral link ID
    """
    seed = f"referral_link:{user_id}:{card_name}"
    return _deterministic_id(seed, length=16)


def generate_agent_discoverable_tool_id(
    tool_name: str,
) -> str:
    """Generate a deterministic ID for an agent discoverable tool instance.

    Only uses the tool name, not arguments, since what matters is that
    the tool was unlocked by the agent.

    Args:
        tool_name: Name of the agent discoverable tool

    Returns:
        16-character hex ID
    """
    seed = f"agent_discoverable_tool:{tool_name}"
    return _deterministic_id(seed, length=16)


def generate_agent_discoverable_tool_call_id(
    tool_name: str,
    arguments: Dict[str, Any],
) -> str:
    """Generate a deterministic ID for an agent discoverable tool call record.

    Args:
        tool_name: Name of the agent discoverable tool
        arguments: Dictionary of arguments given to the tool

    Returns:
        16-character hex ID
    """
    seed = f"agent_discoverable_tool_call:{tool_name}:{json.dumps(arguments, sort_keys=True)}"
    return _deterministic_id(seed, length=16)


def generate_credit_card_order_id(
    credit_card_account_id: str,
    user_id: str,
    reason: str,
) -> str:
    """Generate a deterministic credit card order ID.

    Args:
        credit_card_account_id: The credit card account ID being replaced
        user_id: The user's ID
        reason: Reason for replacement

    Returns:
        Credit card order ID in format 'ccord_xxxxxxxxxxxx'
    """
    seed = f"credit_card_order:{credit_card_account_id}:{user_id}:{reason}"
    return f"ccord_{_deterministic_id(seed, length=12)}"


def generate_closure_reason_id(
    credit_card_account_id: str,
    user_id: str,
) -> str:
    """Generate a deterministic closure reason record ID.

    Args:
        credit_card_account_id: The credit card account ID being closed
        user_id: The user's ID

    Returns:
        Closure reason ID in format 'clsr_xxxxxxxxxxxx'
    """
    seed = f"closure_reason:{credit_card_account_id}:{user_id}"
    return f"clsr_{_deterministic_id(seed, length=12)}"


def generate_account_flag_id(
    credit_card_account_id: str,
    flag_type: str,
    expiration_date: str,
) -> str:
    """Generate a deterministic account flag ID.

    Args:
        credit_card_account_id: The credit card account ID
        flag_type: The type of flag being applied
        expiration_date: The expiration date of the flag

    Returns:
        Account flag ID in format 'ccflag_xxxxxxxxxxxx'
    """
    seed = f"account_flag:{credit_card_account_id}:{flag_type}:{expiration_date}"
    return f"ccflag_{_deterministic_id(seed, length=12)}"


def generate_credit_limit_increase_request_id(
    credit_card_account_id: str,
    user_id: str,
    requested_increase_amount: float,
) -> str:
    """Generate a deterministic credit limit increase request ID.

    Args:
        credit_card_account_id: The credit card account ID
        user_id: The user's ID
        requested_increase_amount: The requested increase amount

    Returns:
        CLI request ID in format 'cli_xxxxxxxxxxxx'
    """
    seed = f"cli_request:{credit_card_account_id}:{user_id}:{requested_increase_amount:.2f}"
    return f"cli_{_deterministic_id(seed, length=12)}"


def generate_bank_account_transaction_id(
    account_id: str,
    date: str,
    description: str,
    amount: float,
    transaction_type: str,
) -> str:
    """Generate a deterministic bank account transaction ID.

    Args:
        account_id: The bank account ID (e.g., 'chk_lj82d4f1a9')
        date: Transaction date in MM/DD/YYYY format
        description: Transaction description
        amount: Transaction amount (negative for debits, positive for credits)
        transaction_type: Type of transaction (e.g., 'atm_withdrawal', 'atm_fee', 'fee_rebate')

    Returns:
        Transaction ID in format 'btxn_xxxxxxxxxxxx'
    """
    seed = f"bank_txn:{account_id}:{date}:{description}:{amount:.2f}:{transaction_type}"
    return f"btxn_{_deterministic_id(seed, length=12)}"


def generate_debit_card_order_id(
    account_id: str,
    user_id: str,
    delivery_option: str,
) -> str:
    """Generate a deterministic debit card order ID.

    Args:
        account_id: The checking account ID the card is linked to
        user_id: The user's ID
        delivery_option: Delivery option (STANDARD, EXPEDITED, RUSH)

    Returns:
        Debit card order ID in format 'dcord_xxxxxxxxxxxx'
    """
    seed = f"debit_card_order:{account_id}:{user_id}:{delivery_option}"
    return f"dcord_{_deterministic_id(seed, length=12)}"


def generate_debit_card_id(
    account_id: str,
    user_id: str,
    issue_date: str,
) -> str:
    """Generate a deterministic debit card ID.

    Args:
        account_id: The checking account ID the card is linked to
        user_id: The user's ID
        issue_date: The date the card was issued (for uniqueness)

    Returns:
        Debit card ID in format 'dbc_xxxxxxxxxxxx'
    """
    seed = f"debit_card:{account_id}:{user_id}:{issue_date}"
    return f"dbc_{_deterministic_id(seed, length=12)}"
