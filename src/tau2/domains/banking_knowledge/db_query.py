"""Database query utilities for the knowledge domain.

This module provides functions to query and modify a TransactionalDB instance
with flexible constraint-based filtering. All operations are in-memory.

Usage:
    from tau2.domains.banking_knowledge.db_query import query_db, add_to_db
    from tau2.domains.banking_knowledge.data_model import TransactionalDB

    db = TransactionalDB.load("db.json")
    records = query_db("users", db=db, status="active")
"""

import json
import operator
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from tau2.domains.banking_knowledge.data_model import TransactionalDB


# =============================================================================
# Internal helper functions
# =============================================================================


def _load_databases(db: "TransactionalDB") -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Convert TransactionalDB to dict format for querying.

    Args:
        db: TransactionalDB instance

    Returns:
        Dictionary mapping database names to their records.
    """
    result = {
        "users": {"data": db.users.data, "notes": db.users.notes},
        "accounts": {"data": db.accounts.data, "notes": db.accounts.notes},
        "referrals": {"data": db.referrals.data, "notes": db.referrals.notes},
        "credit_card_applications": {
            "data": db.credit_card_applications.data,
            "notes": db.credit_card_applications.notes,
        },
    }
    # Add user discoverable tools tables if they exist
    if hasattr(db, "user_discoverable_tools"):
        result["user_discoverable_tools"] = {
            "data": db.user_discoverable_tools.data,
            "notes": db.user_discoverable_tools.notes,
        }
    if hasattr(db, "user_discoverable_tool_calls"):
        result["user_discoverable_tool_calls"] = {
            "data": db.user_discoverable_tool_calls.data,
            "notes": db.user_discoverable_tool_calls.notes,
        }
    # Add agent discoverable tools tables if they exist
    if hasattr(db, "agent_discoverable_tools"):
        result["agent_discoverable_tools"] = {
            "data": db.agent_discoverable_tools.data,
            "notes": db.agent_discoverable_tools.notes,
        }
    if hasattr(db, "task_config"):
        result["task_config"] = {
            "data": db.task_config.data,
            "notes": db.task_config.notes,
        }
    if hasattr(db, "verification_history"):
        result["verification_history"] = {
            "data": db.verification_history.data,
            "notes": db.verification_history.notes,
        }
    if hasattr(db, "credit_card_transaction_history"):
        result["credit_card_transaction_history"] = {
            "data": db.credit_card_transaction_history.data,
            "notes": db.credit_card_transaction_history.notes,
        }
    if hasattr(db, "cash_back_disputes"):
        result["cash_back_disputes"] = {
            "data": db.cash_back_disputes.data,
            "notes": db.cash_back_disputes.notes,
        }
    if hasattr(db, "bank_account_transaction_history"):
        result["bank_account_transaction_history"] = {
            "data": db.bank_account_transaction_history.data,
            "notes": db.bank_account_transaction_history.notes,
        }
    if hasattr(db, "credit_card_accounts"):
        result["credit_card_accounts"] = {
            "data": db.credit_card_accounts.data,
            "notes": db.credit_card_accounts.notes,
        }
    if hasattr(db, "human_transfer_requests"):
        result["human_transfer_requests"] = {
            "data": db.human_transfer_requests.data,
            "notes": db.human_transfer_requests.notes,
        }
    if hasattr(db, "transaction_disputes"):
        result["transaction_disputes"] = {
            "data": db.transaction_disputes.data,
            "notes": db.transaction_disputes.notes,
        }
    if hasattr(db, "credit_card_orders"):
        result["credit_card_orders"] = {
            "data": db.credit_card_orders.data,
            "notes": db.credit_card_orders.notes,
        }
    if hasattr(db, "credit_card_closure_reasons"):
        result["credit_card_closure_reasons"] = {
            "data": db.credit_card_closure_reasons.data,
            "notes": db.credit_card_closure_reasons.notes,
        }
    if hasattr(db, "credit_card_account_flags"):
        result["credit_card_account_flags"] = {
            "data": db.credit_card_account_flags.data,
            "notes": db.credit_card_account_flags.notes,
        }
    if hasattr(db, "debit_cards"):
        result["debit_cards"] = {
            "data": db.debit_cards.data,
            "notes": db.debit_cards.notes,
        }
    if hasattr(db, "debit_card_orders"):
        result["debit_card_orders"] = {
            "data": db.debit_card_orders.data,
            "notes": db.debit_card_orders.notes,
        }
    if hasattr(db, "debit_card_disputes"):
        result["debit_card_disputes"] = {
            "data": db.debit_card_disputes.data,
            "notes": db.debit_card_disputes.notes,
        }
    if hasattr(db, "credit_limit_increase_requests"):
        result["credit_limit_increase_requests"] = {
            "data": db.credit_limit_increase_requests.data,
            "notes": db.credit_limit_increase_requests.notes,
        }
    if hasattr(db, "payment_history"):
        result["payment_history"] = {
            "data": db.payment_history.data,
            "notes": db.payment_history.notes,
        }
    return result


def _get_comparison_op(op_name: str) -> Callable[[Any, Any], bool]:
    """Get a comparison function by operator name.

    Supported operators:
        - eq: equal (default)
        - ne: not equal
        - gt: greater than
        - gte: greater than or equal
        - lt: less than
        - lte: less than or equal
        - contains: substring match (for strings) or item in list
        - startswith: string starts with
        - endswith: string ends with
        - in: value is in a list of options
        - nin: value is not in a list of options

    Args:
        op_name: Name of the operator

    Returns:
        Comparison function
    """
    ops = {
        "eq": operator.eq,
        "ne": operator.ne,
        "gt": operator.gt,
        "gte": operator.ge,
        "lt": operator.lt,
        "lte": operator.le,
        "contains": lambda a, b: b in a if a is not None else False,
        "startswith": lambda a, b: str(a).startswith(str(b))
        if a is not None
        else False,
        "endswith": lambda a, b: str(a).endswith(str(b)) if a is not None else False,
        "in": lambda a, b: a in b,
        "nin": lambda a, b: a not in b,
    }
    return ops.get(op_name, operator.eq)


def _parse_constraint(key: str, value: Any) -> tuple[str, str, Any]:
    """Parse a constraint key into field name and operator.

    Args:
        key: Constraint key, e.g., "amount__gt" or "status"
        value: The value to compare against

    Returns:
        Tuple of (field_name, operator_name, value)
    """
    if "__" in key:
        parts = key.rsplit("__", 1)
        field_name = parts[0]
        op_name = parts[1]
    else:
        field_name = key
        op_name = "eq"

    return field_name, op_name, value


def _record_matches(record: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
    """Check if a record matches all constraints.

    Args:
        record: The record to check
        constraints: Dictionary of constraints (field__op: value)

    Returns:
        True if record matches all constraints
    """
    for key, value in constraints.items():
        field_name, op_name, expected = _parse_constraint(key, value)

        # Get the actual value from record
        actual = record.get(field_name)

        # Get comparison function
        compare = _get_comparison_op(op_name)

        # Perform comparison
        try:
            if not compare(actual, expected):
                return False
        except (TypeError, ValueError):
            # Comparison failed (e.g., comparing incompatible types)
            return False

    return True


# =============================================================================
# Query functions
# =============================================================================


def list_databases(db: "TransactionalDB") -> List[str]:
    """List all available database names.

    Args:
        db: TransactionalDB instance

    Returns:
        List of database names
    """
    databases = _load_databases(db)
    return list(databases.keys())


def get_database(
    db_name: str, db: "TransactionalDB"
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Get all records from a database.

    Args:
        db_name: Name of the database
        db: TransactionalDB instance

    Returns:
        Dictionary mapping record IDs to records, or None if database not found
    """
    databases = _load_databases(db)
    db_entry = databases.get(db_name)
    if db_entry is None:
        return None
    # Handle both old format (just data) and new format (data + notes)
    if isinstance(db_entry, dict) and "data" in db_entry:
        return db_entry["data"]
    return db_entry


def query_db(
    db_name: str,
    db: "TransactionalDB",
    return_ids: bool = False,
    limit: Optional[int] = None,
    **constraints,
) -> Union[List[Dict[str, Any]], List[tuple[str, Dict[str, Any]]]]:
    """Query a database with flexible constraints.

    Supports exact matches and comparison operators:
        - field=value: exact match
        - field__eq=value: exact match (explicit)
        - field__ne=value: not equal
        - field__gt=value: greater than
        - field__gte=value: greater than or equal
        - field__lt=value: less than
        - field__lte=value: less than or equal
        - field__contains=value: substring match or item in list
        - field__startswith=value: string starts with
        - field__endswith=value: string ends with
        - field__in=[values]: value is in list
        - field__nin=[values]: value is not in list

    Args:
        db_name: Name of the database to query
        db: TransactionalDB instance
        return_ids: If True, returns list of (record_id, record) tuples
        limit: Maximum number of results to return
        **constraints: Field constraints as keyword arguments

    Returns:
        List of matching records (or tuples with IDs if return_ids=True)

    Examples:
        >>> query_db("users", db=my_db, status="active")
        >>> query_db("accounts", db=my_db, return_ids=True, type="checking")
    """
    database = get_database(db_name, db)
    if database is None:
        return []

    results = []
    for record_id, record in database.items():
        if _record_matches(record, constraints):
            if return_ids:
                results.append((record_id, record))
            else:
                results.append(record)

            if limit is not None and len(results) >= limit:
                break

    return results


def remove_from_db(
    db_name: str, db: "TransactionalDB", **constraints
) -> List[Dict[str, Any]]:
    """Remove records from a database, based on constraints.

    Supports exact matches and comparison operators:
        - field=value: exact match
        - field__eq=value: exact match (explicit)
        - field__ne=value: not equal
        - field__gt=value: greater than
        - field__gte=value: greater than or equal
        - field__lt=value: less than
        - field__lte=value: less than or equal
        - field__contains=value: substring match or item in list
        - field__startswith=value: string starts with
        - field__endswith=value: string ends with
        - field__in=[values]: value is in list
        - field__nin=[values]: value is not in list

    Args:
        db_name: Name of the database
        db: TransactionalDB instance
        **constraints: Field constraints

    Returns:
        List of removed records
    """
    table = getattr(db, db_name, None)
    if table is None:
        return []

    results = []
    to_pop = []
    for record_id, record in table.data.items():
        if _record_matches(record, constraints):
            results.append(record)
            to_pop.append(record_id)

    for record_id in to_pop:
        del table.data[record_id]

    return results


def add_to_db(
    db_name: str, record_id: str, record: Dict[str, Any], db: "TransactionalDB"
) -> bool:
    """Add a record to a database.

    Args:
        db_name: Name of the database
        record_id: ID for the new record
        record: The record data to add
        db: TransactionalDB instance

    Returns:
        True if successful, False if database not found or record already exists
    """
    table = getattr(db, db_name, None)
    if table is None:
        return False

    if record_id in table.data:
        return False  # Record already exists

    table.data[record_id] = record
    return True


def update_record_in_db(
    db_name: str, db: "TransactionalDB", record_id: str, updates: Dict[str, Any]
) -> tuple[bool, Optional[Dict[str, Any]]]:
    """Update fields in an existing record.

    Args:
        db_name: Name of the database
        db: TransactionalDB instance
        record_id: ID of the record to update
        updates: Dictionary of field names to new values

    Returns:
        Tuple of (success, updated_record). If record not found, returns (False, None).
    """
    table = getattr(db, db_name, None)
    if table is None:
        return False, None

    if record_id not in table.data:
        return False, None

    # Update the fields
    for field, value in updates.items():
        table.data[record_id][field] = value

    return True, table.data[record_id]


# ============================================================================
# Tool wrapper functions (for use by KnowledgeTools and KnowledgeUserTools)
# ============================================================================


def query_database_tool(
    database_name: str, constraints: str = "{}", db: "TransactionalDB" = None
) -> str:
    """Tool wrapper for query_db - handles JSON parsing and formatting.

    Args:
        database_name: Name of the database to query
        constraints: JSON string of field constraints
        db: TransactionalDB instance
    """
    if db is None:
        return "Error: TransactionalDB instance required"

    try:
        available_dbs = list_databases(db)
        if database_name not in available_dbs:
            return f"Error: Database '{database_name}' not found. Available: {available_dbs}"

        try:
            constraint_dict = json.loads(constraints) if constraints else {}
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON: {e}"

        results = query_db(database_name, db=db, return_ids=True, **constraint_dict)

        if not results:
            return f"No records found in '{database_name}'."

        formatted_lines = [f"Found {len(results)} record(s) in '{database_name}':\n"]
        for i, (record_id, record) in enumerate(results, 1):
            formatted_lines.append(f"{i}. Record ID: {record_id}")
            for field, value in record.items():
                formatted_lines.append(f"   {field}: {value}")
            formatted_lines.append("")

        return "\n".join(formatted_lines)

    except Exception as e:
        return f"Error querying database: {str(e)}"


def remove_from_database_tool(
    database_name: str, constraints: str = "{}", db: "TransactionalDB" = None
) -> str:
    """Tool wrapper for remove_from_database - handles JSON parsing and formatting.

    Args:
        database_name: Name of the database to remove from
        constraints: JSON string of field constraints
        db: TransactionalDB instance
    """
    if db is None:
        return "Error: TransactionalDB instance required"

    try:
        available_dbs = list_databases(db)
        if database_name not in available_dbs:
            return f"Error: Database '{database_name}' not found. Available: {available_dbs}"

        try:
            constraint_dict = json.loads(constraints) if constraints else {}
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON: {e}"

        results = remove_from_db(database_name, db=db, **constraint_dict)

        if not results:
            return f"No records found in '{database_name}'."

        formatted_lines = [
            f"Removed {len(results)} record(s) from '{database_name}':\n"
        ]
        for i, record in enumerate(results, 1):
            formatted_lines.append(f"{i}. Removed record:")
            for field, value in record.items():
                formatted_lines.append(f"   {field}: {value}")
            formatted_lines.append("")

        return "\n".join(formatted_lines)

    except Exception as e:
        return f"Error removing from database: {str(e)}"
