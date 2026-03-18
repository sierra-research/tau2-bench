# Database Query Guide

This guide shows how to write custom database functions using `db_query.py`.

## Basic Usage

```python
from tau2.domains.banking_knowledge.db_query import query_db, add_to_db
```

## Writing Custom Functions

### Simple Query Wrapper

```python
def get_credit_card_applications_by_user(user_id: int):
    return query_db("past_credit_card_applications", user_id=user_id)

# Usage:
apps = get_credit_card_applications_by_user(1)
# Returns: [{'card_type': 'gold', 'request_id': 2, 'user_id': 1}]
```

### With Comparison Operators

```python
def get_high_value_transactions(min_amount: float):
    return query_db("transactions", amount__gt=min_amount)

def get_recent_applications(user_id: int, min_request_id: int):
    return query_db(
        "past_credit_card_applications",
        user_id=user_id,
        request_id__gt=min_request_id
    )
```

### As a Tool (for LLM use)

```python
from tau2.domains.banking_knowledge.db_query import query_database_tool

def get_user_applications_tool(user_id: int) -> str:
    return query_database_tool(
        "past_credit_card_applications",
        f'{{"user_id": {user_id}}}'
    )
```

## Available Operators

| Operator | Example | Meaning |
|----------|---------|---------|
| (none) | `status="active"` | Exact match |
| `__gt` | `amount__gt=100` | Greater than |
| `__gte` | `amount__gte=100` | Greater than or equal |
| `__lt` | `amount__lt=100` | Less than |
| `__lte` | `amount__lte=100` | Less than or equal |
| `__ne` | `status__ne="closed"` | Not equal |
| `__contains` | `name__contains="john"` | Substring match |
| `__startswith` | `name__startswith="J"` | Starts with |
| `__endswith` | `email__endswith="@gmail.com"` | Ends with |
| `__in` | `status__in=["a","b"]` | Value in list |
| `__nin` | `status__nin=["x","y"]` | Value not in list |
