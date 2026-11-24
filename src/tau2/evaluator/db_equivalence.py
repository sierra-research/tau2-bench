"""Which DB end states the DB check treats as equal.

`get_dict_hash` keeps list order. Some lists are stored in the order of the
tool call arguments, and nothing gives that order a meaning. This module sorts
those lists before the DB check hashes the database (issue #736).

This is grading logic only, so it lives with the evaluators. Every source file
under a domain package feeds `runtime_rules_digest`, so a change there would
change the released binding revisions.
"""

import json
from typing import Any, Callable, Optional

from tau2.environment.environment import Environment
from tau2.utils import get_dict_hash

# Airline: `book_reservation` stores `passengers` and `payment_history` in the
# order of its arguments, and no tool or policy rule reads that order. The
# order of `flights` is the itinerary, so it is kept.
AIRLINE_ORDER_FREE_RESERVATION_FIELDS = ("passengers", "payment_history")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _airline_hash_data(data: dict[str, Any]) -> dict[str, Any]:
    for reservation in data["reservations"].values():
        for field in AIRLINE_ORDER_FREE_RESERVATION_FIELDS:
            reservation[field] = sorted(reservation[field], key=_canonical_json)
    return data


_HASH_DATA: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "airline": _airline_hash_data,
}


def db_check_hash(environment: Environment) -> Optional[str]:
    """Hash the assistant database the way the DB check compares it.

    Domains without an entry hash the same data as `Environment.get_db_hash`.
    """
    hash_data = _HASH_DATA.get(environment.domain_name)
    if hash_data is None or environment.tools is None:
        return environment.get_db_hash()
    return get_dict_hash(hash_data(environment.tools.db.model_dump()))
