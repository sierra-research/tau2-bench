# Copyright Sierra
"""Data paths, the pinned clock, and the dual-form entity payload contract
for the intake domain.

The clock is pinned (the telecom pinned-clock pattern) so ``get_today`` and
anything else time-derived is deterministic and survives the evaluator's
golden-action replay + DB-hash comparison. Dates are an entity family, so
the pinned clock stays even though nothing stamps timestamps at write time
anymore (design v4 §5.1).
"""

import re
from datetime import datetime

from tau2.utils.utils import DATA_DIR

INTAKE_DATA_DIR = DATA_DIR / "tau2" / "domains" / "intake"
INTAKE_DB_PATH = INTAKE_DATA_DIR / "db.toml"
INTAKE_USER_DB_PATH = INTAKE_DATA_DIR / "user_db.toml"
INTAKE_MAIN_POLICY_PATH = INTAKE_DATA_DIR / "main_policy.md"
INTAKE_TEXT_POLICY_PATH = INTAKE_DATA_DIR / "main_policy_text.md"
INTAKE_TASK_SET_PATH = INTAKE_DATA_DIR / "tasks.json"

# The dual-form entity payload for spoken-form values (relative dates, call
# frame 4.5.0): the sim leads with the spoken form and gives the real
# written value only when the caller asks for the exact date. Without it the
# sim has nothing but the phrase and spells THAT out letter by letter
# (2026-08-26 smoke, intake_dates_hard_01). The annotation is a fixed,
# machine-parseable contract: the generator renders it, and complications
# draw against the SPOKEN part only (what the sim actually says), so the
# written annotation can never flip a feasibility gate or perturb a draw.
ENTITY_DUAL_FORM_TEMPLATE = (
    "{spoken} (the real date this refers to, from your records: {value})"
)
_ENTITY_DUAL_FORM_PATTERN = re.compile(
    r"^(?P<spoken>.+) \(the real date this refers to, from your records: "
    r"(?P<value>\d{4}-\d{2}-\d{2})\)$"
)


def render_dual_form(spoken: str, value: str) -> str:
    """The ``get_entity`` payload for a spoken-form value."""
    rendered = ENTITY_DUAL_FORM_TEMPLATE.format(spoken=spoken, value=value)
    if spoken_form(rendered) != spoken:
        raise ValueError(
            f"Dual-form payload for {spoken!r} / {value!r} does not round-trip"
        )
    return rendered


def spoken_form(entity_value: str) -> str:
    """The part of an entity payload the sim speaks: strips the dual-form
    annotation; identity for plain single-form values."""
    match = _ENTITY_DUAL_FORM_PATTERN.match(entity_value)
    return match.group("spoken") if match else entity_value


def get_now() -> datetime:
    # The intake desk clock is pinned: assume now is 2025-06-12 10:30:00.
    return datetime(2025, 6, 12, 10, 30, 0)
