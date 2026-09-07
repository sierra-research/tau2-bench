# Copyright Sierra
"""User (callee) data model for the intake domain (design v4, outbound).

The callee's knowledge is a flat map from field name to true value
(docs/designs/intake-domain.md §5.3): ``get_entity`` returns it verbatim,
always. The sim never holds a hard value in prose — values enter the
conversation only through the tool, preserving the environment-heavy /
prompt-light property. Nothing user-side writes, so the evaluator's user-DB
hash is trivially equal between the gold and predicted environments; the
reward lives entirely in the agent DB.

State is seeded per task via ``initialization_actions`` (one ``set_entities``
green seed call).
"""

from typing import Annotated, Dict

from pydantic import Field

from tau2.environment.db import DB


class IntakeUserDB(DB):
    """Database for the callee's side of the intake domain."""

    entities: Annotated[
        Dict[str, str],
        Field(
            default_factory=dict,
            description=(
                "The callee's true values, keyed by the record field names "
                "the agent will ask about."
            ),
        ),
    ]
