# Copyright Sierra
"""User (callee) toolkit for the intake domain (design v4, outbound).

One tool (docs/designs/intake-domain.md §5.3): ``get_entity(field)`` returns
the callee's true value for that field, always. It models the person checking
their own records; it never fails on a field the agent was sent to collect,
never returns a stale value, and never points to a document. The non-tool
``set_entities`` method is the green-seed derivation tasks replay through
``initialization_actions``.
"""

from tau2.domains.intake.user_data_model import IntakeUserDB
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class IntakeUserTools(ToolKitBase):
    """Tools the callee can use during a callback call (all READ)."""

    db: IntakeUserDB

    def __init__(self, db: IntakeUserDB) -> None:
        super().__init__(db)

    # ------------------------------------------------------------------
    # Environment derivation functions (initialization_actions, not tools)
    # ------------------------------------------------------------------

    def set_entities(self, entities: dict) -> None:
        """Green seed: replace the callee's true values wholesale."""
        if not isinstance(entities, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and v.strip()
            for k, v in entities.items()
        ):
            raise ValueError("entities must map field names to non-empty string values")
        self.db.entities = dict(entities)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @is_tool(ToolType.READ)
    def get_entity(self, field: str) -> str:
        """
        Look up the exact value of one of your own fields in your records
        (your papers, cards, and files). Use this before answering any factual
        question — your records are always right.

        Args:
            field: The field the caller asked about.

        Returns:
            The exact value from your records.

        Raises:
            ValueError: If you have no record of a field by that name.
        """
        value = self.db.entities.get(field)
        if value is None:
            known = ", ".join(sorted(self.db.entities)) or "(none)"
            raise ValueError(
                f"You have no record of a field named {field!r}. "
                f"Fields in your records: {known}"
            )
        return value
