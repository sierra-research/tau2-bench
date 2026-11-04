"""Pinned diagnostic ideas for authoring, never evidence of judge validation."""

from pathlib import Path
from typing import Annotated

from pydantic import Field, TypeAdapter

from tau2.data_model.judges import ContractModel

# Content digest (SHA-256) of nativeness_authoring_catalog.json; identifies the
# pinned catalog snapshot. tests/test_judges/test_authoring_catalog_pin.py checks it.
SOURCE_REVISION = "4535edcdb13c83df5edc2fc5cbcc1f1502dff6364a3c954937ce158a4b84955a"


class CatalogFactor(ContractModel):
    """Frozen authoring selection metadata, not a calibrated judge contract."""

    factor_id: Annotated[str, Field(description="Closed source-catalog identifier.")]
    category: Annotated[str, Field(description="Source language-feature category.")]
    canonical_question: Annotated[str, Field(description="Unmodified source question.")]
    opportunity: Annotated[str, Field(description="Source applicability description.")]
    allowed: Annotated[str, Field(description="Source passing behavior.")]
    violation: Annotated[str, Field(description="Source failing behavior.")]
    default_severity: Annotated[
        int, Field(description="Source authoring metadata only.")
    ]


def authoring_catalog() -> list[CatalogFactor]:
    """Read the pinned donor's closed selection catalog without research imports."""
    return TypeAdapter(list[CatalogFactor]).validate_json(
        Path(__file__).with_name("nativeness_authoring_catalog.json").read_text()
    )
