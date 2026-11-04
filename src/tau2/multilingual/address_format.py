"""Small semantic address contract shared by regional authoring and reuse."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ADDRESS_FORMAT_VERSION = "regional-address-format-v1"


class AddressParts(BaseModel):
    """Exact local components; storage roles are independent of display order."""

    model_config = ConfigDict(extra="forbid")
    street_lines: list[str] = Field(
        description="Road/premise and optional building/unit lines, excluding named administrative components."
    )
    district: str = Field(
        default="",
        description="Explicit district or submunicipal component when separately known.",
    )
    city: str = Field(
        default="",
        description="Explicit city/municipality; never inferred from a comma position at runtime.",
    )
    region: str = Field(
        default="",
        description="Explicit province/state/metropolitan region; empty when absent in evidence.",
    )
    postal_code: str = Field(
        default="", description="Exact postal identifier, retaining leading zeros."
    )
    country: str = Field(
        default="", description="Country when supplied by the address context."
    )


class AddressForms(BaseModel):
    """Native and romanized components of one address identity."""

    model_config = ConfigDict(extra="forbid")
    language: str = Field(
        description="Language whose local reference ordering applies."
    )
    native: AddressParts = Field(description="Native-script address components.")
    romanized: AddressParts = Field(
        description="Latin-script components of the same address."
    )


class AddressRevisionProvenance(BaseModel):
    """Deterministic address repair, separate from original model generation."""

    model_config = ConfigDict(extra="forbid")
    version: Literal["regional-address-format-v1"] = ADDRESS_FORMAT_VERSION
    parent_map_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Exact input map digest."
    )
    changed_entry_ids: list[str] = Field(
        description="Exact changed source-entry IDs, including dependent prose."
    )


def _localities(parts: AddressParts, language: str) -> list[str]:
    values = (
        [parts.region, parts.city, parts.district]
        if language in {"zh", "ko"}
        else [parts.district, parts.city, parts.region]
    )
    return list(dict.fromkeys(value for value in values if value))


def format_locality(parts: AddressParts, language: str, representation: str) -> str:
    """Render the city/region reference without street, district or postal details."""
    values = _localities(parts.model_copy(update={"district": ""}), language)
    separator = (
        ""
        if language == "zh" and representation == "native"
        else " "
        if language in {"zh", "ko"}
        else ", "
    )
    return separator.join(values)


def format_address(parts: AddressParts, language: str, representation: str) -> str:
    """Render a local written/caller reference, not an international envelope."""
    if representation not in {"native", "romanized"}:
        raise ValueError("Unknown address representation")
    streets = [line for line in parts.street_lines if line]
    if language in {"zh", "ko"}:
        separator = "" if language == "zh" and representation == "native" else " "
        text = separator.join(
            [
                value
                for value in [parts.country, *_localities(parts, language), *streets]
                if value
            ]
        )
        if parts.postal_code:
            text += (
                (
                    "，"
                    if language == "zh" and representation == "native"
                    else ", "
                    if language == "zh"
                    else " ("
                )
                + parts.postal_code
                + (")" if language == "ko" else "")
            )
        return text
    if language == "es":
        locality = " ".join(value for value in [parts.postal_code, parts.city] if value)
        tail = [
            parts.district if parts.district not in {parts.city, parts.region} else "",
            locality,
            parts.region if parts.region != parts.city else "",
            parts.country,
        ]
    else:
        tail = [*_localities(parts, language), parts.postal_code, parts.country]
    return ", ".join(value for value in [*streets, *tail] if value)
