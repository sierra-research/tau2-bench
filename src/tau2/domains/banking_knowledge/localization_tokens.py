"""Keep locally assembled banking reference forms out of model requests."""

import copy
import hashlib
import re
from typing import Literal

from pydantic import BaseModel, Field, JsonValue, create_model, model_serializer

from tau2.multilingual.address_format import AddressForms
from tau2.multilingual.localization import LocalizationModel

_TOKEN = re.compile(r"__LOCALIZED_[A-Za-z0-9_]+__")


class BankingReferenceTokens(LocalizationModel):
    """Local-only exact substitutions, separated by written representation."""

    native: dict[str, str] = Field(default_factory=dict)
    romanized: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def protect(cls, payload: dict) -> "BankingReferenceTokens":
        """Replace completed localized forms with opaque tokens in an outgoing payload."""
        result = cls()
        for entry in payload["completed_mappings"]:
            digest = hashlib.sha256(entry["id"].encode()).hexdigest()[:24]
            forms = entry["forms"]
            for form, suffix in (("native", "N"), ("romanized", "R")):
                value = forms[form]
                if not isinstance(value, str):
                    raise ValueError(
                        "Banking completed reference forms must be strings"
                    )
                token = f"__LOCALIZED_{digest}_{suffix}__"
                getattr(result, form)[token] = value
                forms[form] = token
            forms["english"] = "Exact localized reference retained locally."
            # Component metadata contains the same private values as the display
            # forms; it is not an unprotected second donor-data channel.
            forms.pop("address", None)
        return result

    def restore(self, response: dict) -> dict:
        """Expand only known tokens in their matching representation, without guessing."""
        result = copy.deepcopy(response)

        def expand(value: JsonValue, substitutions: dict[str, str]) -> JsonValue:
            if isinstance(value, dict):
                return {key: expand(item, substitutions) for key, item in value.items()}
            if isinstance(value, list):
                return [expand(item, substitutions) for item in value]
            if not isinstance(value, str):
                return value

            def replace(match: re.Match) -> str:
                token = match.group()
                if token not in substitutions:
                    raise ValueError(
                        f"Unknown or wrong-form banking reference token {token}"
                    )
                return substitutions[token]

            return _TOKEN.sub(replace, value)

        for entry in result["entries"]:
            for form in ("native", "romanized"):
                entry["forms"][form] = expand(entry["forms"][form], getattr(self, form))
                if entry["forms"].get("address") is not None:
                    entry["forms"]["address"][form] = expand(
                        entry["forms"]["address"][form], getattr(self, form)
                    )
            combined = self.native | self.romanized
            entry["forms"]["english"] = expand(entry["forms"]["english"], combined)
            entry["notes"] = expand(entry["notes"], combined)
        return result


class BankingTextForms(LocalizationModel):
    """String-only forms matching banking's authored scalar surfaces."""

    native: str = Field(
        description="Native localized value with native reference tokens."
    )
    romanized: str = Field(
        description="Matching Latin-script value with romanized tokens."
    )
    english: str = Field(description="English explanation of the localized value.")
    address: AddressForms | None = Field(
        default=None,
        description="Explicit components for complete postal addresses; null for other values.",
    )

    @model_serializer(mode="wrap")
    def serialize_address(self, handler):
        """Keep scalar-only generated forms free of unused address metadata."""
        value = handler(self)
        if self.address is None:
            value.pop("address", None)
        return value


class BankingGeneratedValue(LocalizationModel):
    """One requested value; its exact ID is the enclosing required object key."""

    forms: BankingTextForms
    operation: Literal["adapt", "translate", "preserve"]
    notes: str = Field(
        description="English rationale and remaining native-review concerns."
    )


BANKING_RESPONSE_SCHEMA_VERSION = "banking-string-entries-v2"


def banking_response_schema(model: type[BaseModel]) -> dict:
    """Derive a strict provider schema, including nullable/defaulted components."""
    schema = model.model_json_schema()

    def strict(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for value in node.values():
                strict(value)
        elif isinstance(node, list):
            for value in node:
                strict(value)

    strict(schema)
    return schema


def banking_response_model(identifiers: list[str]) -> type[BaseModel]:
    """Generate a strict provider contract requiring every exact ID once as an object key."""
    entries = create_model(
        "BankingRequestedEntries",
        __base__=LocalizationModel,
        **{
            identifier: (BankingGeneratedValue, Field(...))
            for identifier in identifiers
        },
    )
    return create_model(
        "BankingRequestedBatch",
        __base__=LocalizationModel,
        entries=(entries, Field(...)),
    )
