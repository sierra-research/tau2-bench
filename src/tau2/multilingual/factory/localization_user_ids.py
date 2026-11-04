"""Deterministic customer identifiers and their explicitly declared references."""

import re
from copy import deepcopy
from typing import Literal

from pydantic import JsonValue

from tau2.multilingual.factory.localization_identity import ascii_name_parts
from tau2.multilingual.localization import (
    LocalizationEntry,
    LocalizationMap,
    UserIdAlias,
    UserIdRevisionProvenance,
)
from tau2.utils.artifacts import content_digest

_ID_FIELDS = frozenset({"user_id", "customer_id"})
_PROSE_FIELDS = frozenset(
    {
        "reason_for_call",
        "known_info",
        "unknown_info",
        "task_instructions",
        "instructions",
        "ticket",
        "purpose",
        "relevant_policies",
        "notes",
        "communicate_info",
        "nl_assertions",
        "message",
        "summary",
        "expression",
        "persona",
    }
)
_REFERENCE_NOTE = (
    "Customer/user IDs follow the romanized localized name and retain the source "
    "numeric suffix; task and unrelated machine identifiers stay unchanged."
)


def user_id_alias_mapping(aliases: list[UserIdAlias]) -> dict[str, str]:
    """Check one-to-one aliases, including overlap that would make replay ambiguous."""
    forward = {alias.canonical: alias.localized for alias in aliases}
    if len(forward) != len(aliases):
        raise ValueError("Duplicate canonical user ID aliases")
    if len(set(forward.values())) != len(forward):
        raise ValueError("Localized user ID collision")
    if any(value in forward and value != key for key, value in forward.items()):
        raise ValueError("Localized user ID overlaps another canonical user ID")
    return forward


def _replace_tokens(value: str, mapping: dict[str, str]) -> str:
    # Underscores belong to these machine tokens. Prefixes and unrelated IDs must
    # not change just because they happen to contain a customer's identifier.
    return re.sub(
        r"[A-Za-z0-9_]+",
        lambda match: mapping.get(match.group(), match.group()),
        value,
    )


def remap_user_id_references(
    value: JsonValue,
    aliases: list[UserIdAlias],
    *,
    path: tuple[str | int, ...] = (),
) -> JsonValue:
    """Rewrite customer fields, users keys, and exact tokens in instruction fields.

    Calling this twice is harmless. Other dictionary keys and identifiers are
    untouched, even when their spelling equals a customer ID.
    """
    mapping = user_id_alias_mapping(aliases)

    def visit(item: JsonValue, current: tuple[str | int, ...]) -> JsonValue:
        field = current[-1] if current else None
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                target = mapping.get(key, key) if field == "users" else key
                if target in result:
                    raise ValueError("User ID replacement collides with a users key")
                result[target] = visit(child, (*current, key))
            return result
        if isinstance(item, list):
            return [visit(child, (*current, index)) for index, child in enumerate(item)]
        if isinstance(item, str):
            if field in _ID_FIELDS:
                return mapping.get(item, item)
            if any(part in _PROSE_FIELDS for part in current if isinstance(part, str)):
                return _replace_tokens(item, mapping)
        return deepcopy(item)

    return visit(value, path)


def customer_name_components(
    proposal: LocalizationMap, representation: Literal["native", "romanized"]
) -> dict[str, list[str]]:
    """Read exact localized name occurrences, preserving their structured roles.

    Structured given/family roles determine airline and retail ID order. Telecom
    has one full-name field, whose recorded romanized token order is retained.
    Display-name ordering does not change structured first/last-name roles.
    """
    databases = [d for d in proposal.source.documents if d.kind == "database"]
    if len(databases) != 1:
        raise ValueError("User ID localization requires exactly one database")
    document = databases[0]
    database = document.content
    if not isinstance(database, dict):
        raise ValueError("Customer database must be an object")
    names = {}
    for entry in proposal.entries:
        if entry.category == "identity_name":
            for occurrence in entry.occurrences:
                key = (occurrence.document_id, *occurrence.path)
                if key in names:
                    raise ValueError("Duplicate localized name occurrence")
                names[key] = getattr(entry.forms, representation)

    def name_at(path: tuple[str | int, ...]) -> str:
        value = names.get((document.id, *path))
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing {representation} customer name at {path!r}")
        return value

    rows = []
    if proposal.source.domain in {"airline", "retail"}:
        users = database.get("users")
        if not isinstance(users, dict) or not users:
            raise ValueError("Expected customer records in users dictionary")
        for key, user in users.items():
            if not isinstance(user, dict) or user.get("user_id") != key:
                raise ValueError("Source user dictionary key differs from user_id")
            rows.append(
                (
                    key,
                    [
                        name_at(("users", key, "name", role))
                        for role in ("first_name", "last_name")
                    ],
                )
            )
    else:
        customers = database.get("customers")
        if not isinstance(customers, list) or not customers:
            raise ValueError("Expected telecom customer records")
        for index, customer in enumerate(customers):
            if not isinstance(customer, dict):
                raise ValueError("Expected telecom customer object")
            rows.append(
                (
                    customer.get("customer_id"),
                    name_at(("customers", index, "full_name")).split(),
                )
            )
    result = {}
    for canonical, parts in rows:
        if not isinstance(canonical, str) or not canonical or canonical in result:
            raise ValueError("Missing or duplicate canonical customer ID")
        result[canonical] = parts
    return result


def derive_user_id_aliases(proposal: LocalizationMap) -> list[UserIdAlias]:
    """Derive ASCII customer IDs from exact romanized names and source suffixes."""
    aliases = []
    for canonical, parts in customer_name_components(proposal, "romanized").items():
        suffix = (
            re.search(r"([0-9]+)$", canonical) if isinstance(canonical, str) else None
        )
        if suffix is None:
            raise ValueError("Customer ID must have an original numeric suffix")
        aliases.append(
            UserIdAlias(
                canonical=canonical,
                localized="_".join([*ascii_name_parts(parts), suffix.group(1)]),
            )
        )
    user_id_alias_mapping(aliases)
    return sorted(aliases, key=lambda alias: alias.canonical)


def localize_user_id_entry(
    entry: LocalizationEntry, aliases: list[UserIdAlias]
) -> LocalizationEntry:
    """Make every changed reference visible in its original map review row."""
    mapping = user_id_alias_mapping(aliases)

    def replace(value: JsonValue) -> JsonValue:
        if entry.category in {"prose", "policy", "billing"} and isinstance(value, str):
            return _replace_tokens(value, mapping)
        replacements = [
            remap_user_id_references(value, aliases, path=tuple(occurrence.path))
            for occurrence in entry.occurrences
        ]
        if any(replacement != replacements[0] for replacement in replacements[1:]):
            raise ValueError("User ID entry mixes customer and unrelated occurrences")
        return replacements[0]

    native, romanized = replace(entry.forms.native), replace(entry.forms.romanized)
    if native == entry.forms.native and romanized == entry.forms.romanized:
        return entry
    forms = entry.forms.model_copy(
        update={
            "native": native,
            "romanized": romanized,
            "english": (
                _REFERENCE_NOTE
                + " Earlier review explanation (before the customer-ID update): "
                + entry.forms.english
                if entry.category != "protocol"
                else "Customer/user identifier adapted consistently across linked records."
            ),
        }
    )
    return entry.model_copy(
        update={
            "forms": forms,
            "operation": "adapt",
            "notes": (
                _REFERENCE_NOTE
                + " Earlier review notes (before the customer-ID update): "
                + entry.notes
            ),
        }
    )


def finalize_localization_user_ids(proposal: LocalizationMap) -> LocalizationMap:
    """Derive name-based customer IDs for a fresh draft and record their aliases."""
    from tau2.multilingual.factory.localization_caller_names import (
        finalize_localization_caller_names,
    )
    from tau2.multilingual.factory.localization_compile import validate_localization

    proposal = validate_localization(proposal)
    proposal = finalize_localization_caller_names(proposal)
    aliases = derive_user_id_aliases(proposal)
    if proposal.user_id_revision is not None:
        if proposal.user_id_revision.aliases != aliases:
            raise ValueError("Recorded user ID aliases differ from localized names")
        return proposal
    entries = [localize_user_id_entry(entry, aliases) for entry in proposal.entries]
    result = proposal.model_copy(
        update={
            "entries": entries,
            "user_id_revision": UserIdRevisionProvenance(
                parent_map_sha256=content_digest(proposal),
                aliases=aliases,
                changed_entry_count=sum(
                    before != after
                    for before, after in zip(proposal.entries, entries, strict=True)
                ),
            ),
        }
    )
    return validate_localization(result)


def synchronize_revised_user_ids(proposal: LocalizationMap) -> LocalizationMap:
    """Refresh existing aliases after a bounded name revision, before validation.

    Restore old localized tokens to canonical IDs first, then apply newly derived
    aliases. This handles swaps without cascading one person's ID into another's.
    Fresh drafts still use finalize_localization_user_ids.
    """
    previous = proposal.user_id_revision
    if previous is None:
        return proposal
    aliases = derive_user_id_aliases(proposal)
    if aliases == previous.aliases:
        return proposal
    reverse = [
        UserIdAlias(canonical=a.localized, localized=a.canonical)
        for a in previous.aliases
    ]
    entries = []
    for entry in proposal.entries:
        canonical = localize_user_id_entry(entry, reverse)
        revised = localize_user_id_entry(canonical, aliases)
        # Avoid changing review notes for references whose final values are identical.
        if (revised.forms.native, revised.forms.romanized) == (
            entry.forms.native,
            entry.forms.romanized,
        ):
            revised = entry
        else:
            from tau2.multilingual.factory.localization_artifact import (
                automatic_localization,
            )
            from tau2.multilingual.localization import SourceEntry

            automatic = automatic_localization(
                SourceEntry.model_validate(
                    entry.model_dump(include=set(SourceEntry.model_fields))
                ),
                proposal.settings,
            )
            if automatic is not None:
                automatic = localize_user_id_entry(automatic, aliases)
                if (automatic.forms.native, automatic.forms.romanized) != (
                    revised.forms.native,
                    revised.forms.romanized,
                ):
                    raise ValueError(
                        "Revised ID references differ from automatic rules"
                    )
                revised = automatic
        entries.append(revised)
    return proposal.model_copy(
        update={
            "entries": entries,
            "user_id_revision": UserIdRevisionProvenance(
                parent_map_sha256=content_digest(proposal),
                aliases=aliases,
                changed_entry_count=sum(
                    a != b for a, b in zip(proposal.entries, entries, strict=True)
                ),
            ),
        }
    )
