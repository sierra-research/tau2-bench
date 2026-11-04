"""Shared identity components for deterministic regional identifiers."""

import json
import re
import unicodedata

from tau2.multilingual.localization import LocalizationEntry, SourceEntry


def ascii_name_parts(parts: list[str]) -> list[str]:
    """Normalize romanized name components without inventing or reordering them."""
    normalized = [
        re.sub(
            r"[^a-z0-9]",
            "",
            unicodedata.normalize("NFKD", part)
            .encode("ascii", "ignore")
            .decode()
            .lower(),
        )
        for part in parts
    ]
    if not normalized or any(not part for part in normalized):
        raise ValueError("Romanized name cannot form nonempty ASCII components")
    return normalized


def _context_value(entry: SourceEntry, key: str) -> str | None:
    match = re.search(rf"(?:^|;\s*){key}=([^;]+)", entry.context)
    return match.group(1) if match else None


def romanized_identity_parts(
    entry: SourceEntry, completed: list[LocalizationEntry]
) -> list[str] | None:
    """Read completed components for the exact source identity, rejecting ambiguity."""
    marker = "source_identity="
    if marker not in entry.context:
        return None
    identity, _ = json.JSONDecoder().raw_decode(entry.context.split(marker, 1)[1])
    roles = ["full_name"] if identity.get("full_name") else ["first_name", "last_name"]
    owner = _context_value(entry, "entity")
    parts = []
    for role in roles:
        matches = [
            item
            for item in completed
            if item.category == "identity_name"
            and _context_value(item, "role") == role
            and item.source == identity.get(role)
        ]
        owned = [item for item in matches if _context_value(item, "entity") == owner]
        if owner and owned:
            matches = owned
        values = {item.forms.romanized for item in matches}
        if not values:
            return None
        if len(values) != 1:
            raise ValueError("Source identity has ambiguous localized name components")
        value = values.pop()
        if not isinstance(value, str):
            raise ValueError("Localized name component must be text")
        parts.extend(value.split() if role == "full_name" else [value])
    return ascii_name_parts(parts)
