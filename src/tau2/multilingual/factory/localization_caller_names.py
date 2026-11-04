"""Ground explicit caller-name statements in their structured database records."""

import re
from typing import Literal

from tau2.data_model.language import NameOrder
from tau2.multilingual.factory.localization_user_ids import customer_name_components
from tau2.multilingual.localization import (
    CallerNameRevisionProvenance,
    LocalizationEntry,
    LocalizationMap,
)
from tau2.utils.artifacts import content_digest

_NAME = re.compile(r"\A(?:You are|Your name is) (?P<name>[^\r\n.!?]+)\.", re.IGNORECASE)
_USER_ID = re.compile(r"\bYour user id is\s*:?\s*(?P<id>[A-Za-z0-9_]+)", re.IGNORECASE)


def localized_caller_entries(proposal: LocalizationMap) -> list[LocalizationEntry]:
    """Replace only a source-identified caller name; reject ambiguous declarations.

    A source entry must begin with a full-name statement and contain one explicit
    user-ID statement identifying the same DB record. Passenger names, other
    people's names, and unstructured claims are not inferred or replaced.
    """
    database = next(
        d.content for d in proposal.source.documents if d.kind == "database"
    )
    users = database.get("users", {})
    if not users:
        return proposal.entries
    aliases = (
        {a.canonical: a.localized for a in proposal.user_id_revision.aliases}
        if proposal.user_id_revision
        else {}
    )
    names: dict[str, dict[str, list[str]]] = {}

    def rendered_name(
        canonical: str,
        representation: Literal["native", "romanized"],
        current: str,
    ) -> str:
        if representation not in names:
            names[representation] = customer_name_components(proposal, representation)
        parts = names[representation][canonical]
        if proposal.settings.name_order == NameOrder.FAMILY_FIRST:
            parts = list(reversed(parts))
        # Retain the authored script's spacing convention; never infer a new
        # transliteration, substitute a passenger, or manufacture a name part.
        separator = " " if re.search(r"\s", current) else ""
        return separator.join(parts)

    result = []
    for entry in proposal.entries:
        if entry.category != "prose" or not isinstance(entry.source, str):
            result.append(entry)
            continue
        source_name = _NAME.match(entry.source)
        source_ids = list(_USER_ID.finditer(entry.source))
        if source_name is None or not source_ids:
            result.append(entry)
            continue
        if len(source_ids) != 1:
            raise ValueError(f"Ambiguous source caller-ID declarations: {entry.id}")
        canonical = source_ids[0].group("id")
        record = users.get(canonical)
        if record is None:
            raise ValueError(f"Source caller ID has no customer record: {entry.id}")
        source_full = " ".join(
            record["name"][role] for role in ("first_name", "last_name")
        )
        if source_name.group("name") != source_full:
            raise ValueError(
                f"Source caller name disagrees with its customer record; requires review: {entry.id}"
            )
        forms = entry.forms.model_copy(deep=True)
        changed = False
        review_names = {}
        ambiguous_names = set()
        for representation in ("native", "romanized"):
            text = getattr(forms, representation)
            target_name = _NAME.match(text) if isinstance(text, str) else None
            target_ids = list(_USER_ID.finditer(text)) if isinstance(text, str) else []
            if (
                target_name is None
                or len(target_ids) != 1
                or target_ids[0].group("id") not in {canonical, aliases.get(canonical)}
            ):
                raise ValueError(
                    f"Localized caller declaration is missing or ambiguous: {entry.id}"
                )
            name = rendered_name(canonical, representation, target_name.group("name"))
            start, end = target_name.span("name")
            replacement = text[:start] + name + text[end:]
            if replacement != text:
                setattr(forms, representation, replacement)
                old_name = target_name.group("name")
                if old_name in text[:start] + text[end:]:
                    ambiguous_names.add(old_name)
                if old_name in review_names and review_names[old_name] != name:
                    ambiguous_names.add(old_name)
                review_names[old_name] = name
                changed = True
        notes = entry.notes
        if changed:
            # These exact full names were identified as the caller above. If the
            # same name also describes someone else, metadata needs human review.
            if any(old in forms.english or old in notes for old in ambiguous_names):
                raise ValueError(
                    f"Ambiguous caller-name review explanation: {entry.id}"
                )
            pattern = re.compile(
                r"(?<!\w)(?:"
                + "|".join(
                    re.escape(old)
                    for old in sorted(review_names, key=len, reverse=True)
                )
                + r")(?!\w)"
            )
            forms.english = pattern.sub(
                lambda match: review_names[match.group()], forms.english
            )
            notes = pattern.sub(lambda match: review_names[match.group()], notes)
        result.append(
            entry.model_copy(
                update={
                    "forms": forms,
                    "operation": "adapt",
                    "notes": (
                        notes
                        + " Explicit caller name synchronized from its structured localized customer record; all other instruction text retained."
                    ).strip(),
                }
            )
            if changed
            else entry
        )
    return result


def finalize_localization_caller_names(proposal: LocalizationMap) -> LocalizationMap:
    """Record a deterministic caller-name repair separately from model authorship."""
    entries = localized_caller_entries(proposal)
    count = sum(
        before != after for before, after in zip(proposal.entries, entries, strict=True)
    )
    if not count and proposal.caller_name_revision is not None:
        return proposal
    if count and proposal.caller_name_revision is not None:
        raise ValueError(
            "Previously synchronized caller names have changed; requires review"
        )
    return proposal.model_copy(
        update={
            "entries": entries,
            "caller_name_revision": CallerNameRevisionProvenance(
                parent_map_sha256=content_digest(proposal), changed_entry_count=count
            ),
        }
    )
