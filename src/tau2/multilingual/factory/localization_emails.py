"""Deterministic, short regional emails with preserved lookup distinctions."""

import re

from pydantic import Field

from tau2.multilingual.factory.localization_identity import romanized_identity_parts
from tau2.multilingual.localization import (
    LocalizationEntry,
    LocalizationModel,
    LocalizedForms,
    SourceEntry,
    json_content_digest,
)
from tau2.utils.text_match import fold_for_match

EMAIL_RULES_VERSION = "regional-email-v1"
_EMAIL_TOKEN = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
SYNTHETIC_EMAIL_DOMAIN = "example.com"
_EMAIL_DOMAIN = re.compile(r"[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.[a-z]{2,}")


class EmailCandidate(LocalizationModel):
    """One source address and the corresponding localized name components."""

    source: str = Field(min_length=1, description="Canonical complete email address.")
    name_parts: list[str] = Field(
        min_length=1, description="ASCII romanized name components in identity order."
    )


class EmailAssignment(LocalizationModel):
    """One distinct source email and its assigned localized ASCII address."""

    source: str = Field(min_length=1, description="Canonical email lookup value.")
    value: str = Field(min_length=1, description="Localized email lookup value.")


def _stem_and_suffix(candidate: EmailCandidate) -> tuple[str, str]:
    local, separator, _ = candidate.source.rpartition("@")
    if not separator or not _EMAIL_TOKEN.fullmatch(candidate.source):
        raise ValueError(f"Unsupported source email: {candidate.source}")
    match = re.search(r"(\d+)$", local)
    suffix = match.group(1) if match else ""
    shape = local[: -len(suffix)] if suffix else local
    parts = candidate.name_parts
    if any(not re.fullmatch(r"[a-z0-9]+", part) for part in parts):
        raise ValueError("Email name components must be nonempty ASCII tokens")
    separators = re.findall(r"[._-]", shape)
    source_parts = re.split(r"[._-]", shape)
    if not separators:
        stem = "".join(parts)
    elif len(source_parts) == len(parts):
        # Initials remain initials; a missing dot stays missing in deliberate errors.
        adapted = [
            part[:1] if len(original) == 1 else part
            for original, part in zip(source_parts, parts)
        ]
        stem = adapted[0] + "".join(
            separator + part for separator, part in zip(separators, adapted[1:])
        )
    else:
        stem = separators[0].join(parts)
    if not suffix:
        suffix = f"{int(json_content_digest(fold_for_match(candidate.source))[:8], 16) % 10000:04d}"
    return stem, suffix


def email_domain(addresses: list[str]) -> str:
    """Return the one mailbox domain shared by reviewed addresses.

    The regional email domain is a property of reviewed data (a language may
    replace the canonical ``example.com``), so it is read from the addresses a
    map or database already uses. Zero or several domains are ambiguous.
    """
    domains = set()
    for address in addresses:
        if not _EMAIL_TOKEN.fullmatch(address):
            raise ValueError(f"Unsupported email: {address}")
        domains.add(address.rpartition("@")[2].lower())
    if len(domains) != 1:
        raise ValueError(
            f"Email domain is ambiguous: expected one, found {sorted(domains)}"
        )
    return domains.pop()


def allocate_emails(
    candidates: list[EmailCandidate],
    domain: str,
    reserved: list[EmailAssignment] | None = None,
) -> list[EmailAssignment]:
    """Preserve source numeric suffixes and resolve only actual lookup collisions.

    Every allocated address uses ``domain``. Allocation is stable under input
    reordering. Original candidates are reserved before probing four-digit
    alternatives, so a collision cannot steal another source's original number.
    Equal source values retain one cross-domain value.
    """
    if not _EMAIL_DOMAIN.fullmatch(domain):
        raise ValueError(f"Unsupported email domain: {domain}")
    by_source = {}
    aliases = {}
    for candidate in candidates:
        source = fold_for_match(candidate.source)
        proposed = _stem_and_suffix(candidate)
        if source in by_source and by_source[source] != proposed:
            raise ValueError("One source email has conflicting localized names")
        by_source[source] = proposed
        aliases[candidate.source] = source
    occupied = {}
    retained = {}
    for assignment in reserved or []:
        source = fold_for_match(assignment.source)
        key = fold_for_match(assignment.value)
        if key in occupied and occupied[key] != source:
            raise ValueError("Retained emails already contain an ambiguous lookup")
        if source in retained and retained[source] != assignment.value:
            raise ValueError(
                "One retained source email has conflicting localized values"
            )
        occupied[key] = source
        retained[source] = assignment.value
    original_values = {
        fold_for_match(f"{stem}{suffix}@{domain}")
        for stem, suffix in by_source.values()
    }
    result = {}
    for source, (stem, suffix) in sorted(by_source.items()):
        if source in retained:
            result[source] = retained[source]
            continue
        value = f"{stem}{suffix}@{domain}"
        if fold_for_match(value) in occupied:
            start = int(suffix) % 10000
            for offset in range(1, 10001):
                candidate = f"{stem}{(start + offset) % 10000:04d}@{domain}"
                key = fold_for_match(candidate)
                if key not in occupied and key not in original_values:
                    value = candidate
                    break
            else:
                raise ValueError("No free four-digit email suffix for localized name")
        occupied[fold_for_match(value)] = source
        result[source] = value
    return [
        EmailAssignment(source=source, value=result[key])
        for source, key in sorted(aliases.items())
    ]


def _is_email(entry: SourceEntry) -> bool:
    return (
        entry.category == "contact"
        and isinstance(entry.source, str)
        and _EMAIL_TOKEN.fullmatch(entry.source) is not None
    )


def _candidate(
    entry: SourceEntry, completed: list[LocalizationEntry]
) -> EmailCandidate:
    parts = romanized_identity_parts(entry, completed)
    if not parts:
        raise ValueError(f"Email has no resolved localized identity: {entry.id}")
    return EmailCandidate(source=entry.source, name_parts=parts)


def _mapped_email(entry: SourceEntry, value: str) -> LocalizationEntry:
    return LocalizationEntry(
        **entry.model_dump(include=set(SourceEntry.model_fields)),
        forms=LocalizedForms(
            native=value,
            romanized=value,
            english="Synthetic email derived from the regional romanized name and source format, with a short numeric suffix.",
        ),
        operation="adapt" if value != entry.source else "preserve",
        notes="Both forms use the same ASCII identifier. Source punctuation and distinct lookup values are retained; no real mailbox is contacted.",
    )


def localize_email_entries(
    entries: list[SourceEntry], completed: list[LocalizationEntry]
) -> list[LocalizationEntry]:
    """Resolve one generation stage's emails together, reserving retained lookups.

    New addresses use the domain of the map's already-localized emails. Before
    any exist, a single canonical source domain is kept, like the rest of the
    source address format; sources spread over several mailbox domains (the
    banking customer base) are normalized to the synthetic ``example.com``.
    """
    emails = [entry for entry in entries if _is_email(entry)]
    if not emails:
        return []
    reserved = [
        EmailAssignment(source=entry.source, value=entry.forms.native)
        for entry in completed
        if _is_email(entry)
    ]
    if reserved:
        domain = email_domain([row.value for row in reserved])
    else:
        source_domains = {entry.source.rpartition("@")[2].lower() for entry in emails}
        domain = (
            source_domains.pop() if len(source_domains) == 1 else SYNTHETIC_EMAIL_DOMAIN
        )
    assignments = allocate_emails(
        [_candidate(entry, completed) for entry in emails], domain, reserved
    )
    values = {item.source: item.value for item in assignments}
    return [_mapped_email(entry, values[entry.source]) for entry in emails]
