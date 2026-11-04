"""Conservative English-text audit against scoped, explicit entity mappings."""

import hashlib
import json
import re
from typing import Literal

from pydantic import Field, JsonValue

from tau2.domains.banking_knowledge.localization import (
    _ADDRESS_FIELDS,
    _CONTACT_FIELDS,
    _NAME_FIELDS,
    _PROSE_FIELDS,
)
from tau2.multilingual.localization import (
    LocalizationEntry,
    LocalizationMap,
    LocalizationModel,
)

BANKING_PROSE_AUDIT_VERSION = "banking-exact-english-v1"
_ENTITY_CATEGORIES = {"identity_name", "address", "contact", "other"}
_MUTABLE_ARGUMENTS = (
    _NAME_FIELDS
    | _ADDRESS_FIELDS
    | _CONTACT_FIELDS
    | _PROSE_FIELDS
    | {"merchant_name", "employer", "employer_name", "description"}
)


class BankingProseFinding(LocalizationModel):
    """A proposed edit which cannot be justified by explicit source references."""

    entry_id: str = Field(description="Source-bound prose entry.")
    representation: Literal["native", "romanized"] = Field(description="Audited form.")
    reason: Literal["ambiguous_reference", "unresolved_alias", "unverified_edit"] = (
        Field(description="Why this form is not mechanically verified.")
    )
    references: list[str] = Field(
        default_factory=list,
        description="Ambiguous or partial source references requiring resolution.",
    )
    expected: str | None = Field(
        description="Exact deterministic reconstruction, absent when source references are uncertain."
    )
    actual: str = Field(description="Current authored form, retained for review.")


class BankingProseRepairChange(LocalizationModel):
    """One exact source reconstruction, with original generated text retained."""

    entry_id: str = Field(description="Source-bound prose entry.")
    representation: Literal["native", "romanized"] = Field(description="Changed form.")
    previous: str = Field(description="Original generated form.")
    previous_english: JsonValue = Field(
        description="Original generated English explanation."
    )
    previous_notes: str = Field(description="Original generated rationale.")
    reconstructed: str = Field(
        description="Source text with scoped entity substitutions."
    )


class BankingProseRepairProvenance(LocalizationModel):
    """Reconstruction lineage separate from the original LLM generation."""

    version: Literal["banking-exact-english-v1"] = BANKING_PROSE_AUDIT_VERSION
    parent_map_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Input map digest."
    )
    output_map_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Reconstructed map digest."
    )
    changes: list[BankingProseRepairChange] = Field(description="Every changed form.")
    unresolved: list[BankingProseFinding] = Field(
        description="References still requiring explicit resolution."
    )


class BankingProseRepairResult(LocalizationModel):
    """An unapproved candidate and its deterministic reconstruction record."""

    proposal: LocalizationMap = Field(
        description="Candidate; generation lineage remains intact."
    )
    reconstruction: BankingProseRepairProvenance = Field(
        description="Source reconstruction lineage."
    )


def _owner(entry: LocalizationEntry) -> str:
    match = re.search(r"(?:^|; )entity=([^;]*);", entry.context)
    return match.group(1) if match else ""


def _tasks(entry: LocalizationEntry) -> set[int]:
    return {
        occurrence.path[0]
        for occurrence in entry.occurrences
        if occurrence.document_id == "tasks"
        and occurrence.path
        and isinstance(occurrence.path[0], int)
    }


def _references(entry, entities, representation):
    tasks, owner = _tasks(entry), _owner(entry)
    grouped = {}
    for entity in entities:
        in_task = bool(tasks & _tasks(entity))
        in_database = any(o.document_id == "database" for o in entity.occurrences)
        # Another task's injected person belongs to a different source world.
        if not (in_task or in_database or entity.category in {"other", "contact"}):
            continue
        target = getattr(entity.forms, representation)
        if (
            not isinstance(entity.source, str)
            or not isinstance(target, str)
            or not entity.source
        ):
            continue
        same_owner = (
            owner
            and not owner.startswith("task:")
            and owner not in {"contact", "organization"}
            and _owner(entity) == owner
        )
        rank = 3 if same_owner else 2 if in_task else 1 if in_database else 0
        grouped.setdefault(entity.source.casefold(), []).append((rank, target, entity))
    chosen, ambiguous, aliases = {}, set(), {}
    for source, candidates in grouped.items():
        rank = max(item[0] for item in candidates)
        selected = [item for item in candidates if item[0] == rank]
        targets = {item[1] for item in selected}
        if all(target.casefold() == source for target in targets):
            chosen[source] = source
        elif len(targets) == 1:
            chosen[source] = next(iter(targets))
        else:
            ambiguous.add(source)
        for _, target, entity in selected:
            if entity.category == "identity_name" and target != entity.source:
                parts = entity.source.split()
                if len(parts) > 1:
                    for part in (parts[0], parts[-1]):
                        if len(part) >= 3:
                            aliases.setdefault(part.casefold(), set()).add(source)
    return chosen, ambiguous, aliases


def _pattern(values):
    if not values:
        return None
    return re.compile(
        r"(?<!\w)(?:"
        + "|".join(
            re.escape(value)
            for value in sorted(values, key=lambda value: (-len(value), value))
        )
        + r")(?!\w)",
        re.IGNORECASE,
    )


def _reconstruct(text, chosen, ambiguous, aliases):
    pattern = _pattern(set(chosen) | ambiguous)
    unknown, covered = set(), []
    if pattern is None:
        expected = text
    else:

        def replace(match):
            surface = match.group()
            source = surface.casefold()
            covered.append((match.start(), match.end()))
            if source in ambiguous:
                unknown.add(surface)
                return surface
            target = chosen[source]
            return surface if target.casefold() == source else target

        expected = pattern.sub(replace, text)
    residual = list(text)
    for start, end in covered:
        residual[start:end] = " " * (end - start)
    alias_pattern = _pattern(aliases)
    partial = set(alias_pattern.findall("".join(residual))) if alias_pattern else set()
    return expected, unknown, partial


def audit_banking_prose(proposal: LocalizationMap) -> list[BankingProseFinding]:
    """Verify English bytes outside explicit entity edits; never guess aliases.

    JSON-encoded arguments are compared as typed JSON, allowing only entity-value
    edits in the domain's mutable fields. Free prose retains exact punctuation,
    whitespace, negation and every financial clause. Unmapped partial names are
    reported separately rather than silently declared localized.
    """
    if proposal.source.domain != "banking_knowledge":
        raise ValueError("Banking prose audit requires a banking localization map")
    from tau2.domains.banking_knowledge.localization_bindings import (
        validate_prose_bindings,
    )

    validate_prose_bindings(proposal)
    entities = [
        entry for entry in proposal.entries if entry.category in _ENTITY_CATEGORIES
    ]
    findings = []
    cache = {}
    for entry in proposal.entries:
        if entry.category != "prose" or not isinstance(entry.source, str):
            continue
        for representation in ("native", "romanized"):
            key = (tuple(sorted(_tasks(entry))), _owner(entry), representation)
            if key not in cache:
                cache[key] = _references(entry, entities, representation)
            base_chosen, base_ambiguous, aliases = cache[key]
            chosen, ambiguous = dict(base_chosen), set(base_ambiguous)
            for binding in proposal.prose_bindings:
                if binding.entry_id == entry.id:
                    chosen[binding.source.casefold()] = getattr(
                        binding.forms, representation
                    )
                    ambiguous.discard(binding.source.casefold())
            actual = getattr(entry.forms, representation)
            if not isinstance(actual, str):
                raise ValueError(f"Banking prose must remain a string: {entry.id}")
            unknown, partial = set(), set()
            is_arguments = "role=arguments;" in entry.context
            if is_arguments:

                def transform(value, field=""):
                    if isinstance(value, dict):
                        return {k: transform(v, k) for k, v in value.items()}
                    if isinstance(value, list):
                        return [transform(v, field) for v in value]
                    if isinstance(value, str) and field in _MUTABLE_ARGUMENTS:
                        result, uncertain, fragments = _reconstruct(
                            value, chosen, ambiguous, aliases
                        )
                        unknown.update(uncertain)
                        partial.update(fragments)
                        return result
                    return value

                expected = json.dumps(
                    transform(json.loads(entry.source)),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                try:
                    comparable = json.dumps(
                        json.loads(actual), ensure_ascii=False, sort_keys=True
                    )
                except ValueError:
                    comparable = actual
            else:
                expected, unknown, partial = _reconstruct(
                    entry.source, chosen, ambiguous, aliases
                )
                comparable = actual
            reason = (
                "ambiguous_reference"
                if unknown
                else "unresolved_alias"
                if partial
                else "unverified_edit"
                if comparable != expected
                else None
            )
            if reason:
                findings.append(
                    BankingProseFinding(
                        entry_id=entry.id,
                        representation=representation,
                        reason=reason,
                        references=sorted(unknown or partial),
                        expected=None if unknown or partial else expected,
                        actual=actual,
                    )
                )
    return findings


def _map_digest(proposal: LocalizationMap) -> str:
    encoded = json.dumps(
        proposal.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reconstruct_banking_prose(proposal: LocalizationMap) -> BankingProseRepairResult:
    """Restore exact English prose where entity references are unambiguous.

    This offline operation never authors new entity forms, guesses aliases or
    grants approval. It leaves uncertain entries untouched and preserves original
    LLM provenance. Callers must persist the returned reconstruction record and
    resolve remaining findings before compiling a candidate.
    """
    repaired = proposal.model_copy(deep=True)
    entries = {entry.id: entry for entry in repaired.entries}
    changes = []
    originals = {entry.id: entry for entry in proposal.entries}
    for finding in audit_banking_prose(proposal):
        if finding.expected is None:
            continue
        entry = entries[finding.entry_id]
        setattr(entry.forms, finding.representation, finding.expected)
        entry.operation = (
            "preserve"
            if entry.forms.native == entry.source == entry.forms.romanized
            else "adapt"
        )
        changes.append(
            BankingProseRepairChange(
                entry_id=finding.entry_id,
                representation=finding.representation,
                previous=finding.actual,
                previous_english=originals[entry.id].forms.english,
                previous_notes=originals[entry.id].notes,
                reconstructed=finding.expected,
            )
        )
    for identifier in {change.entry_id for change in changes}:
        entries[
            identifier
        ].forms.english = "Original English prose with only explicit source-bound entity substitutions; banking-exact-english-v1."
        entries[
            identifier
        ].notes = "Reconstructed from canonical source text and scoped entity mappings. Original generated forms and rationale are retained in the reconstruction record. Native entity/context review remains pending."
    return BankingProseRepairResult(
        proposal=repaired,
        reconstruction=BankingProseRepairProvenance(
            parent_map_sha256=_map_digest(proposal),
            output_map_sha256=_map_digest(repaired),
            changes=changes,
            unresolved=audit_banking_prose(repaired),
        ),
    )


def validate_banking_prose(proposal: LocalizationMap) -> None:
    """Reject unverified English edits before candidate compilation/publication."""
    findings = audit_banking_prose(proposal)
    if findings:
        first = findings[0]
        raise ValueError(
            f"Banking English prose has {len(findings)} unverified forms: {first.entry_id} ({first.representation}: {first.reason})"
        )
