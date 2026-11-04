"""Preserve source-proven shared household addresses within task worlds."""

from typing import Literal

from pydantic import Field

from tau2.multilingual.localization import (
    EntityReuseProvenance,
    LocalizationEntry,
    LocalizationMap,
    LocalizationModel,
    LocalizedForms,
)
from tau2.utils.artifacts import content_digest


class HouseholdAddressChange(LocalizationModel):
    """A source-proven equality repair, retaining the replaced candidate evidence."""

    entry_id: str = Field(description="Changed address entry.")
    donor_entry_id: str = Field(
        description="Stable canonical entry in the shared household."
    )
    task_indices: list[int] = Field(
        description="Tasks establishing the shared address."
    )
    previous_forms: LocalizedForms = Field(description="Original candidate forms.")
    previous_reuse: EntityReuseProvenance | None = Field(
        description="Original donor lineage."
    )
    forms: LocalizedForms = Field(
        description="Exact replacement from the canonical household entry."
    )
    reuse: EntityReuseProvenance | None = Field(
        description="Replacement donor lineage."
    )


class HouseholdAddressRevision(LocalizationModel):
    """Offline rule provenance; no generation or native approval is implied."""

    version: Literal["banking-household-address-v1"] = "banking-household-address-v1"
    parent_map_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Exact input map."
    )
    output_map_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Exact repaired map."
    )
    changes: list[HouseholdAddressChange] = Field(description="Every candidate change.")


class HouseholdAddressResult(LocalizationModel):
    """An unapproved candidate plus its address equality repair record."""

    proposal: LocalizationMap = Field(description="Repaired draft map.")
    revision: HouseholdAddressRevision = Field(
        description="Deterministic repair evidence."
    )


def _households(proposal: LocalizationMap) -> list[tuple[list[str], list[int]]]:
    groups: dict[tuple[int, str], dict[str, str]] = {}
    for entry in proposal.entries:
        if (
            entry.category != "address"
            or not isinstance(entry.source, str)
            or not entry.source.strip()
        ):
            continue
        for occurrence in entry.occurrences:
            path = occurrence.path
            if (
                occurrence.document_id == "tasks"
                and len(path) == 8
                and isinstance(path[0], int)
                and path[1:6]
                == [
                    "initial_state",
                    "initialization_data",
                    "agent_data",
                    "users",
                    "data",
                ]
                and isinstance(path[6], str)
                and path[7] == "address"
            ):
                groups.setdefault((path[0], entry.source), {})[entry.id] = path[6]
    # Connect only identities explicitly co-occurring at one exact source address.
    # Identical address strings in disjoint task worlds never establish an edge.
    adjacency: dict[str, set[str]] = {}
    evidence: dict[str, set[int]] = {}
    for (task, _), members in groups.items():
        if len(set(members.values())) < 2:
            continue
        for identifier in members:
            adjacency.setdefault(identifier, set()).update(members)
            evidence.setdefault(identifier, set()).add(task)
    components = []
    unseen = set(adjacency)
    while unseen:
        pending = [min(unseen)]
        members = set()
        while pending:
            identifier = pending.pop()
            if identifier in members:
                continue
            members.add(identifier)
            pending.extend(adjacency[identifier] - members)
        unseen -= members
        components.append(
            (sorted(members), sorted(set().union(*(evidence[i] for i in members))))
        )
    return components


def validate_banking_households(proposal: LocalizationMap) -> None:
    """Reject different localized addresses for a source-proven shared household."""
    entries = {entry.id: entry for entry in proposal.entries}
    for members, tasks in _households(proposal):
        for representation in ("native", "romanized"):
            values = {getattr(entries[i].forms, representation) for i in members}
            if len(values) > 1:
                raise ValueError(
                    f"Banking household address differs in tasks {tasks}: {members}"
                )


def harmonize_banking_households(proposal: LocalizationMap) -> HouseholdAddressResult:
    """Reuse a stable household member's complete address without inventing content."""
    if proposal.source.domain != "banking_knowledge":
        raise ValueError("Household address harmonization requires a banking map")
    repaired = proposal.model_copy(deep=True)
    entries: dict[str, LocalizationEntry] = {
        entry.id: entry for entry in repaired.entries
    }
    changes = []
    for members, tasks in _households(proposal):
        donor = entries[members[0]]
        for identifier in members[1:]:
            entry = entries[identifier]
            if (entry.forms.native, entry.forms.romanized) == (
                donor.forms.native,
                donor.forms.romanized,
            ):
                continue
            changes.append(
                HouseholdAddressChange(
                    entry_id=identifier,
                    donor_entry_id=donor.id,
                    task_indices=tasks,
                    previous_forms=entry.forms.model_copy(deep=True),
                    previous_reuse=entry.reuse.model_copy(deep=True)
                    if entry.reuse
                    else None,
                    forms=donor.forms.model_copy(deep=True),
                    reuse=donor.reuse.model_copy(deep=True) if donor.reuse else None,
                )
            )
            entry.forms = donor.forms.model_copy(deep=True)
            entry.reuse = donor.reuse.model_copy(deep=True) if donor.reuse else None
            entry.operation = (
                "preserve"
                if entry.source == entry.forms.native == entry.forms.romanized
                else "adapt"
            )
            entry.notes = (
                f"Shared source household address; exact candidate copied from {donor.id} "
                f"under banking-household-address-v1. Native review remains pending. {donor.notes}"
            )
    validate_banking_households(repaired)
    return HouseholdAddressResult(
        proposal=repaired,
        revision=HouseholdAddressRevision(
            parent_map_sha256=content_digest(proposal),
            output_map_sha256=content_digest(repaired),
            changes=changes,
        ),
    )
