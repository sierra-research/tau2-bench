"""Reuse exact approved entity records before the fixed localization generator."""

from pathlib import Path

from pydantic import Field

from tau2.data_model.language import NameOrder
from tau2.multilingual.address_format import AddressForms, AddressParts, format_address
from tau2.multilingual.factory.localization_inventory import inventory_localization
from tau2.multilingual.localization import (
    EntityReuseProvenance,
    LocalizationEntry,
    LocalizationModel,
    LocalizationSource,
    LocalizedForms,
    NameGender,
    NameGenderEvidence,
    SourceEntry,
)
from tau2.utils.artifacts import ReviewApproval, content_digest


class ReviewedReuseSource(LocalizationModel):
    """A prior map and its separately verified approval, never a proposed approval."""

    map_file: Path = Field(description="Path to an existing compact regional map.")
    approval_file: Path = Field(description="Digest-bound prior human review approval.")


class ReuseCandidate(LocalizationModel):
    """One coherent prior identity assembled from explicit reviewed source paths."""

    name: LocalizedForms = Field(
        description="Approved components assembled in the locale name order."
    )
    address: LocalizedForms = Field(
        description="Address components from the same source identity."
    )
    email: LocalizedForms = Field(
        description="Prior identity email, retained as source evidence."
    )
    name_gender: NameGenderEvidence | None = Field(
        default=None,
        description="Exact donor-name assessment; unknown donors cannot be reused for gender-constrained people.",
    )
    name_ids: list[str] = Field(description="Reviewed name component entry IDs.")
    address_ids: list[str] = Field(description="Reviewed address component entry IDs.")
    email_ids: list[str] = Field(description="Reviewed email entry IDs.")
    lineage: EntityReuseProvenance = Field(
        description="Prior map approval and exact content digest."
    )


def banking_reuse_candidates(
    reference: ReviewedReuseSource, language: str
) -> list[ReuseCandidate]:
    """Read reviewed retail identities without copying IDs or currency conditions."""
    from tau2.multilingual.factory.localization_artifact import RegionalMapArtifact

    artifact = RegionalMapArtifact.model_validate_json(reference.map_file.read_text())
    approval = ReviewApproval.model_validate_json(reference.approval_file.read_text())
    approval.require(artifact)
    if artifact.source.domain != "retail" or artifact.settings.language != language:
        raise ValueError(
            "Banking identity reuse requires a reviewed retail map in the target language"
        )
    from tau2.utils import DATA_DIR

    source = inventory_localization("retail", DATA_DIR)
    current = {d.id: d.sha256 for d in source.documents if d.kind != "tool_text"}
    prior = {d.id: d.sha256 for d in artifact.source.documents if d.kind != "tool_text"}
    if current != prior:
        raise ValueError(
            "Reuse source database, tasks or policies changed since approval"
        )
    authored = {e.id: e for e in artifact.entries}
    rows = {}
    for source_entry in source.entries:
        candidate = authored.get(source_entry.id)
        if candidate is None:
            continue
        if (
            candidate.source != source_entry.source
            or candidate.category != source_entry.category
        ):
            raise ValueError("Reused entry no longer matches its source value")
        entry = LocalizationEntry(
            **source_entry.model_dump(),
            forms=candidate.forms,
            operation=candidate.operation,
            name_gender=candidate.name_gender,
            notes=candidate.notes,
        )
        for occurrence in entry.occurrences:
            path = occurrence.path
            if (
                occurrence.document_id != "database"
                or len(path) < 3
                or path[0] != "users"
            ):
                continue
            rows.setdefault(path[1], {})[tuple(path[2:])] = entry
    result = []
    artifact_digest = content_digest(artifact)
    for user_id, values in sorted(rows.items()):
        names = [values[("name", field)] for field in ("first_name", "last_name")]
        if artifact.settings.name_order == NameOrder.FAMILY_FIRST:
            names.reverse()
        addresses = [
            values[("address", field)]
            for field in ("address1", "address2", "city", "state", "zip", "country")
        ]
        email = values[("email",)]

        def assembled(entries, separator, explanation):
            return LocalizedForms(
                native=separator.join(
                    e.forms.native for e in entries if e.forms.native
                ),
                romanized=separator.join(
                    e.forms.romanized for e in entries if e.forms.romanized
                ),
                english=explanation,
            )

        address_parts = {}
        for representation in ("native", "romanized"):
            fields = [getattr(entry.forms, representation) for entry in addresses]
            from tau2.multilingual.factory.localization_address_components import (
                repair_street_line,
            )

            if repair_street_line(fields[0], language, representation) != fields[0]:
                raise ValueError(
                    "Reviewed donor has malformed local street order; repair and review its source map before reuse"
                )
            parts = AddressParts(
                street_lines=[value for value in fields[:2] if value],
                city=fields[2],
                region=fields[3],
                postal_code=fields[4],
                country=fields[5],
            )
            if language == "ko" and (
                parts.city.endswith("구")
                if representation == "native"
                else parts.city.casefold().endswith("-gu")
            ):
                parts.district, parts.city, parts.region = parts.city, parts.region, ""
                if (
                    parts.city.endswith(("특별시", "광역시", "특별자치시"))
                    if representation == "native"
                    else parts.city.casefold().endswith(
                        ("-teukbyeolsi", "-gwangyeoksi", "-teukbyeoljachisi")
                    )
                ):
                    parts.region = parts.city
            if language in {"zh", "ko"} and parts.street_lines:
                for prefix in dict.fromkeys([parts.region, parts.city, parts.district]):
                    if prefix and parts.street_lines[0].startswith(prefix):
                        remaining = parts.street_lines[0][len(prefix) :]
                        if (
                            language == "zh"
                            and representation == "native"
                            or remaining.startswith(" ")
                        ):
                            parts.street_lines[0] = remaining.strip()
            address_parts[representation] = parts
        address = AddressForms(language=language, **address_parts)

        name = assembled(
            names, " ", "Complete name assembled from a previously approved identity."
        )
        gender = values[("name", "first_name")].name_gender
        if gender is not None:
            gender = gender.model_copy(update={"forms_sha256": content_digest(name)})
        result.append(
            ReuseCandidate(
                name=assembled(
                    names,
                    " ",
                    "Complete name assembled from a previously approved identity.",
                ),
                name_gender=gender,
                address=LocalizedForms(
                    native=format_address(address.native, language, "native"),
                    romanized=format_address(address.romanized, language, "romanized"),
                    address=address,
                    english="Complete address assembled by locale from the same previously approved identity; banking context still needs review.",
                ),
                email=email.forms,
                name_ids=[e.id for e in names],
                address_ids=[e.id for e in addresses],
                email_ids=[email.id],
                lineage=EntityReuseProvenance(
                    source_domain="retail",
                    source_entry_ids=[e.id for e in names],
                    source_artifact_sha256=artifact_digest,
                    approval=approval,
                ),
            )
        )
    return result


def reuse_banking_entities(
    source: LocalizationSource,
    reference: ReviewedReuseSource,
    language: str,
    gender_by_owner: dict[str, NameGender] | None = None,
) -> list[LocalizationEntry]:
    """Assign distinct reviewed donor identities by stable banking owner, never strings alone."""
    import re

    if source.domain != "banking_knowledge":
        raise ValueError("Banking reuse requires a banking source inventory")
    donors = banking_reuse_candidates(reference, language)
    owners = sorted(
        {
            re.search(r"entity=([^;]+)", e.context)[1]
            for e in source.entries
            if e.category == "identity_name"
        }
    )
    # Preserve exact name lookup distinctions, even when the donor database has namesakes.
    unique = []
    seen = set()
    for donor in donors:
        if donor.name.native not in seen:
            unique.append(donor)
            seen.add(donor.name.native)
    if len(owners) > len(unique):
        raise ValueError("Insufficient distinct approved identity candidates")
    assignment = dict(zip(owners, unique, strict=False))
    # Donor human approval alone does not establish compatibility with a new person.
    # Keep address reuse independent; names without evidence go to fixed authoring.
    name_assignment = {}
    used_names = set()
    for owner in owners:
        required = (gender_by_owner or {}).get(owner)
        if required not in {"male", "female"}:
            continue
        for donor in unique:
            if donor.name.native in used_names or donor.name_gender is None:
                continue
            donor.name_gender.require(donor.name)
            if donor.name_gender.target_gender in {required, "unisex"}:
                name_assignment[owner] = donor
                used_names.add(donor.name.native)
                break
    result = []
    for entry in source.entries:
        if entry.category not in {"identity_name", "address"}:
            continue
        match = re.search(r"entity=([^;]+)", entry.context)
        if not match or match[1] not in assignment:
            continue
        if entry.category == "identity_name":
            if match[1] not in name_assignment:
                continue
            donor = name_assignment[match[1]]
        else:
            donor = assignment[match[1]]
        # Different historical/new addresses need distinct reviewed records rather
        # than collapsing every address of one user onto the same donor address.
        if entry.category == "address":
            siblings = [
                e
                for e in source.entries
                if e.category == "address" and f"entity={match[1]};" in e.context
            ]
            if len({str(e.source) for e in siblings}) != 1:
                continue
        forms = donor.name if entry.category == "identity_name" else donor.address
        identifiers = (
            donor.name_ids if entry.category == "identity_name" else donor.address_ids
        )
        result.append(
            LocalizationEntry(
                **entry.model_dump(),
                forms=forms,
                name_gender=(
                    donor.name_gender.model_copy(
                        update={
                            "source_gender": gender_by_owner[match[1]],
                            "source_basis": "explicit_context",
                            "source_evidence": f"Explicit caller-supplied gender requirement for source owner {match[1]}: {gender_by_owner[match[1]]}.",
                            "forms_sha256": content_digest(forms),
                            "reason": "Gender-compatible donor reuse; new banking context remains pending native review.",
                        }
                    )
                    if entry.category == "identity_name"
                    else None
                ),
                operation="adapt"
                if forms.native != entry.source or forms.romanized != entry.source
                else "preserve",
                notes="Reused approved entity components. No new LLM generation for this value. New banking context, name order and address assembly remain pending native review.",
                reuse=donor.lineage.model_copy(
                    update={"source_entry_ids": identifiers}
                ),
            )
        )
    return result


def validate_reused_entries(
    source: LocalizationSource, entries: list[LocalizationEntry]
) -> dict[str, LocalizationEntry]:
    """Reject forged inventory binding or unbound prior approval before generation."""
    sources = {e.id: e for e in source.entries}
    result = {}
    for entry in entries:
        if entry.id in result or entry.id not in sources:
            raise ValueError("Duplicate or unknown reused source entry")
        if (
            SourceEntry.model_validate(
                entry.model_dump(include=set(SourceEntry.model_fields))
            )
            != sources[entry.id]
        ):
            raise ValueError("Reused candidate does not match source inventory")
        if (
            entry.reuse is None
            or entry.reuse.approval.content_sha256 != entry.reuse.source_artifact_sha256
        ):
            raise ValueError("Reused candidate requires digest-bound prior approval")
        if entry.category == "identity_name":
            if entry.name_gender is None:
                raise ValueError(
                    "Reused personal name requires source-bound gender compatibility evidence"
                )
            entry.name_gender.require(entry.forms)
        result[entry.id] = entry
    return result
