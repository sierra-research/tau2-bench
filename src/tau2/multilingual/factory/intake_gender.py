"""Validate frozen source name genders on published intake localization maps."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from tau2.domains.intake.localized import IntakeLocalizationData
from tau2.domains.intake.tasks.banks import load_banks
from tau2.domains.intake.tasks.generator import FreezeManifest
from tau2.multilingual.localization import LocalizationMap, LocalizationModel


class IntakeNameBinding(LocalizationModel):
    """Exact name slot with reviewed source evidence and separate semantic role."""

    id: str = Field(description="Unique audit slot ID; decoys include array position.")
    entry_id: str = Field(description="Owning localization map entry.")
    index: int | None = Field(default=None, description="Name decoy array position.")
    role: Literal["callee", "measured_name", "name_decoy"] = Field(
        description="Identity role; measured names never inherit callee gender."
    )
    source_gender: Literal["male", "female", "unknown"] = Field(
        description="Reviewed bank gender, or unknown for independently assessed decoys."
    )
    source_evidence: str = Field(
        description="Exact frozen-bank evidence or its absence."
    )
    task_ids: list[str] = Field(description="Canonical tasks using this exact role.")


def intake_name_bindings(
    proposal: LocalizationMap, data_root: Path
) -> list[IntakeNameBinding]:
    """Join used identities to pinned bank labels without guessing gender from names."""
    if proposal.source.domain != "intake":
        raise ValueError("Intake gender preservation requires the intake domain")
    directory = data_root / "tau2/domains/intake"
    banks = load_banks(directory / "banks")
    manifest = FreezeManifest.model_validate_json(
        (directory / "tasks.manifest.json").read_text()
    )
    if banks.shas["person_names"] != manifest.bank_shas["person_names"]:
        raise ValueError("Frozen person-name bank changed")
    names = {}
    for name in banks.person_names:
        for value in [name.value, name.alt_name]:
            if value is not None:
                if value in names and names[value].gender != name.gender:
                    raise ValueError("Conflicting reviewed source name gender")
                names[value] = name
    tasks = next(d.content for d in proposal.source.documents if d.id == "tasks")
    selected = IntakeLocalizationData.model_validate(
        next(
            d.content
            for d in proposal.source.documents
            if d.id == "intake_localization"
        )
    )
    result = []
    for entry in proposal.entries:
        rows = {
            point.path[1]
            for point in entry.occurrences
            if point.document_id == "intake_localization"
            and len(point.path) == 3
            and point.path[0] == "stimuli"
            and selected.stimuli[point.path[1]].family == "person_names"
            and point.path[2] in {"written", "decoys"}
        }
        if entry.category == "identity_name":
            role = "callee"
            task_ids = sorted(
                {
                    tasks[p.path[0]]["id"]
                    for p in entry.occurrences
                    if p.document_id == "tasks"
                }
            )
        elif rows:
            role = "name_decoy" if isinstance(entry.source, list) else "measured_name"
            task_ids = sorted(
                selected.stimuli[index].canonical_task_id for index in rows
            )
        else:
            continue
        values = (
            enumerate(entry.source) if role == "name_decoy" else [(None, entry.source)]
        )
        for index, value in values:
            bank = names.get(value)
            if role != "name_decoy" and bank is None:
                raise ValueError(f"Used identity has no reviewed bank gender: {value}")
            evidence = (
                f"Frozen person_names bank: {bank.value!r}, gender={bank.gender.value}; sha256={banks.shas['person_names']}."
                if bank is not None
                else "This selected source decoy has no explicit reviewed gender label. Assess its own naming convention independently; never copy the primary's gender. Mark ambiguous or unisex names explicitly."
            )
            result.append(
                IntakeNameBinding(
                    id=entry.id if index is None else f"{entry.id}:decoy:{index}",
                    entry_id=entry.id,
                    index=index,
                    role=role,
                    source_gender=bank.gender.value if bank is not None else "unknown",
                    source_evidence=evidence,
                    task_ids=task_ids,
                )
            )
    return result


def _colliding_name_decoys(proposal: LocalizationMap) -> list[str]:
    from tau2.domains.intake.folds import FoldKind, fold_value

    result = []
    for entry in proposal.entries:
        if (
            "intake stimulus:intake_person_names_" not in entry.context
            or "dependency=decoys;" not in entry.context
        ):
            continue
        for representation in ("native", "romanized"):
            values = [
                fold_value(FoldKind.NATIVE_NAME, value)
                for value in getattr(entry.forms, representation)
            ]
            if len(values) != len(set(values)):
                result.append(entry.id)
                break
    return result


def validate_intake_gender_coverage(proposal: LocalizationMap, data_root: Path) -> None:
    """Refuse publication with missing, stale or conflicting source-name evidence."""
    indexed = {entry.id: entry for entry in proposal.entries}
    for binding in intake_name_bindings(proposal, data_root):
        entry = indexed[binding.entry_id]
        if (
            proposal.settings.language == "en"
            and entry.forms.native == entry.source
            and entry.forms.romanized == entry.source
        ):
            # Exact source controls preserve identity without a model assessment.
            continue
        if binding.index is None:
            evidence = entry.name_gender
            if evidence is not None:
                evidence.require(entry.forms)
        else:
            item = next(
                (
                    item
                    for item in entry.name_gender_items
                    if item.index == binding.index
                ),
                None,
            )
            evidence = item.evidence if item is not None else None
            if item is not None:
                item.require(entry.forms)
        if evidence is None:
            raise ValueError(
                f"Intake name requires source-gender evidence before publication: {binding.id}"
            )
        if binding.source_gender != "unknown" and (
            evidence.source_gender != binding.source_gender
            or evidence.source_basis != "reviewed_metadata"
        ):
            raise ValueError(
                f"Intake source gender differs from frozen bank: {binding.id}"
            )
    if _colliding_name_decoys(proposal):
        raise ValueError(
            "Intake name decoys collapse independent wrong-name identities"
        )
