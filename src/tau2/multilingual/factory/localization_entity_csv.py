"""Typed, source-bound entity review without scripts or protocol payloads."""

import argparse
import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import Field, JsonValue

from tau2.multilingual.factory.localization_artifact import (
    RegionalMapArtifact,
    read_localization_map,
)
from tau2.multilingual.factory.localization_spreadsheet import EntityReviewRow
from tau2.multilingual.localization import (
    LocalizationEntry,
    LocalizationMap,
    LocalizationModel,
    json_content_digest,
)
from tau2.utils.artifacts import content_digest, write_artifact

_COLUMNS = ("review_type", "difficulty", *tuple(EntityReviewRow.model_fields)[:5])
_EXCLUDED = {"prose", "protocol", "policy", "tool_text", "money"}
_INSTRUCTION_FIELDS = {"spelling_styles", "omission_instruction"}
_PRONUNCIATION_FIELDS = ("token", "reference", "distorted")


ReviewType = Literal[
    "Entity",
    "Entity reference",
    "Decoy",
    "Pronunciation token",
    "Mispronunciation",
    "Spoken digit",
    "Date/time",
    "Identifier",
]


class EntityReferenceOccurrence(LocalizationModel):
    """Every exact scoped reference retained behind a consolidated review pair."""

    entry_id: str = Field(description="Original scoped reference row identity.")
    entry_sha256: str = Field(description="Exact binding digest.")
    source_sha256: str = Field(description="Canonical source inventory digest.")
    context: str = Field(description="Source task scope and role.")
    lineage: str = Field(description="Exact source entry and binding provenance.")
    origin: str = Field(description="Reused or newly generated reference.")
    explanation: str = Field(description="Original per-reference explanation.")


class EntityCsvRow(EntityReviewRow):
    """A review pair classified by its recorded source role."""

    review_type: ReviewType = Field(title="Type", description="Purpose of this row.")
    reference_occurrences: list[EntityReferenceOccurrence] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
        description="All scoped members of a consolidated banking reference pair.",
    )
    difficulty: Literal[
        "Easy", "Hard", "Easy + Hard", "Context only", "Not applicable"
    ] = Field(
        default="Not applicable",
        title="Difficulty",
        description="Frozen measured intake selection tier; incidental context is not a graded capture entity.",
    )


class EntityCsvSource(LocalizationModel):
    """Complete restored map plus the exact compact artifact being reviewed."""

    mapping: LocalizationMap = Field(description="Source-checked complete candidate.")
    artifact_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Compact map digest."
    )


class EntityPair(LocalizationModel):
    """A pair of scalar strings at one explicit source value path."""

    source: str = Field(description="Literal source entity.")
    native: str = Field(description="Localized entity candidate.")
    romanized: str = Field(description="Corresponding romanized candidate.")
    path: list[str | int] = Field(description="Position within the parent map entry.")


class EntityCsvManifest(LocalizationModel):
    """Entity review is a subset and never grants full-map publication approval."""

    version: Literal["localization-entity-csv-v7"] = "localization-entity-csv-v7"
    scope: Literal["entity_pairs_only"] = "entity_pairs_only"
    language: str = Field(description="One language per CSV.")
    map_sha256: dict[str, str] = Field(
        description="Domain to exact compact map digest."
    )
    rows: list[EntityCsvRow] = Field(
        description="Original rows with source lineage and pending decisions."
    )


def _pairs(
    source: JsonValue, native: JsonValue, romanized: JsonValue, path: list[str | int]
) -> list[EntityPair]:
    if isinstance(source, str):
        if not isinstance(native, str) or not isinstance(romanized, str):
            raise ValueError("Entity forms must retain their source string shape")
        return [
            EntityPair(source=source, native=native, romanized=romanized, path=path)
        ]
    if isinstance(source, list):
        if (
            not isinstance(native, list)
            or not isinstance(romanized, list)
            or not len(source) == len(native) == len(romanized)
        ):
            raise ValueError("Entity list forms must retain source alignment")
        return [
            pair
            for i, (before, after, latin) in enumerate(
                zip(source, native, romanized, strict=True)
            )
            for pair in _pairs(before, after, latin, [*path, i])
        ]
    if isinstance(source, dict):
        if (
            not isinstance(native, dict)
            or not isinstance(romanized, dict)
            or not source.keys() == native.keys() == romanized.keys()
        ):
            raise ValueError("Entity object forms must retain source keys")
        if (
            set(_PRONUNCIATION_FIELDS)
            <= source.keys()
            <= {*_PRONUNCIATION_FIELDS, "boundary", "reason"}
        ):
            # Reference respellings are metadata, not spoken entities or errors.
            keys = ("token", "distorted")
        elif source and all(key.isdecimal() for key in source):
            keys = tuple(source)
        else:
            raise ValueError(
                "Unrecognized structured entity; do not export opaque JSON for review"
            )
        return [
            pair
            for key in keys
            for pair in _pairs(source[key], native[key], romanized[key], [*path, key])
        ]
    raise ValueError(
        "Entity review expects strings or explicitly supported entity collections"
    )


def _review_type(entry: LocalizationEntry, pair: EntityPair) -> ReviewType:
    """Use source paths and the inventory's semantic role, not value spelling."""
    fields = {o.path[-1] for o in entry.occurrences if o.path}
    if "decoys" in fields:
        return "Decoy"
    if "digit_words" in fields:
        return "Spoken digit"
    if pair.path and pair.path[-1] in {"token", "distorted"}:
        return "Pronunciation token" if pair.path[-1] == "token" else "Mispronunciation"
    role = entry.context.partition(";")[0]
    if role in {
        "intake field:drop_off_date",
        "intake field:date_of_birth",
        "intake field:appointment_time",
    }:
        return "Date/time"
    if role in {
        "intake field:member_id",
        "intake field:loyalty_number",
        "intake field:vin",
        "intake field:license_plate",
    }:
        return "Identifier"
    return "Entity"


def _difficulty(mapping: LocalizationMap, entry: LocalizationEntry) -> str:
    """Read the frozen measured stimulus assignments rather than guessing difficulty."""
    if mapping.source.domain != "intake":
        return "Not applicable"
    document = next(
        (d for d in mapping.source.documents if d.id == "intake_localization"), None
    )
    if document is None:
        raise ValueError("Intake review requires its frozen selected-stimulus document")
    tiers = {
        document.content["stimuli"][o.path[1]]["tier"]
        for o in entry.occurrences
        if o.document_id == "intake_localization"
        and len(o.path) >= 3
        and o.path[0] == "stimuli"
        and isinstance(o.path[1], int)
    }
    if any(
        o.document_id == "intake_localization" and o.path == ["digit_words"]
        for o in entry.occurrences
    ):
        # The shared spoken-digit detector is used by both frozen task tiers.
        tiers.update(s["tier"] for s in document.content["stimuli"])
    tasks = next((d for d in mapping.source.documents if d.id == "tasks"), None)
    selected = {
        s["canonical_task_id"]: s
        for s in document.content["stimuli"]
        if "canonical_task_id" in s
    }
    for occurrence in entry.occurrences:
        path = occurrence.path
        if (
            tasks is not None
            and occurrence.document_id == "tasks"
            and len(path) == 7
            and isinstance(path[0], int)
            and path[1:3] == ["initial_state", "initialization_actions"]
            and path[4:6] == ["arguments", "entities"]
        ):
            stimulus = selected.get(tasks.content[path[0]]["id"])
            if stimulus and path[-1] == stimulus["field_name"]:
                # Relative-date phrases are caller surfaces of the measured date,
                # even though their canonical selected-stimulus value is ISO.
                tiers.add(stimulus["tier"])
    if tiers == {"easy", "hard"}:
        return "Easy + Hard"
    if tiers == {"easy"}:
        return "Easy"
    if tiers == {"hard"}:
        return "Hard"
    if tiers:
        raise ValueError("Unknown frozen intake difficulty tier")
    return "Context only"


def consolidate_banking_references(rows: list[EntityCsvRow]) -> list[EntityCsvRow]:
    """Review an identical banking reference pair once, preserving every source link."""
    result = []
    grouped = {}
    for row in rows:
        if row.domain != "banking_knowledge" or row.review_type != "Entity reference":
            result.append(row)
            continue
        key = (row.original_entity, row.suggested_entity, row.romanized)
        member = EntityReferenceOccurrence.model_validate(
            row.model_dump(include=set(EntityReferenceOccurrence.model_fields))
        )
        if key not in grouped:
            consolidated = row.model_copy(update={"reference_occurrences": [member]})
            grouped[key] = consolidated
            result.append(consolidated)
        else:
            grouped[key].reference_occurrences.append(member)
    return result


def entity_review_rows(mapping: LocalizationMap) -> list[EntityCsvRow]:
    """Project individual entities and used decoys/pronunciations, excluding prose."""
    rows = []
    source_digest = content_digest(mapping.source)
    for entry in sorted(mapping.entries, key=lambda item: item.id):
        if entry.category in _EXCLUDED or any(
            occurrence.path and occurrence.path[-1] in _INSTRUCTION_FIELDS
            for occurrence in entry.occurrences
        ):
            continue
        for pair in _pairs(entry.source, entry.forms.native, entry.forms.romanized, []):
            pointer = "".join(
                "/" + str(part).replace("~", "~0").replace("/", "~1")
                for part in pair.path
            )
            lineage = {
                "source_entry_id": entry.id,
                "value_path": pair.path,
                "reuse": entry.reuse.model_dump(mode="json") if entry.reuse else None,
            }
            if mapping.source.domain == "intake":
                from tau2.domains.intake.date_display import (
                    DATE_DISPLAY_VERSION,
                    LANGUAGE_DATE_FORMATS,
                    display_date_payload,
                    locale_date_format,
                )

                if mapping.settings.language in LANGUAGE_DATE_FORMATS:
                    style = LANGUAGE_DATE_FORMATS[mapping.settings.language]
                    native = display_date_payload(pair.native, style)
                    romanized = display_date_payload(
                        pair.romanized,
                        locale_date_format(mapping.settings.language, "romanized"),
                    )
                    if native != pair.native or romanized != pair.romanized:
                        lineage["display_projection"] = DATE_DISPLAY_VERSION
                        lineage["date_display_format"] = style
                        pair = pair.model_copy(
                            update={"native": native, "romanized": romanized}
                        )
                from tau2.domains.intake.time_display import (
                    TIME_DISPLAY_VERSION,
                    display_time,
                    display_time_payload,
                    locale_time_format,
                )

                time_style = locale_time_format(mapping.settings.language)
                if time_style is not None:
                    from tau2.domains.intake.utils import spoken_form

                    native_payload = display_time_payload(pair.native, time_style)
                    romanized_payload = display_time_payload(
                        pair.romanized,
                        locale_time_format(mapping.settings.language, "romanized"),
                    )
                    native = (
                        spoken_form(native_payload)
                        if native_payload != pair.native
                        else pair.native
                    )
                    romanized = (
                        spoken_form(romanized_payload)
                        if romanized_payload != pair.romanized
                        else pair.romanized
                    )
                    if (
                        _review_type(entry, pair) == "Decoy"
                        and native_payload != pair.native
                    ):
                        # A meridiem-flip decoy must remain audibly different from
                        # the bare correct expression. Runtime slips include it.
                        native = display_time(pair.native, time_style)
                        romanized = display_time(
                            pair.romanized,
                            locale_time_format(mapping.settings.language, "romanized"),
                        )
                        native_payload = native
                    if native != pair.native or romanized != pair.romanized:
                        lineage["display_projection"] = TIME_DISPLAY_VERSION
                        lineage["time_display_format"] = time_style
                        lineage["caller_time_payload"] = native_payload
                        pair = pair.model_copy(
                            update={"native": native, "romanized": romanized}
                        )
                if (
                    mapping.intake_caller_surfaces
                    and _review_type(entry, pair) == "Date/time"
                ):
                    from tau2.domains.intake.utils import render_time_dual_form

                    selected = next(
                        d
                        for d in mapping.source.documents
                        if d.id == "intake_localization"
                    ).content["stimuli"]
                    linked_ids = {
                        selected[o.path[1]]["canonical_task_id"]
                        for o in entry.occurrences
                        if o.document_id == "intake_localization"
                        and len(o.path) == 3
                        and o.path[0] == "stimuli"
                        and isinstance(o.path[1], int)
                        and o.path[2] == "written"
                    }
                    overrides = {
                        surface.canonical_task_id: surface
                        for surface in mapping.intake_caller_surfaces
                    }
                    authored = [
                        overrides[identifier]
                        for identifier in sorted(linked_ids)
                        if identifier in overrides
                    ]
                    if authored:
                        variants = {
                            (surface.native_spoken, surface.romanized_spoken)
                            for surface in authored
                        }
                        if linked_ids - overrides.keys():
                            variants.add((pair.native, pair.romanized))
                        if len(variants) != 1:
                            raise ValueError(
                                f"Shared time review entry has differing task caller surfaces: {entry.id}"
                            )
                        native, romanized = variants.pop()
                        full = display_time(
                            authored[0].canonical_value, time_style or "en"
                        )
                        lineage["caller_time_payload"] = render_time_dual_form(
                            native, full
                        )
                        lineage["caller_time_payload_romanized"] = (
                            render_time_dual_form(
                                romanized,
                                display_time(
                                    authored[0].canonical_value,
                                    locale_time_format(
                                        mapping.settings.language, "romanized"
                                    )
                                    or "en",
                                ),
                            )
                        )
                        lineage["intake_caller_surfaces"] = [
                            surface.model_dump(mode="json") for surface in authored
                        ]
                        pair = pair.model_copy(
                            update={"native": native, "romanized": romanized}
                        )
            rows.append(
                EntityCsvRow(
                    review_type=_review_type(entry, pair),
                    difficulty=_difficulty(mapping, entry),
                    original_entity=pair.source,
                    suggested_entity=pair.native,
                    domain=mapping.source.domain,
                    category=entry.category,
                    origin="Reused" if entry.reuse else "New",
                    romanized=pair.romanized,
                    explanation=entry.forms.english,
                    context=entry.context,
                    lineage=json.dumps(lineage, ensure_ascii=False, sort_keys=True),
                    language=mapping.settings.language,
                    entry_id=entry.id + pointer,
                    entry_sha256=content_digest(entry),
                    source_sha256=source_digest,
                )
            )
    for binding in sorted(
        mapping.prose_bindings, key=lambda item: (item.entry_id, item.source)
    ):
        if binding.kind == "protected":
            continue
        identifier = "prose-reference:" + json_content_digest(
            {"entry_id": binding.entry_id, "source": binding.source}
        )
        if not isinstance(binding.forms.native, str) or not isinstance(
            binding.forms.romanized, str
        ):
            raise ValueError("Prose entity reference must be a string")
        rows.append(
            EntityCsvRow(
                review_type="Entity reference",
                original_entity=binding.source,
                suggested_entity=binding.forms.native,
                domain=mapping.source.domain,
                category="prose_reference",
                origin="Reused" if binding.kind == "reuse" else "New",
                romanized=binding.forms.romanized,
                explanation=binding.forms.english,
                context=binding.reason,
                lineage=binding.model_dump_json(),
                language=mapping.settings.language,
                entry_id=identifier,
                entry_sha256=content_digest(binding),
                source_sha256=source_digest,
            )
        )
    if len({row.key for row in rows}) != len(rows):
        raise ValueError("Entity CSV contains duplicate source identities")
    return consolidate_banking_references(rows)


def _csv_text(rows: list[EntityCsvRow]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow([EntityCsvRow.model_fields[name].title for name in _COLUMNS])
    writer.writerows([[getattr(row, name) for name in _COLUMNS] for row in rows])
    return stream.getvalue()


def export_entity_csv(
    sources: list[EntityCsvSource], output: Path
) -> list[EntityCsvManifest]:
    """Write immutable CSVs and provenance; retain exact Unicode and CSV quoting."""
    manifests = []
    keys = [
        (source.mapping.settings.language, source.mapping.source.domain)
        for source in sources
    ]
    if not sources or len(set(keys)) != len(keys):
        raise ValueError("Provide one or more distinct language/domain maps")
    for language in sorted({key[0] for key in keys}):
        selected = sorted(
            (
                source
                for source in sources
                if source.mapping.settings.language == language
            ),
            key=lambda source: source.mapping.source.domain,
        )
        manifest = EntityCsvManifest(
            language=language,
            map_sha256={
                source.mapping.source.domain: source.artifact_sha256
                for source in selected
            },
            rows=[
                row for source in selected for row in entity_review_rows(source.mapping)
            ],
        )
        text = _csv_text(manifest.rows)
        path = output / f"localization-{language}.csv"
        sidecar = path.with_suffix(".manifest.json")
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as stream:
                existing = stream.read()
            if existing != text:
                raise FileExistsError(f"CSV already exists or has review edits: {path}")
            if EntityCsvManifest.model_validate_json(sidecar.read_text()) != manifest:
                raise ValueError("CSV provenance differs from the exact map inputs")
        else:
            output.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8-sig", newline="", dir=output, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(text)
            try:
                write_artifact(manifest, sidecar)
                os.link(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        manifests.append(manifest)
    return manifests


def run_entity_csv_export(args: argparse.Namespace) -> None:
    """Load exact compact maps and resolve their canonical source before export."""
    sources = [
        EntityCsvSource(
            mapping=read_localization_map(path),
            artifact_sha256=content_digest(
                RegionalMapArtifact.model_validate_json(path.read_text())
            ),
        )
        for path in args.map
    ]
    manifests = export_entity_csv(sources, args.output)
    logger.info(
        "Exported {} entity pairs across {} language CSVs",
        sum(len(m.rows) for m in manifests),
        len(manifests),
    )


def add_localization_entity_csv_args(commands) -> None:
    """Register the entity-only CSV projection under the factory command group."""
    parser = commands.add_parser(
        "export-localization-entities-csv",
        help="Export classified entity pairs without scripts or protocol JSON",
    )
    parser.add_argument("--map", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(func=run_entity_csv_export)
