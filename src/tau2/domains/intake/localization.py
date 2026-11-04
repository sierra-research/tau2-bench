"""Domain-owned localization of the frozen intake selection, never bank redraws."""

import json
import re
import tomllib
from collections import Counter
from pathlib import Path

from pydantic import TypeAdapter

from tau2.data_model.tasks import Task
from tau2.domains.intake.data_model import IntakeDB
from tau2.domains.intake.environment import IntakeTaskFile
from tau2.domains.intake.folds import FoldKind, fold_value
from tau2.domains.intake.localized import (
    IntakeLocalizationData,
    SelectedPronunciation,
    SelectedStimulus,
)
from tau2.domains.intake.tasks.banks import instantiate_email, load_banks
from tau2.domains.intake.tasks.generator import (
    CANONICAL_BANKS,
    FreezeManifest,
    GeneratorError,
    _gate_task,
    entry_draw_key,
)
from tau2.domains.intake.user_data_model import IntakeUserDB
from tau2.multilingual.localization import (
    IntakeCallerSurface,
    LocalizationSource,
    SourceDocument,
    SourceEntry,
    SourceOccurrence,
    json_content_digest,
)

INTAKE_LOCALIZATION_VERSION = "intake-selected-v1"
INTAKE_ENTRY_AUTHORING_CONTEXT = """intake-selected-v1: Only the frozen task selection is in scope. Reuse reviewed matching entities from supplied reference artifacts where their assigned family, difficulty, pronunciation and spelling hazards fit; otherwise author a draft candidate. Keep internal instructions in English and localize spoken examples and entity forms. Native and romanized views refer to the same entity; romanization is review metadata, never an accepted submitted answer. Preserve numeric medication strength and unit strings, code/checksum rules, protocol field names and required ISO date/time formats. Keep {org} exactly once in the callback opener. Preserve the dual-form date annotation verbatim except its spoken phrase. Localized pronunciation tokens must occur in the mapped written value, their reference is review-only, and their distortion is natural native orthography suitable for TTS. Choose grapheme matching for tokens in unsegmented Chinese/Korean speech and word matching where lexical word boundaries are reliable. Re-author style/omission instructions only where applicable to the new written value and locale; use an empty string when inapplicable rather than importing English letter names. Every candidate, decoy and audio distortion requires native review; do not claim approval."""
INTAKE_POLICIES = (
    "agent_directed_policy",
    "scaffolded_policy",
    "scaffolded_policy_text",
)


def validate_intake_tasks(
    tasks: list[Task], language: str = "en", representation: str = "native"
) -> None:
    """Replay every gold submission and reject caller/record drift in a full frame."""
    cells = Counter()
    for task in tasks:
        parts = task.id.removeprefix("intake_").rsplit("_", 2)
        if (
            len(parts) != 3
            or parts[0] not in CANONICAL_BANKS
            or parts[1] not in {"easy", "hard"}
        ):
            raise ValueError(f"Invalid canonical intake task identity: {task.id}")
        cells[tuple(parts[:2])] += 1
        try:
            _gate_task(task, require_ascii=False)
        except GeneratorError as error:
            raise ValueError(str(error)) from error
        actions = {
            action.func_name: action.arguments
            for action in task.initial_state.initialization_actions
        }
        record = actions["seed_record"]["record"]
        submission = task.evaluation_criteria.actions[0].arguments["fields"]
        for field in record["fields"]:
            if field["name"] not in submission:
                continue
            value = actions["set_entities"]["entities"][field["name"]]
            from tau2.domains.intake.date_display import (
                locale_date_format,
                matches_date_payload,
            )
            from tau2.domains.intake.spell_events import entity_match_texts
            from tau2.domains.intake.utils import spoken_form

            local_date_matches = (
                field["fold"] == FoldKind.DATE.value
                and locale_date_format(language, representation) is not None
                and matches_date_payload(
                    value,
                    submission[field["name"]],
                    locale_date_format(language, representation),
                )
            )
            from tau2.domains.intake.time_display import (
                display_time,
                display_time_payload,
                locale_time_format,
            )
            from tau2.domains.intake.utils import render_time_dual_form

            time_style = locale_time_format(language, representation)
            local_time_matches = (
                field["fold"] == FoldKind.TIME.value
                and time_style is not None
                and (
                    value == display_time_payload(submission[field["name"]], time_style)
                    or value
                    == render_time_dual_form(
                        spoken_form(value),
                        display_time(submission[field["name"]], time_style),
                    )
                )
            )
            if (
                not local_date_matches
                and not local_time_matches
                and value != submission[field["name"]]
                and (
                    spoken_form(value) == value
                    or entity_match_texts(value)[-1] != submission[field["name"]]
                )
            ):
                raise ValueError(
                    f"Task {task.id}: caller value disagrees with golden submission"
                )
            if fold_value(FoldKind(field["fold"]), field["value"]) != fold_value(
                FoldKind(field["fold"]), submission[field["name"]]
            ):
                raise ValueError(
                    f"Task {task.id}: golden submission differs from seeded record"
                )
    expected = {
        (bank, tier): 10 for bank in CANONICAL_BANKS for tier in ("easy", "hard")
    }
    if cells != expected or len({task.id for task in tasks}) != 200:
        raise ValueError(
            "Intake localization requires the frozen 200-task frame, ten slots per family/tier"
        )


def inventory_intake(data_root: Path) -> LocalizationSource:
    """Inventory used task values and context with exact, role-scoped occurrences.

    Protocol keys, format descriptions and strategy instructions stay English.
    Names, context, callback openers and caller scenarios require explicit maps.
    The task source is frozen; no unused bank entry enters this inventory.
    """
    directory = data_root / "tau2" / "domains" / "intake"
    tasks = IntakeTaskFile.model_validate_json(
        (directory / "tasks.json").read_text()
    ).tasks
    validate_intake_tasks(tasks)
    documents = []
    for identifier, filename, kind in (
        ("database", "db.toml", "database"),
        ("user_database", "user_db.toml", "user_database"),
    ):
        content = tomllib.loads((directory / filename).read_text())
        (IntakeDB if kind == "database" else IntakeUserDB).model_validate(content)
        documents.append(
            SourceDocument(
                id=identifier,
                kind=kind,
                source_path=f"tau2/domains/intake/{filename}",
                sha256=json_content_digest(content),
                content=content,
            )
        )
    content = TypeAdapter(list[Task]).dump_python(tasks, mode="json")
    documents.append(
        SourceDocument(
            id="tasks",
            kind="tasks",
            source_path="tau2/domains/intake/tasks.json",
            sha256=json_content_digest(content),
            content=content,
        )
    )
    for name in INTAKE_POLICIES:
        content = (directory / f"{name}.md").read_text()
        documents.append(
            SourceDocument(
                id=f"policy:{name}",
                kind="policy",
                source_path=f"tau2/domains/intake/{name}.md",
                sha256=json_content_digest(content),
                content=content,
            )
        )

    selected = selected_source_data(directory)
    name_genders = {
        value: entry.gender.value
        for entry in load_banks(directory / "banks").person_names
        for value in (entry.value, entry.alt_name)
        if value is not None
    }
    content = selected.model_dump(mode="json")
    documents.append(
        SourceDocument(
            id="intake_localization",
            kind="dependency",
            source_path="tau2/domains/intake/tasks.manifest.json#selected-stimuli-v1",
            sha256=json_content_digest(content),
            content=content,
        )
    )

    entries: dict[str, SourceEntry] = {}

    def walk(document, value, path, role="protocol", category="protocol"):
        if isinstance(value, dict) and value:
            for key, item in value.items():
                child_role, child_category = role, category
                if document.id == "tasks":
                    if key == "callee_full_name":
                        child_role, child_category = "callee", "identity_name"
                    elif key == "org_name":
                        child_role, child_category = "organization", "product_name"
                    elif (
                        key == "fields" and isinstance(item, dict) or key == "entities"
                    ):
                        child_role, child_category = "field", "other"
                    elif key == "value" and "fields" in path:
                        child_role, child_category = f"field:{value['name']}", "other"
                    elif role == "field":
                        child_role = f"field:{key}"
                if document.id == "intake_localization":
                    if key in {
                        "caller_greeting",
                        "agent_opener_template",
                        "clarification_guidance",
                    }:
                        child_role, child_category = key, "prose"
                    elif key in {
                        "decoys",
                        "pronunciations",
                        "spelling_styles",
                        "omission_instruction",
                    }:
                        task_id = value["canonical_task_id"]
                        identifier = (
                            f"other:{json_content_digest([task_id, key, item])}"
                        )
                        entries[identifier] = SourceEntry(
                            id=identifier,
                            category="other" if item else "protocol",
                            source=item,
                            occurrences=[
                                SourceOccurrence(
                                    document_id=document.id, path=[*path, key]
                                )
                            ],
                            context=f"intake stimulus:{task_id}:{key}; entity={task_id}; dependency={key}; selected="
                            + json.dumps(
                                {
                                    name: value[name]
                                    for name in (
                                        "canonical_task_id",
                                        "family",
                                        "tier",
                                        "field_name",
                                        "written",
                                    )
                                },
                                ensure_ascii=False,
                            )
                            + "; "
                            + INTAKE_ENTRY_AUTHORING_CONTEXT,
                        )
                        continue
                    elif key == "written":
                        child_role, child_category = (
                            f"field:{value['field_name']}",
                            "other",
                        )
                    elif key == "digit_words":
                        child_role, child_category = "digit_words", "other"
                        identifier = f"other:{json_content_digest([child_role, item])}"
                        entries[identifier] = SourceEntry(
                            id=identifier,
                            category="other",
                            source=item,
                            occurrences=[
                                SourceOccurrence(
                                    document_id=document.id, path=[*path, key]
                                )
                            ],
                            context="Reviewed spoken digit aliases. Keep ASCII digit keys and localize their spoken names; effort detection only.",
                        )
                        continue
                walk(document, item, [*path, key], child_role, child_category)
        elif isinstance(value, list) and value:
            for index, item in enumerate(value):
                walk(document, item, [*path, index], role, category)
        else:
            if not isinstance(value, str):
                category = "protocol"
            # Deduplicate only matching semantic roles. Equal strings in unrelated
            # fields are separate identities; repeated source occurrences stay linked.
            identifier = f"{category}:{json_content_digest([role, value])}"
            occurrence = SourceOccurrence(document_id=document.id, path=path)
            if identifier in entries:
                entries[identifier].occurrences.append(occurrence)
            else:
                entries[identifier] = SourceEntry(
                    id=identifier,
                    category=category,
                    source=value,
                    occurrences=[occurrence],
                    context=f"intake {role}; {INTAKE_ENTRY_AUTHORING_CONTEXT}",
                )

    for document in documents:
        walk(document, document.content, [])
    # Measurement generation must see the frozen family/tier before authoring a
    # written value, not discover those constraints only in later dependencies.
    for entry in entries.values():
        if isinstance(entry.source, str) and (
            entry.category == "identity_name"
            or entry.context.startswith("intake field:second_guest_full_name;")
        ):
            if entry.source not in name_genders:
                raise ValueError(
                    f"Used intake name lacks reviewed gender: {entry.source}"
                )
            entry.context += (
                "; reviewed_source_gender="
                + name_genders[entry.source]
                + "; Preserve this person's reviewed source gender in the localized name. "
                "A measured second guest is separate from the callee; never copy the callee's gender."
            )
        selected_rows = sorted(
            {
                occurrence.path[1]
                for occurrence in entry.occurrences
                if occurrence.document_id == "intake_localization"
                and len(occurrence.path) == 3
                and occurrence.path[0] == "stimuli"
                and occurrence.path[2] == "written"
            }
        )
        if selected_rows:
            entry.context += "; measured_selection=" + json.dumps(
                [
                    {
                        name: getattr(selected.stimuli[index], name)
                        for name in (
                            "canonical_task_id",
                            "family",
                            "tier",
                            "field_name",
                            "written",
                        )
                    }
                    for index in selected_rows
                ],
                ensure_ascii=False,
            )
    return LocalizationSource(
        domain="intake",
        documents=documents,
        entries=list(entries.values()),
    )


def apply_native_folds(tasks: list[Task]) -> list[Task]:
    """Use Unicode-preserving writes on compiled native records; source is untouched."""
    from tau2.domains.intake.scoring import project_scoring_folds

    compiled = [task.model_copy(deep=True) for task in tasks]
    for task in compiled:
        for action in task.initial_state.initialization_actions:
            if action.func_name != "seed_record":
                continue
            for field in action.arguments["record"]["fields"]:
                if field["fold"] == FoldKind.NAME.value:
                    field["fold"] = FoldKind.NATIVE_NAME.value
                elif field["fold"] == FoldKind.NAME_DIGITS.value:
                    field["fold"] = FoldKind.NATIVE_NAME_DIGITS.value
                elif field["fold"] == FoldKind.NAME_LIST.value:
                    field["fold"] = FoldKind.NATIVE_NAME_LIST.value
    return [project_scoring_folds(task) for task in compiled]


def selected_source_data(directory: Path) -> IntakeLocalizationData:
    """Resolve only the 200 recorded draws and their own decoys/pronunciations."""
    manifest = FreezeManifest.model_validate_json(
        (directory / "tasks.manifest.json").read_text()
    )
    banks = load_banks(directory / "banks")
    from tau2.domains.intake.complications import (
        _OMISSION_FEASIBLE,
        COMPLICATION_LINES,
        LAZY_OMISSION_COMPONENTS,
        _feasible_styles,
    )

    rows = []
    for draw in manifest.tasks:
        key = draw.email_pattern if draw.bank == "emails" else draw.value
        matches = [
            entry
            for entry in getattr(banks, draw.bank)
            if entry.difficulty == draw.tier and entry_draw_key(draw.bank, entry) == key
        ]
        if len(matches) != 1:
            raise ValueError(f"Selected intake bank source drift: {draw.task_id}")
        entry = matches[0]
        if banks.shas[draw.bank] != manifest.bank_shas[draw.bank]:
            raise ValueError(f"Frozen intake bank digest changed: {draw.bank}")
        resolved = (
            instantiate_email(entry, draw.callee_full_name)
            if draw.bank == "emails"
            else entry
        )
        pronunciations = [
            SelectedPronunciation(
                token=p.token,
                reference=p.respelling,
                distorted=p.mispronounced,
                reason=p.operator.value,
            )
            for p in getattr(entry, "pronunciations", [])
            if p.mispronounced
        ]
        spoken = draw.spoken or draw.value
        styles = [
            COMPLICATION_LINES[f"spelling_style.{style.value}"]
            for style in _feasible_styles(draw.bank, spoken)
        ]
        component = LAZY_OMISSION_COMPONENTS.get(draw.bank)
        omission = (
            COMPLICATION_LINES[f"lazy_omission.{component}"]
            if component and _OMISSION_FEASIBLE[component].search(spoken)
            else ""
        )
        rows.append(
            SelectedStimulus(
                canonical_task_id=draw.task_id,
                family=draw.bank,
                tier=draw.tier.value,
                field_name=draw.field_name,
                written=draw.value,
                decoys=resolved.decoys
                or [
                    other.value
                    for other in manifest.tasks
                    if other.bank == draw.bank
                    and other.tier == draw.tier
                    and other.value != draw.value
                ][:1],
                pronunciations=pronunciations,
                spelling_styles=styles,
                omission_instruction=omission,
            )
        )
    return IntakeLocalizationData(
        caller_greeting="Hello?",
        agent_opener_template="Hello, this is {org} calling.",
        clarification_guidance="Give written clarification only when requested. Repetition alone is not a request to spell. Use the exact written value returned by get_entity and the script's natural character or syllable clarification conventions; never volunteer a transliteration or pronunciation guide.",
        digit_words={
            str(i): word
            for i, word in enumerate(
                (
                    "zero",
                    "one",
                    "two",
                    "three",
                    "four",
                    "five",
                    "six",
                    "seven",
                    "eight",
                    "nine",
                )
            )
        },
        stimuli=rows,
    )


def compile_intake_documents(
    documents: list[SourceDocument],
    language: str,
    source_documents: list[SourceDocument],
    representation: str = "native",
    *,
    caller_surfaces: list[IntakeCallerSurface] | None = None,
) -> None:
    """Compile localized identity/context and the shared reviewed call frame together."""
    from tau2.domains.intake.call_frame import (
        USER_TASK_INSTRUCTIONS,
        render_user_scenario,
    )

    source_tasks = TypeAdapter(list[Task]).validate_python(
        next(
            document.content for document in source_documents if document.id == "tasks"
        )
    )
    original_records = {
        task.id: next(
            action.arguments["record"]
            for action in task.initial_state.initialization_actions
            if action.func_name == "seed_record"
        )
        for task in source_tasks
    }
    by_id = {document.id: document for document in documents}
    content = IntakeLocalizationData.model_validate(
        by_id["intake_localization"].content
    )
    surfaces = {surface.canonical_task_id: surface for surface in caller_surfaces or []}
    tasks = TypeAdapter(list[Task]).validate_python(by_id["tasks"].content)
    if language != "en":
        tasks = apply_native_folds(tasks)
    for task in tasks:
        calls = {
            call.func_name: call.arguments
            for call in task.initial_state.initialization_actions
        }
        record = calls["seed_record"]["record"]
        originals = {
            field["name"]: field for field in original_records[task.id]["fields"]
        }
        for field in record["fields"]:
            original = originals[field["name"]]
            if (
                original["fold"] in {FoldKind.DATE.value, FoldKind.TIME.value}
                and field["value"] != original["value"]
            ):
                raise ValueError(
                    f"Task {task.id}: localization changed the protocol date/time value"
                )
            if (
                original["fold"]
                in {FoldKind.EMAIL.value, FoldKind.CODE.value, FoldKind.PHONE.value}
                and field["name"] != "license_plate"
                and isinstance(field["value"], str)
                and not field["value"].isascii()
            ):
                raise ValueError(
                    f"Task {task.id}: protocol identifiers must remain ASCII"
                )
            if (
                original["fold"] == FoldKind.EMAIL.value
                and isinstance(field["value"], str)
                and re.fullmatch(
                    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", field["value"]
                )
                is None
            ):
                raise ValueError(f"Task {task.id}: invalid protocol email")
            if "medication" in field["name"]:
                if medication_strengths(field["value"] or "") != medication_strengths(
                    original["value"] or ""
                ):
                    raise ValueError(
                        f"Task {task.id}: localization changed medication strength or units"
                    )
        from tau2.domains.intake.date_display import (
            display_date_payload,
            locale_date_format,
        )

        if locale_date_format(language, representation) is not None:
            date_fields = {
                field["name"]
                for field in record["fields"]
                if field["fold"] == FoldKind.DATE.value
            }
            for name in date_fields & calls["set_entities"]["entities"].keys():
                calls["set_entities"]["entities"][name] = display_date_payload(
                    calls["set_entities"]["entities"][name],
                    locale_date_format(language, representation),
                )
        from tau2.domains.intake.time_display import (
            TIME_CALLER_GUIDANCE,
            display_time,
            display_time_payload,
            locale_time_format,
        )

        time_style = locale_time_format(language, representation)
        if time_style is not None:
            time_fields = {
                field["name"]
                for field in record["fields"]
                if field["fold"] == FoldKind.TIME.value
            }
            for name in time_fields & calls["set_entities"]["entities"].keys():
                calls["set_entities"]["entities"][name] = display_time_payload(
                    calls["set_entities"]["entities"][name], time_style
                )
            if surface := surfaces.get(task.id):
                from tau2.domains.intake.utils import render_time_dual_form

                stimulus = content.for_task(task.id)
                if stimulus.written != surface.canonical_value:
                    raise ValueError(
                        f"Task {task.id}: caller surface canonical value differs from gold"
                    )
                stimulus.caller_spoken = getattr(surface, f"{representation}_spoken")
                calls["set_entities"]["entities"][stimulus.field_name] = (
                    render_time_dual_form(
                        stimulus.caller_spoken,
                        display_time(surface.canonical_value, time_style),
                    )
                )
        organization = calls["create_callback_order"]["org_name"]
        task.user_scenario = render_user_scenario(
            record["callee_full_name"],
            organization,
            list(calls["set_entities"]["entities"]),
        )
        task.user_scenario.instructions.task_instructions = (
            USER_TASK_INSTRUCTIONS.replace(
                '"Hello?"', json.dumps(content.caller_greeting, ensure_ascii=False), 1
            )
        )
        if language != "en":
            task.user_scenario.instructions.task_instructions += (
                "\n\n# Intake written clarification\n" + content.clarification_guidance
            )
        if time_style and time_fields & calls["set_entities"]["entities"].keys():
            task.user_scenario.instructions.task_instructions += (
                "\n\n" + TIME_CALLER_GUIDANCE
            )
        task.agent_opener = content.agent_opener_template.format(org=organization)
        stimulus = content.for_task(task.id)
        submission = task.evaluation_criteria.actions[0].arguments["fields"]
        if stimulus.written != submission.get(stimulus.field_name):
            raise ValueError(f"Task {task.id}: selected stimulus differs from gold")
        measured = next(
            field for field in record["fields"] if field["name"] == stimulus.field_name
        )
        kind = FoldKind(measured["fold"])
        if any(
            fold_value(kind, decoy) == fold_value(kind, stimulus.written)
            for decoy in stimulus.decoys
        ):
            raise ValueError(f"Task {task.id}: decoy collapses to the correct answer")
        from tau2.domains.intake.check_digits import is_valid_member_id, is_valid_vin

        check = {"vin": is_valid_vin, "member_id": is_valid_member_id}.get(
            stimulus.field_name
        )
        if check is not None and not all(
            check(value) for value in [stimulus.written, *stimulus.decoys]
        ):
            raise ValueError(f"Task {task.id}: invalid localized code checksum")
        if any(p.token not in stimulus.written for p in stimulus.pronunciations):
            raise ValueError(
                f"Task {task.id}: pronunciation token is absent from the written value"
            )
    from tau2.domains.intake.date_display import (
        display_date_payload,
        locale_date_format,
        omission_instruction,
    )

    if locale_date_format(language, representation) is not None:
        style = locale_date_format(language, representation)
        for stimulus in content.stimuli:
            stimulus.date_display_format = style
            if stimulus.family != "dates":
                continue
            stimulus.decoys = [
                display_date_payload(value, style) for value in stimulus.decoys
            ]
            if stimulus.omission_instruction:
                stimulus.omission_instruction = omission_instruction(
                    stimulus.written, style
                )
    from tau2.domains.intake.time_display import (
        display_time,
        locale_time_format,
    )
    from tau2.domains.intake.time_display import (
        omission_instruction as time_omission_instruction,
    )

    time_style = locale_time_format(language, representation)
    if time_style is not None:
        for stimulus in content.stimuli:
            stimulus.time_display_format = time_style
            if stimulus.family != "times":
                continue
            stimulus.decoys = [
                display_time(value, time_style) for value in stimulus.decoys
            ]
            if stimulus.omission_instruction:
                stimulus.omission_instruction = time_omission_instruction(
                    stimulus.written, time_style
                )
                if stimulus.caller_spoken:
                    from tau2.domains.intake.time_display import time_expression

                    initial = time_expression(
                        stimulus.written, time_style
                    ) or display_time(
                        stimulus.written, time_style, include_period=False
                    )
                    stimulus.omission_instruction = (
                        stimulus.omission_instruction.replace(
                            f'"{initial}"', f'"{stimulus.caller_spoken}"'
                        )
                    )
    by_id["intake_localization"].content = content.model_dump(mode="json")
    validate_intake_tasks(tasks, language, representation)
    by_id["tasks"].content = TypeAdapter(list[Task]).dump_python(tasks, mode="json")


def medication_strengths(value: str):
    """Compare numeric strengths without treating a locale decimal comma as a dose change."""
    from decimal import Decimal

    pattern = r"([0-9]+(?:[.,][0-9]+)?)\s*(mg|mcg|g|mL|ml|%)"
    return [
        (Decimal(number.replace(",", ".")), unit)
        for number, unit in re.findall(pattern, value)
    ]
