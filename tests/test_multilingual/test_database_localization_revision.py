"""Targeted revision preserves valid work and repairs failed task semantics."""

import json
from types import SimpleNamespace

import pytest

from tau2.multilingual.factory import database_localization as drafting
from tau2.multilingual.factory.generation_config import FactoryModelConfig
from tau2.multilingual.factory.localization_artifact import RegionalMapArtifact
from tau2.multilingual.factory.localization_compile import validate_localization
from tau2.multilingual.localization import (
    GenerationProvenance,
    LocalizationEntry,
    LocalizationMap,
    LocalizationSource,
    LocalizedForms,
    RegionalSettings,
    SourceDocument,
    SourceEntry,
    SourceOccurrence,
    json_content_digest,
)
from tau2.utils.artifacts import content_digest


@pytest.fixture
def previous():
    user = {
        "user_id": "u1",
        "name": {"first_name": "Ada", "last_name": "Miller"},
        "address": {
            "address1": "1 Main Street",
            "address2": "Unit 2",
            "city": "Chicago",
            "country": "USA",
            "state": "IL",
            "zip": "60601",
        },
        "email": "ada@example.com",
        "payment_methods": {},
        "orders": [],
    }
    task = {
        "id": "t1",
        "user_scenario": {
            "instructions": {
                "domain": "retail",
                "task_instructions": "Ask for help with your order.",
                "reason_for_call": "Help with an order.",
                "known_info": "Try your incorrect postal code first.",
            }
        },
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": "lookup",
                    "name": "find_user_id_by_name_zip",
                    "arguments": {
                        "first_name": "Ada",
                        "last_name": "Miller",
                        "zip": "60602",
                    },
                },
                {
                    "action_id": "inspect",
                    "name": "get_user_details",
                    "arguments": {"user_id": "u1"},
                },
            ],
            "reward_basis": ["DB"],
        },
    }
    documents = [
        SourceDocument(
            id=key,
            kind=kind,
            source_path=key + ".json",
            content=value,
            sha256=json_content_digest(value),
        )
        for key, kind, value in [
            ("db", "database", {"users": {"u1": user}, "orders": {}, "products": {}}),
            ("tasks", "tasks", [task]),
        ]
    ]
    entries = []

    def walk(value, document, path):
        if isinstance(value, dict) and value:
            for key, item in value.items():
                walk(item, document, path + [key])
        elif isinstance(value, list) and value:
            for key, item in enumerate(value):
                walk(item, document, path + [key])
        else:
            role = path[-1] if path else ""
            identifier = (
                "correct-zip"
                if value == "60601"
                else "wrong-zip"
                if value == "60602"
                else "prose"
                if role == "known_info"
                else document + str(path)
            )
            category = (
                "identity_name"
                if role in {"first_name", "last_name"}
                else "address"
                if identifier.endswith("-zip")
                else "prose"
                if identifier == "prose"
                else "protocol"
            )
            entry = SourceEntry(
                id=identifier,
                category=category,
                source=value,
                occurrences=[SourceOccurrence(document_id=document, path=path)],
                context=f"role={role}; entity=u1; Test source context.",
            )
            entries.append(
                drafting.automatic_localization(entry, region)
                or LocalizationEntry(
                    **entry.model_dump(),
                    forms=LocalizedForms(
                        native=value, romanized=value, english="Fixture only."
                    ),
                    operation="preserve",
                    notes="Fixture.",
                )
            )

    region = RegionalSettings(
        language="zh",
        country_code="CN",
        currency_code="CNY",
        currency_scale="1",
        time_zone="Asia/Shanghai",
        name_order="family_first",
    )
    for document in documents:
        walk(document.content, document.id, [])
    return LocalizationMap(
        revision="original",
        settings=region,
        source=LocalizationSource(
            domain="retail",
            documents=documents,
            entries=[
                SourceEntry.model_validate(
                    e.model_dump(include=set(SourceEntry.model_fields))
                )
                for e in entries
            ],
        ),
        entries=entries,
        provenance=GenerationProvenance(
            model="fixture",
            model_args={},
            prompt_version="original-v1",
            call_name="fixture",
            input_sha256="b" * 64,
        ),
    )


def revision(previous, ids):
    return drafting.RegionalRevision(
        previous=previous,
        redraft_ids=ids,
        request=drafting.RegionalDraftRequest(
            source=previous.source,
            settings=previous.settings,
            revision="repaired",
            generation=FactoryModelConfig(model="fixture", reasoning_effort="low"),
        ),
        feedback=[
            drafting.LocalizationReference(
                name="review",
                content="The incorrect postal code must remain incorrect.",
            )
        ],
    )


def test_revision_repairs_wrong_zip_and_selected_prose_only(
    previous, monkeypatch, tmp_path
):
    wrong = next(e for e in previous.entries if e.id == "wrong-zip")
    wrong.forms.native = wrong.forms.romanized = "60601"
    wrong.operation = "adapt"
    with pytest.raises(ValueError, match="authentication"):
        validate_localization(previous)
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        payload = json.loads(kwargs["messages"][1].content)
        rows = []
        for entry in payload["entries"]:
            value = (
                "60602"
                if entry["id"] == "wrong-zip"
                else "Try the incorrect postal code first, then correct it."
            )
            rows.append(
                {
                    "id": entry["id"],
                    "forms": {
                        "native": value,
                        "romanized": value,
                        "english": "Corrected fixture.",
                    },
                    "operation": "preserve" if value == entry["source"] else "adapt",
                    "notes": "Fixed fixture.",
                }
            )
        return SimpleNamespace(content=json.dumps({"entries": rows}))

    monkeypatch.setattr(drafting, "generate", generate)
    result = drafting.revise_database_localization(
        revision(previous, ["wrong-zip", "prose"]),
        tmp_path / "revised.json",
        tmp_path / "checkpoints",
    )
    assert len(calls) == 2
    assert {
        e["id"] for c in calls for e in json.loads(c["messages"][1].content)["entries"]
    } == {"wrong-zip", "prose"}
    first = json.loads(calls[0]["messages"][1].content)
    assert any(e["id"] == "correct-zip" for e in first["completed_mappings"])
    assert all(
        c["call_name"] == "factory_database_localization_revision" for c in calls
    )
    assert all("equally convenient" in c["messages"][0].content for c in calls)
    before = {e.id: e for e in previous.entries}
    assert all(
        e == before[e.id] for e in result.entries if e.id not in {"wrong-zip", "prose"}
    )
    assert result.provenance.parent_map_sha256 == content_digest(previous)
    assert (
        result.provenance.prompt_version
        == drafting.DATABASE_LOCALIZATION_REVISION_VERSION
    )
    assert (
        RegionalMapArtifact.model_validate_json(
            (tmp_path / "revised.json").read_text()
        ).status
        == "draft"
    )
    monkeypatch.setattr(
        drafting,
        "generate",
        lambda **kw: pytest.fail("Validated checkpoints must be reused"),
    )
    again = drafting.revise_database_localization(
        revision(previous, ["wrong-zip", "prose"]),
        tmp_path / "repeated.json",
        tmp_path / "checkpoints",
    )
    assert again == result


@pytest.mark.parametrize("ids", [[], ["unknown"], ["wrong-zip", "wrong-zip"]])
def test_revision_rejects_invalid_selection_before_any_call(
    previous, monkeypatch, tmp_path, ids
):
    monkeypatch.setattr(
        drafting, "generate", lambda **kw: pytest.fail("No model call allowed")
    )
    with pytest.raises(ValueError):
        drafting.revise_database_localization(
            revision(previous, ids), tmp_path / "out.json", tmp_path / "checkpoints"
        )


def test_revision_rejects_stale_source(previous, monkeypatch, tmp_path):
    request = revision(previous, ["wrong-zip"])
    request.request.source = request.request.source.model_copy(deep=True)
    request.request.source.entries[0].context = "Changed source contract"
    monkeypatch.setattr(
        drafting, "generate", lambda **kw: pytest.fail("No model call allowed")
    )
    with pytest.raises(ValueError, match="source"):
        drafting.revise_database_localization(
            request, tmp_path / "out.json", tmp_path / "checkpoints"
        )


def test_failed_final_validation_records_proposal_without_publishing(
    previous, monkeypatch, tmp_path
):
    def generate(**kwargs):
        payload = json.loads(kwargs["messages"][1].content)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "entries": [
                        {
                            "id": e["id"],
                            "forms": {
                                "native": "60601",
                                "romanized": "60601",
                                "english": "Invalid fixture.",
                            },
                            "operation": "adapt",
                            "notes": "Fixture.",
                        }
                        for e in payload["entries"]
                    ]
                }
            )
        )

    monkeypatch.setattr(drafting, "generate", generate)
    output = tmp_path / "bad.json"
    with pytest.raises(ValueError, match="authentication"):
        drafting.revise_database_localization(
            revision(previous, ["wrong-zip"]), output, tmp_path / "checkpoints"
        )
    assert not output.exists()
    failed = drafting.FailedRegionalDraft.model_validate_json(
        (tmp_path / "bad.failed.json").read_text()
    )
    assert "authentication" in failed.validation_error
    assert failed.proposal.revision == "repaired"
    with pytest.raises(ValueError):
        RegionalMapArtifact.model_validate_json(
            (tmp_path / "bad.failed.json").read_text()
        )


@pytest.mark.parametrize(
    "prompt_version",
    [
        drafting.DATABASE_LOCALIZATION_PROMPT_VERSION,
        drafting.DATABASE_LOCALIZATION_REVISION_VERSION,
    ],
)
def test_revision_english_guard_retries_wholesale_translation(
    previous, monkeypatch, tmp_path, prompt_version
):
    source = "You want to change your order and you will provide the information when the agent asks for it. Do not accept another item if the requested item is unavailable."
    entry = SourceEntry(
        id="long-prose",
        category="prose",
        source=source,
        context="English task instructions.",
        occurrences=[SourceOccurrence(document_id="tasks", path=[0])],
    )
    batch = drafting.BatchInput(
        request_sha256="d" * 64,
        entries=[entry],
        completed=[],
        prompt_version=prompt_version,
    )
    request = revision(previous, ["prose"]).request
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        value = (
            "你想更改订单，并且会在客服询问时提供信息。" if len(calls) == 1 else source
        )
        return SimpleNamespace(
            content=json.dumps(
                {
                    "entries": [
                        {
                            "id": entry.id,
                            "forms": {
                                "native": value,
                                "romanized": value,
                                "english": "Fixture.",
                            },
                            "operation": "preserve" if value == source else "translate",
                            "notes": "",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr(drafting, "generate", generate)
    result = drafting._generate_batch(request, batch, tmp_path)
    assert len(calls) == 2
    assert result[0].forms.native == source
    assert (
        json.loads(calls[0]["messages"][1].content)["entries"][0]["text_language"]
        == "en"
    )
    assert "English instruction text" in calls[1]["messages"][-1].content
    # Short entity-only fragments are intentionally outside this conservative guard.
    drafting._require_english_prose("Your name is Ada.", "你的名字是林静。", "short")


def test_revision_state_groups_use_source_city_and_wait_for_city_mapping(previous):
    states = []
    for city in ("Boston", "Seattle"):
        states.append(
            SourceEntry(
                id=city,
                category="address",
                source="NY",
                context=f'role=state; entity={city}; source_address={{"city":"{city}","state":"NY","country":"USA"}};',
                occurrences=[SourceOccurrence(document_id="db", path=[city])],
            )
        )
    assert len(drafting._draft_groups(states)) == 2
    city = states[0].model_copy(
        update={
            "id": "city",
            "source": "Boston",
            "context": "role=city; entity=Boston;",
        }
    )
    street = states[0].model_copy(
        update={"id": "street", "context": "role=address1; entity=Boston;"}
    )
    assert drafting._stage(city) < drafting._stage(states[0]) < drafting._stage(street)
    city_map = LocalizationEntry(
        **city.model_dump(),
        forms=LocalizedForms(
            native="南京", romanized="Nanjing", english="Target city."
        ),
        operation="adapt",
        notes="Fixture.",
    )
    assert drafting._relevant_mappings([states[0]], [city_map]) == [city_map]


def test_revision_city_prose_receives_airport_context(previous):
    airport = SourceEntry(
        id="airport",
        category="airport",
        source="JFK",
        context="Source airport JFK; directory city=New York. Preserve reachable catchment.",
        occurrences=[SourceOccurrence(document_id="db", path=["airport"])],
    )
    completed = LocalizationEntry(
        **airport.model_dump(),
        forms=LocalizedForms(native="PEK", romanized="PEK", english="Beijing Capital."),
        operation="adapt",
        notes="Fixture.",
    )
    prose = next(e for e in previous.source.entries if e.id == "prose").model_copy(
        update={"source": "You live in New York."}
    )
    assert drafting._relevant_mappings([prose], [completed]) == [completed]


def test_targeted_revision_retains_existing_user_id_aliases(
    previous, monkeypatch, tmp_path
):
    from tau2.multilingual.factory.localization_user_ids import (
        finalize_localization_user_ids,
    )

    instructions = (
        "You are Ada Miller. Your user id is u1. Try your incorrect postal code first."
    )
    task_document = next(doc for doc in previous.source.documents if doc.id == "tasks")
    task_document.content[0]["user_scenario"]["instructions"]["known_info"] = (
        instructions
    )
    task_document.sha256 = json_content_digest(task_document.content)
    source_prose = next(
        entry for entry in previous.source.entries if entry.id == "prose"
    )
    source_prose.source = instructions
    prose = next(entry for entry in previous.entries if entry.id == "prose")
    prose.source = instructions
    prose.forms.native = instructions.replace("Ada Miller", "Miller Ada")
    prose.forms.romanized = prose.forms.native
    prose.operation = "adapt"
    previous = finalize_localization_user_ids(previous)
    requested = revision(previous, ["prose"])

    def generate(**kwargs):
        payload = json.loads(kwargs["messages"][1].content)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "entries": [
                        {
                            "id": entry["id"],
                            "forms": {
                                "native": entry["source"].replace(
                                    "Ada Miller", "Other Caller"
                                ),
                                "romanized": entry["source"].replace(
                                    "Ada Miller", "Other Caller"
                                ),
                                "english": "Revised fixture.",
                            },
                            "operation": "adapt",
                            "notes": "Selected prose only.",
                        }
                        for entry in payload["entries"]
                    ]
                }
            )
        )

    monkeypatch.setattr(drafting, "generate", generate)
    result = drafting.revise_database_localization(
        requested, tmp_path / "revised.json", tmp_path / "checks"
    )
    assert result.user_id_revision == previous.user_id_revision
    assert result.caller_name_revision.changed_entry_count == 1
    assert result.caller_name_revision != previous.caller_name_revision
    assert result.provenance.user_id_rules_version == "regional-user-ids-v1"
    revised_prose = next(entry for entry in result.entries if entry.id == "prose")
    expected = instructions.replace("u1", "ada_miller_1").replace(
        "Ada Miller", "Miller Ada"
    )
    assert revised_prose.forms.native == expected
    assert revised_prose.forms.romanized == revised_prose.forms.native
    assert {entry.id: entry for entry in result.entries if entry.id != "prose"} == {
        entry.id: entry for entry in previous.entries if entry.id != "prose"
    }


def test_targeted_revision_does_not_add_unrequested_caller_stage(
    previous, monkeypatch, tmp_path
):
    from tau2.multilingual.factory import localization_caller_names

    monkeypatch.setattr(
        localization_caller_names,
        "localized_caller_entries",
        lambda proposal: pytest.fail("Unfinalized narrow revision must keep its scope"),
    )
    monkeypatch.setattr(
        drafting,
        "_draft_entries",
        lambda request, pending, completed, *args: completed
        + [entry for entry in previous.entries if entry.id == "prose"],
    )
    result = drafting.revise_database_localization(
        revision(previous, ["prose"]), tmp_path / "revised.json", tmp_path / "checks"
    )
    assert result.caller_name_revision is None
    assert result.entries == previous.entries


def test_cli_repairs_explicit_failed_evidence_and_expands_categories(
    previous, monkeypatch, tmp_path
):
    from tau2.multilingual.factory import localization_cli, localization_inventory
    from tau2.multilingual.factory.localization_artifact import read_localization_map
    from tau2.utils.artifacts import write_artifact

    wrong = next(e for e in previous.entries if e.id == "wrong-zip")
    wrong.forms.native = wrong.forms.romanized = "60601"
    wrong.operation = "adapt"
    failed = tmp_path / "previous.failed.json"
    write_artifact(
        drafting.FailedRegionalDraft(
            proposal=previous, validation_error="authentication mismatch"
        ),
        failed,
    )
    with pytest.raises(ValueError):
        read_localization_map(failed)
    monkeypatch.setattr(
        localization_inventory, "inventory_localization", lambda *args: previous.source
    )
    request = localization_cli.DatabaseRevisionCommand(
        previous_file=failed.name,
        previous_kind="failed",
        revision="fixed",
        generation=FactoryModelConfig(model="fixture", reasoning_effort="low"),
        redraft_ids=["wrong-zip"],
        redraft_categories=["prose"],
    )
    path = tmp_path / "request.json"
    write_artifact(request, path)
    calls = []

    def generate(**kwargs):
        payload = json.loads(kwargs["messages"][1].content)
        calls.extend(e["id"] for e in payload["entries"])
        return SimpleNamespace(
            content=json.dumps(
                {
                    "entries": [
                        {
                            "id": e["id"],
                            "forms": {
                                "native": e["source"],
                                "romanized": e["source"],
                                "english": "Fixture.",
                            },
                            "operation": "preserve",
                            "notes": "",
                        }
                        for e in payload["entries"]
                    ]
                }
            )
        )

    monkeypatch.setattr(drafting, "generate", generate)
    output = tmp_path / "fixed.json"
    localization_cli.run_database_revise(
        SimpleNamespace(
            request=path, output=output, checkpoints=tmp_path / "checkpoints"
        )
    )
    assert set(calls) == {"wrong-zip", "prose"}
    assert output.exists()


@pytest.mark.parametrize(
    "prompt_version",
    [
        drafting.DATABASE_LOCALIZATION_PROMPT_VERSION,
        drafting.DATABASE_LOCALIZATION_REVISION_VERSION,
    ],
)
def test_revision_prose_preserves_record_ids_and_clock_values(
    previous, monkeypatch, tmp_path, prompt_version
):
    source = "Your user ID is u1. Contact us on 2024-05-15 at 15:00; use the full keyboard layout."
    task_doc = next(d for d in previous.source.documents if d.kind == "tasks")
    task_doc.content[0]["user_scenario"]["instructions"]["known_info"] = source
    task_doc.sha256 = json_content_digest(task_doc.content)
    entry = next(e for e in previous.source.entries if e.id == "prose")
    entry.source = source
    calls = []

    def generate(**kwargs):
        payload = json.loads(kwargs["messages"][1].content)
        calls.append(payload)
        text = (
            source.replace("u1", "lin_jing_1")
            if len(calls) == 1
            else source.replace("2024-05-15", "2024-05-16")
            if len(calls) == 2
            else source
        )
        return SimpleNamespace(
            content=json.dumps(
                {
                    "entries": [
                        {
                            "id": "prose",
                            "forms": {
                                "native": text,
                                "romanized": text,
                                "english": "Fixture.",
                            },
                            "operation": "adapt" if text != source else "preserve",
                            "notes": "",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr(drafting, "generate", generate)
    request = revision(previous, ["prose"]).request
    batch = drafting.BatchInput(
        request_sha256="a" * 64,
        entries=[entry],
        completed=[],
        prompt_version=prompt_version,
    )
    result = drafting._generate_batch(request, batch, tmp_path)
    assert len(calls) == 3
    assert result[0].forms.native == source
    context = calls[0]["entries"][0]["revision_context"]
    assert context["protected_identifiers"] == ["u1"]
    assert context["wall_clock_literals"] == ["15:00", "2024-05-15"]
    assert context["tasks"][0]["id"] == "t1"
    assert (
        "full keyboard"
        in context["tasks"][0]["user_scenario"]["instructions"]["known_info"]
    )


def test_revision_short_answer_gets_product_context_and_is_not_globally_grouped(
    previous,
):
    db = next(d for d in previous.source.documents if d.kind == "database")
    db.content["products"]["101"] = {
        "product_id": "101",
        "name": "Keyboard",
        "variants": {
            "201": {"options": {"layout": "full"}, "available": True, "price": 20}
        },
    }
    db.content["orders"]["#301"] = {"items": [{"product_id": "101", "item_id": "201"}]}
    tasks = next(d for d in previous.source.documents if d.kind == "tasks")
    tasks.content[0]["user_scenario"]["instructions"]["known_info"] = (
        "Exchange the keyboard in order #301."
    )
    tasks.content[0]["evaluation_criteria"]["communicate_info"] = ["full"]
    entry = SourceEntry(
        id="full-keyboard",
        category="prose",
        source="full",
        context="Scenario/evaluation prose for u1;",
        occurrences=[
            SourceOccurrence(
                document_id="tasks",
                path=[0, "evaluation_criteria", "communicate_info", 0],
            )
        ],
    )
    other = entry.model_copy(
        update={
            "id": "full-zipper",
            "occurrences": [
                SourceOccurrence(
                    document_id="tasks",
                    path=[1, "evaluation_criteria", "communicate_info", 0],
                )
            ],
        }
    )
    assert len(drafting._draft_groups([entry, other])) == 2
    context = drafting._entry_context(previous.source, [entry])[entry.id]
    assert context.referenced_products["101"]["name"] == "Keyboard"
    assert context.tasks[0]["evaluation_criteria"]["communicate_info"] == ["full"]


def _split_previous_zip_inventory(previous):
    """Make one prior scalar cover two exact occurrences, then split its ownership."""
    old = previous.model_copy(deep=True)
    right = next(e for e in old.source.entries if e.id == "wrong-zip")
    doc = next(d for d in old.source.documents if d.id == "tasks")
    doc.content[0]["evaluation_criteria"]["actions"][0]["arguments"]["zip"] = "60601"
    doc.sha256 = json_content_digest(doc.content)
    old.source.entries.remove(right)
    old.entries = [e for e in old.entries if e.id != right.id]
    source = next(e for e in old.source.entries if e.id == "correct-zip")
    row = next(e for e in old.entries if e.id == "correct-zip")
    source.occurrences += right.occurrences
    row.occurrences += right.occurrences
    target = old.source.model_copy(deep=True)
    common = next(e for e in target.entries if e.id == "correct-zip")
    common.occurrences = common.occurrences[:1]
    target.entries.append(
        common.model_copy(
            update={
                "id": "split-zip",
                "occurrences": right.occurrences,
                "context": "role=zip; entity=u1; Refined address owner.",
            }
        )
    )
    return old, target


def test_explicit_rebinding_splits_entries_and_preserves_every_proposal(previous):
    from tau2.multilingual.factory.localization_compile import (
        rebind_localization_inventory,
    )

    old, source = _split_previous_zip_inventory(previous)
    rebound = rebind_localization_inventory(old, source)
    before = {e.id: e for e in old.entries}
    after = {e.id: e for e in rebound.entries}
    assert after["split-zip"].forms == before["correct-zip"].forms
    assert after["split-zip"].operation == before["correct-zip"].operation
    assert after["split-zip"].notes == before["correct-zip"].notes
    for key in before.keys() - {"correct-zip"}:
        assert before[key] == after[key]
    assert old.source != source


@pytest.mark.parametrize(
    "defect", ["document", "missing", "category", "ambiguous_merge"]
)
def test_rebinding_rejects_changed_sources_and_conflicting_proposals(previous, defect):
    from tau2.multilingual.factory.localization_compile import (
        rebind_localization_inventory,
    )

    old, source = _split_previous_zip_inventory(previous)
    if defect == "document":
        source.documents[0].content["users"]["u1"]["email"] = "changed@example.com"
        source.documents[0].sha256 = json_content_digest(source.documents[0].content)
    elif defect == "missing":
        source.entries = [e for e in source.entries if e.id != "split-zip"]
    elif defect == "category":
        next(e for e in source.entries if e.id == "split-zip").category = "contact"
    else:
        split = rebind_localization_inventory(old, source)
        entry = next(e for e in split.entries if e.id == "split-zip")
        entry.forms.native = entry.forms.romanized = "changed"
        entry.operation = "adapt"
        source = old.source
        old = split
    with pytest.raises(ValueError):
        rebind_localization_inventory(old, source)


def test_inventory_revision_uses_original_parent_and_only_selected_model_calls(
    previous, monkeypatch, tmp_path
):
    from tau2.multilingual.factory.localization_compile import (
        rebind_localization_inventory,
    )

    old, source = _split_previous_zip_inventory(previous)
    rebound = rebind_localization_inventory(old, source)
    normal = revision(rebound, ["split-zip"])
    refresh = drafting.RegionalInventoryRevision(parent=old, revision=normal)
    calls = []

    def generate(**kwargs):
        payload = json.loads(kwargs["messages"][1].content)
        calls.extend(e["id"] for e in payload["entries"])
        return SimpleNamespace(
            content=json.dumps(
                {
                    "entries": [
                        {
                            "id": e["id"],
                            "forms": {
                                "native": e["source"],
                                "romanized": e["source"],
                                "english": "Selected fixture.",
                            },
                            "operation": "preserve",
                            "notes": "Regenerated selected split only.",
                        }
                        for e in payload["entries"]
                    ]
                }
            )
        )

    monkeypatch.setattr(drafting, "generate", generate)
    result = drafting.revise_database_localization(
        refresh, tmp_path / "revised.json", tmp_path / "checkpoints"
    )
    assert calls == ["split-zip"]
    assert result.provenance.parent_map_sha256 == content_digest(old)
    assert result.provenance.input_sha256 == content_digest(refresh)
    assert next(e for e in result.entries if e.id == "correct-zip") == next(
        e for e in rebound.entries if e.id == "correct-zip"
    )
    monkeypatch.setattr(
        drafting,
        "generate",
        lambda **kw: pytest.fail("Refreshed checkpoints must resume"),
    )
    replay = drafting.revise_database_localization(
        refresh, tmp_path / "replay.json", tmp_path / "checkpoints"
    )
    assert replay == result
    tampered = normal.model_copy(deep=True)
    tampered.previous.entries[0].notes = "unselected edit"
    with pytest.raises(ValueError, match="rebound"):
        drafting.RegionalInventoryRevision(parent=old, revision=tampered)


def test_cli_inventory_refresh_is_explicit_and_keeps_parent_lineage(
    previous, monkeypatch, tmp_path
):
    from tau2.multilingual.factory import localization_cli, localization_inventory
    from tau2.utils.artifacts import write_artifact

    old, source = _split_previous_zip_inventory(previous)
    snapshot = tmp_path / "prior.snapshot.json"
    write_artifact(old, snapshot)
    monkeypatch.setattr(
        localization_inventory, "inventory_localization", lambda *args: source
    )
    command = localization_cli.DatabaseRevisionCommand(
        previous_file=snapshot.name,
        previous_kind="snapshot",
        revision="refined",
        generation=FactoryModelConfig(model="fixture", reasoning_effort="low"),
        redraft_ids=["split-zip"],
    )
    request = tmp_path / "request.json"
    write_artifact(command, request)
    args = SimpleNamespace(
        request=request,
        output=tmp_path / "out.json",
        checkpoints=tmp_path / "checkpoints",
    )
    captured = []
    monkeypatch.setattr(
        drafting,
        "revise_database_localization",
        lambda value, *args: captured.append(value),
    )
    with pytest.raises(ValueError, match="explicit refresh_inventory"):
        localization_cli.run_database_revise(args)
    assert not captured
    command.refresh_inventory = True
    args.request = tmp_path / "refresh.json"
    write_artifact(command, args.request)
    localization_cli.run_database_revise(args)
    assert len(captured) == 1
    assert isinstance(captured[0], drafting.RegionalInventoryRevision)
    assert captured[0].parent == old
    assert captured[0].revision.request.source == source
    assert captured[0].revision.redraft_ids == ["split-zip"]
    with pytest.raises(ValueError, match="expanded snapshot"):
        localization_cli.DatabaseRevisionCommand(
            previous_file="old.json",
            refresh_inventory=True,
            revision="new",
            generation=FactoryModelConfig(model="fixture", reasoning_effort="low"),
            redraft_ids=["split-zip"],
        )


def test_oversized_invalid_batch_splits_with_exact_coverage_and_resume(
    previous, tmp_path, monkeypatch
):
    entries = [
        e for e in previous.source.entries if e.id in {"correct-zip", "wrong-zip"}
    ]
    request = revision(previous, ["correct-zip", "wrong-zip"]).request
    batch = drafting.BatchInput(request_sha256="a" * 64, entries=entries, completed=[])
    calls = []

    def generate(**kwargs):
        payload = json.loads(kwargs["messages"][1].content)
        selected = payload["entries"]
        calls.append([e["id"] for e in selected])
        rows = (
            []
            if len(selected) > 1
            else [
                {
                    "id": e["id"],
                    "forms": {
                        "native": e["source"],
                        "romanized": e["source"],
                        "english": "Preserved test value.",
                    },
                    "operation": "preserve",
                    "notes": "",
                }
                for e in selected
            ]
        )
        return SimpleNamespace(content=json.dumps({"entries": rows}))

    monkeypatch.setattr(drafting, "generate", generate)
    result = drafting._generate_batch(request, batch, tmp_path)
    assert {e.id for e in result} == {"correct-zip", "wrong-zip"}
    assert [len(c) for c in calls] == [2, 2, 2, 1, 1]
    parent = drafting.BatchArtifact.model_validate_json(
        (tmp_path / f"{content_digest(batch)}.json").read_text()
    )
    assert len(parent.child_batch_sha256s) == 2
    assert len(parent.rejected_candidates) == 3
    for digest in parent.child_batch_sha256s:
        assert (tmp_path / f"{digest}.json").exists()
    assert drafting._generate_batch(request, batch, tmp_path) == result
    assert len(calls) == 5


def test_banking_prose_subdivides_before_oversized_provider_call(
    previous, tmp_path, monkeypatch
):
    request = revision(previous, ["prose"]).request
    request = request.model_copy(
        update={
            "source": request.source.model_copy(update={"domain": "banking_knowledge"})
        }
    )
    original = next(e for e in previous.source.entries if e.id == "prose")
    entries = [original.model_copy(update={"id": f"prose:{i}"}) for i in range(9)]
    batch = drafting.BatchInput(request_sha256="a" * 64, entries=entries, completed=[])
    sizes = []
    monkeypatch.setattr(drafting, "DATABASE_LOCALIZATION_PROSE_MAX_ENTRIES", 2)

    def generate(**kwargs):
        selected = json.loads(kwargs["messages"][1].content)["entries"]
        sizes.append(len(selected))
        return SimpleNamespace(
            content=json.dumps(
                {
                    "entries": {
                        e["id"]: {
                            "forms": {
                                "native": e["source"],
                                "romanized": e["source"],
                                "english": "Unchanged test text.",
                            },
                            "operation": "preserve",
                            "notes": "",
                        }
                        for e in selected
                    }
                }
            )
        )

    monkeypatch.setattr(drafting, "generate", generate)
    result = drafting._generate_batch(request, batch, tmp_path)
    assert len(result) == 9
    assert max(sizes) == 2
    parent = drafting.BatchArtifact.model_validate_json(
        (tmp_path / f"{content_digest(batch)}.json").read_text()
    )
    assert len(parent.child_batch_sha256s) == 2
    assert parent.rejected_candidates == []
