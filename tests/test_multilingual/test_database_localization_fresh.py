"""Fresh regional drafting: dependency order, guarded prose, repair and failures."""

import json
from types import SimpleNamespace

import pytest

from tau2.multilingual.factory import database_localization as drafting
from tau2.multilingual.factory.generation_config import FactoryModelConfig
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


def fresh_request(previous):
    return drafting.RegionalDraftRequest(
        source=previous.source,
        settings=previous.settings,
        revision="repaired",
        generation=FactoryModelConfig(model="fixture", reasoning_effort="low"),
    )


def test_draft_english_guard_retries_wholesale_translation(
    previous, monkeypatch, tmp_path
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
    )
    request = fresh_request(previous)
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
    assert [call["call_name"] for call in calls] == [
        drafting.DATABASE_LOCALIZATION_CALL_NAME,
        drafting.DATABASE_LOCALIZATION_REPAIR_CALL_NAME,
    ]
    assert all(call["reasoning_effort"] == "low" for call in calls)
    # Short entity-only fragments are intentionally outside this conservative guard.
    drafting._require_english_prose("Your name is Ada.", "你的名字是林静。", "short")


def test_draft_state_groups_use_source_city_and_wait_for_city_mapping(previous):
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


def test_draft_city_prose_receives_airport_context(previous):
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


def test_fresh_request_and_checkpoints_record_guarded_prompt(
    previous, monkeypatch, tmp_path
):
    request = fresh_request(previous)
    assert set(request.model_dump()) == {
        "email_rules_version",
        "user_id_rules_version",
        "source",
        "settings",
        "revision",
        "generation",
        "references",
        "reused_entries",
        "batch_size",
        "max_concurrency",
    }
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        payload = json.loads(kwargs["messages"][1].content)
        assert all("text_language" in e for e in payload["entries"])
        assert kwargs["messages"][0].content == drafting.DATABASE_LOCALIZATION_PROMPT
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
                            "notes": "Fixture.",
                        }
                        for e in payload["entries"]
                    ]
                }
            )
        )

    monkeypatch.setattr(drafting, "generate", generate)
    first = drafting.draft_database_localization(
        request, tmp_path / "first.json", tmp_path / "checkpoints"
    )
    assert (
        first.provenance.prompt_version == drafting.DATABASE_LOCALIZATION_PROMPT_VERSION
    )
    assert first.provenance.email_rules_version == "regional-email-v1"
    assert first.provenance.user_id_rules_version == "regional-user-ids-v1"
    assert first.revision == request.revision
    assert first.user_id_revision is not None
    assert first.provenance.input_sha256 == content_digest(request)
    assert first.provenance.parent_map_sha256 is None
    assert calls
    monkeypatch.setattr(
        drafting,
        "generate",
        lambda **kw: pytest.fail("Original checkpoints must remain reusable"),
    )

    assert (
        drafting.draft_database_localization(
            request, tmp_path / "second.json", tmp_path / "checkpoints"
        )
        == first
    )


def test_fresh_draft_resolves_city_before_state(previous, monkeypatch, tmp_path):
    request = fresh_request(previous)
    for entry in request.source.entries:
        if entry.occurrences[0].path[-1] in {"city", "state"}:
            entry.category = "address"
    calls = []

    def generate(**kwargs):
        payload = json.loads(kwargs["messages"][1].content)
        calls.append(payload)
        for entry in payload["entries"]:
            if "role=state;" in entry["context"]:
                assert not any("role=city;" in e["context"] for e in payload["entries"])
                assert any(
                    "role=city;" in e["context"] for e in payload["completed_mappings"]
                )
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
    result = drafting.draft_database_localization(
        request, tmp_path / "draft.json", tmp_path / "checkpoints"
    )
    stages = [[e["context"].split(";")[0] for e in call["entries"]] for call in calls]
    assert stages.index(["role=city"]) < stages.index(["role=state"])
    assert (
        result.provenance.prompt_version
        == drafting.DATABASE_LOCALIZATION_PROMPT_VERSION
    )


def test_fresh_output_reuse_rejects_old_prompt_version(previous, monkeypatch, tmp_path):
    request = fresh_request(previous)
    previous.provenance.input_sha256 = content_digest(request)
    previous.provenance.prompt_version = "regional-database-localization-v1"
    output = tmp_path / "old.json"
    output.write_text("placeholder")
    monkeypatch.setattr(drafting, "read_localization_map", lambda path: previous)
    monkeypatch.setattr(
        drafting, "generate", lambda **kw: pytest.fail("No generation on stale output")
    )
    with pytest.raises(ValueError, match="prompt version"):
        drafting.draft_database_localization(request, output, tmp_path / "checkpoints")


def test_original_final_failure_is_saved_and_reconstructed_without_model_calls(
    previous, monkeypatch, tmp_path
):
    request = fresh_request(previous)

    def generate(**kwargs):
        payload = json.loads(kwargs["messages"][1].content)
        rows = []
        for e in payload["entries"]:
            value = "60601" if e["id"] == "wrong-zip" else e["source"]
            rows.append(
                {
                    "id": e["id"],
                    "forms": {
                        "native": value,
                        "romanized": value,
                        "english": "Fixture.",
                    },
                    "operation": "adapt" if value != e["source"] else "preserve",
                    "notes": "",
                }
            )
        return SimpleNamespace(content=json.dumps({"entries": rows}))

    monkeypatch.setattr(drafting, "generate", generate)
    output = tmp_path / "original.json"
    with pytest.raises(ValueError, match="authentication"):
        drafting.draft_database_localization(request, output, tmp_path / "checkpoints")
    failure = (tmp_path / "original.failed.json").read_bytes()
    monkeypatch.setattr(
        drafting,
        "generate",
        lambda **kw: pytest.fail("Successful original batches must be retained"),
    )
    with pytest.raises(ValueError, match="authentication"):
        drafting.draft_database_localization(request, output, tmp_path / "checkpoints")
    assert (tmp_path / "original.failed.json").read_bytes() == failure
    assert not output.exists()


def test_draft_prose_preserves_record_ids_and_clock_values(
    previous, monkeypatch, tmp_path
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
    request = fresh_request(previous)
    batch = drafting.BatchInput(
        request_sha256="a" * 64,
        entries=[entry],
        completed=[],
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


def test_draft_short_answer_gets_product_context_and_is_not_globally_grouped(
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
