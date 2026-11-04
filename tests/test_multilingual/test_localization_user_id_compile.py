"""Customer-ID aliases preserve task identity and actual account lookup outcomes."""

from copy import deepcopy

import pytest

from tau2.multilingual.factory.localization_compile import (
    compile_localization,
    validate_localization,
)
from tau2.multilingual.factory.localization_user_ids import (
    finalize_localization_user_ids,
)
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

CANONICAL_ID = "emma_kim_9957"
LOCALIZED_ID = "alba_gil_9957"


def _leaves(value, path=()):
    if isinstance(value, dict) and value:
        for key, child in value.items():
            yield from _leaves(child, (*path, key))
    elif isinstance(value, list) and value:
        for index, child in enumerate(value):
            yield from _leaves(child, (*path, index))
    else:
        yield list(path), value


@pytest.fixture
def localized_ids():
    address = {
        "address1": "1 Main Street",
        "address2": "",
        "city": "Chicago",
        "country": "USA",
        "state": "IL",
        "zip": "60601",
    }
    email = "emma.kim9957@example.com"
    database = {
        "users": {
            CANONICAL_ID: {
                "user_id": CANONICAL_ID,
                "name": {"first_name": "Emma", "last_name": "Kim"},
                "address": address,
                "email": email,
                "payment_methods": {},
                "orders": ["#ORDER1"],
            }
        },
        "orders": {
            "#ORDER1": {
                "order_id": "#ORDER1",
                "user_id": CANONICAL_ID,
                "address": address,
                "items": [],
                "status": "pending",
                "fulfillments": [],
                "payment_history": [],
            }
        },
        "products": {},
    }
    task = {
        # Even an unrelated ID with the same spelling must remain unchanged.
        "id": CANONICAL_ID,
        "user_scenario": {
            "instructions": f"You are Emma Kim. Your user id is {CANONICAL_ID}."
        },
        "initial_state": {
            "initialization_data": {
                "agent_data": {"users": {CANONICAL_ID: {"email": email}}}
            }
        },
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": CANONICAL_ID,
                    "name": "find_user_id_by_email",
                    "arguments": {"email": email},
                },
                {
                    "action_id": "inspect",
                    "name": "get_user_details",
                    "arguments": {"user_id": CANONICAL_ID},
                },
                {
                    "action_id": "wrong-email",
                    "name": "find_user_id_by_email",
                    "arguments": {"email": "wrong@example.com"},
                },
            ],
            "reward_basis": ["DB", "NL_ASSERTION"],
        },
    }
    documents = []
    entries = []
    sources = []
    for identifier, kind, content in (
        ("database", "database", database),
        ("tasks", "tasks", [task]),
    ):
        documents.append(
            SourceDocument(
                id=identifier,
                kind=kind,
                source_path=f"tau2/domains/retail/{identifier}.json",
                sha256=json_content_digest(content),
                content=deepcopy(content),
            )
        )
        for path, value in _leaves(content):
            field = path[-1]
            category = (
                "identity_name"
                if field in {"first_name", "last_name"}
                else "prose"
                if field == "instructions"
                else "protocol"
            )
            source = SourceEntry(
                id=f"entry-{len(entries)}",
                category=category,
                source=value,
                occurrences=[SourceOccurrence(document_id=identifier, path=path)],
                context=f"role={field}; entity={CANONICAL_ID}",
            )
            target = {"first_name": "Alba", "last_name": "Gil"}.get(field, value)
            if field == "instructions":
                target = value.replace("Emma Kim", "Alba Gil")
            sources.append(source)
            entries.append(
                LocalizationEntry(
                    **source.model_dump(),
                    forms=LocalizedForms(
                        native=target, romanized=target, english="Fixture value."
                    ),
                    operation="adapt" if target != value else "preserve",
                    notes="Offline fixture.",
                )
            )
    proposal = LocalizationMap(
        revision="es-fixture",
        settings=RegionalSettings(
            language="es",
            country_code="ES",
            currency_code="EUR",
            currency_scale="1",
            time_zone="Europe/Madrid",
            name_order="given_first",
        ),
        source=LocalizationSource(
            domain="retail",
            documents=documents,
            entries=sources,
        ),
        entries=entries,
        provenance=GenerationProvenance(
            model="fixture",
            model_args={},
            prompt_version="fixture-v1",
            call_name="fixture",
            input_sha256="b" * 64,
        ),
    )
    return finalize_localization_user_ids(proposal)


@pytest.mark.parametrize("representation", ["native", "romanized"])
def test_compilation_remaps_customer_graph_and_preserves_task_ids(
    localized_ids, representation
):
    compiled = {
        document.id: document.content
        for document in compile_localization(localized_ids, representation)
    }
    database, task = compiled["database"], compiled["tasks"][0]
    assert set(database["users"]) == {LOCALIZED_ID}
    assert database["users"][LOCALIZED_ID]["user_id"] == LOCALIZED_ID
    assert database["orders"]["#ORDER1"]["user_id"] == LOCALIZED_ID
    assert database["orders"]["#ORDER1"]["order_id"] == "#ORDER1"
    assert task["id"] == CANONICAL_ID
    actions = task["evaluation_criteria"]["actions"]
    assert actions[0]["action_id"] == CANONICAL_ID
    assert actions[1]["arguments"]["user_id"] == LOCALIZED_ID
    assert actions[2]["arguments"]["email"] == "wrong@example.com"
    assert set(task["initial_state"]["initialization_data"]["agent_data"]["users"]) == {
        LOCALIZED_ID
    }
    assert LOCALIZED_ID in task["user_scenario"]["instructions"]
    assert task["evaluation_criteria"]["reward_basis"] == ["DB", "NL_ASSERTION"]


@pytest.mark.parametrize("category", ["protocol", "prose"])
def test_missing_reference_rewrite_is_rejected_before_compilation(
    localized_ids, category
):
    entry = next(
        entry
        for entry in localized_ids.entries
        if entry.category == category
        and isinstance(entry.forms.native, str)
        and LOCALIZED_ID in entry.forms.native
    )
    entry.forms.native = entry.forms.native.replace(LOCALIZED_ID, CANONICAL_ID)
    with pytest.raises(ValueError, match="unlocalized user-ID"):
        compile_localization(localized_ids)


def test_unrelated_task_id_cannot_be_changed_by_customer_alias(localized_ids):
    entry = next(
        entry
        for entry in localized_ids.entries
        if any(
            occurrence.document_id == "tasks" and occurrence.path == [0, "id"]
            for occurrence in entry.occurrences
        )
    )
    entry.forms.native = entry.forms.romanized = LOCALIZED_ID
    entry.operation = "adapt"
    with pytest.raises(ValueError, match="Protocol edits"):
        validate_localization(localized_ids)


def test_declared_aliases_must_match_current_localized_names(localized_ids):
    localized_ids.user_id_revision.aliases[0].localized = "unrelated_person_9957"
    with pytest.raises(ValueError, match="localized customer names"):
        validate_localization(localized_ids)


def test_caller_name_must_match_database_identity(localized_ids):
    entry = next(entry for entry in localized_ids.entries if entry.category == "prose")
    entry.forms.native = entry.forms.native.replace("Alba Gil", "Another Person")
    with pytest.raises(ValueError, match="caller"):
        validate_localization(localized_ids)
