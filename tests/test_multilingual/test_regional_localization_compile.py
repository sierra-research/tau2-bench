"""Source-checked regional edits preserve schemas, lookup behavior, and graph IDs."""

from copy import deepcopy

import pytest

from tau2.multilingual.factory.localization_compile import (
    compile_localization,
    validate_localization,
)
from tau2.multilingual.localization import LocalizationMap, json_content_digest


def _document(identifier, kind, content):
    return {
        "id": identifier,
        "kind": kind,
        "source_path": f"tau2/domains/retail/{identifier}.json",
        "sha256": json_content_digest(content),
        "content": content,
    }


def _entry(identifier, source, paths, native, *, category="identity_name"):
    return {
        "id": identifier,
        "category": category,
        "source": source,
        "occurrences": [
            {"document_id": document, "path": path} for document, path in paths
        ],
        "context": "Synthetic source for compiler tests.",
        "forms": {"native": native, "romanized": native, "english": "Test value."},
        "operation": "adapt",
        "notes": "Reviewed in this offline fixture only.",
    }


@pytest.fixture
def proposal():
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
        "id": "task-1",
        "user_scenario": {
            "instructions": {
                "domain": "retail",
                "task_instructions": "Call about your order.",
                "reason_for_call": "Help with an order.",
                "known_info": "You need help with your order.",
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
                        "zip": "60601",
                    },
                }
            ],
            "reward_basis": ["DB", "NL_ASSERTION"],
        },
    }
    entries = [
        _entry(
            "given-name",
            "Ada",
            [
                ("db", ["users", "u1", "name", "first_name"]),
                (
                    "tasks",
                    [0, "evaluation_criteria", "actions", 0, "arguments", "first_name"],
                ),
            ],
            "静",
        )
    ]
    result = LocalizationMap.model_validate(
        {
            "revision": "zh-fixture-v1",
            "settings": {
                "language": "zh",
                "country_code": "CN",
                "currency_code": "CNY",
                "currency_scale": "1",
                "time_zone": "Asia/Shanghai",
                "name_order": "family_first",
            },
            "source": {
                "domain": "retail",
                "documents": [
                    _document(
                        "db",
                        "database",
                        {"users": {"u1": user}, "orders": {}, "products": {}},
                    ),
                    _document("tasks", "tasks", [task]),
                ],
                "entries": [
                    {
                        key: value
                        for key, value in entry.items()
                        if key not in {"forms", "operation", "notes"}
                    }
                    for entry in entries
                ],
            },
            "entries": entries,
            "provenance": {
                "model": "fixture",
                "model_args": {},
                "prompt_version": "fixture-v1",
                "call_name": "fixture_localization",
                "input_sha256": "b" * 64,
            },
        }
    )
    _cover_preserved_values(result)
    return result


def _append_entry(proposal, entry):
    from tau2.multilingual.localization import LocalizationEntry, SourceEntry

    replaced = set()
    for existing in proposal.source.entries:
        if not existing.id.startswith("untouched-"):
            continue
        for occurrence in existing.occurrences:
            if any(
                occurrence.document_id == new["document_id"]
                and occurrence.path[: len(new["path"])] == new["path"]
                for new in entry["occurrences"]
            ):
                replaced.add(existing.id)
    proposal.entries = [row for row in proposal.entries if row.id not in replaced]
    proposal.source.entries = [
        row for row in proposal.source.entries if row.id not in replaced
    ]
    proposal.entries.append(LocalizationEntry.model_validate(entry))
    proposal.source.entries.append(
        SourceEntry.model_validate(
            {
                key: value
                for key, value in entry.items()
                if key not in {"forms", "operation", "notes"}
            }
        )
    )


def _cover_preserved_values(proposal):
    targets = {
        (occurrence.document_id, *occurrence.path)
        for entry in proposal.source.entries
        for occurrence in entry.occurrences
    }

    def visit(value, document, path):
        if isinstance(value, dict) and value:
            for key, item in value.items():
                visit(item, document, path + [key])
        elif isinstance(value, list) and value:
            for index, item in enumerate(value):
                visit(item, document, path + [index])
        elif (document, *path) not in targets:
            entry = _entry(
                f"untouched-{document}-{path!r}",
                value,
                [(document, path)],
                deepcopy(value),
                category="protocol",
            )
            entry["operation"] = "preserve"
            _append_entry(proposal, entry)

    for document in proposal.source.documents:
        visit(document.content, document.id, [])


def test_compile_applies_all_occurrences_without_mutating_source(proposal):
    proposal.entries[0].forms.romanized = "Jing"
    before = proposal.model_dump_json()
    native = compile_localization(proposal)
    romanized = compile_localization(proposal, representation="romanized")
    for documents, name in [(native, "静"), (romanized, "Jing")]:
        assert documents[0].content["users"]["u1"]["name"]["first_name"] == name
        assert (
            documents[1].content[0]["evaluation_criteria"]["actions"][0]["arguments"][
                "first_name"
            ]
            == name
        )
        assert all(
            document.sha256 == json_content_digest(document.content)
            for document in documents
        )
    assert proposal.model_dump_json() == before
    assert validate_localization(proposal) == proposal


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "extra", "source", "path", "hash", "document", "category"],
)
def test_rejects_incomplete_or_stale_source_contract(proposal, mutation):
    if mutation == "missing":
        proposal.entries.clear()
    elif mutation == "duplicate":
        proposal.entries.append(proposal.entries[0].model_copy(deep=True))
    elif mutation == "extra":
        proposal.entries.append(proposal.entries[0].model_copy(update={"id": "extra"}))
    elif mutation == "source":
        proposal.entries[0].source = "Changed"
    elif mutation == "path":
        proposal.entries[0].occurrences[0].path[-1] = "last_name"
    elif mutation == "hash":
        proposal.source.documents[0].sha256 = "0" * 64
    elif mutation == "document":
        proposal.source.documents.append(
            proposal.source.documents[0].model_copy(deep=True)
        )
    else:
        proposal.entries[0].category = "other"
    with pytest.raises(ValueError):
        compile_localization(proposal)


@pytest.mark.parametrize(
    "path", [["users", "u1", "name"], ["users", "u1", "name", "first_name"], []]
)
def test_rejects_duplicate_or_ancestor_occurrence_targets(proposal, path):
    source = proposal.source.documents[0].content
    for key in path:
        source = source[key]
    _append_entry(proposal, _entry("overlap", source, [("db", path)], deepcopy(source)))
    with pytest.raises(ValueError, match="overlap|duplicate"):
        validate_localization(proposal)


@pytest.mark.parametrize("replacement", [None, 1, ["静"], {"name": "静"}])
def test_both_representations_require_source_shape(proposal, replacement):
    proposal.entries[0].forms.romanized = replacement
    with pytest.raises(ValueError, match="shape|type"):
        compile_localization(proposal)


@pytest.mark.parametrize("category", ["protocol", "date_time"])
def test_preserved_categories_cannot_be_adapted(proposal, category):
    proposal.entries[0].category = category
    proposal.source.entries[0].category = category
    with pytest.raises(ValueError, match="preserv"):
        validate_localization(proposal)


def test_preserve_operation_must_leave_both_values_unchanged(proposal):
    proposal.entries[0].operation = "preserve"
    with pytest.raises(ValueError, match="preserv"):
        validate_localization(proposal)


@pytest.mark.parametrize(
    "document,path,source,replacement",
    [
        ("db", ["users", "u1", "user_id"], "u1", "u2"),
        ("tasks", [0, "id"], "task-1", "task-2"),
        (
            "tasks",
            [0, "evaluation_criteria", "reward_basis"],
            ["DB", "NL_ASSERTION"],
            ["DB", "COMMUNICATE"],
        ),
        (
            "tasks",
            [0, "evaluation_criteria", "actions", 0, "name"],
            "find_user_id_by_name_zip",
            "find_user_id_by_email",
        ),
    ],
)
def test_protocol_invariants_do_not_trust_entry_category(
    proposal, document, path, source, replacement
):
    _append_entry(
        proposal,
        _entry(
            "bad-protocol", source, [(document, path)], replacement, category="other"
        ),
    )
    with pytest.raises(ValueError, match="protocol|task|reward"):
        validate_localization(proposal)


def test_unlinked_identity_change_cannot_break_authentication(proposal):
    proposal.entries[0].occurrences.pop()
    proposal.source.entries[0].occurrences.pop()
    _cover_preserved_values(proposal)
    with pytest.raises(ValueError, match="lookup|authentication"):
        validate_localization(proposal)


def test_currency_amounts_follow_declared_scale(proposal):
    db = proposal.source.documents[0]
    db.content["users"]["u1"]["payment_methods"] = {
        "gift_card_1": {"source": "gift_card", "id": "gift_card_1", "balance": 10.0}
    }
    db.sha256 = json_content_digest(db.content)
    _append_entry(
        proposal,
        _entry(
            "balance",
            10.0,
            [("db", ["users", "u1", "payment_methods", "gift_card_1", "balance"])],
            20.0,
            category="money",
        ),
    )
    proposal.entries = [row for row in proposal.entries if row.source != {}]
    proposal.source.entries = [
        row for row in proposal.source.entries if row.source != {}
    ]
    _cover_preserved_values(proposal)
    proposal.settings.currency_scale = "2"
    assert (
        compile_localization(proposal)[0].content["users"]["u1"]["payment_methods"][
            "gift_card_1"
        ]["balance"]
        == 20.0
    )
    next(row for row in proposal.entries if row.id == "balance").forms.romanized = 19.0
    with pytest.raises(ValueError, match="scale|currency"):
        validate_localization(proposal)


def test_inventory_cannot_omit_unedited_source_content(proposal):
    omitted = proposal.source.entries[-1].id
    proposal.entries = [entry for entry in proposal.entries if entry.id != omitted]
    proposal.source.entries = [
        entry for entry in proposal.source.entries if entry.id != omitted
    ]
    with pytest.raises(ValueError, match="uncovered"):
        validate_localization(proposal)


@pytest.fixture
def airline_proposal(proposal):
    import json
    from pathlib import Path

    db = json.loads(Path("data/tau2/domains/airline/db.json").read_text())
    db["users"] = {}
    db["reservations"] = {}
    db["flights"] = {key: db["flights"][key] for key in ("HAT001", "HAT002")}
    proposal.source.domain = "airline"
    proposal.source.documents = [
        type(proposal.source.documents[0]).model_validate(
            _document("db", "database", db)
        )
    ]
    proposal.entries = []
    proposal.source.entries = []
    for code, target in [("PHL", "PEK"), ("LGA", "PVG"), ("PHX", "CAN")]:
        paths = [
            ("db", ["flights", key, field])
            for key, flight in db["flights"].items()
            for field in ("origin", "destination")
            if flight[field] == code
        ]
        _append_entry(proposal, _entry(code, code, paths, target, category="airport"))
    _cover_preserved_values(proposal)
    return proposal


def test_airport_mapping_preserves_distinct_nodes_and_linked_routes(airline_proposal):
    proposal = airline_proposal
    result = compile_localization(proposal)[0].content
    assert (
        result["flights"]["HAT001"]["destination"]
        == result["flights"]["HAT002"]["origin"]
        == "PVG"
    )
    assert result["flights"]["HAT002"]["scheduled_arrival_time_est"] == "01:30:00+1"
    next(row for row in proposal.entries if row.id == "PHX").forms.native = "PEK"
    with pytest.raises(ValueError, match="graph|merged"):
        validate_localization(proposal)


def test_currency_scale_updates_cabin_prices_without_changing_cabin_keys(
    airline_proposal,
):
    proposal = airline_proposal
    proposal.settings.currency_scale = "2"
    prices = [
        entry
        for entry in proposal.source.entries
        if "prices" in entry.occurrences[0].path
    ]
    assert prices
    for entry in prices:
        _append_entry(
            proposal,
            _entry(
                f"scaled-{entry.id}",
                entry.source,
                [(row.document_id, row.path) for row in entry.occurrences],
                entry.source * 2,
                category="money",
            ),
        )
    result = compile_localization(proposal)[0].content
    original = proposal.source.documents[0].content
    for key, flight in result["flights"].items():
        for date, instance in flight["dates"].items():
            if "prices" not in instance:
                assert instance == original["flights"][key]["dates"][date]
                continue
            assert instance["prices"] == {
                cabin: price * 2
                for cabin, price in original["flights"][key]["dates"][date][
                    "prices"
                ].items()
            }


def test_typed_database_rejects_invalid_localized_status(proposal):
    user = proposal.source.documents[0].content["users"]["u1"]
    user["payment_methods"] = {
        "card1": {
            "source": "credit_card",
            "id": "card1",
            "brand": "visa",
            "last_four": "1234",
        }
    }
    proposal.source.documents[0].sha256 = json_content_digest(
        proposal.source.documents[0].content
    )
    proposal.entries = [row for row in proposal.entries if row.source != {}]
    proposal.source.entries = [
        row for row in proposal.source.entries if row.source != {}
    ]
    _append_entry(
        proposal,
        _entry(
            "card-type",
            "credit_card",
            [("db", ["users", "u1", "payment_methods", "card1", "source"])],
            "cash",
            category="billing",
        ),
    )
    _cover_preserved_values(proposal)
    with pytest.raises(ValueError):
        validate_localization(proposal)
