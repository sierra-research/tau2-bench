"""Full canonical coverage and semantic grouping for regional authoring."""

import json
from pathlib import Path

import pytest

from tau2.multilingual.factory.localization_inventory import inventory_localization

DATA = Path(__file__).resolve().parents[2] / "data"


def _resolve(value, path):
    for key in path:
        value = value[key]
    return value


def _entry_at(source, document_id, path):
    return next(
        entry
        for entry in source.entries
        if any(
            o.document_id == document_id and o.path == path for o in entry.occurrences
        )
    )


@pytest.mark.parametrize("domain", ["airline", "retail", "telecom"])
def test_inventory_covers_every_source_value_once_and_frozen50(domain):
    source = inventory_localization(domain, DATA)
    docs = {doc.id: doc for doc in source.documents}
    tasks = docs["tasks"].content
    subset = json.loads((DATA / "tau2/task_subsets" / f"{domain}_50.json").read_text())
    assert [task["id"] for task in tasks] == subset["task_ids"]
    assert len(tasks) == 50
    occurrences = set()
    for entry in source.entries:
        assert entry.occurrences
        for occurrence in entry.occurrences:
            address = (occurrence.document_id, tuple(occurrence.path))
            assert address not in occurrences
            occurrences.add(address)
            assert (
                _resolve(docs[occurrence.document_id].content, occurrence.path)
                == entry.source
            )

    def covered(document_id, value, path=()):
        if (document_id, path) in occurrences:
            return
        assert isinstance(value, (dict, list)) and value, (document_id, path, value)
        children = value.items() if isinstance(value, dict) else enumerate(value)
        for key, child in children:
            covered(document_id, child, (*path, key))

    for document_id, path in occurrences:
        assert not any((document_id, path[:n]) in occurrences for n in range(len(path)))
    for doc in source.documents:
        covered(doc.id, doc.content)
    assert inventory_localization(domain, DATA) == source


def test_retail_catalog_copies_share_entries_but_duplicate_people_do_not():
    source = inventory_localization("retail", DATA)
    db = next(doc.content for doc in source.documents if doc.id == "database")
    assert len(db["users"]) == 500 and len(db["orders"]) == 1000
    order_id, order = next(iter(db["orders"].items()))
    item = order["items"][0]
    original = _entry_at(source, "database", ["products", item["product_id"], "name"])
    copied = _entry_at(source, "database", ["orders", order_id, "items", 0, "name"])
    assert original.id == copied.id and original.category == "product_name"
    names = {}
    for user_id, user in db["users"].items():
        full = tuple(user["name"].values())
        if full in names:
            left = _entry_at(
                source, "database", ["users", user_id, "name", "first_name"]
            )
            right = _entry_at(
                source, "database", ["users", names[full], "name", "first_name"]
            )
            assert left.id != right.id
            break
        names[full] = user_id
    else:
        pytest.fail("Fixture must contain two distinct users with the same name")
    assert (
        _entry_at(source, "database", ["orders", order_id, "status"]).category
        == "protocol"
    )
    assert (
        _entry_at(
            source, "database", ["orders", order_id, "items", 0, "price"]
        ).category
        == "money"
    )


def test_telecom_preserves_ids_and_groups_primary_phone_with_actual_line():
    source = inventory_localization("telecom", DATA)
    customer = _entry_at(source, "database", ["customers", 0, "phone_number"])
    line = _entry_at(source, "database", ["lines", 1, "phone_number"])
    assert customer.id == line.id and customer.category == "contact"
    assert any(o.document_id == "tasks" for o in customer.occurrences)
    assert (
        _entry_at(source, "database", ["customers", 0, "customer_id"]).category
        == "protocol"
    )
    assert any(doc.kind == "user_database" for doc in source.documents)
    evidence = [entry for entry in source.entries if entry.category == "tool_text"]
    assert any("scale_fee(5)" in str(entry.source) for entry in evidence)
    assert all("runtime wiring" in entry.context for entry in evidence)


def test_airline_airports_and_passenger_names_are_coherent():
    source = inventory_localization("airline", DATA)
    db = next(doc.content for doc in source.documents if doc.id == "database")
    assert len(db["flights"]) == 300 and len(db["reservations"]) == 2000
    airports = [entry for entry in source.entries if entry.category == "airport"]
    assert len(airports) == 20
    for rid, reservation in db["reservations"].items():
        user = db["users"][reservation["user_id"]]
        for idx, passenger in enumerate(reservation["passengers"]):
            if passenger == {**user["name"], "dob": user["dob"]}:
                owner = _entry_at(
                    source, "database", ["users", user["user_id"], "name", "first_name"]
                )
                saved = _entry_at(
                    source,
                    "database",
                    ["reservations", rid, "passengers", idx, "first_name"],
                )
                assert owner.id == saved.id
                return
    pytest.fail("Fixture requires a linked reservation passenger")


def test_source_context_keeps_complete_address_and_component_roles():
    source = inventory_localization("retail", DATA)
    db = next(doc.content for doc in source.documents if doc.id == "database")
    user_id, user = next(iter(db["users"].items()))
    postcode = _entry_at(source, "database", ["users", user_id, "address", "zip"])
    assert "role=zip;" in postcode.context
    assert json.dumps(user["address"], sort_keys=True) in postcode.context
    name = _entry_at(source, "database", ["users", user_id, "name", "first_name"])
    assert "role=first_name;" in name.context
    assert user["name"]["last_name"] in name.context
    assert any(
        entry.category == "prose" and entry.occurrences[0].path[-1] == "expression"
        for entry in source.entries
    )


def test_executable_fees_are_evidence_not_untracked_database_assumptions():
    source = inventory_localization("airline", DATA)
    evidence = [e.source for e in source.entries if e.category == "tool_text"]
    assert "total_price += self._fee(30) * len(passengers)" in evidence
    assert "total_price += self._fee(50) * nonfree_baggages" in evidence
    assert any("scale_fee(canonical_amount)" in str(value) for value in evidence)


def test_tool_evidence_excludes_generic_protocol_vocabulary():
    source = inventory_localization("airline", DATA)
    evidence = [e.source for e in source.entries if e.category == "tool_text"]
    assert "New York" in evidence and "JFK" in evidence
    assert "credit_card" not in evidence
    assert "user_id" not in evidence
    assert "basic_economy" not in evidence


def test_inventory_code_provenance_is_portable_across_data_roots(tmp_path):
    relocated = tmp_path / "data"
    relocated.symlink_to(DATA, target_is_directory=True)
    original = inventory_localization("telecom", DATA)
    moved = inventory_localization("telecom", relocated)
    assert moved == original
    assert all(
        doc.source_path.startswith("../src/tau2/domains/telecom/")
        for doc in original.documents
        if doc.kind == "tool_text"
    )


def test_airport_context_carries_cross_airport_constraints_from_frozen_tasks():
    source = inventory_localization("airline", DATA)
    airports = [entry for entry in source.entries if entry.category == "airport"]
    assert len(airports) == 20
    for entry in airports:
        assert '"task_id": "15"' in entry.context
        assert "EWR and PHL are equally convenient" in entry.context
        assert '"task_id": "19"' in entry.context
        assert "You do not accept JFK, only EWR." in entry.context
        assert '"task_id": "24"' in entry.context
        assert "New York (either EWR or JFK)" in entry.context
        assert "West Coast" in entry.context
    assert "Newark" in next(e.context for e in airports if e.source == "EWR")


def test_optional_retail_exchange_difference_is_money(tmp_path):
    directory = tmp_path / "tau2/domains/retail"
    directory.mkdir(parents=True)
    canonical = DATA / "tau2/domains/retail"
    database = json.loads((canonical / "db.json").read_text())
    order_id = next(iter(database["orders"]))
    database["orders"][order_id]["exchange_price_difference"] = 12.5
    (directory / "db.json").write_text(json.dumps(database))
    for name in ("tasks.json", "policy.md"):
        (directory / name).symlink_to(canonical / name)
    (tmp_path / "tau2/task_subsets").symlink_to(
        DATA / "tau2/task_subsets", target_is_directory=True
    )
    source = inventory_localization("retail", tmp_path)
    entry = _entry_at(
        source, "database", ["orders", order_id, "exchange_price_difference"]
    )
    assert entry.category == "money" and entry.source == 12.5


def test_retail_state_entries_follow_each_address_city():
    source = inventory_localization("retail", DATA)
    db = next(doc.content for doc in source.documents if doc.id == "database")
    profile = ["users", "ivan_santos_6635", "address", "state"]
    shipping = ["orders", "#W6893533", "address", "state"]
    assert _resolve(db, profile[:-1])["city"] == "Dallas"
    assert _resolve(db, shipping[:-1])["city"] == "Austin"
    user_state = _entry_at(source, "database", profile)
    order_state = _entry_at(source, "database", shipping)
    assert user_state.source == order_state.source == "TX"
    assert user_state.id != order_state.id
    assert '"city": "Dallas"' in user_state.context
    assert '"city": "Austin"' in order_state.context
    for entry in source.entries:
        if entry.category != "address" or not entry.context.startswith("role=state;"):
            continue
        cities = set()
        for occurrence in entry.occurrences:
            if occurrence.document_id == "database":
                address = _resolve(db, occurrence.path[:-1])
                cities.add((address["city"], address["country"]))
        assert len(cities) <= 1, entry.id
