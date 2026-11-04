# Copyright (c) 2026 Sierra
"""Semantic regressions in published task facts and product option meanings."""

import json
import re
from decimal import Decimal
from pathlib import Path

import pytest

from tau2.multilingual.runtime import load_language

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def retail_source():
    return json.loads((ROOT / "data/tau2/domains/retail/db.json").read_text())


@pytest.mark.parametrize("language", ["es", "pt", "hi", "zh", "ko"])
def test_retail107_preserves_implicit_theme_and_difficulty(language):
    loaded = load_language(language, "retail", mode="voice")
    task = next(
        t for t in loaded.tasks if loaded.metadata[t.id].canonical_task_id == "107"
    )
    artifact = json.loads(
        (
            ROOT / f"data/tau2/multilingual/{language}/localization/retail.json"
        ).read_text()
    )
    entry = next(
        e
        for e in artifact["entries"]
        if e["category"] == "prose"
        and e["source"].startswith("You received hiking boots that seem")
    )
    preference = entry["source"].split("your jigsaw ", 1)[1]
    for text in [
        task.user_scenario.instructions.reason_for_call,
        entry["forms"]["native"],
        entry["forms"]["romanized"],
    ]:
        assert text.endswith(preference)


@pytest.mark.parametrize(
    "language,expected",
    [
        (
            "es",
            {
                ("armrest", "none"): "sin reposabrazos",
                ("connectivity", "none"): "sin conectividad",
                ("features", "none"): "sin funciones adicionales",
                ("backlight", "none"): "sin retroiluminación",
            },
        ),
        (
            "pt",
            {
                ("armrest", "none"): "sem apoio de braço",
                ("connectivity", "none"): "sem conectividade",
                ("features", "none"): "sem recursos adicionais",
                ("backlight", "none"): "sem iluminação",
            },
        ),
        (
            "zh",
            {
                ("armrest", "none"): "无扶手",
                ("connectivity", "none"): "无连接功能",
                ("features", "none"): "无附加功能",
                ("backlight", "none"): "无背光",
                ("ventilation", "high"): "高通风",
                ("ventilation", "low"): "低通风",
                ("ventilation", "medium"): "中等通风",
                ("speed settings", "high"): "高速",
                ("speed settings", "low"): "低速",
                ("brightness", "medium"): "中等亮度",
                ("brightness", "high"): "高亮度",
                ("skin tone", "medium"): "中等肤色",
                ("size", "medium"): "中号",
                ("material", "stainless steel"): "不锈钢",
                ("color", "stainless steel"): "不锈钢色",
                ("material", "wood"): "木材",
                ("color", "wood"): "木色",
                ("type", "electric"): "电热式",
                ("stovetop compatibility", "electric"): "电陶炉",
            },
        ),
        (
            "ko",
            {
                ("brightness", "medium"): "중간",
                ("ventilation", "medium"): "중간",
                ("skin tone", "medium"): "중간 피부톤",
                ("size", "medium"): "중형",
                ("type", "electric"): "전기식",
                ("type", "gas"): "가스식",
                ("stovetop compatibility", "electric"): "전기레인지용",
                ("stovetop compatibility", "gas"): "가스레인지용",
            },
        ),
    ],
)
def test_product_option_meaning_depends_on_attribute(retail_source, language, expected):
    database = load_language(language, "retail", mode="voice").scenario.database
    seen = set()
    for product_id, product in retail_source["products"].items():
        for item_id, variant in product["variants"].items():
            options = database["products"][product_id]["variants"][item_id]["options"]
            for key, value in variant["options"].items():
                if (key, value) in expected:
                    assert options[key] == expected[key, value], (product_id, item_id)
                    seen.add((key, value))
    assert seen == expected.keys()
    for order_id, order in retail_source["orders"].items():
        for original, localized in zip(
            order["items"], database["orders"][order_id]["items"], strict=True
        ):
            for key, value in original["options"].items():
                if (key, value) in expected:
                    assert localized["options"][key] == expected[key, value], order_id


def test_mandarin_move_reason_matches_old_customer_address():
    loaded = load_language("zh", "retail", mode="voice")
    task = next(
        t for t in loaded.tasks if loaded.metadata[t.id].canonical_task_id == "39"
    )
    action = next(
        a for a in task.evaluation_criteria.actions if a.name == "get_user_details"
    )
    old_state = loaded.scenario.database["users"][action.arguments["user_id"]][
        "address"
    ]["state"]
    instructions = task.user_scenario.instructions
    assert f"moved from {old_state}" in instructions.reason_for_call
    assert f"moved from {old_state}" in instructions.known_info


@pytest.mark.parametrize("language", ["es", "pt", "hi", "zh", "ko"])
def test_retail19_scoring_amounts_match_database(language):
    loaded = load_language(language, "retail", mode="voice")
    task = next(
        t for t in loaded.tasks if loaded.metadata[t.id].canonical_task_id == "19"
    )
    database = loaded.scenario.database
    action = next(
        a
        for a in task.evaluation_criteria.actions
        if a.name == "return_delivered_order_items"
    )
    items = database["orders"][action.arguments["order_id"]]["items"]
    refund = sum(
        Decimal(str(i["price"]))
        for i in items
        if i["item_id"] in action.arguments["item_ids"]
    )
    exchange_ids = [
        a.arguments["product_id"]
        for a in task.evaluation_criteria.actions
        if a.name == "get_product_details"
    ]
    savings = Decimal(0)
    for item in items:
        if item["product_id"] in exchange_ids:
            cheapest = min(
                Decimal(str(v["price"]))
                for v in database["products"][item["product_id"]]["variants"].values()
                if v["available"]
            )
            savings += Decimal(str(item["price"])) - cheapest
    criteria = task.evaluation_criteria
    assert list(map(Decimal, criteria.communicate_info)) == [refund, savings]
    for assertion, amount in zip(
        criteria.nl_assertions, [refund, savings], strict=True
    ):
        numbers = re.findall(r"\d+(?:\.\d+)?", assertion.replace(",", ""))
        assert list(map(Decimal, numbers)) == [amount]


def test_korean_initial_booking_route_matches_airport_mapping():
    loaded = load_language("ko", "airline", mode="voice")
    task = next(
        t for t in loaded.tasks if loaded.metadata[t.id].canonical_task_id == "2"
    )
    catalog = json.loads(
        (
            ROOT / "data/tau2/multilingual/ko/localization/airport-catalog.json"
        ).read_text()
    )
    airport = next(row for row in catalog["airports"] if row["source_iata"] == "SFO")
    assert airport["iata"] == "HND"
    for text in [
        task.user_scenario.instructions.reason_for_call,
        task.user_scenario.instructions.task_instructions,
    ]:
        assert airport["native_city"] in text
        assert "광주" not in text


def test_korean_retail49_preserves_caller_city():
    loaded = load_language("ko", "retail", mode="voice")
    task = next(
        t for t in loaded.tasks if loaded.metadata[t.id].canonical_task_id == "49"
    )
    action = next(
        a for a in task.evaluation_criteria.actions if a.name == "get_user_details"
    )
    address = loaded.scenario.database["users"][action.arguments["user_id"]]["address"]
    known = task.user_scenario.instructions.known_info
    assert address["city"] in known
    assert address["zip"] in known


@pytest.mark.parametrize("language", ["es", "pt", "hi", "zh", "ko"])
def test_reference_address_preserves_saved_source_address(retail_source, language):
    control = load_language("en", "retail", mode="voice")
    originals = {control.metadata[t.id].canonical_task_id: t for t in control.tasks}
    loaded = load_language(language, "retail", mode="voice")
    orders = loaded.scenario.database["orders"]
    fields = ("address1", "address2", "city", "country", "state", "zip")
    checked = 0
    for task in loaded.tasks:
        canonical_id = loaded.metadata[task.id].canonical_task_id
        original = originals[canonical_id]
        for source, action in zip(
            original.evaluation_criteria.actions or [],
            task.evaluation_criteria.actions or [],
            strict=True,
        ):
            if source.name not in (
                "modify_pending_order_address",
                "modify_user_address",
            ):
                continue
            source_address = {key: source.arguments[key] for key in fields}
            matching_orders = [
                order_id
                for order_id, order in retail_source["orders"].items()
                if order["address"] == source_address
            ]
            if not matching_orders:
                continue
            expected = [orders[order_id]["address"] for order_id in matching_orders]
            actual = {key: action.arguments[key] for key in fields}
            assert actual in expected, canonical_id
            checked += 1
    assert checked >= 6


@pytest.mark.parametrize("language", ["es", "pt", "hi", "zh", "ko"])
def test_accidental_shipment_region_identifies_target_order(language):
    loaded = load_language(language, "retail", mode="voice")
    order = loaded.scenario.database["orders"]["#W5565470"]
    region = order["address"]["state"]
    if language == "pt":
        assert region == "PA"
        region = "Pará"
    tasks = {loaded.metadata[t.id].canonical_task_id: t for t in loaded.tasks}
    for task_id in ("25", "26"):
        reason = tasks[task_id].user_scenario.instructions.reason_for_call
        assert f"order sent to {region} by accident" in reason


def test_korean_hidden_address_city_uniquely_identifies_saved_address():
    loaded = load_language("ko", "retail", mode="voice")
    task = next(
        t for t in loaded.tasks if loaded.metadata[t.id].canonical_task_id == "86"
    )
    action = next(
        a for a in task.evaluation_criteria.actions if a.name == "modify_user_address"
    )
    database = loaded.scenario.database
    customer = database["users"][action.arguments["user_id"]]
    addresses = [customer["address"]] + [
        database["orders"][order_id]["address"] for order_id in customer["orders"]
    ]
    candidates = {
        json.dumps(address, sort_keys=True)
        for address in addresses
        if address["city"] == action.arguments["city"]
    }
    assert len(candidates) == 1
    target = json.loads(next(iter(candidates)))
    assert all(action.arguments[key] == value for key, value in target.items())
    reason = task.user_scenario.instructions.reason_for_call
    assert f"your {target['city']} address" in reason
    assert target["address1"] not in reason


@pytest.mark.parametrize(
    "language,task_id",
    [("es", "22"), ("hi", "17"), ("hi", "22"), ("ko", "17"), ("ko", "22")],
)
def test_existing_trip_origin_uses_reservation_airport_city(language, task_id):
    loaded = load_language(language, "airline", mode="voice")
    task = next(
        t for t in loaded.tasks if loaded.metadata[t.id].canonical_task_id == task_id
    )
    reservation = loaded.scenario.database["reservations"]["FQ8APE"]
    origin = loaded.scenario.airports[reservation["origin"]]
    assert f"upcoming trip from {origin} to " in (
        task.user_scenario.instructions.reason_for_call
    )


def test_mandarin_passenger_assertion_matches_booking_identity():
    loaded = load_language("zh", "airline", mode="voice")
    task = next(
        t for t in loaded.tasks if loaded.metadata[t.id].canonical_task_id == "8"
    )
    booking = next(
        a for a in task.evaluation_criteria.actions if a.name == "book_reservation"
    )
    assertions = "\n".join(task.evaluation_criteria.nl_assertions)
    for passenger in booking.arguments["passengers"]:
        name = passenger["last_name"] + passenger["first_name"]
        assert f"{name} DOB {passenger['dob']}" in assertions
