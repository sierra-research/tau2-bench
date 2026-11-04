"""The opt-in banking extension has a local domestic debit shipping rule."""

import json
import re
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from tau2.data_model.language import TaskMetadata
from tau2.domains.banking_knowledge.environment import get_knowledge_base
from tau2.domains.banking_knowledge.regional import scenario_environment_kwargs
from tau2.multilingual.factory.localization_artifact import RegionalMapArtifact
from tau2.multilingual.factory.localization_inventory import inventory_localization
from tau2.multilingual.factory.localization_publish import build_regional_bundle
from tau2.multilingual.factory.pack_extension import (
    BANKING_DATE_AND_SHIPPING_VERSION,
    BANKING_DEBIT_SHIPPING_VERSION,
    BANKING_ES_EXTENSION_VERSION,
    BANKING_ES_SCENARIO_TEXT_VERSION,
    BANKING_KO_EXTENSION_VERSION,
    BANKING_OTHER_SCENARIO_TEXT_VERSION,
    BANKING_PT_EXTENSION_VERSION,
    BANKING_PT_SCENARIO_TEXT_VERSION,
    BANKING_ZH_EXTENSION_VERSION,
    DEBIT_CARD_ORDER_DOCUMENT_ID,
    publish_pack_extension,
)
from tau2.multilingual.runtime import load_language
from tau2.utils import DATA_DIR
from tau2.utils.artifacts import write_artifact

DATA = Path(__file__).resolve().parents[2] / "data/tau2/multilingual_extensions"
COUNTRIES = (
    ("hi", "India", "भारत", r"\b\d{6}\b"),
    ("ko", "South Korea", "대한민국", r"\(\d{5}\)$"),
    ("zh", "China", "中国", r"\d{6}$"),
    ("es", "Spain", "España", r"\b\d{5}\b"),
    ("pt", "Brazil", "Brasil", r"\b\d{5}-\d{3}\b"),
)


@pytest.mark.parametrize("language,country,marker,postcode", COUNTRIES)
def test_banking_shipping_rule_is_published_and_used_by_runtime(
    language, country, marker, postcode
):
    old = load_language(
        language, "banking_knowledge", mode="text", pack_extension="banking"
    )
    new = load_language(
        language, "banking_knowledge", mode="text", pack_extension="banking-dates-v1"
    )
    canonical = get_knowledge_base().documents[DEBIT_CARD_ORDER_DOCUMENT_ID]
    assert "valid US domestic address (international shipping is not available)" in (
        canonical.content
    )
    assert DEBIT_CARD_ORDER_DOCUMENT_ID not in (
        old.scenario.knowledge_document_overrides
    )
    expected_revision = {
        "es": BANKING_ES_EXTENSION_VERSION,
        "pt": BANKING_PT_EXTENSION_VERSION,
        "ko": BANKING_KO_EXTENSION_VERSION,
        "zh": BANKING_ZH_EXTENSION_VERSION,
        "hi": f"{BANKING_DATE_AND_SHIPPING_VERSION}+{BANKING_OTHER_SCENARIO_TEXT_VERSION}",
    }.get(language, BANKING_DATE_AND_SHIPPING_VERSION)
    assert new.pack.revision.endswith(expected_revision)
    scenario_suffix = {
        "es": BANKING_ES_SCENARIO_TEXT_VERSION,
        "pt": BANKING_PT_SCENARIO_TEXT_VERSION,
        "hi": BANKING_OTHER_SCENARIO_TEXT_VERSION,
        "ko": BANKING_OTHER_SCENARIO_TEXT_VERSION,
        "zh": BANKING_OTHER_SCENARIO_TEXT_VERSION,
    }.get(language, BANKING_DEBIT_SHIPPING_VERSION)
    assert new.scenario.revision.endswith(scenario_suffix)
    replacement = new.scenario.knowledge_document_overrides[
        DEBIT_CARD_ORDER_DOCUMENT_ID
    ]
    expected = canonical.content.replace(
        "valid US domestic address (international shipping is not available)",
        f"valid domestic address in {country} "
        "(international shipping is not available)",
        1,
    )
    assert replacement.content == expected
    assert replacement.title == canonical.title
    assembled = scenario_environment_kwargs(new.scenario)["knowledge_base"]
    assert assembled.documents[DEBIT_CARD_ORDER_DOCUMENT_ID] == replacement
    assert (
        scenario_environment_kwargs(old.scenario)["knowledge_base"]
        .documents[DEBIT_CARD_ORDER_DOCUMENT_ID]
        .content
        == canonical.content
    )

    gold_orders = []
    for task in new.scenario.tasks:
        for action in task.evaluation_criteria.actions:
            arguments = action.arguments or {}
            if (
                action.name == "call_discoverable_agent_tool"
                and arguments.get("agent_tool_name") == "order_debit_card_5739"
            ):
                gold_orders.append((task, json.loads(arguments["arguments"])))
    assert len(gold_orders) == 15
    assert {task.id for task, _ in gold_orders} == {
        f"task_{number:03}_{language}"
        for number in (77, 78, 79, 80, 81, 82, 86, 87, 88, 90, 91, 92)
    }
    for task, order in gold_orders:
        address = order["shipping_address"]
        assert marker in address
        assert re.search(postcode, address)
        initialization = task.initial_state.initialization_data
        users = (
            initialization.agent_data["users"]["data"]
            if initialization is not None
            else new.scenario.database["users"]["data"]
        )
        address_on_file = users[order["user_id"]]["address"]
        assert marker in address_on_file
        assert re.search(postcode, address_on_file)


@pytest.mark.parametrize("language,country,marker,postcode", COUNTRIES)
def test_shipping_overlay_is_reproducible_from_approved_bundle(
    tmp_path, language, country, marker, postcode
):
    source = DATA / f"banking/{language}/regional/banking_knowledge.json"
    if language in {"ko", "zh"}:
        mapping = RegionalMapArtifact.model_validate_json(
            (
                DATA_DIR / f"tau2/multilingual/{language}/localization/"
                "banking_knowledge.name-orthography-v1.json"
            ).read_bytes()
        )
        metadata = TypeAdapter(dict[str, TaskMetadata]).validate_json(
            (
                DATA / f"banking/{language}/tasks/banking_knowledge.metadata.json"
            ).read_bytes()
        )
        source = tmp_path / f"{language}-banking-names-regional.json"
        write_artifact(
            build_regional_bundle(
                mapping,
                inventory_localization("banking_knowledge", DATA_DIR),
                metadata,
                DATA_DIR,
            ),
            source,
        )
    output = tmp_path / language
    kwargs = dict(
        name="banking-dates-v1",
        language=language,
        revision={
            "es": BANKING_ES_EXTENSION_VERSION,
            "pt": BANKING_PT_EXTENSION_VERSION,
            "ko": BANKING_KO_EXTENSION_VERSION,
            "zh": BANKING_ZH_EXTENSION_VERSION,
            "hi": f"{BANKING_DATE_AND_SHIPPING_VERSION}+{BANKING_OTHER_SCENARIO_TEXT_VERSION}",
        }.get(language, BANKING_DATE_AND_SHIPPING_VERSION),
        bundle_paths=[source],
        output=output,
    )
    publish_pack_extension(**kwargs)
    published = DATA / f"banking-dates-v1/{language}"
    for relative in (
        "extension.json",
        "regional/banking_knowledge.json",
        "tasks/banking_knowledge.metadata.json",
    ):
        assert (output / relative).read_bytes() == (published / relative).read_bytes()
