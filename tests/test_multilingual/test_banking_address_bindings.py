"""Banking prose bindings read explicit address components, never punctuation."""

import pytest

from tau2.domains.banking_knowledge.localization_bindings import address_component
from tau2.multilingual.address_format import AddressForms, AddressParts, format_locality
from tests.test_multilingual.test_banking_prose import entry, proposal


@pytest.mark.parametrize(
    "address,city,region",
    [
        ("Calle de Prueba 12, Madrid, Madrid, 28001, España", "Madrid", "Madrid"),
        ("Rua Teste 12, São Paulo, SP, 01310-100, Brasil", "São Paulo", "SP"),
        (
            "시험로 12, 서울특별시, 서울특별시, 03000, 대한민국",
            "서울특별시",
            "서울특별시",
        ),
        ("测试路 12号, 北京市, 北京市, 100000, 中国", "北京市", "北京市"),
        ("12 Test Road, Delhi, Delhi 110001, India", "Delhi", "Delhi"),
    ],
)
def test_locality_is_derived_exactly_from_existing_address(address, city, region):
    value = entry("address", "Source", address, category="address")
    parts = AddressParts(street_lines=["Road 12"], city=city, region=region)
    value.forms.address = AddressForms(language="es", native=parts, romanized=parts)
    assert address_component(value, "address_city").native == city
    assert address_component(value, "address_region").native == region
    assert address_component(value, "address_city_region").native == format_locality(
        parts, "es", "native"
    )


def test_address_component_uses_explicit_locality_independent_of_punctuation():
    from tau2.domains.banking_knowledge.localization_bindings import address_component

    value = entry(
        "address",
        "US address",
        "24 अशोक रोड, नई दिल्ली, दिल्ली 110001, भारत",
        "address",
        romanized="24 Ashoka Road, New Delhi, Delhi 110001, India",
    )
    from tau2.multilingual.address_format import AddressForms, AddressParts

    value.forms.address = AddressForms(
        language="hi",
        native=AddressParts(
            street_lines=["24 अशोक रोड"],
            city="नई दिल्ली",
            region="दिल्ली",
            postal_code="११०००१",
        ),
        romanized=AddressParts(
            street_lines=["24 Ashoka Road"],
            city="New Delhi",
            region="Delhi",
            postal_code="110001",
        ),
    )
    assert address_component(value, "address_city").romanized == "New Delhi"
    value.forms.native = "24 अशोक रोड, नई दिल्ली, दिल्ली, 110001, भारत"
    value.forms.romanized = "24 Ashoka Road, New Delhi, Delhi, 110001, India"
    assert (
        address_component(value, "address_city_region").romanized == "New Delhi, Delhi"
    )


def test_address_component_rejects_invalid_part_and_preserves_unicode_digits():
    from tau2.domains.banking_knowledge.localization_bindings import address_component

    value = entry(
        "address",
        "US address",
        "24 अशोक रोड, नई दिल्ली, दिल्ली ११०००१, भारत",
        "address",
        romanized="24 Ashoka Road, New Delhi, Delhi 110001, India",
    )
    from tau2.multilingual.address_format import AddressForms, AddressParts

    value.forms.address = AddressForms(
        language="hi",
        native=AddressParts(
            street_lines=["24 अशोक रोड"],
            city="नई दिल्ली",
            region="दिल्ली",
            postal_code="११०००१",
        ),
        romanized=AddressParts(
            street_lines=["24 Ashoka Road"],
            city="New Delhi",
            region="Delhi",
            postal_code="110001",
        ),
    )
    assert address_component(value, "address_region").native == "दिल्ली"
    assert value.forms.address.native.postal_code == "११०००१"
    with pytest.raises(ValueError, match="Unknown"):
        address_component(value, "ignored")
    value.forms.address = None
    with pytest.raises(ValueError, match="Explicit address components required"):
        address_component(value, "address_city")


def test_absent_address_component_metadata_does_not_change_old_serialization():
    from tau2.multilingual.localization import LocalizedForms, ProseEntityBinding

    b = ProseEntityBinding(
        entry_id="text",
        source="SOURCE",
        forms=LocalizedForms(native="SOURCE", romanized="SOURCE", english="test"),
        kind="protected",
        reason="protocol",
    )
    assert "source_part" not in b.model_dump(mode="json")
    from tau2.domains.banking_knowledge.localization_bindings import (
        validate_prose_bindings,
    )

    p = proposal(entry("text", "SOURCE", "SOURCE"))
    p.prose_bindings = [b.model_copy(update={"source_part": "address_city"})]
    with pytest.raises(ValueError, match="requires a reuse"):
        validate_prose_bindings(p)


def test_address_metadata_does_not_leak_private_donor_components():
    from tau2.domains.banking_knowledge.localization_tokens import (
        BankingReferenceTokens,
        banking_response_model,
        banking_response_schema,
    )

    payload = {
        "completed_mappings": [
            {
                "id": "address",
                "forms": {
                    "native": "private native",
                    "romanized": "private roman",
                    "english": "private explanation",
                    "address": {"private": "component"},
                },
            }
        ]
    }
    BankingReferenceTokens.protect(payload)
    assert "address" not in payload["completed_mappings"][0]["forms"]
    schema = banking_response_schema(banking_response_model(["address"]))
    for model in schema["$defs"].values():
        if "properties" in model:
            assert set(model["required"]) == set(model["properties"])
