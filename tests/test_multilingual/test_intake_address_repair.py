"""Intake reorders proven components without changing unit roles or decoy errors."""

from tau2.multilingual.factory.intake_address_repair import reorder_intake_address


def test_spanish_portal_and_road_number_remain_distinct():
    before = "Vivienda 2, portal 5, bajo B, Travesía de La Güeria, 14, El Entrego, San Martín del Rey Aurelio, 33940 Asturias"
    after = "Travesía de La Güeria, 14, Vivienda 2, portal 5, bajo B, 33940 El Entrego, San Martín del Rey Aurelio, Asturias"
    assert reorder_intake_address(before, "es", "intake_addresses_hard_03") == after
    assert reorder_intake_address(after, "es", "intake_addresses_hard_03") == after
    decoy = before.replace("La Güeria", "La Guía")
    assert reorder_intake_address(
        decoy, "es", "intake_addresses_hard_03"
    ) == after.replace("La Güeria", "La Guía")


def test_brazilian_apartment_is_not_used_as_premise_number():
    before = "Apto 12-B, nº 7, Passagem São Benedito do Murutucu, Belém - PA, 66083-410"
    after = "Passagem São Benedito do Murutucu, nº 7, Apto 12-B, Belém - PA, 66083-410"
    assert reorder_intake_address(before, "pt", "intake_addresses_hard_06") == after
    assert reorder_intake_address(after, "pt", "intake_addresses_hard_06") == after


def test_correct_and_other_locale_surfaces_remain_exact():
    for language, value in [
        ("hi", "फ्लैट 2/1, 96 छत्रपति एन्क्लेव गली, नई दिल्ली, दिल्ली 110055"),
        ("zh", "浙江省杭州市西湖区文二西路741号，邮编310030"),
        ("ko", "서울특별시 송파구 올림픽로 741"),
        ("pt", "Rua das Sibipirunas, 1296, Campinas, SP"),
    ]:
        assert (
            reorder_intake_address(value, language, "intake_addresses_easy_01") == value
        )


def test_authoring_rejects_reintroduced_order_defect():
    from types import SimpleNamespace

    import pytest

    from tau2.multilingual.factory.intake_address_repair import (
        validate_intake_address_forms,
    )
    from tau2.multilingual.localization import LocalizedForms

    entry = SimpleNamespace(
        context="intake stimulus:intake_addresses_hard_06:decoys; dependency=decoys;"
    )
    value = "Apto 12-B, nº 7, Passagem São Benedito do Murutucu, Belém - PA, 66083-410"
    forms = LocalizedForms(native=[value], romanized=[value], english="fixture")
    with pytest.raises(ValueError, match="source-style"):
        validate_intake_address_forms(entry, forms, "pt")
    repaired = reorder_intake_address(value, "pt", "intake_addresses_hard_06")
    forms.native = forms.romanized = [repaired]
    validate_intake_address_forms(entry, forms, "pt")


def test_address_decoy_validation_uses_actual_locale_fold():
    import pytest

    from tau2.multilingual.factory.intake_authoring import validate_intake_forms
    from tau2.multilingual.localization import LocalizationEntry, LocalizedForms
    from tests.test_multilingual.test_intake_temporal_authoring import temporal_entry

    entry = temporal_entry("addresses")
    written = LocalizationEntry(
        id="test-address",
        category="other",
        source="10 Main St",
        occurrences=[
            entry.occurrences[0].model_copy(update={"path": ["stimuli", 90, "written"]})
        ],
        context="intake field:home_address;",
        forms=LocalizedForms(
            native="10 Main St", romanized="10 Main St", english="Address"
        ),
        operation="preserve",
        notes="Test",
    )
    identical = LocalizedForms(
        native=["10 MAIN ST"], romanized=["10 MAIN ST"], english="Decoy"
    )
    with pytest.raises(ValueError, match="collapses"):
        validate_intake_forms(entry, identical, [written], language="ko")
    punctuation = LocalizedForms(
        native=["10 Main St."], romanized=["10 Main St."], english="Decoy"
    )
    validate_intake_forms(entry, punctuation, [written], language="ko")
    with pytest.raises(ValueError, match="collapses"):
        validate_intake_forms(entry, punctuation, [written], language="en")
