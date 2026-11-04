"""Native speech and local plate checks preserve immutable protocol fields."""

import pytest

from tau2.multilingual.factory.intake_authoring import (
    validate_intake_forms,
    validate_pronunciation_script,
)
from tau2.multilingual.localization import LocalizedForms
from tests.test_multilingual.test_banking_prose import entry


@pytest.mark.parametrize(
    "language,token,wrong", [("ko", "민수", "민서"), ("zh", "李明", "李民")]
)
def test_native_script_and_plain_reference_checks(language, token, wrong):
    values = [dict(token=token, reference=token, distorted=wrong)]
    validate_pronunciation_script(values, language, "native")
    values[0]["distorted"] = "Latin"
    with pytest.raises(ValueError, match="Latin"):
        validate_pronunciation_script(values, language, "native")
    values[0]["distorted"] = wrong
    values[0]["reference"] = token + "-guide"
    with pytest.raises(ValueError):
        validate_pronunciation_script(values, language, "native")


@pytest.mark.parametrize("language,plate", [("ko", "123가4567"), ("zh", "京A12345")])
def test_local_license_plate_script_does_not_relax_ascii_protocols(language, plate):
    value = entry("plate", "AB123", plate, category="other")
    value.context = "intake field:license_plate; measured_selection=test"
    forms = LocalizedForms(native=plate, romanized="A12345", english="Test")
    validate_intake_forms(value, forms, language=language)
    value.context = "intake field:vin; measured_selection=test"
    with pytest.raises(ValueError, match="ASCII"):
        validate_intake_forms(value, forms, language=language)


def test_decimal_comma_preserves_dose_but_changed_strength_or_unit_does_not():
    from tau2.domains.intake.localization import medication_strengths

    assert medication_strengths("Clonidina 0,1 mg comprimido") == medication_strengths(
        "Clonidine 0.1 mg tablet"
    )
    assert medication_strengths("Clonidina 0,2 mg comprimido") != medication_strengths(
        "Clonidine 0.1 mg tablet"
    )
    assert medication_strengths("Clonidina 0,1 mcg comprimido") != medication_strengths(
        "Clonidine 0.1 mg tablet"
    )


@pytest.mark.parametrize("language", ["es", "pt", "ko", "zh"])
def test_compiler_rejects_source_phonetic_guides_in_localized_candidates(language):
    from tau2.multilingual.factory.localization_compile import validate_localization
    from tests.test_domains.test_intake.test_localization_intake import identity_map

    candidate = identity_map()
    candidate.settings.language = language
    pronunciation = next(
        entry
        for entry in candidate.entries
        if "dependency=pronunciations;" in entry.context and entry.forms.native
    )
    pronunciation.forms.native[0]["reference"] = "STRESS-guide"
    pronunciation.operation = "adapt"
    with pytest.raises(ValueError, match="pronunciation"):
        validate_localization(candidate)
