"""Published intake maps keep local contact shapes and distinct business identities."""

import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pytest

from tau2.multilingual.factory.localization_artifact import read_localization_map
from tau2.utils.utils import DATA_DIR

BUSINESS_ROLES = {
    "intake organization",
    "intake field:destination_property",
    "intake field:company_name",
    "intake field:employer_name",
    "intake field:aftermarket_brand",
}


def intake_map(language: str):
    return read_localization_map(
        Path(DATA_DIR) / f"tau2/multilingual/{language}/localization/intake.json"
    )


def business_identity_collisions(proposal) -> dict[str, list[str]]:
    """Distinct source businesses that collapse to one localized identity."""
    entries = [
        e for e in proposal.entries if e.context.split(";", 1)[0] in BUSINESS_ROLES
    ]
    collisions: dict[str, list[str]] = defaultdict(list)
    for form in ("native", "romanized"):
        groups = defaultdict(list)
        for entry in entries:
            value = getattr(entry.forms, form)
            key = " ".join(unicodedata.normalize("NFC", value).casefold().split())
            groups[key].append(entry)
        for group in groups.values():
            if len({e.source for e in group}) > 1:
                for entry in group:
                    collisions[entry.id].append(form)
    return dict(collisions)


def is_indian_phone(value: str) -> bool:
    """Indian mobile or major-city landline shape; not subscriber existence."""
    if not value.isascii():
        return False
    parts = re.split(r"\s+ext\.?\s*", value, flags=re.I)
    if len(parts) > 2 or (len(parts) == 2 and not parts[1].isdigit()):
        return False
    digits = re.sub(r"\D", "", parts[0])
    if parts[0].lstrip().startswith(("+", "(+")):
        if not digits.startswith("91"):
            return False
        digits = digits[2:]
    elif digits.startswith("0"):
        digits = digits[1:]
    mobile = bool(re.fullmatch(r"[6-9]\d{9}", digits))
    landline = bool(re.fullmatch(r"(?:11|22|33|44|20|40|80)\d{8}", digits))
    return landline if len(parts) == 2 else mobile or landline


@pytest.mark.parametrize(
    "value",
    ["9876543210", "+91 98765 43210", "+91 11 2345 6789 ext. 7", "011 2345 6789"],
)
def test_indian_phone_shape_accepts_local_formats(value):
    assert is_indian_phone(value)


@pytest.mark.parametrize(
    "value",
    ["(011) 555-0107", "(080) 555-0195", "+1 (415) 555-0100", "9876543210 ext. 7"],
)
def test_indian_phone_shape_rejects_transplanted_or_impossible_values(value):
    assert not is_indian_phone(value)


def test_hindi_intake_has_local_phone_plate_shapes_and_distinct_businesses():
    proposal = intake_map("hi")
    phones = [
        e
        for e in proposal.entries
        if e.context.startswith("intake field:contact_phone;")
    ]
    plates = [
        e
        for e in proposal.entries
        if e.context.startswith("intake field:license_plate;")
    ]
    assert len(phones) == 50
    assert len(plates) == 5
    assert all(is_indian_phone(e.forms.native) for e in phones)
    for entry in plates:
        assert entry.forms.native == entry.forms.romanized
        compact = re.sub(r"[ -]", "", entry.forms.native)
        assert compact.isascii()
        # Delhi also uses DL1C-style prefixes (one district digit).
        # https://transport.delhi.gov.in/sites/default/files/Transport/generic_multiple_files/new_zonal_office_informations_0.pdf
        assert re.fullmatch(r"(?:DL\d{1,2}|[A-Z]{2}\d{2})[A-Z]{1,3}\d{4}", compact)
    assert business_identity_collisions(proposal) == {}


def test_collapsed_business_identity_is_detected_in_both_forms():
    changed = intake_map("hi").model_copy(deep=True)
    entities = [
        e
        for e in changed.entries
        if e.context.startswith("intake field:destination_property;")
    ]
    entities[1].forms = entities[0].forms.model_copy(deep=True)
    collisions = business_identity_collisions(changed)
    assert {entities[0].id, entities[1].id} <= set(collisions)
    assert set(collisions[entities[0].id]) == {"native", "romanized"}


@pytest.mark.parametrize(
    "language,country_prefix",
    [("hi", "91"), ("es", "34"), ("pt", "55"), ("zh", "86"), ("ko", "82")],
)
def test_localized_phone_difficulty_does_not_transplant_another_country(
    language, country_prefix
):
    phones = [
        entry
        for entry in intake_map(language).entries
        if entry.context.startswith("intake field:contact_phone;")
    ]
    assert len(phones) == 50
    for entry in phones:
        for representation in ("native", "romanized"):
            value = getattr(entry.forms, representation)
            if "+" in value:
                assert re.sub(r"\D", "", value).startswith(country_prefix), (
                    language,
                    entry.id,
                    value,
                )
