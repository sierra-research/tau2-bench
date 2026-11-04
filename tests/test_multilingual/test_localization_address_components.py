"""Street repairs conserve exact components and never reinterpret units or PINs."""

import pytest

from tau2.multilingual.factory.localization_address_components import repair_street_line


@pytest.mark.parametrize(
    "language,representation,before,after",
    [
        ("es", "native", "616 Calle de Alcalá", "Calle de Alcalá 616"),
        (
            "es",
            "native",
            "548 Gran Via de les Corts Catalanes",
            "Gran Via de les Corts Catalanes 548",
        ),
        ("pt", "native", "993 Rua das Acácias", "Rua das Acácias, 993"),
        ("zh", "native", "986号锦江路", "锦江路986号"),
        ("zh", "romanized", "986 Hao Jinjiang Lu", "Jinjiang Lu 986 Hao"),
        ("zh", "native", "975 干将东路", "干将东路975号"),
        ("zh", "romanized", "975 Ganjiang Dong Lu", "Ganjiang Dong Lu 975 Hao"),
        ("ko", "native", "570 종로", "종로 570"),
        ("ko", "romanized", "570 Jong-ro", "Jong-ro 570"),
    ],
)
def test_known_road_components_reorder_without_numeric_changes(
    language, representation, before, after
):
    assert repair_street_line(before, language, representation) == after
    assert repair_street_line(after, language, representation) == after


@pytest.mark.parametrize(
    "language,value",
    [
        ("hi", "986 प्रभात रोड"),
        ("en", "986 Sunset Drive"),
        ("es", "Calle 9, 123"),
        ("es", "28005"),
        ("zh", "986室"),
        ("ko", "570호"),
        ("pt", "993 Caixa Postal"),
        ("zh", "9K2X7M4P1N8Q3R5T6A"),
        ("es", "12B Unknown Place"),
    ],
)
def test_valid_other_locale_and_ambiguous_nonroad_values_are_not_reinterpreted(
    language, value
):
    assert repair_street_line(value, language, "native") == value
