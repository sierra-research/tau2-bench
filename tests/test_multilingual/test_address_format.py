"""Address ordering follows semantic components, never comma positions."""

import pytest

from tau2.multilingual.address_format import (
    AddressParts,
    format_address,
    format_locality,
)


@pytest.mark.parametrize(
    "language,parts,expected",
    [
        (
            "zh",
            dict(
                street_lines=["金水区农业路72号4号楼1103室"],
                city="郑州市",
                region="河南省",
                postal_code="450008",
                country="中国",
            ),
            "中国河南省郑州市金水区农业路72号4号楼1103室，450008",
        ),
        (
            "ko",
            dict(
                street_lines=["팔달로 204"],
                district="완산구",
                city="전주시",
                region="전북특별자치도",
                postal_code="55000",
                country="대한민국",
            ),
            "대한민국 전북특별자치도 전주시 완산구 팔달로 204 (55000)",
        ),
        (
            "es",
            dict(
                street_lines=["Calle Mayor 72", "4.º B"],
                city="Madrid",
                region="Madrid",
                postal_code="28013",
                country="España",
            ),
            "Calle Mayor 72, 4.º B, 28013 Madrid, España",
        ),
        (
            "pt",
            dict(
                street_lines=["Rua das Flores, 72"],
                city="Curitiba",
                region="PR",
                postal_code="80020-100",
                country="Brasil",
            ),
            "Rua das Flores, 72, Curitiba, PR, 80020-100, Brasil",
        ),
        (
            "hi",
            dict(
                street_lines=["72 एमजी रोड"],
                city="पुणे",
                region="महाराष्ट्र",
                postal_code="411001",
                country="भारत",
            ),
            "72 एमजी रोड, पुणे, महाराष्ट्र, 411001, भारत",
        ),
    ],
)
def test_locale_rendering_preserves_components(language, parts, expected):
    assert format_address(AddressParts(**parts), language, "native") == expected


def test_municipality_printed_once_without_erasing_storage_role():
    parts = AddressParts(
        street_lines=["渝北区金开大道88号"], city="重庆市", region="重庆市"
    )
    assert format_address(parts, "zh", "native") == "重庆市渝北区金开大道88号"
    assert parts.city == parts.region == "重庆市"
    assert format_locality(parts, "zh", "native") == "重庆市"


def test_romanization_is_not_assumed_international_postal_english():
    parts = AddressParts(
        street_lines=["Nongye Lu 72 Hao"], city="Zhengzhou Shi", region="Henan Sheng"
    )
    assert (
        format_address(parts, "zh", "romanized")
        == "Henan Sheng Zhengzhou Shi Nongye Lu 72 Hao"
    )


def test_spanish_optional_district_is_preserved():
    parts = AddressParts(
        street_lines=["Calle Mayor 12"],
        district="Centro",
        city="Madrid",
        region="Madrid",
        postal_code="28013",
        country="España",
    )
    assert (
        format_address(parts, "es", "native")
        == "Calle Mayor 12, Centro, 28013 Madrid, España"
    )
