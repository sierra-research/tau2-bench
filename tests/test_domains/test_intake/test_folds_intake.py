"""Tests for the published intake fold rules.

Two properties carry the reward's validity, and both are asserted
exhaustively over every entity bank:

1. **Orthography invariance** — for every bank value, plausible alternate
   surface forms permitted by that field's rule canonicalize to the SAME
   stored form. Dates and times permit outer whitespace only.
2. **No decoy collisions** — every decoy canonicalizes to a DIFFERENT stored
   form than every real entry of its bank. Phone entries differing only by
   extension deliberately share a scoring key.
"""

import re
import unicodedata
from functools import partial

import pytest

from tau2.domains.intake.folds import (
    _MEDICATION_STRENGTH_WORDS,
    _STRENGTH_WORDS_BY_LANGUAGE,
    FoldKind,
    fold_amount,
    fold_code,
    fold_email,
    fold_field,
    fold_name,
    fold_name_list,
    fold_phone,
    fold_value,
)
from tau2.domains.intake.tasks.banks import (
    Banks,
    instantiate_email,
    load_banks,
)

# ---------------------------------------------------------------------------
# Unit behavior per fold rule
# ---------------------------------------------------------------------------


def test_fold_phone_orthography_variants_agree():
    variants = ["555 123 2002", "555-123-2002", "(555) 123-2002", "5551232002"]
    assert {fold_phone(v) for v in variants} == {"5551232002"}


def test_fold_phone_normalizes_dialing_prefixes():
    # Country prefixes and international dialing markers do not decide reward.
    assert (
        fold_phone("+44 20 7946 0958")
        == fold_phone("0044 20 7946 0958")
        == fold_phone("44 20 7946 0958")
        == fold_phone("20 7946 0958")
        == "2079460958"
    )
    # NANP country code: '+1 (206) 555-0111' == '1-206-555-0111' == the
    # 10-digit domestic form.
    assert (
        fold_phone("+1 (206) 555-0111")
        == fold_phone("1-206-555-0111")
        == fold_phone("(206) 555-0111")
        == "2065550111"
    )
    # Both prefixes together: international dialing of a NANP number.
    assert fold_phone("001 206 555 0111") == "2065550111"
    # A remaining leading zero (e.g. a UK trunk '0') still carries identity.
    assert fold_phone("020 7946 0958") == "02079460958"
    assert fold_phone("555-010-2211 ext. 34") == fold_phone("555-010-2211 x34")
    assert fold_phone("555-010-2211 ext. 34") == fold_phone("555-010-2211")
    assert fold_phone("+34 91 794 6023") == fold_phone("91 794 6023")
    assert fold_phone("+34 91 482 63 17 ext. 7") == fold_phone("91 482 63 17")
    assert fold_phone("+34 91 482 63 17 ext. 7") != fold_phone("91 482 63 18")
    assert fold_phone("010-8264-0117-7") == fold_phone("010-8264-0117")
    assert fold_phone("139 1056 0179") == "13910560179"
    assert fold_phone("139 1056 0179") != fold_phone("138 1056 0179")


def test_fold_phone_digit_swap_stays_distinct():
    assert fold_phone("555-123-2002") != fold_phone("555-123-2020")


def test_korean_mobile_trunk_zero_matches_international_format_only():
    korean = partial(fold_value, FoldKind.KOREAN_PHONE)
    assert korean("010-3998-0651") == korean("+82 10 3998 0651")
    assert korean("010-3998-0651") != korean("010-3998-0652")
    assert korean("010-3998-0651") != korean("011-3998-0651")
    assert fold_phone("010-3998-0651") != fold_phone("+82 10 3998 0651")


def test_chinese_landline_trunk_zero_matches_international_format_only():
    chinese = partial(fold_value, FoldKind.CHINESE_PHONE)
    assert chinese("010 6839 0235") == chinese("+86 10 6839 0235")
    assert chinese("021 6127 3244") == chinese("+86 21 6127 3244")
    assert chinese("010 6839 0235") != chinese("010 6839 0236")
    assert chinese("010 6839 0235") != chinese("021 6839 0235")
    assert chinese("139 6208 0651") == fold_phone("139 6208 0651")
    assert chinese("0139 6208 0651") != chinese("139 6208 0651")
    assert fold_phone("010 6839 0235") != fold_phone("+86 10 6839 0235")


def test_fold_phone_without_digits_returns_input():
    assert fold_phone(" no phone ") == "no phone"


def test_fold_code_orthography_variants_agree():
    variants = ["MBR-7802", "mbr 7802", "mbr-7802", "MBR7802"]
    assert {fold_code(v) for v in variants} == {"MBR7802"}
    assert fold_code("1ftfw1et5dfc10312") == "1FTFW1ET5DFC10312"


def test_fold_name_orthography_variants_agree():
    variants = [
        "Ana de la Vega-Marchetti",
        "ana de la vega-marchetti",
        "ANA DE LA VEGA MARCHETTI",
        "Ana de la Vega Marchetti",
    ]
    assert {fold_name(v) for v in variants} == {"ana de la vega marchetti"}
    assert fold_name("Renée Vásquez") == fold_name("Renee Vasquez")
    assert fold_name("Dr. O'Brien") == fold_name("dr obrien")
    assert fold_name("14 Larkspur Lane, Apt 3B") == fold_name("14 Larkspur Lane Apt 3B")


def test_fold_name_ampersand_and_digit_boundaries_agree():
    # '&' is spoken 'and': a writer hearing the name cannot know which the
    # record used.
    assert fold_name("Q7 Vellum & Daughters plc") == fold_name(
        "Q7 Vellum and Daughters plc"
    )
    # A digit next to a letter is a boundary the writer may or may not mark.
    assert (
        fold_name("Qventra K9-Plus Analytics")
        == fold_name("Qventra K9 Plus Analytics")
        == fold_name("Qventra K9Plus Analytics")
    )
    assert fold_name("Apt 3B") == fold_name("Apt 3 B")
    # '%' is spoken 'percent'.
    assert fold_name("Prilocaine 2% cream") == fold_name("prilocaine 2 percent cream")
    # Different digits stay different.
    assert fold_name("nGauge X2 Systems") != fold_name("nGauge X3 Systems")


def test_fold_name_preserves_abbreviated_and_written_units():
    assert fold_name("Tramadol, 50 mg tablet") == "tramadol 50 mg tablet"
    assert fold_name("Tramadol, 50 milligram tablet") == "tramadol 50 milligram tablet"
    assert (
        fold_name("Tramadol, 50 milligrams tablet") == "tramadol 50 milligrams tablet"
    )


@pytest.mark.parametrize("strength", [".5", "0.5", "1.25"])
def test_fold_name_preserves_decimal_medication_strength(strength):
    value = f"Medicine {strength} mg tablet"
    assert fold_name(value) == f"medicine {strength} mg tablet"
    assert fold_name(value) != fold_name(value.replace(".", ""))
    assert fold_name("J. P. Rowan") == "j p rowan"


def test_fold_email_folds_case_only():
    assert fold_email(" Anna.Baker@VeltaMail.com ") == "anna.baker@veltamail.com"
    # Separators in an email local part are identity-bearing:
    # they must NOT fold together.
    assert fold_email("ilse-marie_ab@qorvexmail.io") != fold_email(
        "ilse-marie.ab@qorvexmail.io"
    )


def test_date_format_is_preserved_except_outer_whitespace():
    assert fold_value(FoldKind.DATE, "  2025-07-16\n") == "2025-07-16"
    variants = [
        "2025/7/16",
        "July 16, 2025",
        "16 July 2025",
        "Jul 16 2025",
        "16th of July 2025",
        "20250716",
        "07/16/2025",
        "16/07/2025",
        "16.07.2025",
        "2025-7-16",
    ]
    for value in variants:
        assert fold_value(FoldKind.DATE, value) == value
        assert fold_value(FoldKind.DATE, value) != "2025-07-16"


@pytest.mark.parametrize(
    "value", ["07/06/2025", "13/13/2025", "2025-02-30", "sometime soon"]
)
def test_invalid_dates_are_not_repaired(value):
    assert fold_value(FoldKind.DATE, value) == value


def test_fold_name_list_preserves_order_and_canonicalizes_separators():
    variants = [
        "Cabergoline 0.5 mg tablet; Zonisamide 100 mg capsule",
        "Cabergoline 0.5 mg tablet, Zonisamide 100 mg capsule",
        "cabergoline 0.5 mg tablet;zonisamide 100 mg capsule",
    ]
    assert {fold_name_list(v) for v in variants} == {
        "cabergoline 0.5 mg tablet; zonisamide 100 mg capsule"
    }
    # Order is caller-pinned, never sorted away.
    assert fold_name_list("B; A") == "b; a"


def test_time_format_is_preserved_except_outer_whitespace():
    assert fold_value(FoldKind.TIME, "  9:00 AM\n") == "9:00 AM"
    for value in ["09:00 am", "9 AM", "9 a.m.", "9:00 am", "09:00 AM"]:
        assert fold_value(FoldKind.TIME, value) == value
        assert fold_value(FoldKind.TIME, value) != "9:00 AM"
    for value in ["14:30", "00:15", "23:05", "noon", "12 noon", "midnight"]:
        assert fold_value(FoldKind.TIME, value) == value


def test_time_meridiem_and_minutes_are_identity_bearing():
    assert fold_value(FoldKind.TIME, "7:05 AM") != fold_value(FoldKind.TIME, "7:05 PM")
    assert fold_value(FoldKind.TIME, "10:05 AM") != fold_value(
        FoldKind.TIME, "10:50 AM"
    )


@pytest.mark.parametrize("value", ["2:30", "12:00", "2:75 PM", "14:30 PM", "whenever"])
def test_invalid_times_are_not_repaired(value):
    assert fold_value(FoldKind.TIME, value) == value


def test_fold_amount_orthography_variants_agree():
    variants = ["1250", "1,250", "$1,250.00", "1250.00", "$ 1250 USD"]
    assert {fold_amount(v) for v in variants} == {"1250"}
    assert fold_amount("89.07") == fold_amount("$89.07") == "89.07"
    assert fold_amount("350 dollars") == "350"


def test_fold_amount_digits_stay_identity_bearing():
    assert fold_amount("125.5") != fold_amount("125.05")
    assert fold_amount("89.07") != fold_amount("89.70")
    assert fold_amount("not a number") == "not a number"


def test_fold_field_explicit_none_sentinel():
    assert fold_field(FoldKind.EMAIL, None, allows_none=True) is None
    assert fold_field(FoldKind.EMAIL, "none", allows_none=True) is None
    assert fold_field(FoldKind.NAME, " N/A ", allows_none=True) is None
    # On a field that does not accept none the token stays literal (and
    # correctly fails the DB match instead of silently becoming a sentinel).
    assert fold_field(FoldKind.NAME, "none", allows_none=False) == "none"


@pytest.mark.parametrize(
    "kind,gold,alternate,wrong",
    [
        (FoldKind.NAME_DIGITS, "Aarav 7", "Aarav ७", "Aarav ८"),
        (FoldKind.NATIVE_NAME_DIGITS, "आरव 7", "आरव ७", "आरव ८"),
        (FoldKind.CODE, "MBR-702", "MBR-७०२", "MBR-७०३"),
        (
            FoldKind.PHONE,
            "+91 11 4555 0197 ext. 6",
            "+९१ ११ ४५५५ ०१९७ ext. ६",
            "+९१ ११ ४५५५ ०१९८ ext. ७",
        ),
        (FoldKind.DATE, "2025-07-16", "२०२५-०७-१६", "२०२५-०७-१७"),
        (FoldKind.TIME, "7:05 AM", "७:०५ AM", "७:०६ AM"),
        (
            FoldKind.NATIVE_ADDRESS,
            "1518 अशोक ड्राइव, रांची",
            "१५१८ अशोक ड्राइव रांची",
            "१५१९ अशोक ड्राइव रांची",
        ),
        (FoldKind.NATIVE_MEDICATION, "दवा 0.5 mg", "दवा ०.५ mg", "दवा ०५ mg"),
    ],
)
def test_included_folds_normalize_decimal_digit_scripts(kind, gold, alternate, wrong):
    assert fold_value(kind, gold) == fold_value(kind, alternate)
    assert fold_value(kind, gold) != fold_value(kind, wrong)


def test_digit_script_exclusions_and_non_decimal_numerals():
    for kind in (FoldKind.NAME, FoldKind.NATIVE_NAME, FoldKind.EMAIL):
        assert fold_value(kind, "x7") != fold_value(kind, "x७")
    assert fold_value(FoldKind.NAME_DIGITS, "X7") != fold_value(
        FoldKind.NAME_DIGITS, "X⁷"
    )


def test_native_address_only_ignores_commas():
    assert fold_value(FoldKind.NATIVE_ADDRESS, "A-B, C") == fold_value(
        FoldKind.NATIVE_ADDRESS, "A-B C"
    )
    for address in ("18,28001", "18，28001"):
        assert fold_value(FoldKind.NATIVE_ADDRESS, address) == fold_value(
            FoldKind.NATIVE_ADDRESS, "18 28001"
        )
    assert fold_value(FoldKind.NATIVE_ADDRESS, "北京市，朝阳区、三里屯") == fold_value(
        FoldKind.NATIVE_ADDRESS, "北京市 朝阳区 三里屯"
    )
    assert fold_value(FoldKind.NATIVE_ADDRESS, "A-B C") != fold_value(
        FoldKind.NATIVE_ADDRESS, "A B C"
    )
    assert fold_value(FoldKind.NATIVE_ADDRESS, "A. B") != fold_value(
        FoldKind.NATIVE_ADDRESS, "A B"
    )
    # A property name can contain a comma too; its written form stays exact.
    assert fold_value(FoldKind.NATIVE_NAME, "Casa Verde, Madrid") != fold_value(
        FoldKind.NATIVE_NAME, "Casa Verde Madrid"
    )


def test_korean_name_spacing_is_limited_to_hangul_and_name_fields():
    korean = partial(fold_value, FoldKind.KOREAN_NAME)
    assert korean("김 지수") == korean("김지수")
    assert korean("호텔 한빛정원") == korean("호텔한빛정원")
    assert korean("김지수") != korean("김지훈")
    assert fold_value(FoldKind.NATIVE_NAME, "호텔 한빛정원") != fold_value(
        FoldKind.NATIVE_NAME, "호텔한빛정원"
    )
    assert korean("Hotel A") != korean("HotelA")


def test_korean_postcode_parentheses_fold_without_label_or_digit_repair():
    korean = partial(fold_value, FoldKind.KOREAN_ADDRESS)
    gold = "경기도 성남시 분당구 미금로 45 (13618)"
    assert korean(gold) == korean("경기도 성남시 분당구 미금로 45 13618")
    assert korean(gold) != korean("경기도 성남시 분당구 미금로 45 13619")
    assert korean(gold) != korean("경기도 성남시 분당구 미금로 45 우편번호 13618")
    assert korean("(13618) 미금로 45") != korean("13618 미금로 45")
    assert fold_value(FoldKind.NATIVE_ADDRESS, gold) != fold_value(
        FoldKind.NATIVE_ADDRESS, "경기도 성남시 분당구 미금로 45 13618"
    )


@pytest.mark.parametrize(
    ("code", "name"),
    [
        ("SP", "São Paulo"),
        ("PR", "Paraná"),
        ("MG", "Minas Gerais"),
        ("RJ", "Rio de Janeiro"),
        ("RS", "Rio Grande do Sul"),
        ("PE", "Pernambuco"),
        ("PA", "Pará"),
        ("SC", "Santa Catarina"),
    ],
)
def test_portuguese_address_state_slot_equivalence(code: str, name: str):
    portuguese = partial(fold_value, FoldKind.PORTUGUESE_ADDRESS)
    assert portuguese(f"Rua das Flores 7, Campinas - {code}, 13025-090") == (
        portuguese(f"Rua das Flores 7, Campinas - {name}, 13025-090")
    )
    assert portuguese(f"Rua das Flores 7, Campinas, {code}") == portuguese(
        f"Rua das Flores 7, Campinas, {name}"
    )
    assert portuguese(f"Rua Ipê-Roxo 7, Campinas - {code}, 13025-090") == (
        portuguese(f"Rua Ipê-Roxo 7, Campinas, {name}, 13025-090")
    )
    assert portuguese(f"Rua Ipê-Roxo 7, Campinas - {code}, 13025-090") != (
        portuguese(f"Rua Ipê Roxo 7, Campinas, {name}, 13025-090")
    )
    assert portuguese(f"Rua {name} 7, Campinas, {code}") != portuguese(
        f"Rua {code} 7, Campinas, {code}"
    )
    assert fold_value(FoldKind.NATIVE_ADDRESS, f"Campinas - {code}") != (
        fold_value(FoldKind.NATIVE_ADDRESS, f"Campinas - {name}")
    )


@pytest.mark.parametrize(
    ("kind", "written", "digits"),
    [
        (FoldKind.MEDICATION, "one hundred milligrams", "100 mg"),
        (FoldKind.HINDI_MEDICATION, "दस मिलीग्राम", "10 mg"),
        (FoldKind.KOREAN_MEDICATION, "십밀리그램", "10 mg"),
        (FoldKind.CHINESE_MEDICATION, "一百毫克", "100 mg"),
        (FoldKind.PORTUGUESE_MEDICATION, "dez miligramas", "10 mg"),
        (FoldKind.SPANISH_MEDICATION, "diez miligramos", "10 mg"),
    ],
)
def test_medication_strength_number_words_only(
    kind: FoldKind, written: str, digits: str
):
    medication = partial(fold_value, kind)
    assert medication(f"Drug {written} tablet") == medication(f"Drug {digits} tablet")
    assert medication(f"Drug {written} tablet") != medication("Drug 11 mg tablet")
    assert fold_value(FoldKind.NATIVE_NAME, written) != fold_value(
        FoldKind.NATIVE_NAME, digits
    )


def test_medication_number_words_do_not_cross_language_or_field():
    assert fold_value(FoldKind.PORTUGUESE_MEDICATION, "Drug dez mg") != fold_value(
        FoldKind.PORTUGUESE_MEDICATION, "Drug diez mg"
    )
    assert fold_value(FoldKind.NATIVE_MEDICATION, "Drug dez mg") != fold_value(
        FoldKind.NATIVE_MEDICATION, "Drug 10 mg"
    )
    assert fold_value(FoldKind.NATIVE_ADDRESS, "Rua dez") != fold_value(
        FoldKind.NATIVE_ADDRESS, "Rua 10"
    )


def test_every_published_medication_strength_word_has_numeric_equivalence():
    kinds = {
        "en": FoldKind.MEDICATION,
        "hi": FoldKind.HINDI_MEDICATION,
        "ko": FoldKind.KOREAN_MEDICATION,
        "zh": FoldKind.CHINESE_MEDICATION,
        "pt": FoldKind.PORTUGUESE_MEDICATION,
        "es": FoldKind.SPANISH_MEDICATION,
    }
    for language, alternatives in _STRENGTH_WORDS_BY_LANGUAGE.items():
        kind = kinds[language]
        for word in alternatives.split("|"):
            digits = _MEDICATION_STRENGTH_WORDS[word.casefold()]
            assert fold_value(kind, f"Medicine {word} mg tablet") == fold_value(
                kind, f"Medicine {digits} mg tablet"
            )


def test_attested_mandarin_two_hundred_fifty_milligrams_full_medication():
    assert fold_value(
        FoldKind.CHINESE_MEDICATION, "特比萘芬片，两百五十 milligram"
    ) == fold_value(FoldKind.CHINESE_MEDICATION, "特比萘芬片 250 mg")


def test_fullwidth_decimal_comma_in_milligram_strength():
    for kind in (FoldKind.MEDICATION, FoldKind.CHINESE_MEDICATION):
        assert fold_value(kind, "Medicine 0，1 mg") == fold_value(
            kind, "Medicine 0.1 mg"
        )
        assert fold_value(kind, "Medicine 0，1 mg") != fold_value(
            kind, "Medicine 01 mg"
        )


def test_native_medication_ignores_separator_commas_and_spanish_unit_spelling():
    gold = "Clonidina 0,1 mg comprimido"
    for alternate in (
        "Clonidina, 0,1 miligramo, comprimido",
        "Clonidina 0.1 miligramos comprimido",
    ):
        assert fold_value(FoldKind.NATIVE_MEDICATION, gold) == fold_value(
            FoldKind.NATIVE_MEDICATION, alternate
        )
    assert fold_value(FoldKind.NATIVE_MEDICATION, gold) != fold_value(
        FoldKind.NATIVE_MEDICATION, "Clonidina 0,2 mg comprimido"
    )
    assert fold_value(FoldKind.NATIVE_MEDICATION, gold) != fold_value(
        FoldKind.NATIVE_MEDICATION, "Clonidina 01 mg comprimido"
    )
    assert fold_value(FoldKind.NATIVE_NAME, "Clonidina, 1 mg") != fold_value(
        FoldKind.NATIVE_NAME, "Clonidina 1 mg"
    )
    assert fold_value(
        FoldKind.NATIVE_MEDICATION, "Clonidina 0,1 mg comprimido"
    ) == fold_value(FoldKind.NATIVE_MEDICATION, "Clonidina, 0,1 miligrama, comprimido")


@pytest.mark.parametrize(
    ("kind", "gold", "alternate", "wrong_dose"),
    [
        (
            FoldKind.MEDICATION,
            "Clonidine 0.1 mg tablet",
            "Clonidine 0.1 milligrams tablet",
            "Clonidine 0.2 milligrams tablet",
        ),
        (
            FoldKind.NATIVE_MEDICATION,
            "बिसोप्रोलोल 5 mg टैबलेट",
            "बिसोप्रोलोल ५ मिलीग्राम टैबलेट",
            "बिसोप्रोलोल ६ मिलीग्राम टैबलेट",
        ),
        (
            FoldKind.NATIVE_MEDICATION,
            "बिसोप्रोलोल 5 mg टैबलेट",
            "बिसोप्रोलोल 5 एमजी टैबलेट",
            "बिसोप्रोलोल 6 एमजी टैबलेट",
        ),
        (
            FoldKind.NATIVE_MEDICATION,
            "아지트로마이신 250 mg 정",
            "아지트로마이신 250밀리그램정",
            "아지트로마이신 500밀리그램정",
        ),
        (
            FoldKind.NATIVE_MEDICATION,
            "昂丹司琼片 4 mg",
            "昂丹司琼片4毫克",
            "昂丹司琼片8毫克",
        ),
        (
            FoldKind.NATIVE_MEDICATION,
            "Clonidina 0.1 mg comprimido",
            "Clonidina 0,1 miligramos comprimido",
            "Clonidina 0,2 miligramos comprimido",
        ),
        (
            FoldKind.NATIVE_MEDICATION,
            "Clonidina 0.1 mg comprimido",
            "Clonidina 0,1 miligrama comprimido",
            "Clonidina 0,2 miligrama comprimido",
        ),
    ],
)
def test_medication_milligram_units_across_languages(
    kind: FoldKind, gold: str, alternate: str, wrong_dose: str
):
    assert fold_value(kind, gold) == fold_value(kind, alternate)
    assert fold_value(kind, gold) != fold_value(kind, wrong_dose)


# ---------------------------------------------------------------------------
# Exhaustive bank sweep
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def banks() -> Banks:
    return load_banks()


def _deaccent(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _name_alternates(value: str) -> list[str]:
    return [
        value.upper(),
        value.lower(),
        _deaccent(value),
        value.replace("-", " "),
        f"  {value}  ",
    ]


def _phone_alternates(value: str) -> list[str]:
    base = re.sub(r"(?i)\s*(?:ext\.?|x)\s*\d+\s*$", "", value)
    plus = base.strip().startswith("+")
    digits = "".join(ch for ch in base if ch.isdigit())
    grouped = f"{digits[:3]} {digits[3:6]} {digits[6:]}"
    alternates = [
        base,
        ("+" if plus else "") + digits,
        ("+" if plus else "") + grouped,
        base.replace("-", " ").replace("(", "").replace(")", ""),
    ]
    if plus:
        # '+44…' == '0044…' == '44…' — three writings of the same dialing.
        alternates.extend(["00" + digits, digits])
    elif len(digits) == 10:
        # A NANP number with the country code written out.
        alternates.extend([f"+1 {base}", f"1-{digits}"])
    return alternates


def _code_alternates(value: str) -> list[str]:
    key = "".join(ch for ch in value if ch.isalnum())
    return [value.lower(), key, key.lower(), " ".join([key[:3], key[3:]])]


def _outer_whitespace_alternates(value: str) -> list[str]:
    return [f" {value}", f"{value} ", f"\t{value}\n"]


def _name_bank_spaces(banks: Banks) -> dict[str, list[tuple[str, list[str]]]]:
    """Every NAME-folded compare space: (value, decoy values) pairs per bank."""
    return {
        bank_name: [(e.value, e.decoys) for e in getattr(banks, bank_name)]
        for bank_name in (
            "person_names",
            "addresses",
            "coined",
            "medications",
            "rate_plans",
            "insurance_plans",
            "shops",
            "properties",
            "vehicles",
        )
    }


def _assert_space(label, pairs, fold, alternates) -> None:
    folded = {}
    for value, _ in pairs:
        key = fold(value)
        assert key not in folded, (
            f"{label}: {value!r} and {folded[key]!r} fold together as {key!r}"
        )
        folded[key] = value
    for value, decoys in pairs:
        for alternate in alternates(value):
            assert fold(alternate) == fold(value), (
                f"{label}: alternate {alternate!r} of {value!r} folds differently"
            )
        for decoy in decoys:
            assert fold(decoy) not in folded, (
                f"{label}: decoy {decoy!r} of {value!r} folds onto a real entry"
            )


def test_name_banks_fold_invariant_and_collision_free(banks: Banks):
    for label, pairs in _name_bank_spaces(banks).items():
        _assert_space(label, pairs, fold_name, _name_alternates)


def test_person_alt_names_stay_distinct_from_current_names(banks: Banks):
    """A former name (the different-key decoy path) must never fold onto the
    current name, or the broken route would silently work."""
    for entry in banks.person_names:
        if entry.alt_name:
            assert fold_name(entry.alt_name) != fold_name(entry.value)


def test_phone_bank_fold_invariant_and_collision_free(banks: Banks):
    # The bank contains five intentionally different extensions on the same
    # domestic number. They now share a scoring key, but no decoy may match
    # any bank truth after the requested extension fold.
    truth_keys = {fold_phone(entry.value) for entry in banks.phones}
    for entry in banks.phones:
        for alternate in _phone_alternates(entry.value):
            assert fold_phone(alternate) == fold_phone(entry.value)
        for decoy in entry.decoys:
            assert fold_phone(decoy) not in truth_keys


def test_code_bank_fold_invariant_and_collision_free(banks: Banks):
    _assert_space(
        "codes",
        [(e.value, e.decoys) for e in banks.codes],
        fold_code,
        _code_alternates,
    )


def test_date_bank_preserves_its_recorded_format(banks: Banks):
    for entry in banks.dates:
        for alternate in _outer_whitespace_alternates(entry.value):
            assert fold_value(FoldKind.DATE, alternate) == entry.value


def test_time_bank_fold_invariant_and_collision_free(banks: Banks):
    _assert_space(
        "times",
        [(e.value, e.decoys) for e in banks.times],
        partial(fold_value, FoldKind.TIME),
        _outer_whitespace_alternates,
    )


def _amount_alternates(value: str) -> list[str]:
    alternates = [f"${value}", f"{value} USD", f"  {value}  "]
    if "." not in value:
        alternates.append(f"{value}.00")
        if len(value) == 4:
            alternates.append(f"{value[0]},{value[1:]}")
    else:
        whole, cents = value.split(".")
        alternates.append(f"{whole}.{cents}0")
    return alternates


def test_amount_bank_fold_invariant_and_collision_free(banks: Banks):
    _assert_space(
        "amounts",
        [(e.value, e.decoys) for e in banks.amounts],
        fold_amount,
        _amount_alternates,
    )


def test_every_email_instantiation_is_wellformed_and_collision_free(banks: Banks):
    """Exhaustive: every email pattern x every bank person (current and former
    names) instantiates to a valid address whose decoy stays distinct under
    the published email fold, and case variants fold together."""
    owners = [e.value for e in banks.person_names] + [
        e.alt_name for e in banks.person_names if e.alt_name
    ]
    for entry in banks.emails:
        for owner in owners:
            drawn = instantiate_email(entry, owner)  # raises if malformed
            assert fold_email(drawn.value.upper()) == fold_email(drawn.value)
            assert fold_email(drawn.decoys[0]) != fold_email(drawn.value)


def test_fold_value_dispatch_covers_every_kind():
    for kind in FoldKind:
        assert fold_value(kind, "Sample 1") is not None
