"""Tests for the published intake fold rules (design doc §7).

Two properties carry the reward's validity, and both are asserted
exhaustively over every entity bank:

1. **Orthography invariance** — for every bank value, plausible alternate
   surface forms (case variants, separator variants, diacritic-stripped,
   date reformattings) canonicalize to the SAME stored form.
2. **No fold collisions** — every decoy canonicalizes to a DIFFERENT stored
   form than its truth (asserted in the strong form: no decoy folds onto ANY
   real entry of its bank), and all entries of a bank stay distinct under
   the fold.
"""

import unicodedata
from datetime import date

import pytest

from tau2.domains.intake.folds import (
    FoldKind,
    fold_amount,
    fold_code,
    fold_date,
    fold_email,
    fold_field,
    fold_name,
    fold_name_list,
    fold_phone,
    fold_time,
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
    # '+', the international '00' prefix, and the bare country code are three
    # writings of the same number; none of them may decide the reward.
    assert (
        fold_phone("+44 20 7946 0958")
        == fold_phone("0044 20 7946 0958")
        == fold_phone("44 20 7946 0958")
        == "442079460958"
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


def test_fold_phone_digit_swap_stays_distinct():
    assert fold_phone("555-123-2002") != fold_phone("555-123-2020")


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


def test_fold_email_folds_case_only():
    assert fold_email(" Anna.Baker@VeltaMail.com ") == "anna.baker@veltamail.com"
    # Separators in a local part are identity-bearing (§4 hard emails):
    # they must NOT fold together.
    assert fold_email("ilse-marie_ab@qorvexmail.io") != fold_email(
        "ilse-marie.ab@qorvexmail.io"
    )


def test_fold_date_unambiguous_forms_normalize():
    variants = [
        "2025-07-16",
        "2025/7/16",
        "July 16, 2025",
        "16 July 2025",
        "Jul 16 2025",
        "16th of July 2025",
        "20250716",
        # Year-last numerics with a day > 12 have exactly one month reading.
        "07/16/2025",
        "16/07/2025",
        "16.07.2025",
    ]
    assert {fold_date(v) for v in variants} == {"2025-07-16"}


def test_fold_date_ambiguous_or_invalid_forms_stay_as_given():
    # Both components could be the month: genuinely ambiguous, not folded.
    assert fold_date("07/06/2025") == "07/06/2025"
    assert fold_date("13/13/2025") == "13/13/2025"  # no valid month reading
    assert fold_date("2025-02-30") == "2025-02-30"  # not a calendar date
    assert fold_date("sometime soon") == "sometime soon"


def test_fold_name_list_preserves_order_and_canonicalizes_separators():
    variants = [
        "Cabergoline 0.5 mg tablet; Zonisamide 100 mg capsule",
        "Cabergoline 0.5 mg tablet, Zonisamide 100 mg capsule",
        "cabergoline 0.5 mg tablet;zonisamide 100 mg capsule",
    ]
    assert {fold_name_list(v) for v in variants} == {
        "cabergoline 05 mg tablet; zonisamide 100 mg capsule"
    }
    # Order is caller-pinned, never sorted away.
    assert fold_name_list("B; A") == "b; a"


def test_fold_time_orthography_variants_agree():
    variants = ["9:00 AM", "09:00 am", "9 AM", "9 a.m.", "  9:00 AM  "]
    assert {fold_time(v) for v in variants} == {"9:00 AM"}
    # Unambiguous 24-hour forms normalize.
    assert fold_time("14:30") == "2:30 PM"
    assert fold_time("00:15") == "12:15 AM"
    assert fold_time("23:05") == "11:05 PM"
    # The unambiguous words are their own meridiem.
    assert fold_time("noon") == fold_time("12 noon") == fold_time("12:00 PM")
    assert fold_time("midnight") == fold_time("12:00 AM")


def test_fold_time_meridiem_is_identity_bearing():
    assert fold_time("7:05 AM") != fold_time("7:05 PM")
    assert fold_time("10:05 AM") != fold_time("10:50 AM")


def test_fold_time_ambiguous_forms_stay_as_given():
    # A bare 12-hour time has no meridiem: genuinely ambiguous, not folded
    # (the meridiem was already pinned by policy — failing is correct).
    assert fold_time("2:30") == "2:30"
    assert fold_time("12:00") == "12:00"
    assert fold_time("2:75 PM") == "2:75 PM"  # not a clock time
    assert fold_time("14:30 PM") == "14:30 PM"  # contradictory reading
    assert fold_time("whenever") == "whenever"


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
    plus = value.strip().startswith("+")
    digits = "".join(ch for ch in value if ch.isdigit())
    grouped = f"{digits[:3]} {digits[3:6]} {digits[6:]}"
    alternates = [
        ("+" if plus else "") + digits,
        ("+" if plus else "") + grouped,
        value.replace("-", " ").replace("(", "").replace(")", ""),
    ]
    if plus:
        # '+44…' == '0044…' == '44…' — three writings of the same dialing.
        alternates.extend(["00" + digits, digits])
    elif len(digits) == 10 and "ext" not in value.lower():
        # A NANP number with the country code written out.
        alternates.extend([f"+1 {value}", f"1-{digits}"])
    return alternates


def _code_alternates(value: str) -> list[str]:
    key = "".join(ch for ch in value if ch.isalnum())
    return [value.lower(), key, key.lower(), " ".join([key[:3], key[3:]])]


def _date_alternates(value: str) -> list[str]:
    parsed = date.fromisoformat(value)
    month = parsed.strftime("%B")
    return [
        value,
        f"{parsed.year}/{parsed.month}/{parsed.day}",
        f"{month} {parsed.day}, {parsed.year}",
        f"{parsed.day} {month} {parsed.year}",
    ]


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
    _assert_space(
        "phones",
        [(e.value, e.decoys) for e in banks.phones],
        fold_phone,
        _phone_alternates,
    )


def test_code_bank_fold_invariant_and_collision_free(banks: Banks):
    _assert_space(
        "codes",
        [(e.value, e.decoys) for e in banks.codes],
        fold_code,
        _code_alternates,
    )


def test_date_bank_fold_invariant(banks: Banks):
    for entry in banks.dates:
        for alternate in _date_alternates(entry.value):
            assert fold_date(alternate) == entry.value


def _time_alternates(value: str) -> list[str]:
    hm, meridiem = value.split(" ")
    hour, minute = hm.split(":")
    return [
        f"{hm} {meridiem.lower()}",
        f"{hm} {meridiem[0].lower()}.{meridiem[1].lower()}.",
        f"{hour.zfill(2)}:{minute} {meridiem}",
        f"  {value}  ",
    ]


def test_time_bank_fold_invariant_and_collision_free(banks: Banks):
    _assert_space(
        "times",
        [(e.value, e.decoys) for e in banks.times],
        fold_time,
        _time_alternates,
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
