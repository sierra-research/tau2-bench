"""Tests for the intake domain's check-digit families (VIN, mod-97 member ID)."""

import pytest

from tau2.domains.intake.check_digits import (
    format_member_id,
    is_valid_member_id,
    is_valid_vin,
    member_id_check_digits,
    vin_check_digit,
)

# The classic ISO 3779 worked example: check digit X at position 9.
KNOWN_VIN = "1M8GDM9AXKP042788"


def test_vin_check_digit_known_example():
    assert vin_check_digit(KNOWN_VIN) == "X"
    assert is_valid_vin(KNOWN_VIN)
    assert is_valid_vin(KNOWN_VIN.lower())  # case never carries identity


def test_vin_single_digit_error_is_caught():
    """Any single-digit slip is caught. (Letters transliterate mod 11, so a
    letter swapped for its own transliteration value is the known ISO 3779
    blind spot — digits carry no such alias.)"""
    for position, char in enumerate(KNOWN_VIN):
        if position == 8 or not char.isdigit():
            continue
        mutated = list(KNOWN_VIN)
        mutated[position] = str((int(char) + 1) % 10)
        assert not is_valid_vin("".join(mutated)), f"position {position}"
    # A wrong check digit itself is caught too.
    assert not is_valid_vin(KNOWN_VIN[:8] + "3" + KNOWN_VIN[9:])


@pytest.mark.parametrize(
    "bad", ["1M8GDM9AXKP04278", "1M8GDM9AXKP0427889", "1M8GDM9AXKPO42788"]
)
def test_vin_malformed_rejected(bad):
    """Wrong length or the never-valid letters I/O/Q."""
    assert not is_valid_vin(bad)
    with pytest.raises(ValueError, match="17 characters"):
        vin_check_digit(bad)


def test_member_id_round_trip():
    member_id = format_member_id("605811")
    assert member_id.startswith("MBR-605811-")
    assert is_valid_member_id(member_id)


def test_member_id_folds_case_and_separators():
    member_id = format_member_id("A7K2Q9")
    folded = member_id.replace("-", " ").lower()
    assert is_valid_member_id(folded)


def test_member_id_single_character_error_is_caught():
    member_id = format_member_id("605811")
    mutated = member_id.replace("605811", "605812")
    assert not is_valid_member_id(mutated)


def test_member_id_transposition_is_caught():
    """Mod-97 catches adjacent transpositions — the exact error read-backs
    trade on."""
    assert is_valid_member_id(format_member_id("605811"))
    core = "605811"
    swapped = core[0] + core[2] + core[1] + core[3:]  # 605811 -> 650811
    original = format_member_id(core)
    forged = f"MBR-{swapped}-{original[-2:]}"
    assert not is_valid_member_id(forged)


def test_member_id_malformed_core_rejected():
    with pytest.raises(ValueError, match="4-12 characters"):
        member_id_check_digits("ab!")
    assert not is_valid_member_id("XYZ-1234-00")  # wrong prefix
    assert not is_valid_member_id("MBR-1-23")  # core too short
