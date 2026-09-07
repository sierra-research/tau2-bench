"""Tests for the oddball-token pronunciation table: the fixed extractor
classes, extractor <-> table alignment on load (missing/orphan/class-drift
failures), the review packet, and class coverage."""

import shutil

import pytest
import yaml

from tau2.domains.intake.tasks.banks import INTAKE_BANKS_DIR
from tau2.domains.intake.tasks.token_pronunciations import (
    ODDBALL_SOURCE_BANKS,
    TOKEN_PRONUNCIATIONS_FILENAME,
    OddballTokenClass,
    build_token_packet,
    classify_token,
    extract_oddball_tokens,
    load_token_pronunciations,
)

# ---------------------------------------------------------------------------
# The fixed extractor classes
# ---------------------------------------------------------------------------


def test_classify_token_fixed_classes():
    assert classify_token("GmbH") is OddballTokenClass.MIXED_CASE_INNER
    assert classify_token("XyloNine") is OddballTokenClass.MIXED_CASE_INNER
    assert classify_token("gCaorach") is OddballTokenClass.MIXED_CASE_INNER
    assert classify_token("xDrive") is OddballTokenClass.MIXED_CASE_INNER
    assert classify_token("PPO") is OddballTokenClass.ALL_CAPS
    assert classify_token("MICE") is OddballTokenClass.ALL_CAPS
    assert classify_token("II") is OddballTokenClass.ALL_CAPS
    assert classify_token("B&B") is OddballTokenClass.AMPERSAND
    # Letter-digit mixes are the ignored class (owner decision) unless the
    # owner graduated them (packet review 2026-08-26).
    assert classify_token("3B") is None
    assert classify_token("Q7") is None
    assert classify_token("RAV4") is None
    assert classify_token("228i") is OddballTokenClass.LETTER_DIGIT
    assert classify_token("4MATIC") is OddballTokenClass.LETTER_DIGIT
    assert classify_token("sDrive28i") is OddballTokenClass.LETTER_DIGIT
    assert classify_token("Dr4gline") is OddballTokenClass.LETTER_DIGIT
    # Plain words and lone capitals are not oddballs.
    assert classify_token("Lane") is None
    assert classify_token("wholesale") is None
    assert classify_token("A") is None


def test_extraction_is_deterministic_and_nonempty():
    first = extract_oddball_tokens()
    second = extract_oddball_tokens()
    assert first == second
    assert first.tokens
    assert first.ignored_letter_digit == sorted(set(first.ignored_letter_digit))
    # The ignored class never leaks into the table classes.
    assert not set(first.ignored_letter_digit) & set(first.tokens)


# ---------------------------------------------------------------------------
# Loader validation (extractor <-> table alignment)
# ---------------------------------------------------------------------------


def test_checked_in_table_loads_and_aligns():
    entries = load_token_pronunciations()
    extraction = extract_oddball_tokens()
    assert {entry.token for entry in entries} == set(extraction.tokens)
    for entry in entries:
        assert entry.token_class is extraction.tokens[entry.token]
        assert entry.respelling.isascii()
        assert entry.note


def test_every_class_is_populated():
    per_class = {entry.token_class for entry in load_token_pronunciations()}
    assert per_class == set(OddballTokenClass)


@pytest.fixture()
def banks_copy(tmp_path):
    target = tmp_path / "banks"
    shutil.copytree(INTAKE_BANKS_DIR, target)
    return target


def _rewrite_table(banks_dir, mutate):
    path = banks_dir / TOKEN_PRONUNCIATIONS_FILENAME
    payload = yaml.safe_load(path.read_text())
    mutate(payload)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def test_orphan_entry_fails_loud(banks_copy):
    _rewrite_table(
        banks_copy,
        lambda payload: payload["entries"].append(
            {
                "token": "NOTINBANKS",
                "respelling": "not-in-banks",
                "class": "all_caps",
                "note": "orphan",
            }
        ),
    )
    with pytest.raises(ValueError, match="orphan entries"):
        load_token_pronunciations(banks_copy)


def test_missing_entry_fails_loud(banks_copy):
    _rewrite_table(banks_copy, lambda payload: payload["entries"].pop())
    with pytest.raises(ValueError, match="missing entries"):
        load_token_pronunciations(banks_copy)


def test_class_drift_fails_loud(banks_copy):
    def flip(payload):
        entry = next(e for e in payload["entries"] if e["class"] == "mixed_case_inner")
        entry["class"] = "all_caps"

    _rewrite_table(banks_copy, flip)
    with pytest.raises(ValueError, match="classifies as"):
        load_token_pronunciations(banks_copy)


def test_duplicate_entry_fails_loud(banks_copy):
    _rewrite_table(
        banks_copy,
        lambda payload: payload["entries"].append(dict(payload["entries"][0])),
    )
    with pytest.raises(ValueError, match="Duplicate oddball-token entry"):
        load_token_pronunciations(banks_copy)


def test_stale_graduate_fails_loud(banks_copy, monkeypatch):
    import tau2.domains.intake.tasks.token_pronunciations as tp

    monkeypatch.setattr(
        tp, "GRADUATED_LETTER_DIGIT", tp.GRADUATED_LETTER_DIGIT | {"Zz9zz"}
    )
    with pytest.raises(ValueError, match="stale after a bank edit"):
        load_token_pronunciations(banks_copy)


# ---------------------------------------------------------------------------
# The review packet
# ---------------------------------------------------------------------------


def test_token_packet_renders_table_and_ignored_list():
    packet = build_token_packet()
    extraction = extract_oddball_tokens()
    assert packet.total == len(load_token_pronunciations())
    assert sum(packet.per_class.values()) == packet.total
    assert packet.ignored == len(extraction.ignored_letter_digit)
    for entry in load_token_pronunciations():
        assert f"| {entry.token} |" in packet.markdown
    for ignored in extraction.ignored_letter_digit[:5]:
        assert ignored in packet.markdown
    for bank in ODDBALL_SOURCE_BANKS:
        assert bank in packet.markdown
    # Deterministic render.
    assert packet.markdown == build_token_packet().markdown
