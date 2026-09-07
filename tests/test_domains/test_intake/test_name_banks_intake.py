"""Tests for the census/SSA person-name bank pipeline (`tau2 intake-names`).

The checked-in banks must be exactly what `build` produces from the
checked-in extracts (byte-identical determinism), the extracts must match
their recorded provenance, and the emitted names must honor the decoy /
gender / ASCII contracts the rest of the machine relies on.
"""

import hashlib
import json
import shutil

import pytest

from tau2.domains.intake.tasks.banks import INTAKE_BANKS_DIR, load_banks
from tau2.domains.intake.tasks.name_banks import (
    DEFAULT_BUILD_SEED,
    NAME_SOURCES_DIR,
    NameBanksManifest,
    build_name_banks,
    restore_surname_casing,
)


@pytest.fixture(scope="module")
def manifest() -> NameBanksManifest:
    return NameBanksManifest.model_validate(
        json.loads((NAME_SOURCES_DIR / "name_banks.manifest.json").read_text())
    )


def test_extract_shas_match_provenance():
    """The checked-in extracts are exactly the files provenance.json (and
    the build manifest) hash — a silent edit to an extract fails here."""
    provenance = json.loads((NAME_SOURCES_DIR / "provenance.json").read_text())
    assert set(provenance["extract_shas"]) == {
        "ssa_given_names.csv",
        "census_surnames.csv",
        "pronunciations.csv",
        "wordlist_hits.csv",
    }
    for filename, expected in provenance["extract_shas"].items():
        actual = hashlib.sha256((NAME_SOURCES_DIR / filename).read_bytes()).hexdigest()
        assert actual == expected, filename


def test_provenance_pins_all_five_sources():
    """Extract v2 records URL + retrieval date + sha256 for every raw source,
    the lexicons and wordlist pinned at fixed commits (design doc §2.1)."""
    provenance = json.loads((NAME_SOURCES_DIR / "provenance.json").read_text())
    assert provenance["extractor_version"] == "2"
    for source in ("ssa", "census", "cmudict", "wikipron", "wordlist"):
        raw = provenance[source]
        assert raw["url"].startswith("https://"), source
        assert len(raw["sha256"]) == 64, source
        assert raw["retrieved"], source
    # The pinned commits are part of the URL: a repin is a visible diff.
    assert "/cmusphinx/cmudict/" in provenance["cmudict"]["url"]
    assert "/CUNY-CL/wikipron/" in provenance["wikipron"]["url"]
    assert "/freebsd/freebsd-src/" in provenance["wordlist"]["url"]


def test_manifest_records_pool_stats_above_the_floors(manifest: NameBanksManifest):
    """The lexicon-grounded hard pools stay 10-20x oversubscribed (design
    doc §1); a collapse means a broken filter, not thinner data."""
    from tau2.domains.intake.tasks.name_banks import (
        HARD_GIVEN_POOL_FLOOR,
        TAIL_SURNAME_POOL_FLOOR,
    )

    stats = manifest.pool_stats
    assert stats.hard_given_admissible >= HARD_GIVEN_POOL_FLOOR
    assert stats.tail_surname_admissible >= TAIL_SURNAME_POOL_FLOOR
    assert stats.hard_given_admissible <= stats.hard_given_band
    assert stats.tail_surname_admissible <= stats.tail_surname_band
    assert sum(manifest.pronunciation_sources.values()) > 0


def test_build_reproduces_checked_in_banks_byte_identically(tmp_path):
    """Same extracts + same seed -> the exact checked-in bank bytes."""
    banks_copy = tmp_path / "banks"
    shutil.copytree(INTAKE_BANKS_DIR, banks_copy)
    sources_copy = tmp_path / "name_sources"
    shutil.copytree(NAME_SOURCES_DIR, sources_copy)

    outcome = build_name_banks(
        seed=DEFAULT_BUILD_SEED, banks_dir=banks_copy, sources_dir=sources_copy
    )
    assert outcome.written == []  # byte-identical: nothing rewritten
    assert (banks_copy / "person_names.yaml").read_bytes() == (
        INTAKE_BANKS_DIR / "person_names.yaml"
    ).read_bytes()


def test_restore_surname_casing_rules():
    assert restore_surname_casing("SMITH") == "Smith"
    assert restore_surname_casing("GARCIA") == "Garcia"  # never "García"
    assert restore_surname_casing("OBRIEN") == "O'Brien"
    assert restore_surname_casing("OSULLIVAN") == "O'Sullivan"
    assert restore_surname_casing("MCDONALD") == "McDonald"
    assert restore_surname_casing("MCVERRY") == "McVerry"
    # MAC is not auto-cased: Macias/Machado are not Mac-names.
    assert restore_surname_casing("MACIAS") == "Macias"
    assert restore_surname_casing("DANGELO") == "D'Angelo"
    # Everything the rules emit is ASCII.
    for upper in ("GARCIA", "MUNOZ", "PENA", "OBRIEN", "MCGARVIN"):
        assert restore_surname_casing(upper).isascii()


def test_person_decoys_follow_the_shared_component_convention():
    """decoys[0]: same given name, different surname; decoys[1]: different
    given name, same surname — the contract the locale identity derivation
    (localization.py) encodes as decoy_surname_variant/decoy_given_variant."""
    banks = load_banks()
    for entry in banks.person_names:
        given, surname = entry.value.split(" ", 1)
        assert len(entry.decoys) == 2, entry.value
        d0_given, d0_surname = entry.decoys[0].split(" ", 1)
        assert d0_given == given and d0_surname != surname, entry.value
        d1_given, d1_surname = entry.decoys[1].split(" ", 1)
        assert d1_surname == surname and d1_given != given, entry.value
        if entry.alt_name:
            alt_given, alt_surname = entry.alt_name.split(" ", 1)
            assert alt_given == given and alt_surname != surname, entry.value


def test_constructed_surnames_are_flagged_in_manifest(
    manifest: NameBanksManifest,
):
    """Every hyphenated hard surname is a flagged construction, and every
    flagged construction is a hard-tier value surname."""
    banks = load_banks()
    hyphenated = {
        entry.value.split(" ", 1)[1]
        for entry in banks.person_names
        if "-" in entry.value.split(" ", 1)[1]
    }
    assert hyphenated == set(manifest.constructed_surnames)
    assert all(
        e.difficulty.value == "hard"
        for e in banks.person_names
        if "-" in e.value.split(" ", 1)[1]
    )


def test_generated_name_bank_is_ascii_and_sized():
    banks = load_banks()
    assert len(banks.person_names) == 80  # the 40/40 flat bank contract
    by_difficulty = {"easy": 0, "hard": 0}
    for entry in banks.person_names:
        by_difficulty[entry.difficulty.value] += 1
        strings = [entry.value, *entry.decoys]
        if entry.alt_name:
            strings.append(entry.alt_name)
        assert all(s.isascii() for s in strings), entry.value
    assert by_difficulty == {"easy": 40, "hard": 40}
