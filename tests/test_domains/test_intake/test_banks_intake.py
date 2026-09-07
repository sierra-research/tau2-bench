"""Tests for the intake entity banks: loading, fail-loud validation, and the
flat 40 easy / 40 hard bank contract (design doc §4)."""

import shutil

import pytest
import yaml
from pydantic import ValidationError

from tau2.domains.intake.tasks.banks import (
    BANK_CONTRACT_PER_TIER,
    BANK_NAMES,
    INTAKE_BANKS_DIR,
    CodeFamily,
    CoinedRole,
    DateKind,
    Difficulty,
    Gender,
    assert_bank_contract,
    load_banks,
)

EXPECTED_BANKS = {
    "person_names",
    "codes",
    "phones",
    "dates",
    "times",
    "addresses",
    "properties",
    "amounts",
    "emails",
    "medications",
    "insurance_plans",
    "vehicles",
    "rate_plans",
    "coined",
    "shops",
}


def test_load_banks_ok_and_provenance():
    banks = load_banks()
    assert set(banks.shas) == EXPECTED_BANKS
    assert set(BANK_NAMES) == EXPECTED_BANKS
    assert len(BANK_NAMES) == 15
    assert all(len(sha) == 64 for sha in banks.shas.values())


def test_retired_banks_are_gone():
    """providers.yaml and parts.yaml were retired 2026-08-24 (design §11):
    providers duplicated person_names, parts are near-zero-hazard dictionary
    compounds. Exactly the 15 contract banks may exist on disk — plus the
    shared oddball-token pronunciation table (mispronunciation design doc
    sec. 8), which is token-level authored data, not an entity bank."""
    on_disk = {path.stem for path in INTAKE_BANKS_DIR.glob("*.yaml")}
    assert on_disk == EXPECTED_BANKS | {"token_pronunciations"}


def test_checked_in_banks_honor_the_flat_contract():
    """Every bank ships exactly 40 easy + 40 hard values (design doc §4)."""
    banks = load_banks()
    assert_bank_contract(banks)
    for bank_name in BANK_NAMES:
        entries = getattr(banks, bank_name)
        for tier in Difficulty:
            count = sum(1 for e in entries if e.difficulty is tier)
            assert count == BANK_CONTRACT_PER_TIER, (bank_name, tier)


def test_banks_cover_both_difficulties_everywhere():
    """Every drawable pool must offer both difficulties — except VINs, which
    are hard by design (design doc §4)."""
    banks = load_banks()

    def levels(entries):
        return {entry.difficulty for entry in entries}

    both = {Difficulty.EASY, Difficulty.HARD}
    for bank_name in BANK_NAMES:
        if bank_name == "codes":
            continue
        assert levels(getattr(banks, bank_name)) == both, bank_name
    for family in CodeFamily:
        entries = [c for c in banks.codes if c.family is family]
        assert entries, family
        expected = {Difficulty.HARD} if family is CodeFamily.VIN else both
        assert levels(entries) == expected, family
    for role in CoinedRole:
        assert levels([c for c in banks.coined if c.role is role]) == both, role
    # Dates: relative entries are hard and spoken by construction.
    for kind in (DateKind.BIRTH_DATE, DateKind.FUTURE_DATE):
        assert [d for d in banks.dates if d.kind is kind]
    relative = [d for d in banks.dates if d.kind is DateKind.RELATIVE_DATE]
    assert relative
    assert all(d.spoken and d.difficulty is Difficulty.HARD for d in relative)


def test_decoy_carrying_banks_always_carry_decoys():
    """The value grammars whose confusability the decoys encode must plant at
    least one decoy per entry. Reviewed exceptions: hard addresses, dates,
    and the former capture-pool medications — their hazard is the value
    itself, not a planted neighbor (decoys have no runtime role in v4; they
    anchor the fold suite's collision sweep)."""
    banks = load_banks()
    for bank_name in (
        "person_names",
        "codes",
        "phones",
        "times",
        "amounts",
        "insurance_plans",
        "vehicles",
        "rate_plans",
        "coined",
        "shops",
        "properties",
    ):
        assert all(entry.decoys for entry in getattr(banks, bank_name)), bank_name


@pytest.fixture
def banks_copy(tmp_path):
    """A private copy of the checked-in banks that tests may corrupt."""
    target = tmp_path / "banks"
    shutil.copytree(INTAKE_BANKS_DIR, target)
    return target


def corrupt(banks_dir, bank_name, mutate):
    path = banks_dir / f"{bank_name}.yaml"
    raw = yaml.safe_load(path.read_text())
    mutate(raw)
    path.write_text(yaml.safe_dump(raw, allow_unicode=True))


def test_missing_bank_file_fails_loud(banks_copy):
    (banks_copy / "phones.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="Missing entity bank"):
        load_banks(banks_copy)


def test_unknown_field_fails_loud(banks_copy):
    corrupt(
        banks_copy, "person_names", lambda raw: raw["entries"][0].update(nickname="Zed")
    )
    with pytest.raises(ValidationError):
        load_banks(banks_copy)


def test_wrong_entity_name_fails_loud(banks_copy):
    corrupt(banks_copy, "phones", lambda raw: raw.update(entity="telephones"))
    with pytest.raises(ValueError, match="declares entity"):
        load_banks(banks_copy)


def test_duplicate_entry_fails_loud(banks_copy):
    corrupt(banks_copy, "emails", lambda raw: raw["entries"].append(raw["entries"][0]))
    with pytest.raises(ValueError, match="Duplicate entries"):
        load_banks(banks_copy)


def test_decoy_colliding_with_real_entry_fails_loud(banks_copy):
    def mutate(raw):
        raw["entries"][0]["decoys"] = [raw["entries"][1]["value"]]

    corrupt(banks_copy, "phones", mutate)
    with pytest.raises(ValueError, match="colliding with real entries"):
        load_banks(banks_copy)


def test_invalid_vin_check_digit_fails_loud(banks_copy):
    def mutate(raw):
        vins = [e for e in raw["entries"] if e["family"] == "vin"]
        vin = vins[0]["value"]
        vins[0]["value"] = vin[:-1] + ("7" if vin[-1] != "7" else "5")

    corrupt(banks_copy, "codes", mutate)
    with pytest.raises(ValidationError, match="Invalid VIN check digit"):
        load_banks(banks_copy)


def test_invalid_member_id_check_digits_fail_loud(banks_copy):
    def mutate(raw):
        members = [e for e in raw["entries"] if e["family"] == "member_id"]
        members[0]["decoys"] = ["MBR-000000-00"]

    corrupt(banks_copy, "codes", mutate)
    with pytest.raises(ValidationError, match="Invalid member-ID check digits"):
        load_banks(banks_copy)


def test_relative_date_without_spoken_form_fails_loud(banks_copy):
    def mutate(raw):
        relatives = [e for e in raw["entries"] if e["kind"] == "relative_date"]
        relatives[0].pop("spoken")

    corrupt(banks_copy, "dates", mutate)
    with pytest.raises(ValidationError, match="spoken form"):
        load_banks(banks_copy)


# ---------------------------------------------------------------------------
# The flat bank contract fails loud (the generator refuses to draw)
# ---------------------------------------------------------------------------


def test_short_bank_violates_the_contract(banks_copy):
    """Dropping one easy value must fail naming the bank and the shortfall."""

    def mutate(raw):
        easy = next(e for e in raw["entries"] if e["difficulty"] == "easy")
        raw["entries"].remove(easy)

    corrupt(banks_copy, "amounts", mutate)
    with pytest.raises(ValueError, match=r"bank amounts: 39 easy values \(short by 1"):
        assert_bank_contract(load_banks(banks_copy))


def test_overfull_bank_violates_the_contract(banks_copy):
    """An extra hard value must fail too — 'draw 10 of the 40' has to mean
    the same reviewed 40 forever."""

    def mutate(raw):
        raw["entries"].append(
            {"value": "999.99", "difficulty": "hard", "decoys": ["99.99"]}
        )

    corrupt(banks_copy, "amounts", mutate)
    with pytest.raises(ValueError, match=r"bank amounts: 41 hard values \(over by 1"):
        assert_bank_contract(load_banks(banks_copy))


# ---------------------------------------------------------------------------
# Caller gender (name_genders.py convention: every caller-facing given name
# carries an explicit, reviewed gender — fail loud, never guess)
# ---------------------------------------------------------------------------


def test_every_person_name_carries_a_gender():
    banks = load_banks()
    assert all(e.gender in (Gender.MALE, Gender.FEMALE) for e in banks.person_names)
    assert {e.gender for e in banks.person_names} == {Gender.MALE, Gender.FEMALE}


def test_missing_gender_fails_loud(banks_copy):
    corrupt(banks_copy, "person_names", lambda raw: raw["entries"][0].pop("gender"))
    with pytest.raises(ValidationError):
        load_banks(banks_copy)


def test_unknown_gender_fails_loud(banks_copy):
    corrupt(
        banks_copy,
        "person_names",
        lambda raw: raw["entries"][0].update(gender="nonbinary"),
    )
    with pytest.raises(ValidationError):
        load_banks(banks_copy)


def test_inconsistent_given_name_gender_fails_loud(banks_copy):
    """Two entries sharing a given name but disagreeing on gender is an
    authoring slip: the callee's pinned voice would be a coin flip."""

    def mutate(raw):
        first = raw["entries"][0]["value"].split()[0]
        peers = [e for e in raw["entries"] if e["value"].split()[0] == first]
        clone = dict(peers[0])
        clone["value"] = f"{first} Zzyzxson"
        clone["gender"] = "male" if peers[0]["gender"] == "female" else "female"
        clone.pop("alt_name", None)
        clone["decoys"] = []
        # Keep the pronunciation columns aligned with the mutated value so
        # the gender check (not the token-alignment check) is what fires.
        surname_pron = dict(peers[0]["pronunciations"][-1])
        surname_pron["token"] = "Zzyzxson"
        clone["pronunciations"] = [peers[0]["pronunciations"][0], surname_pron]
        raw["entries"].append(clone)

    corrupt(banks_copy, "person_names", mutate)
    with pytest.raises(ValueError, match="one given name must carry one gender"):
        load_banks(banks_copy)


# ---------------------------------------------------------------------------
# ASCII-only banks (owner directive 2026-08-23; identity-ASCII-fold convention)
# ---------------------------------------------------------------------------


def test_banks_are_ascii_only():
    """Every bank string is pure ASCII — values, decoys, respellings — with
    the single reviewed exemption of the ``phonemes`` pronunciation column
    (WikiPron rows are IPA; they are converter INPUT provenance, never spoken,
    captured, or serialized into a task).

    Accented characters at best fold away silently (the NAME fold strips
    them) and at worst romanize as digraphs (ü -> ue), so an ASR's spelling
    folds to a different token than the bank value: a fold false-negative in
    ENGLISH runs. The census/SSA person-name pipeline emits ASCII by
    construction (apostrophes and hyphens are ASCII and stay); this test
    keeps every hand-authored bank at the same standard, superseding the
    earlier digraph-prone-character screen."""

    def strings(node, path=""):
        if isinstance(node, str):
            yield path, node
        elif isinstance(node, dict):
            for key, child in node.items():
                if key == "phonemes":
                    continue  # the one reviewed non-ASCII column (IPA)
                yield from strings(child, f"{path}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                yield from strings(child, f"{path}[{index}]")

    for path in sorted(INTAKE_BANKS_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        offenders = [
            f"{where}: {text!r}" for where, text in strings(raw) if not text.isascii()
        ]
        assert not offenders, f"{path.name} carries non-ASCII: {offenders[:5]}"
