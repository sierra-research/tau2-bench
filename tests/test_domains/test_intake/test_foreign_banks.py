"""Tests for the foreign-token bank pipeline (`tau2 intake-names
extract-foreign / build-foreign / foreign-packet`, design doc §8b): the
checked-in properties/vehicles banks must be exactly what the builder
produces from the checked-in extracts, the curated tables must stay aligned
with the extracted candidate set, every covered token must carry a
reviewable column, and the review packet must render every decision (pins,
tiers, shadowed bulk matches, the applicability proposal)."""

import hashlib
import json
import shutil

import pytest

from tau2.domains.intake.tasks.banks import (
    INTAKE_BANKS_DIR,
    PronunciationSource,
    foreign_value_tokens,
    load_banks,
)
from tau2.domains.intake.tasks.foreign_banks import (
    AUTHORED_FOREIGN_PRONUNCIATIONS,
    FOREIGN_BANKS,
    LANGUAGE_PINS,
    TIER3_ENGLISH_TOKENS,
    WIKTIONARY_FOREIGN_PRONUNCIATIONS,
    ForeignBanksManifest,
    build_foreign_banks,
    build_foreign_packet,
    is_foreign_candidate,
    lexicon_key_folds,
)
from tau2.domains.intake.tasks.loanwords import LoanLanguage
from tau2.domains.intake.tasks.name_banks import DEFAULT_BUILD_SEED, NAME_SOURCES_DIR
from tau2.domains.intake.tasks.pronunciation import NO_VARIANT_TOKENS


@pytest.fixture(scope="module")
def manifest() -> ForeignBanksManifest:
    return ForeignBanksManifest.model_validate(
        json.loads((NAME_SOURCES_DIR / "foreign_banks.manifest.json").read_text())
    )


def test_foreign_extract_shas_match_provenance():
    provenance = json.loads((NAME_SOURCES_DIR / "foreign_provenance.json").read_text())
    assert set(provenance["extract_shas"]) == {
        "foreign_tokens.csv",
        "foreign_pronunciations.csv",
    }
    for filename, expected in provenance["extract_shas"].items():
        actual = hashlib.sha256((NAME_SOURCES_DIR / filename).read_bytes()).hexdigest()
        assert actual == expected, filename


def test_foreign_provenance_pins_all_lexicons():
    provenance = json.loads((NAME_SOURCES_DIR / "foreign_provenance.json").read_text())
    assert set(provenance["lexicons"]) == {lang.value for lang in LoanLanguage}
    for source in provenance["lexicons"].values():
        assert source["url"].startswith(
            "https://raw.githubusercontent.com/CUNY-CL/wikipron/"
        )
        assert source["sha256"]
        assert source["retrieved"]


def test_build_reproduces_checked_in_banks_byte_identically(tmp_path):
    """Same extracts + same seed -> the exact checked-in bank bytes."""
    banks_copy = tmp_path / "banks"
    shutil.copytree(INTAKE_BANKS_DIR, banks_copy)
    sources_copy = tmp_path / "name_sources"
    shutil.copytree(NAME_SOURCES_DIR, sources_copy)

    outcome = build_foreign_banks(
        seed=DEFAULT_BUILD_SEED, banks_dir=banks_copy, sources_dir=sources_copy
    )
    assert outcome.written == []  # byte-identical: nothing rewritten
    for bank in FOREIGN_BANKS:
        assert (banks_copy / f"{bank}.yaml").read_bytes() == (
            INTAKE_BANKS_DIR / f"{bank}.yaml"
        ).read_bytes()


def test_candidate_rule():
    english = {"hotel", "house", "grand", "nord"}
    assert is_foreign_candidate("Alpenblick", english)
    assert not is_foreign_candidate("Hotel", english)  # English-covered
    assert not is_foreign_candidate("Nord", english)
    assert not is_foreign_candidate("GLB", english)  # all-caps: oddball table
    assert not is_foreign_candidate("xDrive", english)  # mixed-case: oddball
    assert not is_foreign_candidate("gCaorach", english)  # eclipsis: oddball
    assert not is_foreign_candidate("Q3", english)  # digit-bearing
    assert not is_foreign_candidate("B&B", english)
    assert not is_foreign_candidate("y", english)  # too short


def test_lexicon_key_folding():
    assert "soestre" in lexicon_key_folds("søstre")  # Germanic oe convention
    assert "krummen" in lexicon_key_folds("krümmen")  # plain strip
    assert "kruemmen" in lexicon_key_folds("krümmen")  # Germanic ue
    assert "zlamany" in lexicon_key_folds("złamany")  # ł -> l
    assert "weisses" in lexicon_key_folds("weißes")  # ß -> ss
    assert lexicon_key_folds("plain") == {"plain"}


def test_curated_tables_cover_the_right_banks():
    for table in (
        LANGUAGE_PINS,
        TIER3_ENGLISH_TOKENS,
        WIKTIONARY_FOREIGN_PRONUNCIATIONS,
        AUTHORED_FOREIGN_PRONUNCIATIONS,
    ):
        assert set(table) == set(FOREIGN_BANKS)


def test_every_covered_token_column_is_complete(manifest: ForeignBanksManifest):
    """Every column-bearing token has phonemes + respelling, a citation
    exactly when curated, a mispronounced variant unless deliberately
    allow-listed, and a language matching its entry pin for bulk rows."""
    banks = load_banks()
    for bank in FOREIGN_BANKS:
        pins = LANGUAGE_PINS[bank]
        no_variant = NO_VARIANT_TOKENS[bank]
        seen_no_variant = set()
        for entry in getattr(banks, bank):
            pin = pins.get(entry.value)
            assert (entry.language is not None) == (pin is not None)
            if pin is not None:
                assert entry.language is pin
            for column in entry.pronunciations:
                assert column.phonemes and column.respelling
                needs_citation = column.source in (
                    PronunciationSource.WIKTIONARY,
                    PronunciationSource.AUTHORED,
                )
                assert (column.citation is not None) == needs_citation
                if column.source is PronunciationSource.WIKTIONARY:
                    assert column.citation.startswith("https://en.wiktionary.org/")
                if column.source is PronunciationSource.WIKIPRON:
                    assert column.language is pin
                if column.mispronounced is None:
                    assert column.token in no_variant
                    assert column.operator is None
                    seen_no_variant.add(column.token)
                else:
                    assert column.operator is not None
                    assert column.mispronounced != column.token.casefold()
        # Both directions: every allow-listed token exists and drew nothing.
        assert seen_no_variant == set(no_variant)


def test_tier3_tokens_carry_no_column(manifest: ForeignBanksManifest):
    banks = load_banks()
    for bank in FOREIGN_BANKS:
        columned = {
            column.token
            for entry in getattr(banks, bank)
            for column in entry.pronunciations
        }
        assert columned.isdisjoint(TIER3_ENGLISH_TOKENS[bank])
        # Tier-3 tokens actually appear in the bank (stale-list guard).
        bank_tokens = {
            token
            for entry in getattr(banks, bank)
            for token in foreign_value_tokens(entry.value)
        }
        assert TIER3_ENGLISH_TOKENS[bank] <= bank_tokens


def test_repeated_tokens_share_one_column(manifest: ForeignBanksManifest):
    """Tokens appearing in several entries (Auberge, Pousada) carry the
    SAME drawn column everywhere — one draw per (bank, token)."""
    banks = load_banks()
    for bank in FOREIGN_BANKS:
        by_token: dict[str, tuple] = {}
        for entry in getattr(banks, bank):
            for column in entry.pronunciations:
                key = (
                    column.source,
                    column.phonemes,
                    column.respelling,
                    column.mispronounced,
                    column.operator,
                    column.language,
                )
                assert by_token.setdefault(column.token, key) == key


def test_manifest_matches_banks(manifest: ForeignBanksManifest):
    assert manifest.seed == DEFAULT_BUILD_SEED
    banks = load_banks()
    for bank in FOREIGN_BANKS:
        manifest_tokens = [column.token for column in manifest.columns[bank]]
        assert len(manifest_tokens) == len(set(manifest_tokens))
        bank_tokens = {
            column.token
            for entry in getattr(banks, bank)
            for column in entry.pronunciations
        }
        assert set(manifest_tokens) == bank_tokens
        assert manifest.pins[bank] == {
            value: language.value for value, language in LANGUAGE_PINS[bank].items()
        }
        assert manifest.no_variant_tokens[bank] == sorted(NO_VARIANT_TOKENS[bank])


def test_shadowed_bulk_matches_are_recorded(manifest: ForeignBanksManifest):
    """Curated rows that override a pinned-lexicon fold match (the false
    friends: Krummen/krümmen, Zaguan, Animas, Velho) are flagged for the
    packet — precedence is a reviewed decision, not a silent one."""
    shadowed = {
        column.token
        for columns in manifest.columns.values()
        for column in columns
        if column.shadowed_bulk_language is not None
    }
    assert {"Krummen", "Zaguan", "Animas", "Velho"} <= shadowed
    # Only curated rows can shadow.
    for columns in manifest.columns.values():
        for column in columns:
            if column.shadowed_bulk_language is not None:
                assert column.source in (
                    PronunciationSource.WIKTIONARY,
                    PronunciationSource.AUTHORED,
                )


def test_packet_renders_every_decision(manifest: ForeignBanksManifest):
    packet = build_foreign_packet()
    assert packet.tokens == sum(len(cols) for cols in manifest.columns.values())
    assert packet.per_source["wiktionary"] == 2
    text = packet.markdown
    assert "## Operator applicability (owner review requested)" in text
    for bank in FOREIGN_BANKS:
        for value in LANGUAGE_PINS[bank]:
            assert value in text
        for token in TIER3_ENGLISH_TOKENS[bank]:
            assert token in text
    assert "shadows a" in text  # shadowed bulk matches are visible
    assert "gCaorach" in text  # oddball delegation is visible
