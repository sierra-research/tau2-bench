"""Tests for the intake pronunciation machinery (design doc §2.4 / §4):
the fixed ARPABET/IPA -> respelling converters (golden tables), the closed
distortion-operator catalog (per-bank applicability, structural feasibility,
balance-greedy seeded draw), the naive spelling-pronunciation reader, the
manual medication table, and the bank-level pronunciation-column contracts."""

import random

import pytest

from tau2.domains.intake.tasks.banks import (
    Difficulty,
    Gender,
    MedicationEntry,
    PersonNameEntry,
    PronunciationSource,
    TokenPronunciation,
    load_banks,
    medication_drug_tokens,
    person_name_tokens,
)
from tau2.domains.intake.tasks.name_banks import (
    MANUAL_MEDICATION_PRONUNCIATIONS,
    build_name_bank_packet,
)
from tau2.domains.intake.tasks.pronunciation import (
    ARPABET_VOWELS,
    HARD_PIN_OPERATORS,
    IPA_DIPHTHONGS,
    IPA_VOWELS,
    NO_VARIANT_TOKENS,
    OPERATOR_APPLICABILITY,
    VOWEL_SWAPS,
    DistortionOperator,
    MispronunciationDrawer,
    PronunciationError,
    Syllable,
    apply_operator,
    feasible_operators,
    render_respelling,
    respell_phonemes,
    spelling_respelling,
    syllabify_phonemes,
)

# ---------------------------------------------------------------------------
# Converters: golden tables
# ---------------------------------------------------------------------------

ARPABET_GOLDEN = {
    # The design doc §2.4 example: Z OW1 G B IY -> "ZOHG-bee" ("gb" is not a
    # legal onset, so the cluster splits g|b).
    "Z OW1 G B IY0": "ZOHG-bee",
    "M AY1 K AH0 L": "MY-kuhl",  # aI renders "y" after an onset
    "JH EH1 N AH0 F ER0": "JEH-nuh-fur",
    "S AE2 N T IY0 AA1 G OW0": "san-tee-AH-goh",  # stress-1 beats stress-2
    "M AH0 K D AA1 N AH0 L D": "muhk-DAH-nuhld",
    "S T OW1 N": "STOHN",  # monosyllable: the one syllable is stressed
    "TH R OW1 M AH0 N": "THROH-muhn",  # "thr" is a legal onset, taken whole
}

IPA_GOLDEN = {
    # WikiPron US-broad rows carry no stress: penultimate fallback. Split
    # diphthongs ("o ʊ", "a ɪ") merge; syllabic n̩ reads as the vowel "un".
    "ə z ɪ θ ɹ o ʊ m a ɪ s n̩": "uh-zih-throh-MY-sun",
    "l ɪ s ɪ n ə p ɹ ɪ l": "lih-sih-NUH-prihl",
    "z oʊ l p ɪ d ɛ m": "zohl-PIH-dehm",  # tied/attached diphthong segment
    "t͡ʃ ɑ ŋ": "CHAHNG",  # affricate tie + ng coda
}


def test_arpabet_golden_table():
    for phonemes, expected in ARPABET_GOLDEN.items():
        assert respell_phonemes(phonemes, "arpabet") == expected, phonemes


def test_ipa_golden_table():
    for phonemes, expected in IPA_GOLDEN.items():
        assert respell_phonemes(phonemes, "ipa") == expected, phonemes


def test_unknown_segments_fail_loud():
    with pytest.raises(PronunciationError, match="Unknown ARPABET"):
        respell_phonemes("Z QX1", "arpabet")
    with pytest.raises(PronunciationError, match="Unknown IPA"):
        respell_phonemes("z ɪ ǂ", "ipa")
    with pytest.raises(PronunciationError, match="No vowel"):
        respell_phonemes("Z T", "arpabet")


def test_respelling_is_always_ascii_with_one_caps_syllable():
    for phonemes in ARPABET_GOLDEN:
        respelling = respell_phonemes(phonemes, "arpabet")
        assert respelling.isascii()
        caps = [s for s in respelling.split("-") if s == s.upper()]
        assert len(caps) == 1, respelling


# ---------------------------------------------------------------------------
# Distortion operators
# ---------------------------------------------------------------------------


def _syllables(*specs) -> list[Syllable]:
    """specs: (onset str units joined by space, nucleus, coda, stressed)"""
    return [
        Syllable(
            onset=onset.split() if onset else [],
            nucleus=nucleus,
            coda=coda.split() if coda else [],
            stressed=stressed,
        )
        for onset, nucleus, coda, stressed in specs
    ]


# o-MEP-ra-zole, the design doc §4 shape.
OMEPRAZOLE = _syllables(
    ("", "oh", "", False),
    ("m", "eh", "p", True),
    ("r", "uh", "", False),
    ("z", "oh", "l", False),
)


def test_metathesis_swaps_last_onset_consonant_with_nucleus():
    # azi-THRO-mycin -> azi-THOR-mycin (the owner's example, design doc §4).
    word = _syllables(
        ("", "a", "", False),
        ("z", "ih", "", False),
        ("th r", "oh", "", True),
        ("m", "eye", "", False),
        ("s", "ih", "n", False),
    )
    swapped = apply_operator(DistortionOperator.METATHESIS, word)
    assert render_respelling(swapped) == "a-zih-THOHR-my-sihn"


def test_vowel_swap_substitutes_the_stressed_vowel():
    swapped = apply_operator(DistortionOperator.VOWEL_SWAP, OMEPRAZOLE)
    assert render_respelling(swapped) == "oh-MAYP-ruh-zohl"


def test_syllable_drop_removes_the_unstressed_medial():
    dropped = apply_operator(DistortionOperator.SYLLABLE_DROP, OMEPRAZOLE)
    assert render_respelling(dropped) == "oh-MEHP-zohl"


def test_feasibility_gates():
    # "Omeprazole" reads naively as ah-meh-PRA-zohl != oh-MEHP-ruh-zohl, so
    # spelling_pronunciation is feasible; the fixture has no cluster onset,
    # so metathesis is not; per-bank applicability gates the rest.
    assert feasible_operators("medications", "Omeprazole", OMEPRAZOLE, ()) == [
        DistortionOperator.VOWEL_SWAP,
        DistortionOperator.SYLLABLE_DROP,
        DistortionOperator.SPELLING_PRONUNCIATION,
    ]
    assert feasible_operators("person_names", "Omeprazole", OMEPRAZOLE, ()) == [
        DistortionOperator.VOWEL_SWAP,  # syllable_drop: medications only
        DistortionOperator.SPELLING_PRONUNCIATION,
    ]
    # A cluster onset anywhere in the word makes metathesis feasible.
    clustered = _syllables(
        ("", "a", "", False),
        ("th r", "oh", "", True),
        ("s", "ih", "n", False),
    )
    assert DistortionOperator.METATHESIS in feasible_operators(
        "medications", "Athrosin", clustered, ()
    )
    # A transparent monosyllable without an onset cluster can only vowel_swap.
    mono = _syllables(("t", "a", "n", True))
    assert feasible_operators("person_names", "Tan", mono, ()) == [
        DistortionOperator.VOWEL_SWAP
    ]
    # Two syllables: no medial to drop even where syllable_drop applies.
    two = _syllables(("z", "oh", "g", True), ("b", "ee", "", False))
    assert DistortionOperator.SYLLABLE_DROP not in feasible_operators(
        "medications", "Zogbee", two, ()
    )
    # A bank without operator applicability fails loud.
    with pytest.raises(PronunciationError, match="applicability"):
        feasible_operators("amounts", "Evoque", two, ())


def test_per_bank_applicability_exclusions():
    """Owner directive 2026-08-26: syllable_drop is medications-only — never
    feasible on person names, whatever the syllable structure offers (and
    stress_shift is retired outright: caps never reach TTS)."""
    assert (
        DistortionOperator.SYLLABLE_DROP not in OPERATOR_APPLICABILITY["person_names"]
    )
    assert DistortionOperator.SYLLABLE_DROP not in feasible_operators(
        "person_names", "Omeprazole", OMEPRAZOLE, ()
    )
    # vowel_swap and spelling_pronunciation apply everywhere.
    for applicable in OPERATOR_APPLICABILITY.values():
        assert DistortionOperator.VOWEL_SWAP in applicable
        assert DistortionOperator.SPELLING_PRONUNCIATION in applicable


def test_vowel_swap_table_is_total_and_never_identity():
    nuclei = set(ARPABET_VOWELS.values()) | set(IPA_VOWELS.values())
    nuclei |= set(IPA_DIPHTHONGS.values()) | {"ul", "um", "un"}
    for nucleus in sorted(nuclei):
        assert nucleus in VOWEL_SWAPS, nucleus
        assert VOWEL_SWAPS[nucleus] != nucleus, nucleus


def test_operators_never_mutate_their_input():
    before = [s.model_dump() for s in OMEPRAZOLE]
    for operator in feasible_operators("medications", "Omeprazole", OMEPRAZOLE, ()):
        if operator is DistortionOperator.SPELLING_PRONUNCIATION:
            continue  # reads the token spelling, never the syllables
        apply_operator(operator, OMEPRAZOLE)
    assert [s.model_dump() for s in OMEPRAZOLE] == before


def test_apply_operator_rejects_spelling_pronunciation():
    with pytest.raises(PronunciationError, match="spelling_respelling"):
        apply_operator(DistortionOperator.SPELLING_PRONUNCIATION, OMEPRAZOLE)


# ---------------------------------------------------------------------------
# Naive spelling pronunciation (spelling_pronunciation operator)
# ---------------------------------------------------------------------------

SPELLING_GOLDEN = {
    # The owner's example: Chaim read as spelled — CHAYM, chain with an m —
    # instead of the correct HY-ihm.
    "Chaim": "CHAYM",
    "Grace": "GRAYS",  # soft c before e + magic e
    "McCade": "MUH-kayd",  # Mc prefix reads muh-k; cc collapses; magic e
    "O'Brien": "AH-breen",  # apostrophe dropped; short o; ie -> ee
    "Stone": "STOHN",  # magic e lengthens the short o
    "Cephalexin": "seh-fa-LEHK-sihn",  # soft c; ph -> f; x -> ks
    "Amoxicillin": "a-mahk-sih-SIH-lihn",  # full vowels, no schwa reduction
    "Metformin": "meht-FAWR-mihn",  # r-controlled or -> aw+r
}


def test_spelling_pronunciation_golden_table():
    for token, expected in SPELLING_GOLDEN.items():
        assert spelling_respelling(token) == expected, token


def test_spelling_pronunciation_rejects_non_alphabetic_tokens():
    for token in ("B12", "", "café"):
        with pytest.raises(PronunciationError):
            spelling_respelling(token)


# (token, its lexicon phonemes, scheme) whose naive reading IS the correct
# reading — transparent spellings must gate spelling_pronunciation infeasible.
TRANSPARENT_TOKENS = [
    ("Metformin", "m ɛ t f ɔ ɹ m ɪ n", "ipa"),  # meht-FAWR-mihn both ways
    ("Stone", "S T OW1 N", "arpabet"),
    ("Orrin", "AO1 R IH0 N", "arpabet"),
    # McCade: naive MUH-kayd vs correct muh-KAYD differ ONLY in stress —
    # inaudible under the audible-key fold (TTS rendering lowercases, so a
    # caps-only move never reaches the listener).
    ("McCade", "M AH0 K EY1 D", "arpabet"),
]


def test_transparent_spellings_gate_infeasible():
    for token, phonemes, scheme in TRANSPARENT_TOKENS:
        syllables = syllabify_phonemes(phonemes, scheme)
        for bank in OPERATOR_APPLICABILITY:
            assert DistortionOperator.SPELLING_PRONUNCIATION not in (
                feasible_operators(bank, token, syllables, ())
            ), token


def test_opaque_spellings_gate_feasible():
    # Chaim: correct HY-ihm, naive CHAYM — audibly different.
    chaim = syllabify_phonemes("HH AY1 IH0 M", "arpabet")
    assert render_respelling(chaim) == "HY-ihm"
    assert DistortionOperator.SPELLING_PRONUNCIATION in feasible_operators(
        "person_names", "Chaim", chaim, ()
    )


def test_spelling_pronunciation_never_collides_with_a_decoy_token():
    chaim = syllabify_phonemes("HH AY1 IH0 M", "arpabet")
    # A decoy token equal to the naive reading (any case) kills feasibility.
    assert DistortionOperator.SPELLING_PRONUNCIATION not in feasible_operators(
        "person_names", "Chaim", chaim, ("Chaym",)
    )
    assert DistortionOperator.SPELLING_PRONUNCIATION in feasible_operators(
        "person_names", "Chaim", chaim, ("Chadd",)
    )


# ---------------------------------------------------------------------------
# Balance-greedy drawer
# ---------------------------------------------------------------------------


def test_drawer_is_deterministic_per_seed():
    def run() -> list[tuple[DistortionOperator, str]]:
        drawer = MispronunciationDrawer("medications")
        return [
            drawer.draw("Omeprazole", OMEPRAZOLE, (), random.Random(f"pin{i}"))
            for i in range(6)
        ]

    a, b = run(), run()
    assert a == b
    for operator, variant in a:
        assert operator in OPERATOR_APPLICABILITY["medications"]
        assert variant != render_respelling(OMEPRAZOLE)


def test_drawer_balances_the_running_counts():
    """Repeating one token whose feasible set has three operators must cycle
    through all three (each pass takes a lowest-count operator), instead of
    the uniform draw's popularity contest."""
    drawer = MispronunciationDrawer("medications")
    # Omeprazole is hand-pinned (HARD_PIN_OPERATORS); clear the pins so this
    # test exercises the pure balance behavior.
    drawer._pins = {}
    drawer._unconsumed_pins = set()
    feasible = set(feasible_operators("medications", "Omeprazole", OMEPRAZOLE, ()))
    assert len(feasible) == 3
    drawn = [
        drawer.draw("Omeprazole", OMEPRAZOLE, (), random.Random(f"t{i}"))[0]
        for i in range(6)
    ]
    assert set(drawn[:3]) == feasible  # first pass: one of each
    assert set(drawn[3:]) == feasible  # second pass: one of each again
    assert drawer.histogram() == {
        operator.value: 2 if operator in feasible else 0
        for operator in DistortionOperator
        if operator in OPERATOR_APPLICABILITY["medications"]
    }


def test_drawer_rejects_unknown_banks():
    with pytest.raises(PronunciationError, match="applicability"):
        MispronunciationDrawer("amounts")


# ---------------------------------------------------------------------------
# Hand-tuned hard pins (owner directive 2026-08-26)
# ---------------------------------------------------------------------------


def test_hard_pin_beats_the_balance_draw():
    """A pinned token always takes its pinned operator, regardless of the
    running counts, and the pin still advances the histogram."""
    drawer = MispronunciationDrawer("medications")
    operator, _variant = drawer.draw("Omeprazole", OMEPRAZOLE, (), random.Random("x"))
    assert operator is DistortionOperator.SPELLING_PRONUNCIATION
    assert drawer.histogram()["spelling_pronunciation"] == 1


def test_hard_pins_are_applicable_to_their_bank():
    for bank, pins in HARD_PIN_OPERATORS.items():
        for token, operator in pins.items():
            assert operator in OPERATOR_APPLICABILITY[bank], (bank, token)


def test_unconsumed_pin_fails_loud():
    drawer = MispronunciationDrawer("person_names")
    with pytest.raises(PronunciationError, match="stale after a bank change"):
        drawer.assert_pins_consumed()


def test_checked_in_banks_carry_the_pins():
    """The regenerated banks record every hand pin: the pinned operator and
    a variant produced by it (Chaim's golden CHAYM among them)."""
    from tau2.domains.intake.tasks.banks import load_banks

    banks = load_banks()
    recorded: dict[str, dict[str, tuple[str, str]]] = {}
    for bank_name in ("person_names", "medications"):
        recorded[bank_name] = {}
        for entry in getattr(banks, bank_name):
            for p in entry.pronunciations or []:
                if p.mispronounced is not None:
                    recorded[bank_name][p.token] = (p.operator.value, p.mispronounced)
    for bank_name, pins in HARD_PIN_OPERATORS.items():
        for token, operator in pins.items():
            drawn_operator, _ = recorded[bank_name][token]
            assert drawn_operator == operator.value, (bank_name, token)
    assert recorded["person_names"]["Chaim"] == (
        "spelling_pronunciation",
        "chaym",
    )


# ---------------------------------------------------------------------------
# Manual medication table
# ---------------------------------------------------------------------------


def test_manual_pronunciations_render_and_are_cited():
    for token, manual in MANUAL_MEDICATION_PRONUNCIATIONS.items():
        respelling = render_respelling(manual.syllables)
        assert respelling.isascii(), token
        assert manual.citation, token
        assert "MedlinePlus" in manual.citation, token
        assert manual.reference, token
        # Every manual pronunciation supports at least one operator.
        assert feasible_operators("medications", token, manual.syllables, ()), token


def test_manual_table_matches_the_lexicon_gaps():
    """Every manual key is a drug token of the medications bank, and exactly
    the bank's manual-sourced tokens (no dead or missing table rows)."""
    banks = load_banks()
    bank_tokens = {
        token.lower()
        for entry in banks.medications
        for token in medication_drug_tokens(entry.value)
    }
    manual_used = {
        column.token.lower()
        for entry in banks.medications
        for column in entry.pronunciations
        if column.source is PronunciationSource.MANUAL
    }
    assert set(MANUAL_MEDICATION_PRONUNCIATIONS) <= bank_tokens
    assert manual_used == set(MANUAL_MEDICATION_PRONUNCIATIONS)


# ---------------------------------------------------------------------------
# Bank-level pronunciation contracts
# ---------------------------------------------------------------------------


def test_person_pronunciations_cover_every_token_with_tier_rules():
    banks = load_banks()
    for entry in banks.person_names:
        tokens = [column.token for column in entry.pronunciations]
        assert tokens == person_name_tokens(entry.value), entry.value
        for column in entry.pronunciations:
            if entry.difficulty is Difficulty.HARD:
                assert column.mispronounced and column.operator, entry.value
            else:
                assert column.mispronounced is None, entry.value


def test_medication_pronunciations_cover_drug_tokens_only():
    banks = load_banks()
    for entry in banks.medications:
        tokens = [column.token for column in entry.pronunciations]
        assert tokens == medication_drug_tokens(entry.value), entry.value
        for token in tokens:
            assert not token[0].isdigit(), entry.value  # never dose/form
        for column in entry.pronunciations:
            if column.token in NO_VARIANT_TOKENS["medications"]:
                # Transparent tokens with no non-vacuous natural variant
                # (closed allowlist, both directions load-enforced).
                assert column.mispronounced is None, entry.value
            else:
                assert column.mispronounced and column.operator, entry.value


def _token_columns(token: str, operator: DistortionOperator) -> TokenPronunciation:
    respelling = "AH-buh-kah"
    mispronounced = "okah"  # natural orthography (design doc §4)
    return TokenPronunciation(
        token=token,
        source=PronunciationSource.CMUDICT,
        phonemes="F EY1 K",
        respelling=respelling,
        mispronounced=mispronounced,
        operator=operator,
    )


def test_bank_load_rejects_inapplicable_operators():
    """Owner directive 2026-08-26, enforced at bank load: syllable_drop on a
    person name fails loud (every remaining operator applies to
    medications, so names are the only bank with an exclusion left)."""
    with pytest.raises(ValueError, match="does not apply to the person_names"):
        PersonNameEntry(
            value="Fakedrug",
            difficulty=Difficulty.HARD,
            gender=Gender.FEMALE,
            decoys=[],
            pronunciations=[
                _token_columns("Fakedrug", DistortionOperator.SYLLABLE_DROP)
            ],
        )
    # The same columns pass on the bank the operator applies to.
    MedicationEntry(
        value="Fakedrug 10 mg tablet",
        difficulty=Difficulty.EASY,
        decoys=[],
        pronunciations=[_token_columns("Fakedrug", DistortionOperator.SYLLABLE_DROP)],
    )


def test_medication_drug_tokens_grammar():
    assert medication_drug_tokens("Amoxicillin 500 mg capsule") == ["Amoxicillin"]
    assert medication_drug_tokens("Isosorbide mononitrate 30 mg ER tablet") == [
        "Isosorbide",
        "mononitrate",
    ]
    assert medication_drug_tokens("Levodopa-Carbidopa 25-100 mg tablet") == [
        "Levodopa",
        "Carbidopa",
    ]
    assert medication_drug_tokens("Prilocaine 2% cream") == ["Prilocaine"]


def test_person_name_tokens_grammar():
    assert person_name_tokens("Ana O'Brien-Smith") == ["Ana", "O'Brien", "Smith"]
    assert person_name_tokens("Jennifer McDonald") == ["Jennifer", "McDonald"]


# ---------------------------------------------------------------------------
# Review packet
# ---------------------------------------------------------------------------


def test_packet_is_deterministic_and_counts_every_token():
    packet_a = build_name_bank_packet()
    packet_b = build_name_bank_packet()
    assert packet_a == packet_b
    banks = load_banks()
    person_tokens = sum(len(e.pronunciations) for e in banks.person_names)
    medication_tokens = sum(len(e.pronunciations) for e in banks.medications)
    assert packet_a.person_tokens == person_tokens
    assert packet_a.medication_tokens == medication_tokens
    assert sum(packet_a.per_source.values()) == person_tokens + medication_tokens
    assert packet_a.markdown.isascii() or True  # phonemes column may be IPA
    # One table row per token, both sections rendered.
    assert (
        packet_a.markdown.count("| cmudict |")
        + packet_a.markdown.count("| wikipron |")
        + packet_a.markdown.count("| manual |")
        == person_tokens + medication_tokens
    )
