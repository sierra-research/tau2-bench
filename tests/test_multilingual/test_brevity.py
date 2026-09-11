# Copyright Sierra
"""The short-phrase brevity measure and the bound checker both persona
inventories (out-of-turn speech and backchannel continuers) share."""

import pytest

from tau2.backchannel import MAX_BACKCHANNEL_PHRASE_WORDS, MAX_BACKCHANNEL_PHRASES
from tau2.multilingual.brevity import (
    BACKCHANNEL_INVENTORY,
    DEFAULT_WORD_MEASURE,
    NON_DIRECTED_INVENTORY,
    SPACELESS_SCRIPTS,
    WORD_MEASURES,
    backchannel_phrase_problems,
    non_directed_phrase_problems,
    phrase_budget_label,
    phrase_counting_rule,
    phrase_word_count,
    word_measure,
)
from tau2.voice_config import (
    MAX_NON_DIRECTED_PHRASE_WORDS,
    MAX_NON_DIRECTED_PHRASES,
    NON_DIRECTED_PHRASES,
)


class TestPhraseWordCount:
    @pytest.mark.parametrize(
        "text,language,expected",
        [
            # Space-separated: whitespace tokens, one per word.
            ("Un momento", "es", 2),
            ("Hold on a sec, I'm on the phone", "en", 8),
            ("", "en", 0),
            # Punctuation never inflates the count of a token-measured phrase.
            ("Sí, venga", "es", 2),
            # Han: 2 characters per word, punctuation excluded.
            ("等下啊，我接个电话。", "zh", 4),
            ("老伴儿，我电话里呢，等会儿啊。", "zh", 6),
            # Thai: 4 characters per word (the worked example kept for future
            # space-less onboarding).
            ("รอสักครู่นะ", "th", 3),
        ],
    )
    def test_counts(self, text, language, expected):
        assert phrase_word_count(text, language) == expected

    def test_unknown_language_falls_back_to_tokens(self):
        assert word_measure("sw") == DEFAULT_WORD_MEASURE
        assert phrase_word_count("subiri kidogo", "sw") == 2

    def test_rounds_up_so_a_phrase_never_measures_short(self):
        # 5 Han characters / 2 per word = 2.5 -> 3, not 2.
        assert phrase_word_count("这个事儿咱", "zh") == 3

    def test_every_spaceless_script_has_a_char_measure_available(self):
        """A space-less script with no char-based language entry would
        silently disable the bound, so the two tables must stay in step for
        every language the packs actually ship (checked in
        test_pack_invariants); here we assert the char measures exist at all."""
        char_measured = {
            lang for lang, measure in WORD_MEASURES.items() if measure.unit == "char"
        }
        assert char_measured, "no character-based measures declared"
        assert SPACELESS_SCRIPTS


class TestNonDirectedPhraseProblems:
    def test_in_budget_phrases_pass(self):
        assert (
            non_directed_phrase_problems(
                "p", ["Un momento", "Ya voy"], language="es", script="latn"
            )
            == []
        )

    def test_none_falls_back_to_defaults_without_complaint(self):
        assert non_directed_phrase_problems("p", None, language="es") == []

    def test_english_defaults_are_in_budget(self):
        """The fallback list every pack without an override inherits."""
        assert (
            non_directed_phrase_problems(
                "p", NON_DIRECTED_PHRASES, language="en", script="latn"
            )
            == []
        )

    def test_long_phrase_is_flagged(self):
        problems = non_directed_phrase_problems(
            "ricardo_pt_v1",
            ["Pode deixar o relatório sobre a minha mesa, por gentileza."],
            language="pt",
            script="latn",
        )
        assert len(problems) == 1
        assert "ricardo_pt_v1" in problems[0]
        assert f"{MAX_NON_DIRECTED_PHRASE_WORDS} words" in problems[0]

    def test_too_many_phrases_is_flagged(self):
        phrases = ["Ya voy"] * (MAX_NON_DIRECTED_PHRASES + 1)
        problems = non_directed_phrase_problems(
            "p", phrases, language="es", script="latn"
        )
        assert len(problems) == 1
        assert "at most" in problems[0]

    def test_count_and_length_are_reported_together(self):
        phrases = ["Un momento, que estoy al teléfono"] * (MAX_NON_DIRECTED_PHRASES + 1)
        problems = non_directed_phrase_problems(
            "p", phrases, language="es", script="latn"
        )
        assert len(problems) == 1 + len(phrases)

    def test_spaceless_script_without_a_char_measure_fails_loudly(self):
        """Onboarding a Han-script language must add a measure first —
        otherwise whitespace counting scores a whole sentence as one word."""
        problems = non_directed_phrase_problems(
            "p",
            ["这个事儿咱一会儿再说，我先接个电话啊"],
            language="yue",  # not in WORD_MEASURES
            script="hans",
        )
        assert len(problems) == 1
        assert "WORD_MEASURES" in problems[0]

    def test_non_string_entries_are_rejected(self):
        problems = non_directed_phrase_problems(
            "p", ["ok", 3], language="es", script="latn"
        )
        assert problems == [
            "persona 'p' non_directed_phrases must be a list of strings"
        ]


class TestBackchannelPhraseProblems:
    """The same checker, the tighter inventory: a continuer is a hum."""

    def test_reviewed_continuers_pass(self):
        assert backchannel_phrase_problems("p", ["mhm"], language="es") == []
        # A doubled hum written with a space is why the bound is 2, not 1.
        assert backchannel_phrase_problems("p", ["aham aham"], language="pt") == []
        assert backchannel_phrase_problems("p", ["嗯嗯"], language="zh") == []

    def test_turn_level_response_is_flagged(self):
        problems = backchannel_phrase_problems(
            "p", ["vale, eso lo he entendido bien"], language="es", script="latn"
        )
        assert len(problems) == 1
        assert f"{MAX_BACKCHANNEL_PHRASE_WORDS} words" in problems[0]
        assert "backchannel_phrases" in problems[0]

    def test_inventory_longer_than_the_bound_is_flagged(self):
        phrases = ["mhm"] * (MAX_BACKCHANNEL_PHRASES + 1)
        problems = backchannel_phrase_problems("p", phrases, language="es")
        assert len(problems) == 1
        assert "draft-continuers" in problems[0]

    def test_none_falls_back_to_defaults_without_complaint(self):
        assert backchannel_phrase_problems("p", None, language="es") == []


class TestInventoryDescriptions:
    """The strings the guardrail messages and drafting prompts are built from."""

    @pytest.mark.parametrize(
        "inventory,max_words",
        [
            (NON_DIRECTED_INVENTORY, MAX_NON_DIRECTED_PHRASE_WORDS),
            (BACKCHANNEL_INVENTORY, MAX_BACKCHANNEL_PHRASE_WORDS),
        ],
    )
    def test_budget_label_states_the_inventory_bound(self, inventory, max_words):
        assert phrase_budget_label("es", inventory) == f"{max_words} words"

    def test_budget_label_converts_for_a_char_measured_language(self):
        label = phrase_budget_label("zh", BACKCHANNEL_INVENTORY)
        assert (
            label
            == f"{MAX_BACKCHANNEL_PHRASE_WORDS} words (~4 spoken characters in 'zh')"
        )

    @pytest.mark.parametrize("language", ["es", "zh", "th"])
    def test_counting_rule_states_a_concrete_ceiling(self, language):
        rule = phrase_counting_rule(language, BACKCHANNEL_INVENTORY)
        assert "ceiling" in rule
        # The prompt must state the ceiling in the units the checker counts.
        unit = "CHARACTERS" if word_measure(language).unit == "char" else "tokens"
        assert unit in rule
