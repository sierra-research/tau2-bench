# Copyright Sierra
"""Prompt-parity render matrix: 16 languages x both prompt arms x both
simulators, plus the unit seams behind it.

The parity contract (grounded in the es arm-diff and round-1 annotation):

- ENGLISH INSTRUCTION LANGUAGE, TARGET-LANGUAGE EXAMPLES: in english prompt
  mode the instructional prose stays English, but every inline example
  utterance (disfluency palette, confirmations, check-ins, …) renders in the
  persona's language from pack data — the English palettes are what models
  copy verbatim into target-language calls.
- The language directive is visible near the TOP of the prompt (before any
  behavioral instruction), and a terse reminder anchors the very END (after
  the scenario block).
- English personas render today's prompts unchanged: fixed English defaults,
  no directive, no reminder. That covers BOTH the plain ``PersonaConfig`` and
  the English language pack's own personas — they speak the language the
  prompt is written in, so there is nothing to redirect them to.
"""

import pytest

from tau2.data_model.persona import PersonaConfig
from tau2.multilingual.localization_catalog import (
    GUIDELINE_EXAMPLE_CATALOG,
    ExampleJoinStyle,
    get_guideline_example_kind,
    render_guideline_example,
)
from tau2.multilingual.registry import get_language_pack, list_language_packs
from tau2.user.user_simulator import (
    UserSimulator,
    get_end_of_prompt_reminder,
    get_target_language_directive,
    insert_language_directive,
    substitute_example_slots,
)
from test_multilingual.conftest import first_persona, make_voice_sim

NON_ENGLISH = [lang for lang in list_language_packs() if lang != "en"]

# The exact English renderings of the highest-risk palettes: these joined
# strings must never survive into a multilingual prompt.
ENGLISH_PALETTE_LEFTOVERS = [
    '"um", "uh", "you know", "like", "I mean"',
    '"Uh huh", "Yeah", "Okay", "Got it"',
    '"Hello? Are you still there?"',
    '"This is ridiculous, I\'ll try calling back later"',
    '"So, um, I was wondering if you could, you know, help me out"',
    '"It\'s not working"',
    '"I have a problem" or "Something\'s wrong with my account"',
]

DIRECTIVE_HEADER = "## LANGUAGE OF THE CALL (MANDATORY)"
REMINDER_HEADER = "## FINAL REMINDER"
SCENARIO_SENTINEL = "parity-scenario-sentinel"


def voice_sim(persona_config):
    return make_voice_sim(
        persona_config,
        llm="gpt-4o-mini",
        instructions=SCENARIO_SENTINEL,
    )


def text_sim(persona_config):
    return UserSimulator(
        llm="dummy",
        instructions=SCENARIO_SENTINEL,
        persona_config=persona_config,
    )


class TestExampleSlotRendering:
    def test_unknown_kind_raises(self):
        with pytest.raises(KeyError, match="Unknown guideline example kind"):
            substitute_example_slots("x <EXAMPLE:definitely_not_a_kind> y", None)

    def test_english_default_renders_without_native_note(self):
        kind = get_guideline_example_kind("conversational_confirmations")
        english = render_guideline_example(kind, kind.english_items, native=False)
        assert english == '"Uh huh", "Yeah", "Okay", "Got it"'
        native = render_guideline_example(
            kind, ["ja", "oke", "mm", "goed"], native=True
        )
        assert native.startswith('"ja", "oke", "mm", "goed"')
        assert "those take precedence" in native

    def test_chain_render_interleaves_fixed_connectors(self):
        kind = get_guideline_example_kind("make_agent_work_chain")
        assert render_guideline_example(kind, kind.english_items, native=False) == (
            "\"It's not working\" → (agent asks what's not working) → "
            '"The app" → (agent asks which app) → "Your mobile app"'
        )

    def test_appended_example_empty_for_english(self):
        kind = get_guideline_example_kind("one_at_a_time_example")
        assert render_guideline_example(kind, [], native=False) == ""
        assert render_guideline_example(kind, ["toy zin"], native=True) == (
            ' For example: "toy zin"'
        )

    def test_every_catalog_kind_round_trips(self):
        """Every kind renders for both the English default and a native list
        of min_items utterances (the drafting floor)."""
        for kind_id, kind in GUIDELINE_EXAMPLE_CATALOG.items():
            if kind.english_items:
                assert render_guideline_example(
                    kind, kind.english_items, native=False
                ), kind_id
            native_items = [f"toy-{kind_id}-{i}" for i in range(kind.min_items)]
            rendered = render_guideline_example(kind, native_items, native=True)
            if kind.min_items:
                assert f"toy-{kind_id}-0" in rendered, kind_id

    def test_chain_kinds_carry_enough_connectors(self):
        for kind_id, kind in GUIDELINE_EXAMPLE_CATALOG.items():
            if kind.join_style == ExampleJoinStyle.CHAIN:
                assert len(kind.chain_connectors) == kind.max_items - 1, kind_id


class TestGenderedExampleSlots:
    """A pack may mark a kind as gendered language mechanics (hi first-person
    verb agreement: करती हूँ vs करता हूँ). The slot renders the male realization
    for male speakers and the shared palette for everyone else.

    No shipped pack carries ``male_utterances`` today (ja, the original
    carrier, was dropped from the benchmark), so the selection machinery is
    exercised by injecting a gendered variant of the hi pack's
    frustrated_endings kind — Hindi genuinely marks speaker gender in exactly
    these forms."""

    SLOT = "<EXAMPLE:frustrated_endings>"
    FEMALE_ENDING = "मेरे पास इसके लिए time नहीं है, रखती हूँ"
    MALE_ENDING = "मेरे पास इसके लिए time नहीं है, रखता हूँ"

    @pytest.fixture
    def gendered_hi_pack(self, monkeypatch):
        """The hi pack with a gendered frustrated_endings palette injected."""
        from tau2.multilingual.schema import GuidelineExample

        loc = get_language_pack("hi").localization
        gendered = GuidelineExample(
            kind="frustrated_endings",
            utterances=["ये तो हद है, बाद में call करती हूँ", self.FEMALE_ENDING],
            male_utterances=["ये तो हद है, बाद में call करता हूँ", self.MALE_ENDING],
        )
        monkeypatch.setattr(
            loc,
            "guideline_examples",
            [e for e in loc.guideline_examples if e.kind != gendered.kind] + [gendered],
        )

    def _gendered_kinds(self, language):
        loc = get_language_pack(language).localization
        return [e for e in (loc.guideline_examples if loc else []) if e.male_utterances]

    def test_male_and_female_render_differently(self, gendered_hi_pack):
        male = substitute_example_slots(self.SLOT, "hi", "male")
        female = substitute_example_slots(self.SLOT, "hi", "female")
        assert male != female
        assert self.MALE_ENDING in male
        assert self.FEMALE_ENDING in female

    def test_unspecified_gender_gets_the_shared_palette(self, gendered_hi_pack):
        assert substitute_example_slots(self.SLOT, "hi", None) == (
            substitute_example_slots(self.SLOT, "hi", "female")
        )

    def test_non_gendered_kind_is_identical_for_both(self, gendered_hi_pack):
        slot = "<EXAMPLE:vague_openers>"
        assert substitute_example_slots(slot, "hi", "male") == (
            substitute_example_slots(slot, "hi", "female")
        )

    def test_gender_is_a_no_op_for_packs_without_variants(self):
        for language in NON_ENGLISH:
            if self._gendered_kinds(language):
                continue
            assert substitute_example_slots(self.SLOT, language, "male") == (
                substitute_example_slots(self.SLOT, language, "female")
            ), language

    def test_variants_stay_parallel_in_every_shipped_pack(self):
        for language in NON_ENGLISH:
            for example in self._gendered_kinds(language):
                assert len(example.male_utterances) == len(example.utterances), (
                    f"{language}/{example.kind}"
                )

    def test_voice_prompt_selects_by_persona_gender(self, gendered_hi_pack):
        """End-to-end: the persona's gender tag reaches the rendered prompt —
        gendered pack data must not be dead data."""
        pack = get_language_pack("hi")
        male = voice_sim(pack.personas["imran_hindi_v1"]).system_prompt
        female = voice_sim(pack.personas["rishika_hindi_v1"]).system_prompt
        assert self.MALE_ENDING in male
        assert self.FEMALE_ENDING not in male
        assert self.FEMALE_ENDING in female
        assert self.MALE_ENDING not in female


class TestDirectiveInsertion:
    def test_inserted_before_first_section(self):
        guidelines = "# Title\n\nIntro prose.\n\n## First Section\n- rule\n"
        out = insert_language_directive(guidelines, "## DIRECTIVE\nSpeak X.")
        assert out.index("## DIRECTIVE") < out.index("## First Section")
        assert out.index("Intro prose.") < out.index("## DIRECTIVE")

    def test_none_directive_passthrough(self):
        guidelines = "# Title\n\n## First Section\n"
        assert insert_language_directive(guidelines, None) == guidelines

    def test_no_sections_appends(self):
        out = insert_language_directive("just prose", "## D\nx")
        assert out.endswith("## D\nx")


class TestEndReminder:
    def test_english_persona_gets_none(self):
        assert get_end_of_prompt_reminder(PersonaConfig()) is None
        assert get_end_of_prompt_reminder(None) is None

    @pytest.mark.parametrize("language", NON_ENGLISH)
    def test_pack_persona_gets_the_versioned_english_reminder(self, language):
        reminder = get_end_of_prompt_reminder(first_persona(language))
        assert reminder
        assert get_language_pack(language).display_name in reminder
        for token in ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"):
            assert token in reminder, f"'{language}' reminder dropped {token}"


class TestEnglishPersonaUnchanged:
    """English personas must render today's prompts: fixed English palettes,
    no directive, no reminder, no slot markers."""

    @pytest.mark.parametrize("factory", [voice_sim, text_sim])
    def test_no_multilingual_blocks(self, factory):
        prompt = factory(PersonaConfig()).system_prompt
        assert DIRECTIVE_HEADER not in prompt
        assert REMINDER_HEADER not in prompt
        assert "<EXAMPLE:" not in prompt
        assert "<SPOKEN_VALUES_SPELLOUT>" not in prompt

    def test_voice_prompt_keeps_english_palettes(self):
        prompt = voice_sim(PersonaConfig()).system_prompt
        assert '"um", "uh", "you know", "like", "I mean"' in prompt
        assert '"Uh huh", "Yeah", "Okay", "Got it"' in prompt


class TestEnglishPackPersona:
    """The ENGLISH language pack's personas speak the prompt's own language.

    ``data/tau2/multilingual/en/pack.yaml`` personas are
    ``MultilingualPersonaConfig``s carrying ``language='en'``, so they used to
    trip the same ``getattr(persona, 'language')`` gate as every other pack and
    render both target-language blocks with ``language_name='English'`` —
    "every word you speak to the agent MUST be in English … never in English",
    plus native-orthography and code-switching rules aimed at a non-English
    speaker. The English baseline arm must read like the plain-persona arm.
    """

    def test_pack_persona_carries_the_english_language_code(self):
        # Guards the premise: if the en pack ever stopped tagging its personas
        # with a language, the suppression below would be vacuously true.
        assert getattr(first_persona("en"), "language", None) == "en"

    @pytest.mark.parametrize("factory", [voice_sim, text_sim])
    def test_no_directive_and_no_reminder(self, factory):
        persona = first_persona("en")
        assert get_target_language_directive(persona) is None
        assert get_end_of_prompt_reminder(persona) is None
        prompt = factory(persona).system_prompt
        assert DIRECTIVE_HEADER not in prompt
        assert REMINDER_HEADER not in prompt
        assert "never in English" not in prompt

    @pytest.mark.parametrize("factory", [voice_sim, text_sim])
    def test_special_token_contract_survives(self, factory):
        """The dropped reminder restated the special-token contract; the
        guidelines' own sections must still carry all three tokens."""
        prompt = factory(first_persona("en")).system_prompt
        for token in ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"):
            assert token in prompt

    @pytest.mark.parametrize("factory", [voice_sim, text_sim])
    def test_non_english_pack_persona_still_gets_both(self, factory):
        """The counterpart: suppression is keyed on the English pack alone."""
        persona = first_persona("ko")
        assert get_target_language_directive(persona) is not None
        assert get_end_of_prompt_reminder(persona) is not None
        prompt = factory(persona).system_prompt
        assert DIRECTIVE_HEADER in prompt
        assert REMINDER_HEADER in prompt


class TestRenderMatrix:
    """The full matrix: every language x both simulators."""

    @pytest.mark.parametrize("language", NON_ENGLISH)
    def test_voice_prompt_parity(self, language):
        persona = first_persona(language)
        prompt = voice_sim(persona).system_prompt

        # Every slot filled; no marker leaks.
        assert "<EXAMPLE:" not in prompt
        assert "<SPOKEN_VALUES_SPELLOUT>" not in prompt
        assert "<PERSONA_GUIDELINES>" not in prompt

        # No English palette leftovers: the pack's example slots must win.
        for leftover in ENGLISH_PALETTE_LEFTOVERS:
            assert leftover not in prompt, (
                f"{language}: English palette leaked: {leftover}"
            )

        # The directive is near the TOP: before the first behavioral
        # section and inside the first 15% of the prompt.
        assert DIRECTIVE_HEADER in prompt
        directive_at = prompt.index(DIRECTIVE_HEADER)
        assert directive_at < prompt.index("## Core Voice Call Principles")
        assert directive_at < len(prompt) * 0.15
        assert prompt.count(DIRECTIVE_HEADER) == 1

        # The reminder anchors the very END, after the scenario block.
        reminder = get_end_of_prompt_reminder(persona)
        assert reminder and prompt.rstrip().endswith(reminder.rstrip())
        assert prompt.index(SCENARIO_SENTINEL) < prompt.rindex(reminder[:20])

    @pytest.mark.parametrize("language", NON_ENGLISH)
    def test_text_prompt_parity(self, language):
        persona = first_persona(language)
        prompt = text_sim(persona).system_prompt

        assert "<EXAMPLE:" not in prompt
        assert DIRECTIVE_HEADER in prompt
        directive_at = prompt.index(DIRECTIVE_HEADER)
        first_rule_section = prompt.index("## ", directive_at + 10)
        assert directive_at < first_rule_section
        assert directive_at < len(prompt) * 0.20

        reminder = get_end_of_prompt_reminder(persona)
        assert reminder and prompt.rstrip().endswith(reminder.rstrip())

    @pytest.mark.parametrize("language", NON_ENGLISH)
    def test_scaffold_is_the_english_template(self, language):
        """The pack's translated scaffold strings are retired-arm data: the
        localization block and persona header render from the fixed English
        scaffold for every language."""
        prompt = voice_sim(first_persona(language)).system_prompt
        assert "## PERSONA AND LANGUAGE" in prompt
        display = get_language_pack(language).display_name
        assert f"## LANGUAGE LOCALIZATION ({display})" in prompt
