# Copyright Sierra
"""Per-pack spoken-localization data: schema, rendering, prompt assembly,
and the coverage guard over every shipped pack.

Localization grounding: de voiced 'underscore' in English instead of
'Unterstrich'; pl hallucinated a symbol readout ('archeśnika'); fr 'tiret du
bas' was mistranscribed as a hyphen; ko mixed languages inside one date/number
('May 십칠', '이십two'); vi ran 'May21' together; pl callers said 'user ID' /
'refund' / 'future trip' in English mid-sentence (7/10 languages flagged
untranslated tokens). The pack's ``localization`` block is the structured fix:
symbol readouts + single-language date/numeral examples + a domain glossary,
selected from the closed catalogs in ``tau2.multilingual.localization_catalog``
and injected into BOTH user-simulator prompts (text and voice).

Self-honorification grounding prevents the caller from applying 성함 — the
honorific word for the listener's name — to itself. ``honorific_self_reference``
is the structured fix, rendered in English prompt mode in both text and voice.
"""

import pytest

import tau2.multilingual.registry as ml_registry
from tau2.data_model.persona import PersonaConfig
from tau2.multilingual.localization_catalog import (
    GUIDELINE_EXAMPLE_CATALOG,
    HONORIFIC_CONCEPT_CATALOG,
    SYMBOL_CATALOG,
    get_domain_term_catalog,
)
from tau2.multilingual.registry import (
    get_language_pack,
    get_localization_guidelines,
    list_language_packs,
)
from tau2.multilingual.schema import (
    AVOID_REDUNDANT_LEAD_RE,
    AmountReadout,
    DomainTermGloss,
    GuidelineExample,
    HonorificSelfReference,
    LanguagePack,
    LocalizationPackConfig,
    MultilingualPersonaConfig,
    PhoneNumberReadout,
    SpellingAlphabet,
    SpokenFormExample,
    SpokenValueExample,
    SpokenValueKind,
    SymbolReadout,
)
from tau2.user.user_simulator import ENGLISH_SPELLOUT_SECTION, UserSimulator
from test_multilingual.conftest import make_voice_sim

NON_ENGLISH = [lang for lang in list_language_packs() if lang != "en"]

SPELLOUT_HEADER = "## Speaking Special Characters and Numbers"


def sample_localization() -> LocalizationPackConfig:
    return LocalizationPackConfig(
        symbol_readouts=[
            SymbolReadout(symbol="_", spoken=["unterstrich-xx"]),
            SymbolReadout(symbol="@", spoken=["at-xx", "affe-xx"]),
        ],
        date_examples=[
            SpokenFormExample(written="May 17", spoken="siebzehnter-mai-xx")
        ],
        number_examples=[SpokenFormExample(written="21", spoken="einundzwanzig-xx")],
        domain_glossaries={
            "airline": [
                DomainTermGloss(term_id="user_id", native="benutzerkennung-xx"),
                DomainTermGloss(
                    term_id="refund",
                    native="rückerstattung-xx",
                    note="formal register",
                ),
            ],
            "telecom": [
                DomainTermGloss(term_id="roaming", native="roaming-xx"),
            ],
        },
        phone_number_readout=PhoneNumberReadout(
            rule="Phone numbers are read in xx pairs.",
            examples=[
                SpokenFormExample(written="0176 23", spoken="null-eins-xx pairs")
            ],
        ),
        amount_readout=AmountReadout(
            rule="Amounts put the currency word last; the decimal comma is "
            "spoken 'komma-xx'.",
            examples=[
                SpokenFormExample(written="12,50", spoken="zwölf-komma-fünfzig-xx"),
                SpokenFormExample(written="100", spoken="hundert-xx"),
            ],
        ),
        spelling_alphabet=SpellingAlphabet(
            rule="Spell with xx first-name anchors.",
            avoid="the NATO/anglo 'B for boy' alphabet",
            anchor_examples=[
                SpokenFormExample(written="A", spoken="A wie Anton-xx"),
                SpokenFormExample(written="B", spoken="B wie Berta-xx"),
                SpokenFormExample(written="M", spoken="M wie Martha-xx"),
            ],
        ),
        spoken_value_examples=[
            SpokenValueExample(
                kind=kind,
                written=f"{kind.value}-written-xx",
                spoken=f"{kind.value} gesprochen-xx",
            )
            for kind in SpokenValueKind
        ],
    )


class TestLocalizationSchema:
    def test_unknown_symbol_rejected(self):
        with pytest.raises(ValueError, match="Unknown localization symbol"):
            SymbolReadout(symbol="%", spoken=["prozent"])

    def test_unknown_domain_term_rejected(self):
        with pytest.raises(ValueError, match="Unknown localization domain term"):
            LocalizationPackConfig(
                domain_glossaries={
                    "airline": [
                        DomainTermGloss(term_id="totally_made_up_term", native="x")
                    ]
                }
            )

    def test_unknown_glossary_domain_rejected(self):
        with pytest.raises(ValueError, match="Unknown glossary domain"):
            LocalizationPackConfig(
                domain_glossaries={
                    "no_such_domain": [DomainTermGloss(term_id="user_id", native="x")]
                }
            )

    def test_duplicate_symbol_rejected(self):
        with pytest.raises(ValueError, match="Duplicate localization symbol"):
            LocalizationPackConfig(
                symbol_readouts=[
                    SymbolReadout(symbol="@", spoken=["a"]),
                    SymbolReadout(symbol="@", spoken=["b"]),
                ]
            )

    def test_duplicate_term_rejected(self):
        with pytest.raises(ValueError, match="Duplicate localization domain term"):
            LocalizationPackConfig(
                domain_glossaries={
                    "airline": [
                        DomainTermGloss(term_id="refund", native="a"),
                        DomainTermGloss(term_id="refund", native="b"),
                    ]
                }
            )

    def test_avoid_repeating_the_do_not_use_label_rejected(self):
        """`to_prompt_text` prefixes its own 'Do NOT use:' label, so an avoid
        string that repeats it rendered 'Do NOT use: Do NOT use the NATO…' in
        eight shipped packs. Gated on the DATA, not by a renderer heuristic."""
        anchors = [
            SpokenFormExample(written="A", spoken="A wie Anton-xx"),
            SpokenFormExample(written="B", spoken="B wie Berta-xx"),
            SpokenFormExample(written="M", spoken="M wie Martha-xx"),
        ]
        for bad in (
            "Do NOT use the NATO alphabet.",
            "do not use the NATO alphabet.",
            "Don't use the NATO alphabet.",
        ):
            with pytest.raises(ValueError, match="must not begin with"):
                SpellingAlphabet(rule="r", avoid=bad, anchor_examples=anchors)
        # A noun phrase is fine, and so is a later 'Do NOT' in a second
        # sentence (zh carries one) — only the redundant LEAD is rejected.
        assert SpellingAlphabet(
            rule="r",
            avoid="The NATO alphabet. Do NOT give the name given-name-first.",
            anchor_examples=anchors,
        ).avoid.startswith("The NATO")

    def test_empty_spoken_rejected(self):
        with pytest.raises(ValueError):
            SymbolReadout(symbol="@", spoken=[])
        with pytest.raises(ValueError, match="non-empty"):
            SymbolReadout(symbol="@", spoken=["  "])


class TestGenderedGuidelineExamples:
    """Speaker gender as language MECHANICS (ja review, 2026-07-21): a kind
    may carry a male realization parallel to the shared palette."""

    def test_male_variant_selected_for_male_speakers_only(self):
        example = GuidelineExample(
            kind="frustrated_endings",
            utterances=["ごめんなさい、切りますね", "また後でかけ直します"],
            male_utterances=["すみません、切ります", "また後でかけ直します"],
        )
        assert example.items_for_gender("male") == example.male_utterances
        assert example.items_for_gender("female") == example.utterances
        assert example.items_for_gender(None) == example.utterances

    def test_shared_palette_when_no_variant(self):
        example = GuidelineExample(kind="frustrated_endings", utterances=["a", "b"])
        assert example.items_for_gender("male") == ["a", "b"]

    def test_variant_must_be_parallel(self):
        with pytest.raises(ValueError, match="parallel item-for-item"):
            GuidelineExample(
                kind="frustrated_endings",
                utterances=["a", "b"],
                male_utterances=["a2"],
            )

    def test_no_op_variant_rejected(self):
        with pytest.raises(ValueError, match="duplicates utterances"):
            GuidelineExample(
                kind="frustrated_endings",
                utterances=["a", "b"],
                male_utterances=["a", "b"],
            )

    def test_empty_variant_utterance_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            GuidelineExample(
                kind="frustrated_endings",
                utterances=["a", "b"],
                male_utterances=["a2", "  "],
            )

    def test_lookup_threads_gender_through_the_pack(self):
        loc = LocalizationPackConfig(
            guideline_examples=[
                GuidelineExample(
                    kind="frustrated_endings",
                    utterances=["f1", "f2"],
                    male_utterances=["m1", "m2"],
                ),
                GuidelineExample(
                    kind="vague_openers", utterances=["shared1", "shared2"]
                ),
            ]
        )
        assert loc.guideline_example_items("frustrated_endings", "male") == [
            "m1",
            "m2",
        ]
        assert loc.guideline_example_items("frustrated_endings", "female") == [
            "f1",
            "f2",
        ]
        assert loc.guideline_example_items("frustrated_endings") == ["f1", "f2"]
        assert loc.guideline_example_items("vague_openers", "male") == [
            "shared1",
            "shared2",
        ]
        assert loc.guideline_example_items("not_carried", "male") is None


class TestLocalizationRendering:
    def test_voice_mode_renders_all_sections(self):
        text = sample_localization().to_prompt_text(
            "Testlang", mode="voice", domain="airline"
        )
        assert "## LANGUAGE LOCALIZATION (Testlang)" in text
        # Single-language consistency rule + worked examples.
        assert "ENTIRELY in Testlang" in text
        assert "'May 17' → “siebzehnter-mai-xx”" in text
        assert "'21' → “einundzwanzig-xx”" in text
        # Speech conventions: phone grouping, amounts, spelling alphabet.
        assert "### Phone numbers" in text
        assert "'0176 23' → “null-eins-xx pairs”" in text
        assert "### Amounts and prices" in text
        assert "'12,50' → “zwölf-komma-fünfzig-xx”" in text
        assert "### Spelling out a misheard name or code" in text
        assert "Do NOT use: the NATO/anglo 'B for boy' alphabet" in text
        assert "'B' → “B wie Berta-xx”" in text
        # Glossary keyed by the catalog's English surface terms.
        assert "user ID → “benutzerkennung-xx”" in text
        assert "refund → “rückerstattung-xx” (formal register)" in text

    def test_block_does_not_render_symbols_or_value_examples(self):
        """The guidelines' spellout section (to_spellout_section) is the
        single owner of the symbol table and the worked value readouts —
        the persona-slot block must not duplicate them in either mode."""
        for mode in ("voice", "text"):
            text = sample_localization().to_prompt_text("Testlang", mode=mode)
            assert "unterstrich-xx" not in text
            assert "at-xx" not in text
            assert SPELLOUT_HEADER not in text
            assert "gesprochen-xx" not in text

    def test_glossary_selected_by_domain(self):
        """Each run renders ONLY its domain's glossary; no domain, no
        glossary (the rest of the localization block still renders)."""
        loc = sample_localization()
        telecom = loc.to_prompt_text("Testlang", mode="voice", domain="telecom")
        assert "roaming → “roaming-xx”" in telecom
        assert "benutzerkennung-xx" not in telecom
        no_domain = loc.to_prompt_text("Testlang", mode="voice")
        assert no_domain is not None
        assert "benutzerkennung-xx" not in no_domain
        assert "roaming-xx" not in no_domain

    def test_text_mode_omits_spoken_only_sections(self):
        text = sample_localization().to_prompt_text(
            "Testlang", mode="text", domain="airline"
        )
        assert "unterstrich-xx" not in text
        assert "siebzehnter-mai-xx" not in text  # spoken examples are voice-only
        assert "### Phone numbers" not in text  # spoken-only conventions
        assert "### Amounts and prices" not in text
        assert "Spelling out a misheard" not in text
        # The consistency rule and the glossary still apply to typed chat.
        assert "Never mix English and Testlang words" in text
        assert "user ID → “benutzerkennung-xx”" in text

    def test_empty_config_renders_none(self):
        assert LocalizationPackConfig().to_prompt_text("Testlang") is None

    def test_registry_helper_none_paths(self):
        assert get_localization_guidelines(None) is None
        assert get_localization_guidelines("zz-no-such-pack") is None


class TestSpelloutSection:
    def test_renders_native_table_and_examples(self):
        section = sample_localization().to_spellout_section("Testlang")
        assert section.startswith(SPELLOUT_HEADER)
        assert "voicing each symbol the native Testlang way" in section
        assert '- _ (underscore) = "unterstrich-xx"' in section
        assert '- @ (at sign) = "at-xx" or "affe-xx"' in section
        # One worked example per kind, labeled like the English section.
        assert '- Email: "email gesprochen-xx"' in section
        assert '- User ID: "user_id gesprochen-xx"' in section
        assert '- Phone: "phone gesprochen-xx"' in section
        assert '- Spelling name: "spelling_name gesprochen-xx"' in section
        assert '- Account number: "account_number gesprochen-xx"' in section
        assert '- Website: "website gesprochen-xx"' in section
        # Persona-neutral contract is stated, register stays persona-owned.
        assert "your persona controls tone and register" in section

    def test_partial_data_renders_none(self):
        """Native table + English examples (or vice versa) would be worse
        than the English fallback — partial data renders nothing."""
        config = sample_localization()
        assert (
            config.model_copy(update={"spoken_value_examples": []}).to_spellout_section(
                "Testlang"
            )
            is None
        )
        assert (
            config.model_copy(update={"symbol_readouts": []}).to_spellout_section(
                "Testlang"
            )
            is None
        )

    def test_english_default_is_persona_neutral(self):
        """The fixed English fallback shows pure readout mechanics — the
        old 'Yeah,'/'um'/'uh' dressing belongs to personas, not guidelines."""
        assert ENGLISH_SPELLOUT_SECTION.startswith(SPELLOUT_HEADER)
        for filler in ('"Yeah', ", um,", ", uh,"):
            assert filler not in ENGLISH_SPELLOUT_SECTION
        assert '- Email: "it\'s john underscore doe at gmail dot com"' in (
            ENGLISH_SPELLOUT_SECTION
        )

    def test_registry_helper(self):
        from tau2.multilingual.registry import get_spellout_section

        assert get_spellout_section(None) is None
        assert get_spellout_section("zz-no-such-pack") is None


HONORIFIC_HEADING = "### Talking about yourself, not the other person"


class TestHonorificSelfReference:
    """Self-honorification (ko annotation + 71/300 measured ko telecom calls):
    the caller applied 성함 — the honorific word for 'name', only ever used
    ABOUT the listener — to ITSELF. The pack now carries the honorific/plain
    pairs and the block renders in BOTH prompt modes."""

    PAIRS = [
        HonorificSelfReference(
            concept_id="name", honorific="HON-NAME", plain="name-xx"
        ),
        HonorificSelfReference(concept_id="age", honorific="HON-AGE", plain="age-xx"),
    ]

    def _config(self, **overrides) -> LocalizationPackConfig:
        return sample_localization().model_copy(
            update={"honorific_self_reference": self.PAIRS, **overrides}
        )

    def test_unknown_concept_rejected_at_pack_load(self):
        """The closed catalog is enforced on the load path (model_validate is
        what the pack loader runs) — a pack cannot introduce a pair for an
        un-reviewed concept."""
        with pytest.raises(ValueError, match="Unknown honorific concept"):
            LocalizationPackConfig.model_validate(
                {
                    "honorific_self_reference": [
                        {
                            "concept_id": "totally_made_up_concept",
                            "honorific": "a",
                            "plain": "b",
                        }
                    ]
                }
            )

    def test_duplicate_concept_rejected(self):
        with pytest.raises(ValueError, match="Duplicate honorific concept"):
            LocalizationPackConfig(
                honorific_self_reference=[
                    HonorificSelfReference(concept_id="name", honorific="a", plain="b"),
                    HonorificSelfReference(concept_id="name", honorific="c", plain="d"),
                ]
            )

    def test_identical_forms_rejected(self):
        """A no-op pair would teach a distinction the language does not make."""
        with pytest.raises(ValueError, match="identical"):
            HonorificSelfReference(concept_id="name", honorific="x", plain="x")

    def test_empty_word_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            HonorificSelfReference(concept_id="name", honorific="  ", plain="b")

    @pytest.mark.parametrize("mode", ["voice", "text"])
    def test_renders_in_both_modes(self, mode):
        """Applying an other-directed honorific to yourself is ungrammatical
        typed or spoken — the block is NOT voice-gated."""
        text = self._config().to_prompt_text("Testlang", mode=mode, domain="airline")
        assert HONORIFIC_HEADING in text
        assert "apply only to the other person" in text
        assert "Do NOT use: HON-NAME, HON-AGE" in text
        assert "- name → “name-xx” (not “HON-NAME”)" in text
        assert "- age → “age-xx” (not “HON-AGE”)" in text

    def test_renders_nothing_when_absent(self):
        """A pack that declares no pairs renders EXACTLY what it renders
        today — the whole delta is the honorific section."""
        for mode in ("voice", "text"):
            without = sample_localization().to_prompt_text(
                "Testlang", mode=mode, domain="airline"
            )
            with_pairs = self._config().to_prompt_text(
                "Testlang", mode=mode, domain="airline"
            )
            assert HONORIFIC_HEADING not in without
            assert "HON-NAME" not in without
            # The only change is the inserted section: drop it and the two
            # renderings are byte-identical.
            kept = [
                section
                for section in with_pairs.split("\n\n")
                if HONORIFIC_HEADING not in section
            ]
            assert "\n\n".join(kept) == without

    def test_pairs_alone_render_the_block(self):
        """The gate counts the honorific pairs: a pack carrying nothing but
        them still renders (and an empty config still renders None)."""
        only = LocalizationPackConfig(honorific_self_reference=self.PAIRS)
        assert HONORIFIC_HEADING in only.to_prompt_text("Testlang")
        assert LocalizationPackConfig().to_prompt_text("Testlang") is None

    # The ONLY languages with evidence of the self-honorification trap. Adding
    # a language here is a reviewed decision: a speculative pair teaches the
    # simulator a distinction its language may not make.
    HONORIFIC_LANGS = {"ko", "zh"}

    @pytest.mark.parametrize("language", NON_ENGLISH)
    def test_only_evidenced_languages_carry_pairs(self, language):
        loc = get_language_pack(language).localization
        assert bool(loc.honorific_self_reference) == (
            language in self.HONORIFIC_LANGS
        ), (
            f"pack '{language}' honorific_self_reference disagrees with the "
            "reviewed language set"
        )

    @pytest.mark.parametrize("language", sorted(HONORIFIC_LANGS))
    @pytest.mark.parametrize("mode", ["voice", "text"])
    def test_shipped_pack_block_renders(self, language, mode):
        """The REAL pack data renders the block in english prompt mode (what
        every production run uses) — a Latin-script pack renders none of it."""
        pack = get_language_pack(language)
        text = pack.localization.to_prompt_text(
            pack.display_name, mode=mode, domain="telecom"
        )
        assert HONORIFIC_HEADING in text
        for pair in pack.localization.honorific_self_reference:
            assert f"“{pair.plain}” (not “{pair.honorific}”)" in text
        es = get_language_pack("es")
        assert HONORIFIC_HEADING not in es.localization.to_prompt_text(
            es.display_name, mode=mode, domain="telecom"
        )

    def test_ko_pack_names_the_measured_word(self):
        """Regression anchor for the measured defect: 성함 self-applied in
        71/300 ko telecom calls. The rule must name it and give 이름."""
        pack = get_language_pack("ko")
        text = pack.localization.to_prompt_text(
            pack.display_name, mode="voice", domain="telecom"
        )
        assert "- name → “이름” (not “성함”)" in text
        assert "Do NOT use: 성함" in text

    def test_every_catalog_concept_is_referenced(self):
        """Closed-catalog coverage guard, the unreferenced direction: a
        concept no pack uses is dead scaffold and must be removed."""
        referenced = {
            pair.concept_id
            for language in NON_ENGLISH
            for pair in get_language_pack(
                language
            ).localization.honorific_self_reference
        }
        assert referenced == set(HONORIFIC_CONCEPT_CATALOG), (
            "honorific concept catalog and pack usage disagree "
            f"(unreferenced: {sorted(set(HONORIFIC_CONCEPT_CATALOG) - referenced)})"
        )

    def test_the_english_scaffold_renders_the_rule(self):
        """The whole point of not needing translated scaffold strings: every
        run renders the English scaffold, and the rule must still appear."""
        rendered = get_localization_guidelines("ko", mode="voice", domain="telecom")
        assert HONORIFIC_HEADING in rendered
        assert "성함" in rendered


@pytest.fixture
def synthetic_pack_env(clean_registries):
    """Register a synthetic 'xx' pack carrying a localization block.

    Rides ``clean_registries`` for the registry snapshot/restore AND the
    synthetic 'xx' variety-catalog entries (the v3 directive renders the
    persona's locale/variety, so the pack persona pins a locale).
    """
    persona = MultilingualPersonaConfig(
        persona_id="tara_xxloc_v1",
        display_name="Tara",
        short_description="Localization test persona",
        language="xx",
        locale="XX-TS",
        voice_id="fake_voice_id_123",
        pragmatics_clauses=["Speak Testlang."],
    )
    ml_registry.register_language_pack(
        LanguagePack(
            language="xx",
            display_name="Testlang",
            personas={persona.persona_id: persona},
            localization=sample_localization(),
        )
    )
    yield persona


class TestPromptAssembly:
    """The localization data must actually REACH the user-sim prompts."""

    def test_text_system_prompt_carries_localization(self, synthetic_pack_env):
        sim = UserSimulator(
            llm="dummy",
            instructions="scenario",
            persona_config=synthetic_pack_env,
            domain="airline",
        )
        prompt = sim.system_prompt
        assert "## LANGUAGE LOCALIZATION (Testlang)" in prompt
        assert "user ID → “benutzerkennung-xx”" in prompt
        # Text mode: no spoken-only sections.
        assert "unterstrich-xx" not in prompt
        # The persona's own guidelines still land too.
        assert "Speak Testlang." in prompt

    def _voice_sim(self, persona_config):
        return make_voice_sim(persona_config, llm="gpt-4o-mini", domain="airline")

    def test_voice_system_prompt_carries_localization(self, synthetic_pack_env):
        prompt = self._voice_sim(synthetic_pack_env).system_prompt
        assert "## LANGUAGE LOCALIZATION (Testlang)" in prompt
        assert "'May 17' → “siebzehnter-mai-xx”" in prompt
        assert "### Phone numbers" in prompt
        assert "'B' → “B wie Berta-xx”" in prompt
        assert "user ID → “benutzerkennung-xx”" in prompt

    def test_voice_spellout_slot_rendered_natively_exactly_once(
        self, synthetic_pack_env
    ):
        """English prompt mode (the default): the guidelines' spellout
        section renders the pack's native symbol table + worked examples in
        place of the English ones — single owner, no English leftovers and
        no second copy from the persona-slot block."""
        prompt = self._voice_sim(synthetic_pack_env).system_prompt
        assert prompt.count(SPELLOUT_HEADER) == 1
        assert '- _ (underscore) = "unterstrich-xx"' in prompt
        assert '- Email: "email gesprochen-xx"' in prompt
        # No English symbol table or worked examples remain.
        assert '- @ = "at"' not in prompt
        assert "john underscore doe" not in prompt
        # The slot marker itself never leaks into a rendered prompt.
        assert "<SPOKEN_VALUES_SPELLOUT>" not in prompt
        # The native symbol table appears exactly once.
        assert prompt.count("unterstrich-xx") == 1

    def test_voice_plain_english_persona_gets_english_spellout(self):
        prompt = self._voice_sim(PersonaConfig()).system_prompt
        assert prompt.count(SPELLOUT_HEADER) == 1
        assert '- @ = "at"' in prompt
        assert '- Email: "it\'s john underscore doe at gmail dot com"' in prompt
        assert "<SPOKEN_VALUES_SPELLOUT>" not in prompt
        assert "LANGUAGE LOCALIZATION" not in prompt

    def test_plain_english_persona_unaffected(self):
        sim = UserSimulator(
            llm="dummy",
            instructions="scenario",
            persona_config=PersonaConfig(),
        )
        assert "LANGUAGE LOCALIZATION" not in sim.system_prompt


class TestShippedPackCoverage:
    """Coverage guard: every shipped non-English pack carries a FULL
    localization block (all catalog symbols, all catalog terms, worked
    date/number examples), so no language silently loses the fix."""

    def test_there_are_registered_languages(self):
        assert NON_ENGLISH, "no non-English packs registered — discovery is broken"

    @pytest.mark.parametrize("language", NON_ENGLISH)
    def test_pack_has_full_localization_block(self, language):
        pack = get_language_pack(language)
        assert pack.localization is not None, (
            f"pack '{language}' has no localization block — run "
            f"`tau2 factory draft-localization --lang {language}`"
        )
        loc = pack.localization
        assert {r.symbol for r in loc.symbol_readouts} == set(SYMBOL_CATALOG), (
            f"pack '{language}' does not cover the full symbol catalog"
        )
        assert "airline" in loc.domain_glossaries, (
            f"pack '{language}' has no airline glossary — run "
            f"`tau2 factory draft-localization --lang {language}`"
        )
        for glossary_domain, glosses in loc.domain_glossaries.items():
            assert {g.term_id for g in glosses} == set(
                get_domain_term_catalog(glossary_domain)
            ), (
                f"pack '{language}' does not cover the full "
                f"'{glossary_domain}' domain-term catalog"
            )
        assert len(loc.date_examples) >= 2, f"'{language}': need >=2 date examples"
        assert len(loc.number_examples) >= 2, f"'{language}': need >=2 number examples"

    @pytest.mark.parametrize("language", NON_ENGLISH)
    def test_pack_has_speech_conventions(self, language):
        """Every shipped pack carries the speech conventions + worked value
        examples — otherwise the default (English-prompt-mode) arm silently
        falls back to English phone/amount/spelling behavior. Backfill with
        `tau2 factory draft-localization --lang <language>` (upgrade path)."""
        loc = get_language_pack(language).localization
        assert loc is not None, f"pack '{language}' has no localization block"
        assert loc.phone_number_readout is not None, (
            f"pack '{language}' is missing phone_number_readout"
        )
        assert loc.amount_readout is not None, (
            f"pack '{language}' is missing amount_readout"
        )
        assert loc.spelling_alphabet is not None, (
            f"pack '{language}' is missing spelling_alphabet"
        )
        assert {e.kind for e in loc.spoken_value_examples} == set(SpokenValueKind), (
            f"pack '{language}' does not cover all spoken-value example kinds"
        )

    @pytest.mark.parametrize("language", NON_ENGLISH)
    def test_rendered_label_is_not_duplicated(self, language):
        """Regression guard over the SHIPPED data: the rendered spelling
        section must say 'Do NOT use:' exactly once. Eight packs (de es hi it
        ko nl pl vi) rendered 'Do NOT use: Do NOT use the NATO…' until the
        2026-07-27 correction, and it went unnoticed because nobody rendered
        the section."""
        pack = get_language_pack(language)
        if pack.localization is None or pack.localization.spelling_alphabet is None:
            pytest.skip("no spelling alphabet")
        text = pack.localization.to_prompt_text(pack.display_name, mode="voice")
        label_line = next(
            line for line in text.split("\n") if line.startswith("Do NOT use:")
        )
        # Only the LEAD is the defect. A later "… — Japanese callers do not
        # use it" (ja, ru) is meaningful prose, not a repeated label.
        assert not AVOID_REDUNDANT_LEAD_RE.match(
            label_line[len("Do NOT use:") :].strip()
        ), f"pack '{language}' renders a duplicated prohibition: {label_line}"

    @pytest.mark.parametrize("language", NON_ENGLISH)
    def test_pack_localization_renders_both_modes(self, language):
        pack = get_language_pack(language)
        if pack.localization is None:
            pytest.skip("covered by test_pack_has_full_localization_block")
        for mode in ("voice", "text"):
            text = pack.localization.to_prompt_text(pack.display_name, mode=mode)
            assert text and pack.display_name in text

    @pytest.mark.parametrize("language", NON_ENGLISH)
    def test_pack_spellout_section_renders(self, language):
        pack = get_language_pack(language)
        if pack.localization is None:
            pytest.skip("covered by test_pack_has_full_localization_block")
        section = pack.localization.to_spellout_section(pack.display_name)
        assert section and section.startswith(
            "## Speaking Special Characters and Numbers"
        ), f"pack '{language}' renders no spellout section"

    @pytest.mark.parametrize("language", NON_ENGLISH)
    def test_pack_has_full_guideline_example_palettes(self, language):
        """Every shipped pack covers the FULL guideline-example catalog —
        otherwise the English behavioral palettes (disfluencies,
        confirmations, check-ins, …) leak verbatim into target-language
        calls. Backfill with `tau2 factory draft-localization --lang
        <language>` (upgrade path)."""
        loc = get_language_pack(language).localization
        assert loc is not None, f"pack '{language}' has no localization block"
        assert {e.kind for e in loc.guideline_examples} == set(
            GUIDELINE_EXAMPLE_CATALOG
        ), f"pack '{language}' does not cover the full guideline-example catalog"

    # Languages whose reviewed native convention genuinely reads emails/URLs
    # token by token (comma-separated readouts are correct there). Everyone
    # else reads them as a continuous flow of words — per-token commas are
    # the non-native pattern the comma convention rejects.
    PER_TOKEN_EMAIL_LANGS = {"ja"}

    @pytest.mark.parametrize("language", NON_ENGLISH)
    def test_email_website_readouts_flow_without_commas(self, language):
        """The comma convention: email/website spoken forms are flowing words
        — never 'juan, guion bajo, perez, arroba, gmail, punto, com'.
        Comma+space stays reserved for character-by-character SPELLING (a
        spelled 'doble ve, doble ve, doble ve' is fine), and a clause comma
        in the carrier phrase is natural prose. The per-token tell is a comma
        directly adjacent to a SYMBOL readout ('… , punto, …'): symbols are
        words in the flow, never spelled characters."""
        import re

        if language in self.PER_TOKEN_EMAIL_LANGS:
            pytest.skip("reviewed native convention is genuinely per-token")
        loc = get_language_pack(language).localization
        assert loc is not None, f"pack '{language}' has no localization block"
        # The pack's native readouts, plus the catalog's English symbol names
        # — packs that keep an English loanword readout ('dot', 'underscore')
        # must still flow it without per-token commas.
        symbol_words = [s for r in loc.symbol_readouts for s in r.spoken] + [
            d.english_name for d in SYMBOL_CATALOG.values()
        ]
        for example in loc.spoken_value_examples:
            if example.kind.value not in ("email", "website"):
                continue
            for word in symbol_words:
                pattern = re.compile(
                    rf"[,、،]\s*{re.escape(word)}|{re.escape(word)}\s*[,、،]",
                    re.IGNORECASE,
                )
                assert not pattern.search(example.spoken), (
                    f"pack '{language}' {example.kind.value} readout reads "
                    f"the symbol '{word}' with per-token commas: "
                    f"{example.spoken!r} — re-draft with `tau2 factory "
                    f"draft-localization --lang {language} --force`"
                )

    def test_english_pack_needs_no_localization(self):
        en = get_language_pack("en")
        if en is None:
            pytest.skip("no English pack registered")
        assert en.localization is None


def test_build_user_threads_the_environments_domain(synthetic_pack_env):
    """The REAL build path (runner/build.py) must hand the environment's
    domain to the user sim — a threading regression there renders NO glossary
    in every run while constructor-level tests all stay green."""
    from tau2.registry import registry
    from tau2.runner.build import build_user

    for domain, wanted in [
        ("airline", "benutzerkennung-xx"),
        ("telecom", "roaming-xx"),
    ]:
        environment = registry.get_env_constructor(domain)()
        task = registry.get_tasks_loader(domain)()[0]
        user = build_user(
            "user_simulator",
            environment,
            task,
            llm="dummy",
            persona_config=synthetic_pack_env,
        )
        assert user.domain == domain, domain
        assert wanted in user.system_prompt, domain
