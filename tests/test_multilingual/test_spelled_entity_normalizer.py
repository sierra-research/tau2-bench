# Copyright Sierra
"""Pre-TTS spelled-entity normalization: pure-function goldens, schema
validation, the opt-in gate, and the shipped-pack coverage guard.

Portuguese letter names and digits can be mispronounced when spelling IDs.
The mechanism is opt-in per pack: of the surviving benchmark languages only
pt enables it; every other language must be completely untouched.
"""

from types import SimpleNamespace
from typing import Optional

import pytest

from tau2.agent.base.voice import VoiceMixin
from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.audio_effects import (
    ChannelEffectsConfig,
    SourceEffectsConfig,
    SpeechEffectsConfig,
)
from tau2.data_model.message import UserMessage
from tau2.data_model.voice import (
    ElevenLabsTTSConfig,
    SpeechEnvironment,
    SynthesisConfig,
    VoiceSettings,
)
from tau2.data_model.voice_personas import DEFAULT_PERSONA_NAME
from tau2.multilingual.registry import (
    get_language_pack,
    get_spelled_entity_tables,
    list_language_packs,
)
from tau2.multilingual.schema import LocalizationPackConfig, SymbolReadout
from tau2.multilingual.spelled_entity_normalizer import (
    SpelledEntityTables,
    normalize_spelled_entities,
)
from tau2.voice.utils.elevenlabs_utils import elevenlabs_language_code

# Languages whose packs ENABLE spelled-entity normalization. Deliberately a
# frozen guard: enabling a new language requires native-speaker validation of
# its tables first; extend this set in the same PR.
ENABLED_LANGUAGES = {"pt"}


@pytest.fixture(scope="module")
def pt_tables() -> SpelledEntityTables:
    tables = get_spelled_entity_tables("pt")
    assert tables is not None
    return tables


class TestGoldenExpansions:
    """Round-1 failure shapes expand into native words."""

    def test_pt_id_spelling_run(self, pt_tables):
        # Round-1 pt failure family: spelled letters/digits misrendered
        # (S -> 'éfi', 2 -> 'dô'). The run expands to native letter names.
        assert (
            normalize_spelled_entities("É E, H, G, L, P, 3.", pt_tables)
            == "É é, agá, gê, éli, pê, três."
        )

    def test_pt_user_id(self, pt_tables):
        assert (
            normalize_spelled_entities("Meu ID é mia_li_3668.", pt_tables)
            == "Meu ID é êmi, i, á, underline, éli, i, underline, "
            "três, seis, seis, oito."
        )

    def test_pt_reservation_and_flight(self, pt_tables):
        assert (
            normalize_spelled_entities("Reserva VAAOXJ, voo HAT123.", pt_tables)
            == "Reserva vê, á, á, ó, xis, jota, voo agá, á, tê, um, dois, três."
        )
        assert (
            normalize_spelled_entities("Código 4WQ150.", pt_tables)
            == "Código quatro, dáblio, quê, um, cinco, zero."
        )

    def test_pt_digit_string(self, pt_tables):
        # Bare ID-like digit runs (>= 3 digits) expand digit-by-digit.
        assert (
            normalize_spelled_entities("Meu número é 48123.", pt_tables)
            == "Meu número é quatro, oito, um, dois, três."
        )

    def test_hyphen_and_space_spelling_runs(self, pt_tables):
        assert (
            normalize_spelled_entities("E-H-G-L-P", pt_tables) == "é, agá, gê, éli, pê"
        )
        assert normalize_spelled_entities("M I A", pt_tables) == "êmi, i, á"

    def test_idempotent_on_all_goldens(self, pt_tables):
        goldens = [
            "É E, H, G, L, P, 3.",
            "Meu ID é mia_li_3668.",
            "Reserva VAAOXJ, voo HAT123.",
            "Meu número é 48123.",
            "E-H-G-L-P",
        ]
        for text in goldens:
            once = normalize_spelled_entities(text, pt_tables)
            assert normalize_spelled_entities(once, pt_tables) == once


class TestProsePassesThrough:
    """Ordinary prose numbers, dates, amounts, and solitary capitals are
    byte-identical — the deliberately-narrow contract."""

    PT_PROSE = [
        "A passagem custou R$ 1.250,90 no dia 05/08 às 14h30.",
        # Currency word immediately after a digit run is a prose amount.
        "Paguei 250 reais pela passagem.",
        # Space-grouped thousands are a prose number, not an ID.
        "A fatura veio a 1 250 no total.",
        "Vou viajar em maio, dia 17, com duas malas.",
        "Reservei em 2026 para o fim do ano.",
        "Vou esperar 2-3 horas no aeroporto.",
        # Sentence boundary after a gate letter is prose, not spelling.
        "Chegamos ao portão E. O agente disse que sim.",
    ]
    # Solitary capitals must never match (German noun capitalization; the
    # function is generic even though no Germanic pack ships the flag).
    SINGLE_CAPITALS = [
        "Ich buche einen Flug nach Berlin am Montag.",
        "Der Flug geht ab Terminal B heute.",
        "A empresa cancelou o voo.",
    ]

    @pytest.mark.parametrize("text", PT_PROSE + SINGLE_CAPITALS)
    def test_pt_prose_untouched(self, text, pt_tables):
        assert normalize_spelled_entities(text, pt_tables) == text

    def test_seven_char_code_untouched(self, pt_tables):
        # Reservation codes are EXACTLY six characters in the airline DB.
        text = "O código é VAAOXJZ hoje."
        assert normalize_spelled_entities(text, pt_tables) == text


class TestSchemaValidation:
    """Loader-level rules: tables complete whenever present; the flag only
    with complete tables and an '_' readout."""

    LETTERS = {
        chr(c): f"letter-{chr(c).lower()}" for c in range(ord("A"), ord("Z") + 1)
    }
    DIGITS = {str(d): f"digit-{d}" for d in range(10)}
    UNDERSCORE = SymbolReadout(symbol="_", spoken=["underline-xx"])

    def test_partial_letter_table_rejected(self):
        partial = dict(self.LETTERS)
        del partial["Q"]
        with pytest.raises(ValueError, match="letter_names must cover exactly"):
            LocalizationPackConfig(letter_names=partial)

    def test_empty_value_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            LocalizationPackConfig(letter_names={**self.LETTERS, "A": "  "})

    def test_flag_requires_both_tables(self):
        with pytest.raises(ValueError, match="requires complete"):
            LocalizationPackConfig(
                spelled_entity_normalization=True,
                letter_names=self.LETTERS,
                symbol_readouts=[self.UNDERSCORE],
            )

    def test_flag_requires_underscore_readout(self):
        with pytest.raises(ValueError, match="'_' symbol"):
            LocalizationPackConfig(
                spelled_entity_normalization=True,
                letter_names=self.LETTERS,
                digit_names=self.DIGITS,
            )

    def test_flag_off_allows_staged_tables(self):
        config = LocalizationPackConfig(
            letter_names=self.LETTERS, digit_names=self.DIGITS
        )
        assert config.spelled_entity_tables() is None

    def test_flag_on_builds_tables(self):
        config = LocalizationPackConfig(
            spelled_entity_normalization=True,
            letter_names=self.LETTERS,
            digit_names=self.DIGITS,
            symbol_readouts=[self.UNDERSCORE],
        )
        tables = config.spelled_entity_tables()
        assert tables is not None
        assert tables.letter_names["S"] == "letter-s"
        assert tables.digit_names["3"] == "digit-3"
        # The most-natural (first) spoken form of each symbol readout.
        assert tables.symbol_names["_"] == "underline-xx"

    def test_hint_without_letter_names_rejected(self):
        with pytest.raises(ValueError, match="requires a letter_names table"):
            LocalizationPackConfig(letter_name_tts={"S": "letter-ess"})

    def test_hint_key_must_be_a_letter(self):
        with pytest.raises(ValueError, match="must be letters A-Z"):
            LocalizationPackConfig(
                letter_names=self.LETTERS, letter_name_tts={"ß": "scharfes-s"}
            )

    def test_hint_empty_value_rejected(self):
        with pytest.raises(ValueError, match="letter_name_tts values must be"):
            LocalizationPackConfig(
                letter_names=self.LETTERS, letter_name_tts={"S": " "}
            )

    def test_hint_colliding_with_another_letter_name_rejected(self):
        """A hint that IS a letter name would re-match on the next pass."""
        with pytest.raises(ValueError, match="must not repeat a letter_names"):
            LocalizationPackConfig(
                letter_names=self.LETTERS, letter_name_tts={"S": "letter-f"}
            )

    def test_hints_reach_the_tables(self):
        config = LocalizationPackConfig(
            spelled_entity_normalization=True,
            letter_names=self.LETTERS,
            digit_names=self.DIGITS,
            symbol_readouts=[self.UNDERSCORE],
            letter_name_tts={"S": "letter-ess-hint"},
        )
        tables = config.spelled_entity_tables()
        assert tables is not None
        assert tables.spoken_letter("S") == "letter-ess-hint"
        # Un-hinted letters keep the orthographic name.
        assert tables.spoken_letter("A") == "letter-a"


class TestNativeLetterNameRuns:
    """The simulator spells in native words, so capital-based families never
    fire. Brazilian target forms use "érri"/"êni", not "érrê"/"ênê"."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            # A name spelled out mid-turn (task ...24d09eb3, Renata Moreira).
            (
                "Soletrando: érre, é, êne, á, tê, á",
                "Soletrando: érri, é, êni, á, tê, á",
            ),
            # Doubled letters and a bracketed pause inside the run.
            (
                "Vou soletrar: éle, á, érre, i, ésse, ésse, á",
                "Vou soletrar: éli, á, érri, i, éssi, éssi, á",
            ),
            # Digit words are run tokens too, so a letter beside them is
            # still reached; the digits themselves are already correct.
            ("bê, érre, quatro, cá, nove", "bê, érri, quatro, cá, nove"),
            # Sentence-initial capitalization of a letter name.
            ("Érre, é, êne, á", "Érri, é, êni, á"),
        ],
    )
    def test_native_runs_are_hinted(self, text, expected, pt_tables):
        assert normalize_spelled_entities(text, pt_tables) == expected

    @pytest.mark.parametrize(
        "text",
        [
            # The demonstrative and the pronoun are the unaccented words a
            # naive rewrite would eat; neither is a letter name.
            "Não, esse aí não é o meu, ele já cancelou esse plano.",
            # A letter name needs a run: one alone stays orthographic.
            "A letra é ésse, não outra.",
            # Prose that happens to comma-list vocabulary words ('é' is the
            # verb, 'dois' a count) carries no hinted letter, so it is a
            # no-op by construction.
            "É, dois dias atrás, mais ou menos.",
        ],
    )
    def test_prose_untouched(self, text, pt_tables):
        assert normalize_spelled_entities(text, pt_tables) == text

    def test_expanded_capitals_come_out_hinted(self, pt_tables):
        """The capital families emit the TTS form directly, not the
        orthographic one — the two paths agree on what the voice hears."""
        assert normalize_spelled_entities("R, E, N, A", pt_tables) == "érri, é, êni, á"

    def test_idempotent(self, pt_tables):
        once = normalize_spelled_entities("éle, á, érre, i, ésse", pt_tables)
        assert normalize_spelled_entities(once, pt_tables) == once

    def test_tables_without_hints_are_unaffected(self):
        # No surviving pack ships hint-less tables (pt, the only opt-in,
        # carries hints), so the no-hint path is covered synthetically.
        tables = SpelledEntityTables(
            letter_names={
                chr(c): f"letter-{chr(c).lower()}"
                for c in range(ord("A"), ord("Z") + 1)
            },
            digit_names={str(d): f"digit-{d}" for d in range(10)},
            symbol_names={"_": "underline-xx"},
        )
        assert tables.letter_name_tts == {}
        text = "letter-a, letter-b, letter-c"
        assert normalize_spelled_entities(text, tables) == text


class TestPackCoverageGuard:
    """The shipped-pack opt-in surface is pinned explicitly."""

    def test_enabled_languages_exactly_pt(self):
        enabled = {
            lang
            for lang in list_language_packs()
            if get_spelled_entity_tables(lang) is not None
        }
        assert enabled == ENABLED_LANGUAGES

    def test_all_other_packs_completely_untouched(self):
        untouched = set(list_language_packs()) - ENABLED_LANGUAGES
        assert untouched  # sanity: the benchmark has more languages
        for lang in untouched:
            loc = get_language_pack(lang).localization
            if loc is None:
                continue
            assert loc.spelled_entity_normalization is False
            assert loc.letter_names == {}
            assert loc.digit_names == {}
            assert loc.letter_name_tts == {}

    def test_pronunciation_hints_only_where_normalization_runs(self):
        """A hint on a pack with the flag off would be dead data: nothing
        reaches the TTS text. pt is the only language with hints today."""
        hinted = {
            lang
            for lang in list_language_packs()
            if get_language_pack(lang).localization is not None
            and get_language_pack(lang).localization.letter_name_tts
        }
        assert hinted == {"pt"}
        assert hinted <= ENABLED_LANGUAGES


def _make_voice_mixin(language: Optional[str], persona_name: str) -> VoiceMixin:
    """A minimal VoiceMixin host with every audio effect disabled."""
    synthesis_config = SynthesisConfig(
        provider_config=ElevenLabsTTSConfig(),
        source_effects_config=SourceEffectsConfig(
            enable_background_noise=False, enable_burst_noise=False
        ),
        speech_effects_config=SpeechEffectsConfig(
            enable_dynamic_muffling=False,
            enable_vocal_tics=False,
            enable_non_directed_phrases=False,
        ),
        channel_effects_config=ChannelEffectsConfig(enable_frame_drops=False),
    )
    voice_settings = VoiceSettings(
        synthesis_config=synthesis_config,
        speech_environment=SpeechEnvironment(
            persona_name=persona_name, language=language
        ),
    )
    return VoiceMixin(voice_settings=voice_settings)


class TestTtsSeamIntegration:
    """The normalization + language pinning applied at the single TTS seam:
    TTS-bound text is expanded, the stored transcript is not, and the
    provider config carries the pinned language code."""

    RAW = "Meu ID é mia_li_3668."
    EXPANDED = (
        "Meu ID é êmi, i, á, underline, éli, i, underline, três, seis, seis, oito."
    )

    def _synthesize(self, monkeypatch, language: str, persona_name: str):
        captured: dict = {}

        def fake_synthesize_voice(text, provider, provider_config):
            captured["text"] = text
            captured["language_code"] = provider_config.language_code
            return AudioData(
                data=b"\x00\x00",
                format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
            )

        monkeypatch.setattr(
            "tau2.agent.base.voice.synthesize_voice", fake_synthesize_voice
        )
        mixin = _make_voice_mixin(language, persona_name)
        message = mixin.synthesize_voice(
            UserMessage(role="user", content=self.RAW),
            state=SimpleNamespace(noise_generator=None),
            add_background_noise=False,
            add_burst_noise=False,
            add_telephony_format=False,
            add_channel_effects=False,
        )
        return captured, message

    def _persona_for(self, language: str) -> str:
        return sorted(get_language_pack(language).personas)[0]

    def test_pt_normalizes_tts_text_and_keeps_transcript(self, monkeypatch):
        captured, message = self._synthesize(monkeypatch, "pt", self._persona_for("pt"))
        assert captured["text"] == self.EXPANDED
        # The delivered-text record mirrors what the TTS received (the
        # vocal-tic markup prior art); the transcript keeps the original.
        assert message.audio_script_gold == self.EXPANDED
        assert message.content == self.RAW

    def test_flag_off_language_is_never_normalized(self, monkeypatch):
        # es never opted in — for a pack without the flag the text must pass
        # through byte-identical.
        captured, message = self._synthesize(monkeypatch, "es", self._persona_for("es"))
        assert captured["text"] == self.RAW
        assert message.audio_script_gold == self.RAW
        assert message.content == self.RAW

    def test_language_pinning_all_packs(self, monkeypatch):
        # Part A: every pack language pins its own ISO 639-1 code.
        for language in ("pt", "es", "hi", "en"):
            captured, _ = self._synthesize(
                monkeypatch, language, self._persona_for(language)
            )
            assert captured["language_code"] == language

    def test_language_pinning_plain_english_persona(self, monkeypatch):
        # No language pack active (language=None) pins 'en'.
        captured, _ = self._synthesize(monkeypatch, None, DEFAULT_PERSONA_NAME)
        assert captured["language_code"] == "en"


class TestElevenLabsLanguageCode:
    def test_identity_for_pack_codes(self):
        for code in ("pt", "es", "hi", "zh", "en"):
            assert elevenlabs_language_code(code) == code

    def test_none_pins_english(self):
        assert elevenlabs_language_code(None) == "en"
