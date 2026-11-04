"""Explicit spelling pronunciation and unchanged ordinary TTS input."""

from pathlib import Path

import pytest

from tau2.data_model.voice import SpelledEntityTables
from tau2.voice.spelled_entity_normalizer import normalize_spelled_entities


@pytest.fixture(scope="module")
def pt_tables():
    return SpelledEntityTables.model_validate_json(
        (
            Path(__file__).parent / "fixtures" / "pt_spelled_entity_tables.json"
        ).read_text()
    )


class TestSpellingExpansions:
    """Only character-separated spelling runs expand into native words."""

    def test_pt_id_spelling_run(self, pt_tables):
        assert (
            normalize_spelled_entities("É E, H, G, L, P, 3.", pt_tables)
            == "É é, agá, gê, éli, pê, três."
        )

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("V, A, A, O, X, J", "vê, á, á, ó, xis, jota"),
            ("H. A. T. 1. 2. 3", "agá, á, tê, um, dois, três"),
            ("4, 8, 1, 2, 3", "quatro, oito, um, dois, três"),
        ],
    )
    def test_explicit_codes_and_digits(self, text, expected, pt_tables):
        assert normalize_spelled_entities(text, pt_tables) == expected

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
        "Espere 120 segundos.",
        "São 300 passageiros.",
        "Meu número é 48123.",
        "Meu ID é mia_li_3668.",
        "Reserva VAAOXJ, voo HAT123, código 4WQ150.",
        "Modelo X-1-Pro.",
        "Código A-1-2026.",
        "Código REF-A-B.",
        "Meu telefone é 555-123-4567.",
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

    def test_long_code_untouched(self, pt_tables):
        text = "O código é VAAOXJZ hoje."
        assert normalize_spelled_entities(text, pt_tables) == text


class TestNativeLetterNameRuns:
    """Already-spelled native letter names receive reviewed PT-BR TTS hints."""

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


def test_incomplete_runtime_normalization_tables_are_rejected(pt_tables):
    raw = pt_tables.model_dump()
    del raw["letter_names"]["Q"]
    with pytest.raises(ValueError, match="uppercase A-Z"):
        SpelledEntityTables.model_validate(raw)


@pytest.mark.parametrize("language", ["pt", "en"])
def test_normalization_revision_is_recorded_and_checked_on_resume(language):
    from tau2.config import SPELLING_NORMALIZATION_VERSION
    from tau2.data_model.run_config import AudioNativeConfig
    from tau2.data_model.simulation import Info, VoiceRunConfig
    from tau2.multilingual.runtime import validate_language_resume
    from tau2.runner.helpers import get_info

    current = get_info(
        VoiceRunConfig(
            domain="retail",
            language=language,
            num_tasks=1,
            audio_native_config=AudioNativeConfig(),
        )
    )
    context = next(iter(current.language_tasks.values()))
    expected = SPELLING_NORMALIZATION_VERSION if language == "pt" else None
    assert context.spelling_normalization_version == expected
    assert Info.model_validate_json(current.model_dump_json()) == current

    previous = current.model_dump(mode="json")
    for entry in previous["language_tasks"].values():
        entry.pop("spelling_normalization_version")
    previous = Info.model_validate(previous)
    if language == "pt":
        with pytest.raises(ValueError, match="language/task/persona content"):
            validate_language_resume(previous, current)
    else:
        validate_language_resume(previous, current)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Espere 120 segundos. Código VAAOXJ.", "Espere 120 segundos. Código VAAOXJ."),
        ("Soletrando: R, E, N, A.", "Soletrando: érri, é, êni, á."),
        ("Soletrando: érre, é, êne, á.", "Soletrando: érri, é, êni, á."),
    ],
)
def test_tts_boundary_keeps_original_transcript(monkeypatch, pt_tables, raw, expected):
    pytest.importorskip("elevenlabs")
    from types import SimpleNamespace

    from tau2.agent.base.voice import VoiceMixin
    from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
    from tau2.data_model.message import UserMessage
    from tau2.data_model.voice import SpeechEnvironment, SynthesisConfig, VoiceSettings

    captured = []

    def synthesize(text, provider, provider_config):
        captured.append((text, provider_config.language_code))
        return AudioData(
            data=b"\x00\x00" * 160,
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
        )

    monkeypatch.setattr("tau2.agent.base.voice.synthesize_voice", synthesize)
    config = SynthesisConfig(spelled_entity_tables=pt_tables)
    config.provider_config.language_code = "pt"
    config.speech_effects_config.enable_vocal_tics = False
    config.speech_effects_config.enable_non_directed_phrases = False
    mixin = VoiceMixin(
        voice_settings=VoiceSettings(
            synthesis_config=config, speech_environment=SpeechEnvironment()
        )
    )
    result = mixin.synthesize_voice(
        UserMessage(role="user", content=raw),
        state=SimpleNamespace(noise_generator=None),
        add_background_noise=False,
        add_burst_noise=False,
        add_telephony_format=False,
        add_channel_effects=False,
        apply_speech_effects=False,
    )
    assert captured == [(expected, "pt")]
    assert result.content == raw
    assert result.audio_script_gold == expected


@pytest.mark.parametrize("language", ["pt", "en"])
def test_prepared_context_requires_current_normalization_revision(
    language, monkeypatch
):
    from tau2.config import SPELLING_NORMALIZATION_VERSION
    from tau2.data_model.run_config import AudioNativeConfig
    from tau2.data_model.simulation import VoiceRunConfig
    from tau2.multilingual import runtime

    config = VoiceRunConfig(
        domain="retail", language=language, audio_native_config=AudioNativeConfig()
    )
    loaded = runtime.load_language(language, "retail", mode="voice")
    task = loaded.tasks[0]
    context = runtime.resolve_task(loaded, task, config.seed or 0)

    def forbidden(*args, **kwargs):
        pytest.fail("Prepared calls must preserve their resolved persona assignment")

    # Prepared calls still verify the complete DB/policy snapshot before execution.
    monkeypatch.setattr(runtime, "resolve_task", forbidden)
    assert runtime.task_context(config, task, context=context) == context
    revisions = (
        [None, "outdated"]
        if language == "pt"
        else [SPELLING_NORMALIZATION_VERSION, "outdated"]
    )
    for revision in revisions:
        stale = context.model_copy(update={"spelling_normalization_version": revision})
        with pytest.raises(ValueError, match="spelling normalization revision"):
            runtime.task_context(config, task, context=stale)
