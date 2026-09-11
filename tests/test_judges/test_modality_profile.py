# Copyright Sierra
"""Per-factor modality profile: catalog assignments + harness filtering.

Guards the Item-3 contract of docs/designs/text-mode-multilingual.md:

- the catalog's modality assignments are conservative and CLOSED: exactly two
  voice-only judge factors (speech-framed rubrics), no text-only judge
  factors, everything else ``both``;
- the deterministic factors carry explicit modality flags (both universal
  factors are voice-only; the zh script factor is text-only);
- pack-assembled judge factors inherit their catalog modality;
- the harness filters factors by the run's mode in BOTH directions: a
  voice-only factor never scores a text transcript and vice versa.
"""

from tau2.data_model.message import AssistantMessage, Tick
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import StructuredUserInstructions, Task, UserScenario
from tau2.judges.nativeness.factors import (
    DEFAULT_FACTORS,
    TEXT_ONLY_FACTORS_BY_LANGUAGE,
    enabled_deterministic_factor_ids_for,
    judge_factors_for,
)
from tau2.judges.nativeness.harness import (
    NativenessJudgeSettings,
    _deterministic_factors_for,
    evaluate_nativeness,
)
from tau2.multilingual.factor_catalog import FactorModality
from tau2.multilingual.nativeness_catalog import JUDGE_FACTOR_CATALOG

NO_JUDGE = NativenessJudgeSettings(llm_judge=False)


def _text_sim(agent_text: str) -> SimulationRun:
    return SimulationRun(
        id="s1",
        task_id="t1",
        start_time="x",
        end_time="y",
        duration=1.0,
        termination_reason="agent_stop",
        messages=[AssistantMessage(role="assistant", content=agent_text)],
    )


def _voice_sim(agent_text: str) -> SimulationRun:
    sim = _text_sim("")
    sim.messages = None
    sim.ticks = [
        Tick(
            tick_id=0,
            timestamp="0",
            agent_chunk=AssistantMessage.voice(
                content=agent_text, utterance_ids=["a1"]
            ),
        )
    ]
    return sim


def _task() -> Task:
    return Task(
        id="t1",
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call="book",
                task_instructions="help",
            )
        ),
    )


# The ONLY judge factors whose rubric polarity is speech-framed. Adding a
# factor here (or flipping one to text-only) is a scoring change: bump
# NATIVENESS_RUBRIC_VERSION and update docs/designs/text-mode-multilingual.md.
VOICE_ONLY_JUDGE_FACTOR_IDS = {"register_diglossia", "discourse_markers"}


class TestCatalogAssignments:
    def test_voice_only_judge_factors_are_exactly_the_reviewed_two(self):
        voice_only = {
            factor_id
            for factor_id, definition in JUDGE_FACTOR_CATALOG.items()
            if definition.modality is FactorModality.VOICE
        }
        assert voice_only == VOICE_ONLY_JUDGE_FACTOR_IDS

    def test_no_text_only_judge_factors(self):
        # Text-only signal lives in deterministic checkers (zh script), not in
        # the judged catalog; new text-only factors are proposals only.
        assert not [
            factor_id
            for factor_id, definition in JUDGE_FACTOR_CATALOG.items()
            if definition.modality is FactorModality.TEXT
        ]

    def test_every_other_judge_factor_defaults_to_both(self):
        for factor_id, definition in JUDGE_FACTOR_CATALOG.items():
            if factor_id not in VOICE_ONLY_JUDGE_FACTOR_IDS:
                assert definition.modality is FactorModality.BOTH, factor_id

    def test_universal_deterministic_factors_are_voice_only(self):
        assert {factor.id: factor.modality for factor in DEFAULT_FACTORS} == {
            "backchannel_frequency": FactorModality.VOICE,
            "email_symbol_verbalization": FactorModality.VOICE,
        }

    def test_text_only_deterministic_factors_are_text_flagged(self):
        for language, factors in TEXT_ONLY_FACTORS_BY_LANGUAGE.items():
            for factor in factors:
                assert factor.modality is FactorModality.TEXT, (language, factor.id)


class TestModalityApplies:
    def test_both_applies_everywhere(self):
        assert FactorModality.BOTH.applies(is_voice=True)
        assert FactorModality.BOTH.applies(is_voice=False)

    def test_voice_and_text_are_exclusive(self):
        assert FactorModality.VOICE.applies(is_voice=True)
        assert not FactorModality.VOICE.applies(is_voice=False)
        assert FactorModality.TEXT.applies(is_voice=False)
        assert not FactorModality.TEXT.applies(is_voice=True)


def _with_voice_only_factors(monkeypatch, language: str = "es") -> None:
    """Inject the two voice-only catalog factors into a kept pack.

    No surviving pack carries them (ar/fr, the original carriers, were
    dropped), so the inheritance and filtering seams are exercised by
    appending rubrics to a registered pack's nativeness block."""
    from tau2.multilingual.registry import get_language_pack
    from tau2.multilingual.schema import NativenessPackFactorRubric

    nativeness = get_language_pack(language).nativeness
    extra = [
        NativenessPackFactorRubric(
            factor_id=factor_id,
            nuance="test nuance",
            native_does="native behavior",
            ai_likely_does="ai behavior",
        )
        for factor_id in ("register_diglossia", "discourse_markers")
    ]
    monkeypatch.setattr(
        nativeness, "judge_factors", [*nativeness.judge_factors, *extra]
    )


class TestPackFactorsInheritModality:
    def test_pack_factors_inherit_voice_only_catalog_modality(self, monkeypatch):
        _with_voice_only_factors(monkeypatch, "es")
        by_id = {factor.id: factor for factor in judge_factors_for("es")}
        assert by_id["register_diglossia"].modality is FactorModality.VOICE
        assert by_id["discourse_markers"].modality is FactorModality.VOICE

    def test_es_factors_apply_to_both_modes(self):
        for factor in judge_factors_for("es"):
            assert factor.modality is FactorModality.BOTH, factor.id


class TestHarnessFiltering:
    def test_voice_run_gets_voice_factors_not_text(self):
        ids = {f.id for f in _deterministic_factors_for("zh", is_voice=True)}
        assert "email_symbol_verbalization" in ids
        assert "script_consistency" not in ids

    def test_text_run_gets_text_factors_not_voice(self):
        ids = {f.id for f in _deterministic_factors_for("zh", is_voice=False)}
        assert ids == {"script_consistency"}

    def test_backchannel_requires_language_spec_and_voice(self):
        assert "backchannel_frequency" in {
            f.id for f in _deterministic_factors_for("en", is_voice=True)
        }
        assert not _deterministic_factors_for("en", is_voice=False)
        # A language without a backchannel spec never gets the factor (every
        # kept pack language has one, so an unregistered code stands in).
        assert "backchannel_frequency" not in {
            f.id for f in _deterministic_factors_for("sw", is_voice=True)
        }

    def test_reviewed_packs_schedule_only_selected_voice_diagnostics(self):
        for language in ("es", "pt", "ko", "zh"):
            assert enabled_deterministic_factor_ids_for(language) == {
                "email_symbol_verbalization"
            }
            assert {
                factor.id
                for factor in _deterministic_factors_for(language, is_voice=True)
            } == {"email_symbol_verbalization"}
        assert enabled_deterministic_factor_ids_for("hi") == set()
        assert _deterministic_factors_for("hi", is_voice=True) == []

    def test_text_run_drops_voice_only_judge_factor_end_to_end(self, monkeypatch):
        _with_voice_only_factors(monkeypatch, "es")
        text_info = evaluate_nativeness(
            _text_sim("hola, ¿en qué puedo ayudarle?"),
            _task(),
            "es",
            "latn",
            settings=NO_JUDGE,
        )
        assert text_info is not None
        assert "discourse_markers" not in {
            check.id for check in text_info.factor_checks
        }

        voice_info = evaluate_nativeness(
            _voice_sim("hola, ¿en qué puedo ayudarle?"),
            _task(),
            "es",
            "latn",
            settings=NO_JUDGE,
        )
        assert voice_info is not None
        # Present on voice (recorded DEFERRED here because the LLM judge is off).
        assert "discourse_markers" in {check.id for check in voice_info.factor_checks}
