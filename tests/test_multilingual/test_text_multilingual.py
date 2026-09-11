# Copyright Sierra
"""Text-mode (half-duplex) multilingual: runtime wiring, guidelines, nativeness.

These guard the text-mode counterpart of the voice pipeline:

- ``--user-persona-id`` selects a language-pack persona in TEXT mode (no
  ``--audio-native``): the user simulator gets the localized text guidelines +
  persona pragmatics, and the agent is localized (agent_language_clause +
  localized opening greeting).
- The completed text ``SimulationRun`` carries a ``SpeechEnvironment`` so
  nativeness scoring resolves the run's language (the whole nativeness harness
  is already mode-agnostic).
- Every registered language pack ships text guidelines (coverage guard, with an
  explicit burn-down allowlist for packs not yet backfilled).
- Prerequisite-bug regression: runtime persona guidelines are injected into the
  text user simulator's system prompt (the base guidelines carry the slot).
"""

import pytest

from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.persona import PersonaConfig, Verbosity
from tau2.data_model.simulation import SimulationRun, TextRunConfig
from tau2.evaluator.evaluator import get_simulation_language_info
from tau2.multilingual.registry import (
    get_agent_language_clause,
    get_language_pack,
    list_language_packs,
)
from tau2.registry import registry
from tau2.runner.build import build_text_orchestrator
from tau2.user.user_simulator import UserSimulator

# Packs that predate the text-mode factory (Call C) and have not been backfilled
# with text guidelines yet. Shrink this as each is backfilled; the coverage
# guard requires every pack NOT listed here to ship text guidelines, so it is
# immediately load-bearing for new languages while tracking the burn-down.
TEXT_GUIDELINES_BACKFILL_PENDING = {
    "ar",
    "de",
    "en",
    "fr",
    "hi",
    "it",
    "ja",
    "ko",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "tr",
    "vi",
    "zh",
}


def _airline_hi_task(task_id: str = "7_hi"):
    return next(t for t in registry.get_tasks_loader("airline_hi")() if t.id == task_id)


class TestTextPersonaInjection:
    def test_minimal_verbosity_reaches_text_system_prompt(self):
        """Prerequisite bug: the base text guidelines now carry the
        <PERSONA_GUIDELINES> slot, so runtime persona guidelines are injected
        (previously silently dropped in text mode)."""
        sim = UserSimulator(
            llm="dummy",
            instructions="scenario",
            persona_config=PersonaConfig(verbosity=Verbosity.MINIMAL),
        )
        assert "MINIMAL VERBOSITY" in sim.system_prompt
        assert "<PERSONA_GUIDELINES>" not in sim.system_prompt  # slot replaced

    def test_default_persona_injects_nothing(self):
        """A default persona has no guidelines text -> nothing injected, and the
        slot is still cleanly removed (English text runs unaffected)."""
        sim = UserSimulator(llm="dummy", instructions="scenario")
        assert "<PERSONA_GUIDELINES>" not in sim.system_prompt


class TestTextGuidelinesRouting:
    def test_every_persona_reads_the_english_guidelines(self):
        """The localized text guidelines belong to the retired native
        prompt-language arm: a language-pack persona still reads the ENGLISH
        base guidelines (slot intact), never the pack's translated file."""
        from tau2.multilingual.registry import get_multilingual_persona

        sim = UserSimulator(
            llm="dummy",
            instructions="scenario",
            persona_config=get_multilingual_persona("camila_es_v1"),
        )
        guidelines = sim.global_simulation_guidelines
        assert "<PERSONA_GUIDELINES>" in guidelines
        assert "Chat de Texto" not in guidelines


class TestTextOrchestratorWiring:
    def test_user_persona_id_resolves_language_pack_persona(self):
        from tau2.multilingual.schema import MultilingualPersonaConfig

        config = TextRunConfig(
            domain="airline", agent="llm_agent", user_persona_id="hi"
        )
        orch = build_text_orchestrator(config, _airline_hi_task(), seed=42)

        # The user simulator got the language-pack persona + language signal.
        assert isinstance(orch.user.persona_config, MultilingualPersonaConfig)
        assert orch.user.persona_config.language == "hi"
        assert orch.user.speech_environment is not None
        assert orch.user.speech_environment.language == "hi"
        assert orch.user.speech_environment.persona_id is not None

    def test_agent_is_localized(self):
        config = TextRunConfig(
            domain="airline", agent="llm_agent", user_persona_id="hi"
        )
        orch = build_text_orchestrator(config, _airline_hi_task(), seed=42)

        # Agent carries the run language and injects the pack's language clause.
        assert getattr(orch.agent, "language", None) == "hi"
        assert orch.agent._get_agent_language_clause() is not None
        # Localized opening greeting (not the English default).
        assert orch.first_agent_message is not None
        assert orch.first_agent_message.content != "Hi! How can I help you today?"
        assert (
            orch.first_agent_message.content == get_language_pack("hi").agent_greeting
        )

    def test_english_run_is_unaffected(self):
        """No persona id -> no language, English default greeting, plain user."""
        config = TextRunConfig(domain="airline", agent="llm_agent")
        task = next(iter(registry.get_tasks_loader("airline")()))
        orch = build_text_orchestrator(config, task, seed=42)
        assert getattr(orch.agent, "language", None) is None
        assert orch.first_agent_message is None  # falls back to DEFAULT
        assert orch.user.speech_environment is None

    def test_persona_assignment_is_deterministic_and_balanced(self):
        """Bare language code assigns a pack persona per task via the shared
        round-robin; the same (task, seed) is stable."""
        config = TextRunConfig(
            domain="airline", agent="llm_agent", user_persona_id="hi"
        )
        ids = set()
        for tid in ("0_hi", "1_hi", "2_hi", "3_hi"):
            orch = build_text_orchestrator(config, _airline_hi_task(tid), seed=42)
            ids.add(orch.user.persona_config.persona_id)
        # Two personas in the hi pack, both exercised across contiguous tasks.
        assert len(ids) == 2

    def test_agent_prompt_identifies_the_resolved_personas_locale(self):
        """The agent sees the caller's original locale, not a pack-wide guess."""
        config = TextRunConfig(
            domain="airline", agent="llm_agent", user_persona_id="hi"
        )
        orch = build_text_orchestrator(config, _airline_hi_task(), seed=42)
        persona = orch.user.persona_config
        from tau2.multilingual.varieties import locale_display_name

        assert (
            f"The customer is originally from {locale_display_name(persona.locale)} "
            "and speaks Hindi — expect standard urban Hindustani."
            in orch.agent.system_prompt
        )
        assert persona.locale not in orch.agent.system_prompt  # no raw codes
        assert (
            "not necessarily their current physical location"
            in orch.agent.system_prompt
        )


class TestTextNativenessSignal:
    def test_text_simulation_resolves_language_for_nativeness(self):
        """A text SimulationRun whose user carried a language-pack persona
        records a SpeechEnvironment, so get_simulation_language_info (the signal
        attach_nativeness reads) resolves the run's language + script."""
        config = TextRunConfig(
            domain="airline", agent="llm_agent", user_persona_id="hi"
        )
        orch = build_text_orchestrator(config, _airline_hi_task(), seed=42)
        sim = SimulationRun(
            id="x",
            task_id="7_hi",
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:01:00",
            duration=1.0,
            termination_reason="user_stop",
            messages=[],
            speech_environment=orch.user.speech_environment,
        )
        language, script = get_simulation_language_info(sim)
        assert language == "hi"
        assert script == "deva"


def test_every_pack_ships_text_guidelines_except_pending():
    """Coverage guard: every registered pack must ship text guidelines, except
    the explicit backfill-pending allowlist. New languages are gated
    immediately; the allowlist shrinks as old packs are backfilled."""
    for lang in list_language_packs():
        pack = get_language_pack(lang)
        if lang in TEXT_GUIDELINES_BACKFILL_PENDING:
            continue
        assert pack.guidelines_text_path is not None, (
            f"language pack '{lang}' is not in the backfill-pending allowlist but "
            "ships no text guidelines (guidelines_text_path). Regenerate the pack "
            "through the factory (draft Call C authors them; finalize requires "
            "them), or add it to the allowlist if it is a deliberate "
            "voice-only pack."
        )


def test_backfilled_pack_removed_from_pending_allowlist():
    """es has been backfilled, so it must NOT be in the pending allowlist (keeps
    the allowlist honest as a burn-down list)."""
    assert "es" not in TEXT_GUIDELINES_BACKFILL_PENDING
    assert get_language_pack("es").guidelines_text_path is not None


@pytest.mark.parametrize("language", ["es", "hi", "ko", "pt", "zh"])
def test_text_and_voice_agents_share_the_complete_language_clause(language):
    """Script, register, code-switching, and locale rules have one renderer."""
    pack = get_language_pack(language)
    persona = next(iter(pack.personas.values()))
    expected = get_agent_language_clause(language, persona.locale)
    assert expected and pack.agent_language_clause in expected

    text_agent = LLMAgent(
        tools=[],
        domain_policy="policy",
        llm="dummy",
        language=language,
        locale=persona.locale,
    )
    voice_agent = DiscreteTimeAudioNativeAgent(
        tools=[],
        domain_policy="policy",
        language=language,
        locale=persona.locale,
        # Text agents carry no voice, so the parity property is the language
        # clause; the voice-gender disclosure would trail it otherwise.
        disclose_voice_gender=False,
    )

    assert text_agent._get_agent_language_clause() == expected
    assert voice_agent._get_agent_language_clause() == expected
    assert text_agent.system_prompt.endswith(expected)
    assert voice_agent.system_prompt.endswith(expected)


@pytest.mark.parametrize("language", ["es", "hi", "ko", "pt", "zh"])
def test_text_user_prompt_keeps_pack_pragmatics_and_text_localization(language):
    """Text omits audio mechanics only; multilingual behavioral rules remain."""
    from tau2.multilingual.registry import get_localization_guidelines

    pack = get_language_pack(language)
    persona = next(iter(pack.personas.values()))
    prompt = UserSimulator(
        llm="dummy",
        instructions="scenario",
        persona_config=persona,
        domain="airline",
    ).system_prompt

    for clause in persona.pragmatics_clauses:
        assert clause.strip() in prompt
    localization = get_localization_guidelines(language, mode="text", domain="airline")
    assert localization and localization in prompt
