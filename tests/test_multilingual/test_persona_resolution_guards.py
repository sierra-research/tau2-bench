# Copyright Sierra
"""A run that cannot speak as its caller must die, not substitute a stock voice.

Five simulations of the 4,800-call preference pool were written with
``speech_environment.persona_id`` null and a stock English voice
(``arjun_roy``, ``mamadou_diallo``) — Korean and Mandarin scenarios read in an
Indian-English or West-African-English accent. All five came out of refill
invocations that re-ran a handful of tasks into an existing run directory
while omitting ``--user-persona-id <lang>``. Nothing failed, nothing warned:
the sampler treats "no persona asked for" as "plain English run", so the calls
were synthesized, scored, shipped to Drive, and sampled for human annotation.

Two guards close the class, both here:

- :func:`require_persona_override` — an override that names no registered
  pack, pack persona, or stock voice persona is fatal wherever a persona is
  resolved (voice sampling and the text path both route through it), instead
  of falling through to the stock-persona branch.
- :func:`require_caller_persona` — a run whose tasks come from a localized set
  must be configured with that language's persona. This is the one that
  catches the actual incident: the task ids say ``ko``, the run says nothing.

The legitimate stock-persona path — a plain English run over base task ids
with no override — must keep working untouched, so it is asserted here too.
"""

from __future__ import annotations

import pytest

from tau2.data_model.persona import PersonaConfig
from tau2.data_model.simulation import AudioNativeConfig, TextRunConfig, VoiceRunConfig
from tau2.data_model.voice import SynthesisConfig
from tau2.multilingual.registry import (
    PersonaResolutionError,
    require_caller_persona,
    require_persona_override,
    resolve_task_persona,
)
from tau2.multilingual.schema import MultilingualPersonaConfig
from tau2.user_simulation_voice_presets import sample_voice_config

# The four ko/zh simulations found corrupt in the preference pool, plus the
# English one of the same class (an ``airline_en`` task refilled without
# ``--user-persona-id en``). Each is (task_id, language the run needed).
CORRUPTED_TASKS = [
    ("21_ko_identity", "ko"),
    ("6_ko_identity", "ko"),
    ("32_zh_identity", "zh"),
    ("30_zh_identity", "zh"),
    ("9_en", "en"),
]

# A stock English voice persona: the thing the corrupt sims fell back to.
STOCK_PERSONA = "arjun_roy"


class TestRequirePersonaOverride:
    """The three accepted forms pass; anything else raises."""

    @pytest.mark.parametrize(
        "identifier",
        ["ko", "zh", "en", "hi", "jihun_ko_v1", "imran_hindi_v1", STOCK_PERSONA],
    )
    def test_accepts_language_codes_pack_personas_and_stock_voices(self, identifier):
        require_persona_override(identifier)  # must not raise

    @pytest.mark.parametrize("identifier", ["xx", "nope_xx", "korean", "ko_identity"])
    def test_rejects_anything_else(self, identifier):
        with pytest.raises(PersonaResolutionError, match="Unknown user persona"):
            require_persona_override(identifier)

    def test_rejects_a_language_whose_pack_is_not_installed(self, isolated_pack_env):
        """The failure mode the guard exists for: the pack did not resolve.

        Whatever the reason a pack is missing at lookup time — not installed,
        a discovery failure, a registration race — asking for its language
        must raise rather than resolve to a stock English voice. The isolated
        env has only the toy 'tl' pack, so 'ko' resolves to nothing.
        """
        require_persona_override("tl")  # the pack that IS installed here
        with pytest.raises(PersonaResolutionError, match="Unknown user persona 'ko'"):
            require_persona_override("ko")


class TestVoiceSamplerNeverDegradesToStock:
    """``sample_voice_config`` is where the substitution happened."""

    def test_unresolvable_override_raises_instead_of_sampling_stock(
        self, isolated_pack_env
    ):
        """Pre-fix this returned a config and the run carried on."""
        with pytest.raises(PersonaResolutionError):
            sample_voice_config(
                seed=1094288,
                synthesis_config=SynthesisConfig(),
                complexity="regular",
                persona_name="ko",
                task_id="21_ko_identity",
            )

    def test_unknown_persona_name_raises(self):
        with pytest.raises(PersonaResolutionError, match="Unknown user persona"):
            sample_voice_config(
                seed=42,
                synthesis_config=SynthesisConfig(),
                complexity="regular",
                persona_name="nope_xx",
            )

    def test_language_code_still_resolves_to_a_pack_persona(self):
        config = sample_voice_config(
            seed=42,
            synthesis_config=SynthesisConfig(),
            complexity="regular",
            persona_name="ko",
            task_id="21_ko_identity",
        )
        assert config.persona_name.endswith("_ko_v1")
        assert isinstance(config.persona_config, MultilingualPersonaConfig)
        # The field that was null on all five corrupt sims.
        assert config.to_speech_environment(1094288).persona_id == config.persona_name

    def test_plain_english_stock_path_is_untouched(self):
        """No override, no voice-id pin: still the stock-persona sampling."""
        config = sample_voice_config(
            seed=1094288,
            synthesis_config=SynthesisConfig(),
            complexity="regular",
        )
        assert config.persona_name
        assert not isinstance(config.persona_config, MultilingualPersonaConfig)
        assert isinstance(config.persona_config, PersonaConfig)
        assert config.to_speech_environment(1094288).persona_id is None


class TestTextPathNeverDegradesToStock:
    def test_resolve_task_persona_raises_on_unknown_id(self):
        with pytest.raises(PersonaResolutionError, match="Unknown user persona"):
            resolve_task_persona("nope_xx", task_id="3_hi", domain="airline")

    def test_resolve_task_persona_still_passes_stock_personas_through(self):
        assert resolve_task_persona(STOCK_PERSONA, task_id="3") == STOCK_PERSONA


class TestRequireCallerPersona:
    """Localized tasks impose a language; the run must be configured for it."""

    @pytest.mark.parametrize("task_id,language", CORRUPTED_TASKS)
    def test_the_corrupted_sims_would_now_be_refused(self, task_id, language):
        """Each of the five, replayed: no --user-persona-id on localized tasks."""
        with pytest.raises(PersonaResolutionError) as excinfo:
            require_caller_persona(None, [task_id], "airline")
        message = str(excinfo.value)
        assert "no --user-persona-id" in message
        assert f"{language}: 1 task(s) e.g. {task_id}" in message

    @pytest.mark.parametrize("task_id,language", CORRUPTED_TASKS)
    def test_the_matching_persona_is_accepted(self, task_id, language):
        require_caller_persona(language, [task_id], "airline")

    def test_a_pack_persona_id_of_the_right_language_is_accepted(self):
        require_caller_persona("jihun_ko_v1", ["21_ko_identity"], "airline")

    def test_wrong_language_persona_is_refused(self):
        with pytest.raises(PersonaResolutionError, match="ko: 1 task"):
            require_caller_persona("es", ["21_ko_identity"], "airline")

    def test_stock_persona_on_localized_tasks_is_refused(self):
        with pytest.raises(PersonaResolutionError, match="stock English voice"):
            require_caller_persona(STOCK_PERSONA, ["21_ko_identity"], "airline")

    def test_reports_every_offending_language_not_just_the_first(self):
        with pytest.raises(PersonaResolutionError) as excinfo:
            require_caller_persona(
                None, ["21_ko_identity", "6_ko_identity", "32_zh_identity"], "airline"
            )
        message = str(excinfo.value)
        assert "ko: 2 task(s)" in message
        assert "zh: 1 task(s)" in message

    def test_domain_scopes_the_lookup(self):
        """``<domain>_<lang>`` sets are per domain: telecom ids need telecom."""
        with pytest.raises(PersonaResolutionError):
            require_caller_persona(None, ["0_ko_identity"], "telecom")
        require_caller_persona("ko", ["0_ko_identity"], "telecom")

    def test_unresolvable_override_is_refused_even_on_base_tasks(self):
        with pytest.raises(PersonaResolutionError, match="Unknown user persona"):
            require_caller_persona("nope_xx", ["3"], "airline")


class TestLegitimateEnglishRunsStillPass:
    """The stock-persona path must survive the guard."""

    def test_base_english_tasks_without_an_override(self):
        from tau2.registry import registry

        task_ids = [task.id for task in registry.get_tasks_loader("airline")()]
        require_caller_persona(None, task_ids, "airline")

    def test_base_tasks_with_a_stock_persona_override(self):
        require_caller_persona(STOCK_PERSONA, ["0", "1", "2"], "airline")

    def test_empty_task_list(self):
        require_caller_persona(None, [], "airline")


class TestBatchRunnerRefusesBeforeWriting:
    """The guard sits in the batch write path, which is what wrote the five."""

    @staticmethod
    def _korean_tasks():
        from tau2.registry import registry

        return registry.get_tasks_loader("airline_ko_identity")()[:3]

    def test_voice_run_without_the_persona_dies(self, tmp_path):
        from tau2.runner.batch import run_tasks

        config = VoiceRunConfig(
            domain="airline",
            task_set_name="airline_ko_identity",
            audio_native_config=AudioNativeConfig(),
        )
        with pytest.raises(PersonaResolutionError, match="no --user-persona-id"):
            run_tasks(config, self._korean_tasks(), save_path=tmp_path / "results.json")
        # Nothing was written: the run died before it could persist a call.
        assert not (tmp_path / "results.json").exists()
        assert not (tmp_path / "simulations").exists()

    def test_text_run_without_the_persona_dies(self, tmp_path):
        from tau2.runner.batch import run_tasks

        config = TextRunConfig(domain="airline", task_set_name="airline_ko_identity")
        with pytest.raises(PersonaResolutionError):
            run_tasks(config, self._korean_tasks(), save_path=tmp_path / "results.json")


class TestPersonaProvenanceIsRecorded:
    """A results directory must state which caller it was configured to speak
    as — without it, a refill that drops the flag is invisible after the fact
    (the recorded Info of all four corrupt ko/zh runs said nothing about it).
    """

    def test_info_carries_the_configured_persona(self):
        from tau2.runner.helpers import get_info

        info = get_info(
            VoiceRunConfig(
                domain="airline",
                task_set_name="airline_ko_identity",
                user_persona_id="ko",
                audio_native_config=AudioNativeConfig(),
            )
        )
        assert info.user_persona_id == "ko"

    def test_info_records_none_for_plain_english_runs(self):
        from tau2.runner.helpers import get_info

        assert get_info(TextRunConfig(domain="airline")).user_persona_id is None
