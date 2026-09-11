# Copyright Sierra
"""Pack invariants, parametrized over ALL discovered language packs.

A new language gets this entire suite for free — no per-language test file
required. Language-specific *content* assertions (specific phrases, voice
ids, register axes) may still live in optional per-language extras files
(see test_hindi_pack.py).
"""

import pytest

from tau2.multilingual.brevity import (
    backchannel_phrase_problems,
    non_directed_phrase_problems,
)
from tau2.multilingual.invariants import script_regex
from tau2.multilingual.loader import load_language_packs
from tau2.multilingual.registry import get_language_pack, list_language_packs
from tau2.multilingual.schema import LanguagePack
from tau2.user.user_simulator import get_global_user_sim_guidelines_voice

CONTROL_TOKENS = ["###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"]

# ---------------------------------------------------------------------------
# The reviewed pure-continuer inventory.
#
# `backchannel_phrases` is the ONE channel whose content is emitted by uniform
# random draw with no context (the timing is LLM-gated, the phrase is not), so
# every entry must be a pure "I'm listening, keep going" continuer — the
# language's "mm-hmm"/"uh-huh". Acknowledgments (zh 好的/明白了, es vale, tr
# tamam …) stay available at TURN level through the localization block's
# `conversational_confirmations` guideline examples, where the simulator picks
# them in context.
#
# Every phrase below was SELECTED from material already authored in that pack
# (the persona's own list, a sibling persona's list, or the pack's
# conversational_confirmations palette). Pinning the exact strings makes this
# the reviewed set of record: changing a pack's continuer is a deliberate
# edit here too, so an arbitrary string cannot drift back in.
# ---------------------------------------------------------------------------
REVIEWED_CONTINUERS: dict[str, dict[str, list[str]]] = {
    "en": {"matt_en_v1": ["mm-hm"], "lisa_en_v1": ["uh-huh"]},
    "es": {"alejandro_es_v1": ["ajá"], "camila_es_v1": ["ajá"]},
    "hi": {"rishika_hindi_v1": ["हम्म"], "imran_hindi_v1": ["हम्म"]},
    "ko": {"jihun_ko_v1": ["음"], "soyeon_ko_v1": ["음음"]},
    "pt": {"ricardo_pt_v1": ["uhum"], "juliana_pt_v1": ["aham"]},
    "zh": {"jianguo_zh_v1": ["嗯"], "xiaomeng_zh_v1": ["嗯嗯"]},
}


def all_language_codes() -> list[str]:
    load_language_packs()
    return sorted(list_language_packs())


def test_every_shipped_pack_dir_loads_and_registers():
    """Every ``data/tau2/multilingual/<lang>/pack.yaml`` on disk validates and
    registers. With ``extra='forbid'`` on the whole pack schema, this is the
    guard that no shipped pack carries an unknown (typo'd) key anywhere —
    the loader raises on the first invalid pack."""
    from tau2.utils import DATA_DIR

    on_disk = sorted(
        path.parent.name
        for path in (DATA_DIR / "tau2" / "multilingual").glob("*/pack.yaml")
    )
    assert on_disk, "no shipped packs found on disk"
    assert on_disk == all_language_codes()


def test_reviewed_continuers_cover_every_pack_exactly():
    """Closed-catalog coverage: no pack is silently un-reviewed, and no stale
    entry survives a pack being renamed or removed."""
    assert set(REVIEWED_CONTINUERS) == set(all_language_codes())


def pack_script_re(pack):
    """Union regex over the scripts declared by the pack's personas."""
    import re

    codes = sorted({p.script for p in pack.personas.values() if p.script})
    assert codes, f"pack '{pack.language}' declares no persona script codes"
    return re.compile("|".join(script_regex(code).pattern for code in codes))


@pytest.mark.parametrize("language", all_language_codes())
class TestPackInvariants:
    def test_guidelines_file_exists_with_slot_and_tokens(self, language):
        """The pack's localized guidelines file is still well-formed.

        Retired-arm material: no run reads it (the English arm serves the
        stock guidelines to every language), and it lives in the pack's
        ``guidelines/`` subdirectory. It stays valid so the file can be
        resurrected as an ablation without re-drafting 17 translations.
        """
        pack = get_language_pack(language)
        assert pack.guidelines_voice_path is not None
        assert pack.guidelines_voice_path.exists()
        text = pack.guidelines_voice_path.read_text()
        assert text != get_global_user_sim_guidelines_voice()
        assert "<PERSONA_GUIDELINES>" in text
        for token in CONTROL_TOKENS:
            assert token in text, f"control token {token} missing/not ASCII"
        assert pack_script_re(pack).search(text), "guidelines not in native script"

    def test_backchannel_level_declared(self, language):
        # Density is a single knob now; every shipped pack declares low/medium/high.
        pack = get_language_pack(language)
        assert pack.backchannel_level is not None
        assert pack.backchannel_level.value in {"low", "medium", "high"}

    def test_backchannel_phrases_are_reviewed_pure_continuers(self, language):
        """The random-draw channel holds only reviewed pure continuers.

        Guards the failure a Mandarin reviewer caught on live runs: the sim
        drew 明白了 ("I understand how to fix it") out of a mixed inventory
        mid-explanation and the agent ended the call.

        Two rails, because a pinned catalog and a bound catch different
        things: the bounds checker (shared with the factory guardrails) stops
        an out-of-shape inventory in ANY pack including one added tomorrow,
        and the pinned set stops an in-bounds but unreviewed phrase drifting
        into a pack that already has a reviewed continuer.
        """
        pack = get_language_pack(language)
        actual = {
            pid: persona.backchannel_phrases
            for pid, persona in pack.personas.items()
            if persona.backchannel_phrases is not None
        }
        problems: list[str] = []
        for persona in pack.personas.values():
            problems.extend(
                backchannel_phrase_problems(
                    persona.persona_id,
                    persona.backchannel_phrases,
                    language=language,
                    script=persona.script,
                )
            )
        assert not problems, "\n".join(problems)
        assert language in REVIEWED_CONTINUERS, (
            f"pack '{language}' has no reviewed continuer entry — add it to "
            "REVIEWED_CONTINUERS, selecting from the pack's own authored "
            "material, or authoring one with `tau2 factory draft-continuers` "
            "when the pack contains none"
        )
        assert actual == REVIEWED_CONTINUERS[language], (
            f"pack '{language}' backchannel_phrases drifted from the reviewed "
            "continuer set; update REVIEWED_CONTINUERS deliberately"
        )

    def test_rates_sane(self, language):
        pack = get_language_pack(language)
        if pack.default_out_of_turn_events_per_minute is not None:
            assert 0 < pack.default_out_of_turn_events_per_minute < 10

    def test_agent_language_clause_keeps_tools_english(self, language):
        pack = get_language_pack(language)
        if pack.agent_language_clause is None:
            pytest.skip("pack keeps the default agent behavior")
        assert "tool calls" in pack.agent_language_clause

    def test_personas_carry_native_script_content(self, language):
        pack = get_language_pack(language)
        script_re = pack_script_re(pack)
        assert pack.personas, "pack has no personas"
        for persona in pack.personas.values():
            assert persona.pragmatics_clauses, (
                f"{persona.persona_id}: no pragmatics clauses"
            )
            text = persona.to_guidelines_text()
            assert text and "PERSONA AND LANGUAGE" in text
            # tts_voice_prompt is voice-DESIGN material — never in the LLM
            # behavioral prompt (the task persona owns attitude/affect).
            if persona.tts_voice_prompt.strip():
                assert persona.tts_voice_prompt.strip() not in text
            assert script_re.search(text), (
                f"{persona.persona_id}: pragmatics clauses carry no inline "
                "native-script examples"
            )
            if persona.backchannel_phrases is not None:
                assert any(script_re.search(p) for p in persona.backchannel_phrases)
            if persona.non_directed_phrases is not None:
                assert all(script_re.search(p) for p in persona.non_directed_phrases)

    def test_non_directed_phrases_are_clipped(self, language):
        """Out-of-turn speech stays away-from-the-mic length in every pack.

        It is spliced into the live call as one uninterrupted burst, so a
        sentence-length phrase reads as a second conversation rather than a
        glance away from the phone (see tau2.multilingual.brevity). The same
        checker gates factory drafts, so a new pack cannot reintroduce the
        long form.
        """
        pack = get_language_pack(language)
        problems: list[str] = []
        for persona in pack.personas.values():
            problems.extend(
                non_directed_phrase_problems(
                    persona.persona_id,
                    persona.non_directed_phrases,
                    language=language,
                    script=persona.script,
                )
            )
        assert not problems, "\n".join(problems)

    def test_pack_schema_round_trip(self, language):
        # Every shipped pack survives dump -> validate unchanged (guards the
        # serialized contract, incl. redrafted pragmatics content).
        pack = get_language_pack(language)
        reloaded = LanguagePack.model_validate(pack.model_dump(mode="json"))
        assert reloaded == pack

    def test_acoustic_preset_references_resolve(self, language):
        pack = get_language_pack(language)
        for persona in pack.personas.values():
            if persona.acoustic_preset_id is not None:
                assert persona.acoustic_preset_id in pack.acoustic_presets

    def test_acoustic_preset_files_exist_on_disk(self, language):
        # Path/subdir resolution: every referenced WAV must resolve under the
        # same verified dirs sample_voice_config looks under (catches e.g.
        # `hi_IN/...` subdir bugs for every locale, not just one).
        from tau2.voice_config import BACKGROUND_NOISE_CONTINUOUS_DIR, BURST_NOISE_DIR

        pack = get_language_pack(language)
        for preset in pack.acoustic_presets.values():
            for bg_file in preset.background_noise_files:
                path = BACKGROUND_NOISE_CONTINUOUS_DIR / bg_file
                assert path.exists(), f"Missing background noise file: {path}"
            for burst_file in preset.burst_noise_files:
                path = BURST_NOISE_DIR / burst_file
                assert path.exists(), f"Missing burst noise file: {path}"
