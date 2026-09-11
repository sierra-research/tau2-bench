# Copyright Sierra
"""Tier-1 (offline, no-network) tests for the backchannel density knob.

These tests pin the *mechanism* so we stay confident in the knob without
running voice sims or LLM calls:

  - the three levels render distinct, correctly-ordered density values;
  - the correctness GATES are byte-identical across levels (the fix for the
    guardrail failure that let ja/hi/ar fire mid-question / just-started);
  - language surface (continuer phrases, display name) is injected;
  - every shipped language pack declares a valid level (migration is complete);
  - the per-language level assignment matches the intended target table.

The live fire-rate calibration harness that picked these levels
(`tau2 factory backchannel-eval`) is retired — every pack's level is settled
and shipped. Recover it from git history if a level is ever revisited.
"""

import pytest

from tau2.backchannel import (
    BACKCHANNEL_PROFILES,
    BackchannelLevel,
    render_backchannel_prompt,
)
from tau2.multilingual.loader import load_language_packs
from tau2.multilingual.registry import get_language_pack, list_language_packs

ALL_LEVELS = [BackchannelLevel.LOW, BackchannelLevel.MEDIUM, BackchannelLevel.HIGH]

# The four correctness gates that MUST appear in every rendered prompt,
# regardless of level. These are what keep the user from backchanneling
# mid-question, just after another backchannel, or before the agent has said
# anything substantive.
GATE_SUBSTRINGS = [
    "at least",  # min-sentence gate
    "has NOT spoken or backchanneled within",  # refractory gate
    "does NOT contain or end with a question",  # question gate
    # ...which also covers imperatives: "please read the full phone number
    # digit by digit" is not a question, and the question-only wording let the
    # sim backchannel over a direct request for information.
    "a request, or an instruction addressed to the user",
    "has NOT just started",  # just-started gate
    "Say NO if ANY of those fail",
    'Respond with ONLY "YES" or "NO".',
]

# Intended target table (see tau2.backchannel docstring + the matrix audit).
#
# UNIFORM since 2026-07-26: every pack sits at medium. The per-language table
# this replaced was a sequence of single-language retunes (ar high -> medium
# after round-1; zh medium -> high on a native debrief; ja/hi/vi high ->
# medium when telecom's long agent turns overshot, hi at 14.8 continuers/call
# against es/pt's ~3), and each was defensible on its own while the set was
# not: a cross-language comparison could read a difference that came from the
# knob rather than the model. One knob, one setting, and the density question
# moves to annotation.
EXPECTED_PACK_LEVEL = BackchannelLevel.MEDIUM


# --- rendering -------------------------------------------------------------


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_render_keeps_conversation_history_placeholder(level):
    """Downstream .format(conversation_history=...) must still work."""
    prompt = render_backchannel_prompt(level, language_name="Hindi", phrases=["जी"])
    assert "{conversation_history}" in prompt
    # and no other stray unfilled slot
    assert "{" not in prompt.replace("{conversation_history}", "")


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_gates_identical_across_levels(level):
    """The correctness gates appear verbatim at every level."""
    prompt = render_backchannel_prompt(
        level, language_name="Japanese", phrases=["はい"]
    )
    for gate in GATE_SUBSTRINGS:
        assert gate in prompt, f"gate {gate!r} missing at level {level}"


def test_language_surface_injected():
    prompt = render_backchannel_prompt(
        BackchannelLevel.HIGH, language_name="Vietnamese", phrases=["dạ", "vâng", "ừm"]
    )
    assert "Vietnamese-speaking listener" in prompt
    assert '"dạ"' in prompt and '"vâng"' in prompt


def test_only_first_four_phrases_shown():
    prompt = render_backchannel_prompt(
        BackchannelLevel.LOW, language_name="X", phrases=["a", "b", "c", "d", "e"]
    )
    assert '"e"' not in prompt  # only first 4 used as examples


def test_render_accepts_string_level():
    assert render_backchannel_prompt(
        "high", language_name="X", phrases=["x"]
    ) == render_backchannel_prompt(
        BackchannelLevel.HIGH, language_name="X", phrases=["x"]
    )


def test_phrases_none_falls_back_to_english_continuers():
    prompt = render_backchannel_prompt(
        BackchannelLevel.LOW, language_name="X", phrases=None
    )
    assert "uh-huh" in prompt


# --- level differentiation + monotonicity ----------------------------------


def test_levels_render_distinct_prompts():
    rendered = {
        lv: render_backchannel_prompt(lv, language_name="X", phrases=["x"])
        for lv in ALL_LEVELS
    }
    assert len({*rendered.values()}) == 3


def test_profiles_differ_per_level():
    """Each level's density axes are distinct — asserted on the structured
    BackchannelProfile fields, not rendered prose, so it survives template
    rewording (the previous version pinned literal strings and was brittle)."""
    profs = {lv: BACKCHANNEL_PROFILES[lv] for lv in ALL_LEVELS}
    # target frequency distinct across all three levels
    assert len({p.target_per_n for p in profs.values()}) == 3
    # eagerness distinct across all three levels
    assert len({p.eagerness for p in profs.values()}) == 3
    # trigger threshold relaxes from low (more sentences) toward high (fewer)
    assert profs[BackchannelLevel.LOW].min_sentences_word == "two"
    assert profs[BackchannelLevel.HIGH].min_sentences_word == "one"


def test_profile_frequency_monotonic():
    """Higher level => denser (smaller N in '1 per N sentences')."""

    def lo(level):
        return int(BACKCHANNEL_PROFILES[level].target_per_n.split("-")[0])

    assert (
        lo(BackchannelLevel.HIGH)
        < lo(BackchannelLevel.MEDIUM)
        < lo(BackchannelLevel.LOW)
    )


def test_profiles_cover_all_levels():
    assert set(BACKCHANNEL_PROFILES) == set(ALL_LEVELS)


# --- shipped packs ----------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_packs():
    load_language_packs()
    return list_language_packs()


def test_every_pack_declares_a_level(loaded_packs):
    """Migration is complete: no pack relies on the old prose path."""
    missing = [
        code
        for code in loaded_packs
        if get_language_pack(code).backchannel_level is None
    ]
    assert not missing, f"packs without backchannel_level: {missing}"


def test_pack_levels_match_target_table(loaded_packs):
    """EVERY pack, not a fixed list: a new pack shipping at another density
    is the thing this guards, and a table would not have seen it."""
    off_target = {
        code: get_language_pack(code).backchannel_level
        for code in loaded_packs
        if get_language_pack(code).backchannel_level != EXPECTED_PACK_LEVEL
    }
    assert not off_target, f"packs off {EXPECTED_PACK_LEVEL.value}: {off_target}"
