"""Tests for the pre-synthesis pronunciation substitution: the pure swap
function (whole-token, spell-outs untouched, email inheritance, all-caps
case sensitivity, B&B literal matching), the intake map builder
(correct-by-default contents, the mispronounced_term override), and the
runner's domain-keyed dispatch (non-intake domains get no map)."""

from types import SimpleNamespace

import pytest

from tau2.data_model.simulation import SampledComplication
from tau2.domains.intake.complications import COMPLICATION_CATALOG_VERSION
from tau2.domains.intake.pronunciation_map import build_pronunciation_map
from tau2.domains.intake.tasks.banks import load_banks
from tau2.runner.complications import run_pronunciation_map
from tau2.voice.utils.pronunciation_swap import apply_pronunciations

# ---------------------------------------------------------------------------
# The pure swap function
# ---------------------------------------------------------------------------


def test_identity_on_empty_mapping():
    text = "I take Amoxicillin every day."
    assert apply_pronunciations(text, None) == (text, [])
    assert apply_pronunciations(text, {}) == (text, [])
    assert apply_pronunciations("", {"a": "b"}) == ("", [])


def test_whole_token_multi_occurrence():
    swapped, events = apply_pronunciations(
        "Narbut. Yes, Narbut, but not Narbutson.",
        {"Narbut": "NAR-but"},
    )
    assert swapped == "NAR-but. Yes, NAR-but, but not Narbutson."
    assert len(events) == 1
    assert events[0].token == "Narbut"
    assert events[0].replacement == "NAR-but"
    assert events[0].count == 2


def test_case_insensitive_for_name_tokens():
    swapped, events = apply_pronunciations(
        "narbut, NARBUT, and Narbut", {"Narbut": "NAR-but"}
    )
    assert swapped == "NAR-but, NAR-but, and NAR-but"
    assert sum(event.count for event in events) == 3


def test_spell_outs_never_match():
    mapping = {"Narbut": "NAR-but"}
    for spelled in (
        "N-a-r-b-u-t",
        "N, a, r, b, u, t",
        "N. A. R. B. U. T.",
        "capital N, a, r, b, u, t",
    ):
        assert apply_pronunciations(spelled, mapping) == (spelled, [])


def test_emails_inherit_name_swaps():
    swapped, events = apply_pronunciations(
        "It's wendee.narbut@veltamail.com, all lowercase.",
        {"Narbut": "NAR-but", "Wendee": "WEN-dee"},
    )
    assert swapped == "It's WEN-dee.NAR-but@veltamail.com, all lowercase."
    assert {event.token for event in events} == {"wendee", "narbut"}


def test_all_caps_tokens_match_case_sensitively():
    mapping = {"AS": "ay-ess", "IT": "eye-tee", "PPO": "pee-pee-oh"}
    text = "As I said, it is the AS plan, not the PPO one, as it were."
    swapped, events = apply_pronunciations(text, mapping)
    assert swapped == (
        "As I said, it is the ay-ess plan, not the pee-pee-oh one, as it were."
    )
    assert {event.token for event in events} == {"AS", "PPO"}


def test_ampersand_token_is_a_literal_whole_token_match():
    mapping = {"B&B": "bee and bee"}
    swapped, _ = apply_pronunciations("The B&B on Elm Street.", mapping)
    assert swapped == "The bee and bee on Elm Street."
    # A trailing word character breaks the whole-token boundary.
    assert apply_pronunciations("The B&Bs downtown.", mapping) == (
        "The B&Bs downtown.",
        [],
    )


def test_replacements_never_cascade():
    # One pass: a respelling containing another mapping token is not
    # re-scanned.
    swapped, _ = apply_pronunciations(
        "MICE and BAR.", {"MICE": "mice", "BAR": "bar", "ZZ": "zee-zee"}
    )
    assert swapped == "mice and bar."
    again, _ = apply_pronunciations(swapped, {"MICE": "mice", "BAR": "bar"})
    assert again == swapped


def test_deterministic():
    mapping = {"Narbut": "NAR-but", "B&B": "bee and bee", "PPO": "pee-pee-oh"}
    text = "Narbut booked the B&B on his PPO plan."
    assert apply_pronunciations(text, mapping) == apply_pronunciations(text, mapping)


def test_colliding_insensitive_tokens_fail_loud():
    with pytest.raises(ValueError, match="collide case-insensitively"):
        apply_pronunciations("x", {"Narbut": "NAR-but", "narbut": "nar-BOOT"})


# ---------------------------------------------------------------------------
# The intake map builder
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def banks():
    return load_banks()


def test_uncomplicated_map_is_empty(banks):
    """Owner decision 2026-08-26 (audition listen): default TTS readings
    always — no complication means NO swaps, for any bank or the oddball
    table."""
    assert build_pronunciation_map(None) == {}


def _mispronounced_complication(term: str, bad: str) -> SampledComplication:
    return SampledComplication(
        kind="mispronounced_term",
        line="packet display",
        params={"term": term, "respelling": "x", "operator": "vowel_swap"},
        catalog_version=COMPLICATION_CATALOG_VERSION,
        injected=False,
        mispronunciation=bad,
    )


def test_mispronounced_draw_maps_exactly_the_drawn_term(banks):
    term = next(
        p.token
        for entry in banks.medications
        for p in entry.pronunciations
        if p.mispronounced is not None
    )
    bad = "totallyrong"
    mapping = build_pronunciation_map(_mispronounced_complication(term, bad))
    # The drawn term is the map's ONLY entry; the stored variant is already
    # TTS-ready natural orthography and passes through verbatim.
    assert mapping == {term: "totallyrong"}


def test_other_complication_kinds_leave_the_map_alone():
    default = build_pronunciation_map(None)
    other = SampledComplication(
        kind="self_correction",
        line="x",
        params={"decoy_value": "y"},
        catalog_version=COMPLICATION_CATALOG_VERSION,
    )
    assert build_pronunciation_map(other) == default == {}


def test_unknown_term_fails_loud():
    with pytest.raises(ValueError, match="carries no mispronounced"):
        build_pronunciation_map(_mispronounced_complication("NotABankToken", "bad"))


def test_builder_returns_a_fresh_dict_per_call(banks):
    term = next(
        p.token
        for entry in banks.medications
        for p in entry.pronunciations
        if p.mispronounced is not None
    )
    first = build_pronunciation_map(_mispronounced_complication(term, "bad-x"))
    first["Mutated"] = "x"
    assert "Mutated" not in build_pronunciation_map(
        _mispronounced_complication(term, "bad-x")
    )


# ---------------------------------------------------------------------------
# Runner dispatch: non-intake domains untouched
# ---------------------------------------------------------------------------


def test_non_intake_domains_get_no_map():
    for domain in ("airline", "retail", "telecom", "mock", "banking_knowledge"):
        assert run_pronunciation_map(SimpleNamespace(domain=domain), None) is None


def test_intake_dispatch_matches_the_builder():
    assert run_pronunciation_map(
        SimpleNamespace(domain="intake"), None
    ) == build_pronunciation_map(None)
