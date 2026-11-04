"""Preserve the calibrated spell-out and restart detection behavior."""

from tau2.data_model.audio_effects import SpellOutDetection
from tau2.domains.intake.spell_events import SpellEventDetector, value_skeleton


def _detector() -> SpellEventDetector:
    return SpellEventDetector(
        field_skeletons={
            "second_guest_full_name": ["lisasandoval"],
            "contact_phone": ["5550142233"],
        }
    )


def test_detects_comma_separated_letter_run():
    detections = _detector().detect("That's L, I, S, A — Sandoval.")
    assert detections == [
        SpellOutDetection(
            field="second_guest_full_name",
            letters_spoken=4,
            spelled_text="L, I, S, A",
        )
    ]


def test_digit_words_fold_to_digits():
    detections = _detector().detect(
        "It's five, five, five, oh, one, four, two, two, three, three."
    )
    assert len(detections) == 1
    assert detections[0].field == "contact_phone"
    assert detections[0].letters_spoken == 10


def test_runs_for_one_field_merge_within_an_utterance():
    detections = _detector().detect("L, I, S, A... last name S, A, N, D, O, V, A, L.")
    assert len(detections) == 1
    assert detections[0].letters_spoken == 12
    assert " / " in detections[0].spelled_text


def test_unattributed_runs_stamp_nothing():
    assert _detector().detect("That's X, Y, Z, Q as in nothing we track.") == []


def test_plain_speech_is_not_a_spell_out():
    assert _detector().detect("Sure, it's Lisa Sandoval, 555-014-2233.") == []


def test_single_character_is_not_a_run():
    assert _detector().detect("The letter L.") == []


def test_value_skeleton_folds_orthography_and_digit_words():
    assert value_skeleton("Lisa  SANDOVAL!") == "lisasandoval"
    assert value_skeleton("five five five 014-2233") == "5550142233"


def test_restarted_spellout_counts_all_letters_and_the_restart():
    """The spell_correction construct: a faltered attempt (one edit from a
    gold prefix) followed by the complete run — one detection, ALL letters
    counted, restarts=1, runs in utterance order."""
    detections = _detector().detect(
        "L, I, Z — sorry — L, I, S, A, S, A, N, D, O, V, A, L."
    )
    assert detections == [
        SpellOutDetection(
            field="second_guest_full_name",
            letters_spoken=15,
            spelled_text="L, I, Z / L, I, S, A, S, A, N, D, O, V, A, L",
            restarts=1,
        )
    ]


def test_dropped_character_restart_counts():
    # "5, 5, 5, 0, 4" is the prefix "5, 5, 5, 0, 1, 4" with the 1 dropped —
    # a faltered (non-substring) prefix of the phone value.
    detections = _detector().detect(
        "5, 5, 5, 0, 4 — wait, no — 5, 5, 5, 0, 1, 4, 2, 2, 3, 3."
    )
    assert len(detections) == 1
    assert detections[0].field == "contact_phone"
    assert detections[0].letters_spoken == 15
    assert detections[0].restarts == 1


def test_unrelated_run_is_not_a_restart():
    """An unattributed run that is NOT one edit from a gold prefix stays
    unattributed even when a complete run for a field is present."""
    detections = _detector().detect("X, Y, Z — L, I, S, A, S, A, N, D, O, V, A, L.")
    assert len(detections) == 1
    assert detections[0].letters_spoken == 12
    assert detections[0].restarts == 0


def test_faltered_attempt_without_a_complete_run_stamps_nothing():
    """The pairing is per utterance: a faltered attempt alone (the restart
    landing in a later utterance) stamps nothing — an under-count, per the
    detector's stated limits."""
    assert _detector().detect("L, I, Z — sorry.") == []
