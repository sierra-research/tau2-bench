"""The transcript cap: agent text is shown no faster than it is spoken.

Hydra sends each clause's caption just before its audio. These tests pin the
rule: at most floor(rate * seconds played) characters per utterance (with the
proportional rule as a second limit), the held-back tail released only once a
completed reply has played in full, never for a reply cut short, and rate 0
equal to upstream's proportional reveal byte for byte.
"""

import random

import pytest

from tau2.voice.audio_native.smallest.pacing import (
    TRANSCRIPT_REVEAL_CPS_ENV,
    PacedUtteranceTranscript,
    rate_cap_chars,
    resolve_reveal_cps,
)
from tau2.voice.audio_native.smallest.responses import HydraResponses
from tau2.voice.audio_native.tick_result import UtteranceTranscript

BPS = 8000  # 8 kHz u-law, 1 byte per sample
TICK = 1600  # 200 ms
CPS = 15.0  # 3 characters per 200 ms tick
SENTENCE = (
    "I understand you want to change your flight from ATL to PHL to the "
    "cheapest economy class for the next available date."
)  # 120 characters


def paced(cps: float = CPS, text: str = SENTENCE) -> PacedUtteranceTranscript:
    tracker = PacedUtteranceTranscript(
        item_id="r1", reveal_chars_per_s=cps, bytes_per_second=BPS
    )
    tracker.add_transcript(text)
    return tracker


def play(tracker, ticks: int) -> list[str]:
    return [tracker.get_transcript_for_audio(TICK) for _ in range(ticks)]


def test_resolve_reveal_cps_default_explicit_env_and_invalid():
    assert resolve_reveal_cps(env={}) == 15.0
    assert resolve_reveal_cps(env={TRANSCRIPT_REVEAL_CPS_ENV: "  "}) == 15.0
    assert resolve_reveal_cps(env={TRANSCRIPT_REVEAL_CPS_ENV: "0"}) == 0.0
    assert resolve_reveal_cps(env={TRANSCRIPT_REVEAL_CPS_ENV: "17.5"}) == 17.5
    assert resolve_reveal_cps(12, env={TRANSCRIPT_REVEAL_CPS_ENV: "0"}) == 12.0
    for bad in ("fast", "-1", "nan", "inf"):
        with pytest.raises(ValueError, match=TRANSCRIPT_REVEAL_CPS_ENV):
            resolve_reveal_cps(env={TRANSCRIPT_REVEAL_CPS_ENV: bad})
    for bad in (-1.0, float("nan"), True, "15"):
        with pytest.raises(ValueError):
            resolve_reveal_cps(bad)


def test_rate_cap_chars_uses_the_audio_format_rate():
    assert rate_cap_chars(CPS, TICK, BPS) == 3
    assert rate_cap_chars(CPS, 32000, 32000) == 15  # 1 s of 16 kHz PCM16
    assert rate_cap_chars(CPS, 0, BPS) == 0
    assert rate_cap_chars(0.0, TICK, BPS) is None
    with pytest.raises(ValueError):
        rate_cap_chars(CPS, TICK, 0)


def test_text_ahead_of_audio_is_shown_at_the_speaking_rate():
    # Whole sentence received with only its first 200 ms of audio; the rest
    # of the audio arrives in real time, one tick at a time.
    tracker = paced()
    for tick in range(1, 11):
        tracker.add_audio(TICK)
        text = tracker.get_transcript_for_audio(TICK)
        assert text == SENTENCE[3 * (tick - 1) : 3 * tick]
        assert tracker.transcript_chars_shown <= CPS * tracker.audio_bytes_played / BPS


def test_text_and_audio_in_step_are_unchanged_by_the_cap():
    # A voice at 10 chars/s, audio delivered ahead of playback in 1 s bursts:
    # below the cap, so the capped reveal equals the proportional one.
    def run(cps: float) -> list[str]:
        tracker = paced(cps, text="")
        shown = []
        for burst in range(4):
            tracker.add_transcript(SENTENCE[burst * 10 : (burst + 1) * 10])
            tracker.add_audio(5 * TICK)
            shown += play(tracker, 5)
        return shown

    assert run(CPS) == run(0.0)
    assert "".join(run(CPS)) == SENTENCE[:40]


def test_truncated_reply_never_shows_its_unspoken_rest():
    tracker = paced()
    tracker.add_audio(3 * TICK)
    assert "".join(play(tracker, 3)) == SENTENCE[:9]
    tracker.truncated = True
    tracker.completed = True  # a late "completed" must not resurrect it
    assert not tracker.has_spoken_tail
    assert tracker.get_transcript_for_audio(0) == ""
    assert tracker.transcript_chars_shown == 9


def test_completed_reply_releases_its_tail_only_after_all_audio_played():
    # A voice at 20 chars/s (4 per tick) against the 15 chars/s cap.
    tracker = paced(text=SENTENCE[:40])
    tracker.add_audio(10 * TICK)
    first = play(tracker, 8)
    assert "".join(first) == SENTENCE[:24]
    tracker.completed = True
    assert not tracker.has_spoken_tail  # 2 ticks of audio still to play
    last = play(tracker, 2)
    assert "".join(first + last) == SENTENCE[:40]  # tail rides on the last audio
    assert tracker.get_transcript_for_audio(0) == ""


def test_completed_reply_that_already_played_releases_on_a_zero_byte_call():
    tracker = paced(text=SENTENCE[:40])
    tracker.add_audio(5 * TICK)
    assert "".join(play(tracker, 5)) == SENTENCE[:15]
    tracker.completed = True
    assert tracker.has_spoken_tail
    assert tracker.get_transcript_for_audio(0) == SENTENCE[15:40]
    assert not tracker.has_spoken_tail


def test_rate_zero_is_byte_identical_to_the_upstream_reveal():
    rng = random.Random(7)
    for _ in range(200):
        upstream = UtteranceTranscript(item_id="u")
        ours = paced(0.0, text="")
        for _ in range(rng.randint(1, 30)):
            step = rng.random()
            if step < 0.3:
                piece = SENTENCE[: rng.randint(0, 25)]
                upstream.add_transcript(piece)
                ours.add_transcript(piece)
            elif step < 0.6:
                n = rng.randint(0, 4000)
                upstream.add_audio(n)
                ours.add_audio(n)
            else:
                n = rng.randint(0, 2500)
                expected = upstream.get_transcript_for_audio(n)
                assert ours.get_transcript_for_audio(n) == expected
        assert ours.transcript_chars_shown == upstream.transcript_chars_shown


def ledger() -> tuple[HydraResponses, dict]:
    transcripts: dict = {}
    return HydraResponses(transcripts, CPS, BPS), transcripts


def test_ledger_marks_completed_and_cut_short_replies():
    responses, transcripts = ledger()
    for rid, status in (("r1", "completed"), ("r2", "cancelled"), ("r3", "failed")):
        responses.open(rid)
        responses.close(rid, status)
    assert transcripts["r1"].completed and not transcripts["r1"].truncated
    assert transcripts["r2"].truncated and transcripts["r3"].truncated
    responses.open("r4")
    responses.close("r4", None)  # no status: left open
    assert not transcripts["r4"].completed and not transcripts["r4"].truncated


def test_ledger_barge_in_cuts_in_flight_replies_and_keeps_them_cut():
    responses, transcripts = ledger()
    responses.open("r1")
    responses.close("r1", "completed")
    responses.open("r2")
    assert responses.cut() == "r2"  # still streaming: its later audio is skipped
    assert transcripts["r2"].truncated
    assert not transcripts["r1"].truncated  # already done before the cut
    responses.close("r2", "completed")
    assert transcripts["r2"].truncated and not transcripts["r2"].completed
    responses.audio_done("r2")
    responses.open("r3")
    responses.audio_done("r3")
    assert responses.cut() is None  # nothing streaming any more


def test_ledger_keys_events_without_a_response_id():
    responses, transcripts = ledger()
    assert responses.close(None, "completed") is None  # nothing to close
    key = responses.key(None)
    assert key == "smallest_response_1" and responses.streaming == key
    assert responses.key(None) == key
    assert responses.close(None, "completed") == key
    assert transcripts[key].completed
