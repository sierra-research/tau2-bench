"""A cut that catches a word mid-sound shows that word on the cut tick, so the
tick path's clip of the line (utils/transcript/views.py utterance groups,
unchanged) has text for the audio it carries."""

import pytest

from tau2.data_model.message import AssistantMessage, Tick
from tau2.utils.transcript.views import utterance_groups
from tau2.voice.audio_native.avartha_pipecat.audio_timing import speech_spans
from tau2.voice.audio_native.avartha_pipecat.cut_token import UNIT_MS_FLOOR
from tau2.voice.audio_native.avartha_pipecat.discrete_time_adapter import (
    AvarthaTickResult,
    DiscreteTimeAvarthaPipecatAdapter,
    _HeardTranscript,
    _next_word,
)
from tau2.voice.audio_native.avartha_pipecat.provider import (
    AgentAudioEvent,
    AgentAudioTimelineEvent,
    AgentLLMStartedEvent,
    AgentPlaybackEvent,
    AgentStartedSpeakingEvent,
    AgentWordEvent,
    SpeechEvent,
    TranscriptEvent,
)

from .test_adapter import FakeProvider
from .test_audio_timing import BINDING, FIRST_RTP, TICK_RTP, _word

LOUD = (6000).to_bytes(2, "little", signed=True)


def _speech(n):
    """``n`` x 20 ms of audible 24 kHz PCM16 agent audio."""
    return AgentAudioEvent(LOUD * 480 * n, 24000)


def _clocked(index):
    """One tick of audible audio placed on the call's RTP clock."""
    start = FIRST_RTP + index * TICK_RTP
    return AgentAudioEvent(LOUD * 480, 24000, rtp_start=start, rtp_decoded_from=start)


def _quiet(n):
    return AgentAudioEvent(b"\x00\x00" * 480 * n, 24000)


def _run(events_by_tick):
    adapter = DiscreteTimeAvarthaPipecatAdapter(
        tick_duration_ms=20, provider=FakeProvider(events_by_tick)
    )
    adapter.connect("policy", [])
    try:
        return [
            adapter.run_tick(b"\xff" * 160, tick_number=tick)
            for tick in range(1, len(events_by_tick) + 1)
        ]
    finally:
        adapter.disconnect()


def _started(result):
    return result.model_dump(serialize_as_any=True).get("started_word")


def _units(result):
    data = result.model_dump(serialize_as_any=True)
    return data.get("started_word_units_heard"), data.get("started_word_units")


# A code read digit by digit is one word whose sound lasts seconds: the cut
# caught it mid-sound, before its end was reported (seen live, pt).
# 1999 ms of it played: at the default unit (400 ms) with the safety margin
# (x 2.3), two digits surely played.
CUT_MID_CODE = (
    [[TranscriptEvent("Card 6941833."), _speech(1), AgentWordEvent("Card")]]
    + [[_speech(1)] for _ in range(100)]
    + [[SpeechEvent("speech_started")]]
)


@pytest.mark.parametrize("late_audio", [False, True])
def test_a_cut_shows_the_digits_it_heard_of_a_code_and_cuts_its_line(late_audio):
    events = [list(tick) for tick in CUT_MID_CODE]
    if late_audio:
        events[-1].append(_speech(1))  # the line's audio still streaming, stale
    ticks = _run(events)
    cut = ticks[-1]

    assert [t.proportional_transcript for t in ticks] == ["Card"] + [""] * 100 + [" 69"]
    assert isinstance(cut, AvarthaTickResult)
    assert _started(cut) == " 69"
    assert _units(cut) == (2, 7)
    assert cut.was_truncated
    assert cut.item_ids == ["avartha-1"]


def test_a_code_cut_in_its_first_unit_shows_nothing():
    """Less than one unit surely played: none is shown, as before a started
    word was shown at all."""

    ticks = _run(
        [
            [TranscriptEvent("Card 6941833."), _speech(1), AgentWordEvent("Card")],
            [_speech(1)],
            [_speech(1)],
            [SpeechEvent("speech_started")],
        ]
    )

    assert "".join(t.proportional_transcript for t in ticks) == "Card"
    assert all(_started(t) is None for t in ticks)


@pytest.mark.parametrize(
    "sentence, shown",
    [
        ("Total 1,387 dollars.", "Total"),  # an amount, read as number words
        ("Flight 1177 today.", "Flight"),  # a plain 4-digit number
        ("Your confirmation number.", "Your"),  # a long word
    ],
)
def test_a_number_or_a_long_word_shows_nothing(sentence, shown):
    ticks = _run(
        [[TranscriptEvent(sentence), _speech(1), AgentWordEvent(shown)]]
        + [[_speech(1)] for _ in range(100)]
        + [[SpeechEvent("speech_started")]]
    )

    assert "".join(t.proportional_transcript for t in ticks) == shown
    assert all(_started(t) is None for t in ticks)


@pytest.mark.parametrize(
    "sentence, shown, word",
    [
        ("User josé_12345 found.", "User", " josé_"),
        ("Email hao.h18836, right?", "Email", " hao."),
    ],
)
def test_an_identifier_is_never_shown_whole(sentence, shown, word):
    """A dotted or accented identifier is read unit by unit: 60 ms of it shows
    nothing, 2 s shows two units."""

    early = _run(
        [[TranscriptEvent(sentence), _speech(1), AgentWordEvent(shown)]]
        + [[_speech(1)] for _ in range(3)]
        + [[SpeechEvent("speech_started")]]
    )
    late = _run(
        [[TranscriptEvent(sentence), _speech(1), AgentWordEvent(shown)]]
        + [[_speech(1)] for _ in range(101)]
        + [[SpeechEvent("speech_started")]]
    )

    assert all(_started(t) is None for t in early)
    assert _started(late[-1]) == word
    assert _units(late[-1])[0] == 2


def test_an_ordinary_word_records_no_units():
    ticks = _run(
        [
            [TranscriptEvent("Your card number."), _speech(1), AgentWordEvent("Your")],
            [_speech(1)],
            [_speech(1)],
            [_speech(1)],
            [SpeechEvent("speech_started")],
        ]
    )

    assert _started(ticks[4]) == " card"
    assert _units(ticks[4]) == (None, None)


def test_the_line_s_tick_group_holds_the_word_the_cut_caught():
    ticks = [
        Tick(
            tick_id=index,
            timestamp="",
            tick_duration_seconds=0.02,
            agent_chunk=AssistantMessage(
                role="assistant",
                content=result.proportional_transcript or None,
                contains_speech=result.agent_audio_bytes > 0,
                raw_data=result.model_dump(serialize_as_any=True),
                utterance_ids=result.item_ids or None,
            ),
        )
        for index, result in enumerate(_run(CUT_MID_CODE))
    ]

    (group,) = utterance_groups(ticks, "agent")
    assert (group.start_index, group.end_index) == (0, 101)
    assert AssistantMessage.merge_chunks(group.chunks).content == "Card 69"


def test_only_a_tick_showing_a_started_word_records_one():
    ticks = _run(CUT_MID_CODE)

    for tick in ticks[:-1]:
        data = tick.model_dump(serialize_as_any=True)
        assert not {
            "started_word",
            "started_word_units_heard",
            "started_word_units",
        } & set(data)


def test_without_speech_after_the_last_shown_word_a_cut_shows_no_word():
    ticks = _run(
        [
            [TranscriptEvent("Card 6941833."), _speech(1), AgentWordEvent("Card")],
            [SpeechEvent("speech_started"), _speech(1)],
        ]
    )

    assert ticks[1].was_truncated
    assert ticks[1].proportional_transcript == ""
    assert _started(ticks[1]) is None


def test_speech_that_stopped_before_the_cut_shows_no_word():
    """A burst after the shown word (a breath, a decay) and then quiet audio is
    not a word the cut caught."""

    ticks = _run(
        [
            [TranscriptEvent("Card 6941833."), _speech(1), AgentWordEvent("Card")],
            [_speech(1)],
            [_speech(1)],
            # The first quiet tick still plays the held last millisecond.
            [_quiet(1)],
            [_quiet(1)],
            [_quiet(1), SpeechEvent("speech_started")],
        ]
    )

    assert ticks[5].proportional_transcript == ""
    assert _started(ticks[5]) is None


def test_a_cut_after_a_sentence_end_shows_no_word_of_the_next():
    ticks = _run(
        [
            [
                TranscriptEvent("Done."),
                TranscriptEvent(" Next line."),
                _speech(1),
                AgentWordEvent("Done."),
            ],
            [_speech(1)],
            [_speech(1)],
            [SpeechEvent("speech_started")],
        ]
    )

    assert ticks[3].proportional_transcript == ""
    assert _started(ticks[3]) is None


def test_a_cut_keeping_a_mostly_played_word_records_no_started_word():
    """A word whose end was reported is kept by its timing, not inferred."""

    ticks = _run(
        [
            [
                TranscriptEvent("Card 6941833."),
                _speech(1),
                AgentWordEvent("Card"),
                AgentWordEvent("6941833."),
            ],
            [_speech(1)],
            [SpeechEvent("speech_started")],
        ]
    )

    assert "".join(t.proportional_transcript for t in ticks) == "Card 6941833."
    assert all(_started(t) is None for t in ticks)


def test_caller_speech_after_a_finished_line_shows_no_word():
    ticks = _run(
        [
            [
                TranscriptEvent("Done. Next line."),
                _speech(1),
                AgentWordEvent("Done."),
                AgentPlaybackEvent("Done.", cut=False),
            ],
            [],
            [SpeechEvent("speech_started")],
        ]
    )

    assert not ticks[2].was_truncated
    assert ticks[2].proportional_transcript == ""
    assert _started(ticks[2]) is None


@pytest.mark.parametrize(
    "text, start, word",
    [
        ("Card 6941833. Next", 4, " 6941833."),
        ("Done.", 5, ""),
        ("Card, 6941833.", 4, ", 6941833."),
        ("Card. ", 5, ""),
        ("您好！我来", 0, "您"),
        ("好！我来", 0, "好！"),
        ("카드 번호는", 2, " 번호는"),
        ("PayPal로 보내", 0, "PayPal"),
        ("कार्ड नंबर है", 5, " नंबर"),
        ("号码是7091239、6276644。", 3, "7091239、"),
    ],
)
def test_next_word_is_a_spaced_word_or_one_unspaced_character(text, start, word):
    assert _next_word(text, start) == word


def test_speech_counts_frames_above_the_peak_threshold():
    frame = 160 * 2  # 20 ms of 8 kHz PCM16
    quiet = (20).to_bytes(2, "little", signed=True) * (frame // 2)
    loud = LOUD * (frame // 2)

    assert speech_spans(quiet * 2, 8000) == []
    assert speech_spans(loud * 2 + quiet + loud, 8000) == [(0, 320), (480, 640)]


def test_a_word_whose_end_was_reported_keeps_the_mostly_played_rule():
    """Its end mark came: caught at its start (under half played), it is
    dropped as before, not shown as a started word."""

    ticks = _run(
        [
            [
                AgentAudioTimelineEvent(BINDING),
                TranscriptEvent("Hello world."),
                _clocked(0),
                _word("Hello", FIRST_RTP + 400),
            ],
            [_clocked(1), _word("world.", FIRST_RTP + 12 * TICK_RTP)],
            [_clocked(2)],
            [_clocked(3)],
            [_clocked(4), SpeechEvent("speech_started")],
            [],
        ]
    )

    assert "".join(t.proportional_transcript for t in ticks) == "Hello"
    assert all(_started(t) is None for t in ticks)


def test_after_the_agent_cut_its_own_line_no_word_of_that_reply_is_started():
    """The next reply's first word is not inferred either: the cut line's
    text ends where the agent cut it."""

    ticks = _run(
        [
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("La XEWRD9, su ida y vuelta."),
                AgentStartedSpeakingEvent(),
                _speech(1),
                AgentWordEvent("La"),
            ],
            [_speech(1), AgentWordEvent("XEWRD9,")],
            [AgentPlaybackEvent("La XEWRD9", cut=True)],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Perdona, me ha fallado."),
                AgentStartedSpeakingEvent(),
                _speech(1),
            ],
            [_speech(1)],
            [_speech(1)],
            [SpeechEvent("speech_started")],
        ]
    )

    assert ticks[-1].proportional_transcript == ""
    assert _started(ticks[-1]) is None


@pytest.mark.parametrize(
    "sentence, shown, word",
    [
        ("카드 번호는 1234입니다.", "카드", " 번호는"),
        ("कार्ड नंबर है।", "कार्ड", " नंबर"),
    ],
)
def test_a_cut_mid_word_in_korean_and_hindi_shows_the_next_word(sentence, shown, word):
    ticks = _run(
        [
            [TranscriptEvent(sentence), _speech(1), AgentWordEvent(shown)],
            [_speech(1)],
            [_speech(1)],
            [_speech(1)],
            [SpeechEvent("speech_started")],
        ]
    )

    assert ticks[4].proportional_transcript == word
    assert _started(ticks[4]) == word


def _completed_code_then_cut(code_end_tick, cut_after, *, code="12345"):
    """A timed "Card", a timed ``code`` ending ``code_end_tick`` ticks later
    (its sound is (code_end_tick - 1) x 20 ms), then ``cut_after`` ticks of
    the next code's sound and the cut."""

    events = [
        [
            AgentAudioTimelineEvent(BINDING),
            TranscriptEvent(f"Card {code} 678901."),
            _clocked(0),
            _word("Card", FIRST_RTP + TICK_RTP),
        ]
    ]
    events += [[_clocked(index)] for index in range(1, code_end_tick)]
    events[-1].append(_word(code, FIRST_RTP + code_end_tick * TICK_RTP))
    events += [[_clocked(i)] for i in range(code_end_tick, code_end_tick + cut_after)]
    events += [[SpeechEvent("speech_started")]]
    adapter = DiscreteTimeAvarthaPipecatAdapter(
        tick_duration_ms=20, provider=FakeProvider(events)
    )
    adapter.connect("policy", [])
    try:
        ticks = [
            adapter.run_tick(b"\xff" * 160, tick_number=tick)
            for tick in range(1, len(events) + 1)
        ]
        return ticks, adapter._transcript()._unit_ms()
    finally:
        adapter.disconnect()


def test_completed_codes_measure_the_call_s_unit_duration():
    """A timed code right after a timed word measures a unit: its sound, from
    that word's end to its own, over its units (5 digits in 1 s). A code the
    cut catches later is counted at that unit (x 2.3): 1 s of it played, two
    digits."""

    ticks, unit_ms = _completed_code_then_cut(51, 50)

    assert unit_ms == pytest.approx(200.0)
    assert "".join(t.proportional_transcript for t in ticks) == "Card 12345 67"
    assert _started(ticks[-1]) == " 67"
    assert _units(ticks[-1]) == (2, 6)


def test_a_mis_timed_completed_code_cannot_shrink_the_unit_below_the_floor():
    """5 digits in 100 ms would make a unit 20 ms: 1 s of the next code would
    show all of it. The unit is held at 150 ms: two digits."""

    ticks, unit_ms = _completed_code_then_cut(6, 50)

    assert unit_ms == UNIT_MS_FLOOR == 150.0
    assert _started(ticks[-1]) == " 67"


@pytest.mark.parametrize(
    "code, unit_ms",
    [
        ("6941833", 1000 / 7),  # held at the floor below
        ("HAT110", 1000 / 6),
        ("garcia5901", 200.0),
        # Read as words, so they do not measure: the default stays.
        ("p.m.", 400.0),
        ("24-hour", 400.0),
        ("2GB까지", 400.0),
        ("750ml", 400.0),
        ("10000", 400.0),
        ("128GB", 400.0),
        ("1500W", 400.0),
        ("NEVER", 400.0),
        ("3RK2T9", 1000 / 6),
        ("ivan_rossi_8555", 1000 / 8),
    ],
)
def test_only_tokens_surely_read_by_units_measure_the_unit(code, unit_ms):
    ticks, measured = _completed_code_then_cut(51, 1, code=code)

    shown = "".join(t.proportional_transcript for t in ticks)
    assert shown.startswith(f"Card {code}")  # aired by its timed end
    assert measured == pytest.approx(max(unit_ms, UNIT_MS_FLOOR))


def _code_after(*sounds):
    """ "Card 6941833.", "Card" shown, then the given (ticks, loud) runs of
    20 ms of agent audio, then the cut."""

    events = [[TranscriptEvent("Card, 6941833."), _speech(1), AgentWordEvent("Card,")]]
    for count, loud in sounds:
        events += [[_speech(1) if loud else _quiet(1)] for _ in range(count)]
    return _run(events + [[SpeechEvent("speech_started")]])


def test_quiet_before_a_code_is_no_code_sound():
    """1.9 s of quiet audio after "Card," and 60 ms of the code: no digit."""

    ticks = _code_after((96, False), (3, True))

    assert "".join(t.proportional_transcript for t in ticks) == "Card,"
    assert all(_started(t) is None for t in ticks)


def test_a_code_s_sound_runs_from_its_first_to_its_last_speech():
    """1.2 s of quiet, then 500 ms of the code, 400 ms quiet inside it, and
    60 ms more: its sound is 960 ms, one digit at 400 ms x 2.3 (the time
    since "Card," would count two)."""

    ticks = _code_after((60, False), (25, True), (20, False), (3, True))

    assert _started(ticks[-1]) == " 6"
    assert _units(ticks[-1]) == (1, 7)


def test_noting_audio_not_just_counted_as_played_fails_loudly():
    """An internal invariant: the speech clock notes each played byte once,
    in order, right after get_transcript_for_audio counted it."""

    transcript = _HeardTranscript()
    transcript.add_audio(160)
    transcript.get_transcript_for_audio(160)
    transcript.note_played(b"\xff" * 160)

    with pytest.raises(RuntimeError, match="out of order"):
        transcript.note_played(b"\xff" * 160)
