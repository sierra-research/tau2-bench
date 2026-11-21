"""Lines that played but never showed text: words spoken before their
sentence's text came, and a line the agent cut itself."""

import pytest

from tau2.voice.audio_native.avartha_pipecat.discrete_time_adapter import (
    _SpokenTranscript,
)
from tau2.voice.audio_native.avartha_pipecat.provider import (
    AgentAudioTimelineEvent,
    AgentLLMStartedEvent,
    AgentPlaybackEvent,
    AgentStartedSpeakingEvent,
    AgentWordEvent,
    SpeechEvent,
    TranscriptEvent,
)

from .test_adapter import _audio_ticks, _played, _transcripts
from .test_audio_timing import BINDING, FIRST_RTP, TICK_RTP, _audio, _word


def test_words_spoken_before_their_sentence_s_text_place_when_it_comes():
    """Seen live (zh): "抱", "歉", "刚", "才", "听"
    were spoken before the sentence's text came. Dropped, they left the
    cursor behind, every later syllable matched nothing, and the cut line
    showed nothing for 8.6 s of speech."""

    shown = _transcripts(
        [
            [
                AgentLLMStartedEvent(),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                *[AgentWordEvent(s) for s in ("抱", "歉", "刚", "才", "听")],
            ],
            [_audio_ticks(1), TranscriptEvent("抱歉刚才听岔了。您最新的拼读。")],
            [_audio_ticks(1), *[AgentWordEvent(s) for s in ("岔", "了。", "您")]],
            [_audio_ticks(1), SpeechEvent("speech_started")],
        ]
    )

    assert shown == ["", "抱歉刚才听", "岔了。您", ""]


def test_a_word_spoken_before_any_text_is_not_matched_after_its_line_ended():
    shown = _transcripts(
        [
            [_audio_ticks(1), AgentWordEvent("Stray")],
            [_audio_ticks(1), _played("")],
            [AgentLLMStartedEvent(), TranscriptEvent("Stray dogs are fine.")],
            [_audio_ticks(1)],
        ]
    )

    assert shown == ["", "", "", ""]


def test_after_the_agent_cuts_its_own_line_the_next_reply_shows_its_words():
    """Seen live (es): the agent stopped "La XEWRD9, su
    ida y vuelta..." after "La XEWRD9" (a caption with ``cut``, no caller
    speech) and replied again. The unplayed rest of the cut reply stayed
    ahead of the cursor, so none of the next line's 9 s of words showed."""

    shown = _transcripts(
        [
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("La XEWRD9, su ida y vuelta."),
                TranscriptEvent(" No admite cambios."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("La"),
            ],
            [_audio_ticks(1), AgentWordEvent("XEWRD9,")],
            [AgentPlaybackEvent("La XEWRD9", cut=True)],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Perdona, me ha fallado el audio."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Perdona,"),
            ],
            [_audio_ticks(1), AgentWordEvent("me"), AgentWordEvent("ha")],
        ]
    )

    assert shown == ["La", " XEWRD9,", "", " Perdona,", " me ha"]


def test_a_self_cut_keeps_the_next_reply_s_text_that_came_first():
    shown = _transcripts(
        [
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("One two three four."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("One"),
            ],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent(" Sorry, again."),
                AgentPlaybackEvent("One", cut=True),
            ],
            [
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Sorry,"),
                AgentWordEvent("again."),
            ],
        ]
    )

    assert shown == ["One", "", " Sorry, again."]


def test_a_late_caption_of_a_caller_cut_line_is_not_shown_again():
    """Seen live (es): after a barge-in the cut line's
    caption came after the next line started speaking. Read as that line's,
    it put the whole cut reply back into the shown text."""

    shown = _transcripts(
        [
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Buenos días. Para localizar su cuenta."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Buenos"),
            ],
            [_audio_ticks(1), AgentWordEvent("días.")],
            [SpeechEvent("speech_started")],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Perdone, decía algo?"),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentPlaybackEvent("Buenos días. Para", cut=True),
                AgentWordEvent("Perdone,"),
            ],
            [_audio_ticks(1), AgentWordEvent("decía")],
        ]
    )

    assert shown == ["Buenos", " días.", "", " Perdone,", " decía"]


def test_a_self_cut_caption_with_unsent_text_keeps_the_next_reply():
    """The caption adds text the sentences lacked ("2 3") before the drop."""

    shown = _transcripts(
        [
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("One two three four."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("One"),
            ],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent(" Sorry, again."),
                AgentPlaybackEvent("One 2 3", cut=True),
            ],
            [
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Sorry,"),
                AgentWordEvent("again."),
            ],
        ]
    )

    assert "".join(shown[2:]) == " Sorry, again."


def test_a_word_left_after_its_line_s_caption_does_not_open_the_next_reply():
    shown = _transcripts(
        [
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Okay, over there."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Okay,"),
                AgentWordEvent("over"),
            ],
            [
                _audio_ticks(1),
                AgentPlaybackEvent("Okay, over there.", cut=False),
                AgentWordEvent("there"),
            ],
            [],
            [AgentLLMStartedEvent(), TranscriptEvent("There is a fee.")],
            [],
        ]
    )

    assert shown[3:] == ["", ""]


def test_a_timed_word_before_its_text_still_waits_for_its_sound():
    shown = _transcripts(
        [
            [
                AgentAudioTimelineEvent(BINDING),
                AgentLLMStartedEvent(),
                _audio(0),
                _word("Hello", FIRST_RTP + 2 * TICK_RTP + 100),
            ],
            [TranscriptEvent("Hello world."), _audio(1)],
            [_audio(2)],
            [_audio(3)],
            [_audio(4)],
        ]
    )

    # Its end lies in the third tick's audio: placed on arrival of its text it
    # would show on the second.
    assert shown[:3] == ["", "", "Hello"]


def test_a_self_cut_after_a_caller_cut_line_never_captioned_still_drops():
    """A cut line's caption is known by its text, not by order: when it
    never comes, the next line's own cut caption is still a self-cut."""

    shown = _transcripts(
        [
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Buenos días. Para localizar."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Buenos"),
            ],
            [SpeechEvent("speech_started")],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Perdone, decía algo? Otra frase larga."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Perdone,"),
            ],
            [AgentPlaybackEvent("Perdone,", cut=True)],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Un momento."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Un"),
                AgentWordEvent("momento."),
            ],
            [_audio_ticks(1)],
        ]
    )

    assert "".join(shown[4:]) == " Un momento."


def test_two_caller_cut_lines_late_captions_are_both_ignored():
    shown = _transcripts(
        [
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Uno dos tres."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Uno"),
            ],
            [SpeechEvent("speech_started")],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Cuatro cinco seis."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Cuatro"),
            ],
            [SpeechEvent("speech_started")],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Siete ocho."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentPlaybackEvent("Uno dos", cut=True),
                AgentPlaybackEvent("Cuatro cinco", cut=True),
                AgentWordEvent("Siete"),
            ],
            [_audio_ticks(1), AgentWordEvent("ocho.")],
        ]
    )

    assert "".join(shown[4:]) == " Siete ocho."


def test_a_self_cut_caption_never_matches_into_the_next_reply_s_text():
    """Avartha sends a reply's first sentence without a leading space; the
    caption's unsent tail must not land inside the next reply."""

    shown = _transcripts(
        [
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Your booking is confirmed."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                *[AgentWordEvent(w) for w in ("Your", "booking", "is", "confirmed.")],
            ],
            [_audio_ticks(1), AgentWordEvent("and")],
            [AgentLLMStartedEvent(), TranscriptEvent("Second reply here.")],
            [AgentPlaybackEvent("Your booking is confirmed. and your", cut=True)],
            [
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                *[AgentWordEvent(w) for w in ("Second", "reply", "here.")],
            ],
            [_audio_ticks(1)],
        ]
    )

    assert "Second reply here." in "".join(shown[4:])


@pytest.mark.parametrize("words_matched", [False, True])
def test_one_caption_over_a_reply_resumed_after_a_tool_call(words_matched):
    """A reply resumed after a tool call is a second LLM run spoken in the
    same utterance: its caption spans both runs' text."""

    resumed = [_audio_ticks(1)]
    if words_matched:
        resumed += [AgentWordEvent(w) for w in ("Your", "booking", "is", "confirmed.")]
    shown = _transcripts(
        [
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Let me check."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Let"),
            ],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent(" Your booking is confirmed."),
                _audio_ticks(1),
                AgentWordEvent("me"),
                AgentWordEvent("check."),
            ],
            resumed,
            [
                _audio_ticks(1),
                _played("Let me check. Your booking is confirmed."),
            ],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Anything else?"),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Anything"),
                AgentWordEvent("else?"),
            ],
        ]
    )

    assert "".join(shown) == "Let me check. Your booking is confirmed. Anything else?"


def test_a_caption_no_cut_line_owes_ends_the_owed_ones():
    """Captions come in playback order: after one that no cut line owes, an
    old cut line that never sent its caption no longer owes one, so a later
    self-cut opening the same way is still a self-cut."""

    shown = _transcripts(
        [
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Perfecto, su reserva está lista."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Perfecto,"),
            ],
            [SpeechEvent("speech_started")],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Claro, dígame."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Claro,"),
                AgentWordEvent("dígame."),
            ],
            [_audio_ticks(1), _played("Claro, dígame.")],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Perfecto, voy a revisar su vuelo."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Perfecto,"),
            ],
            [AgentPlaybackEvent("Perfecto,", cut=True)],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Un momento."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Un"),
                AgentWordEvent("momento."),
            ],
            [_audio_ticks(1)],
        ]
    )

    assert "".join(shown[6:]) == " Un momento."


@pytest.mark.parametrize("words_matched", [False, True])
def test_a_late_caption_spelled_otherwise_than_its_cut_line_is_still_owed(
    words_matched,
):
    """The caption spells the code out ("X E W R D nueve" for "XEWRD9"), so
    it does not open as the cut line's text did; it does not open as the
    open line's either, so it is the owed one."""

    nxt = [
        AgentLLMStartedEvent(),
        TranscriptEvent("Perdone, decía algo? Dígame."),
        AgentStartedSpeakingEvent(),
        _audio_ticks(1),
    ]
    if words_matched:
        nxt.append(AgentWordEvent("Perdone,"))
    shown = _transcripts(
        [
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Su código es XEWRD9. Gracias."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Su"),
            ],
            [SpeechEvent("speech_started")],
            nxt,
            [
                _audio_ticks(1),
                AgentPlaybackEvent("Su código es X E W R D nueve", cut=True),
            ],
            [_audio_ticks(1), AgentWordEvent("decía"), AgentWordEvent("algo?")],
            [_audio_ticks(1), AgentWordEvent("Dígame.")],
        ]
    )

    assert "".join(shown[2:]).strip() == "Perdone, decía algo? Dígame."


def test_a_caption_spelling_a_character_otherwise_still_ends_its_line():
    """The caption parts from the matched text behind the cursor (’ vs ');
    its rest still lines up with the text after the cursor."""

    shown = _transcripts(
        [
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("I’m checking your booking now, one moment please."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                *[AgentWordEvent(w) for w in ("I’m", "checking", "your", "booking")],
            ],
            [
                _audio_ticks(1),
                _played("I'm checking your booking now, one moment please."),
            ],
            [_audio_ticks(1)],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Found it."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Found"),
                AgentWordEvent("it."),
            ],
            [_audio_ticks(1)],
        ]
    )

    assert "".join(shown) == (
        "I’m checking your booking now, one moment please. Found it."
    )


# Captions come one per line, in playback order, and name no utterance. A
# self-cut caption cuts only its own reply: an earlier line's caption that
# comes late, after a later reply started speaking, leaves that reply's text.


def _line(text, word=None):
    return [
        AgentLLMStartedEvent(),
        TranscriptEvent(text),
        AgentStartedSpeakingEvent(),
        _audio_ticks(1),
        AgentWordEvent(word or text.split()[0]),
    ]


_REST = [_audio_ticks(1), AgentWordEvent("reply"), AgentWordEvent("continues.")]


def test_a_late_self_cut_caption_keeps_the_speaking_reply():
    shown = _transcripts(
        [
            _line("Done."),
            _line("Fresh reply continues."),
            [AgentPlaybackEvent("Done.", cut=True)],
            _REST,
        ]
    )

    assert shown == ["Done.", " Fresh", "", " reply continues."]


def test_two_late_cut_captions_each_find_their_own_line():
    shown = _transcripts(
        [
            _line("One."),
            _line("Two."),
            _line("Fresh reply continues."),
            [AgentPlaybackEvent("One.", cut=True)],
            [AgentPlaybackEvent("Two.", cut=True)],
            _REST,
        ]
    )

    assert shown == ["One.", " Two.", " Fresh", "", "", " reply continues."]


def test_a_late_self_cut_caption_behind_an_owed_one_keeps_the_speaking_reply():
    shown = _transcripts(
        [
            _line("Stop."),
            [SpeechEvent("speech_started")],
            _line("Done."),
            _line("Fresh reply continues."),
            [AgentPlaybackEvent("Stop.", cut=True)],
            [AgentPlaybackEvent("Done.", cut=True)],
            _REST,
        ]
    )

    assert shown == ["Stop.", "", " Done.", " Fresh", "", "", " reply continues."]


def test_late_captions_of_two_lines_a_caller_cut_keep_the_next_reply():
    """Only the newest line owes its caption at the cut; the older line's
    caption is taken for it, and the newer one's is then found nowhere."""

    shown = _transcripts(
        [
            _line("One."),
            _line("Stop."),
            [SpeechEvent("speech_started")],
            _line("Fresh reply continues."),
            [AgentPlaybackEvent("One.", cut=True)],
            [AgentPlaybackEvent("Stop.", cut=True)],
            _REST,
        ]
    )

    assert shown == ["One.", " Stop.", "", " Fresh", "", "", " reply continues."]


def test_a_late_caption_of_a_line_cut_before_its_text_drops_nothing():
    """The caption's text is not found (its line's text never came), so it
    may be an earlier line's and cuts nothing."""

    shown = _transcripts(
        [
            [
                AgentLLMStartedEvent(),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Hello"),
            ],
            [SpeechEvent("speech_started")],
            _line("Fresh response continues."),
            [AgentPlaybackEvent("Hello", cut=True)],
            [_audio_ticks(1), AgentWordEvent("response"), AgentWordEvent("continues.")],
        ]
    )

    assert shown == ["", "", "Fresh", "", " response continues."]


def test_a_self_cut_after_a_line_that_sent_no_caption_finds_its_reply():
    """ "Done." sent no caption, so the line now due starts there; the
    self-cut caption "Old" is found at the next reply and cuts that one.
    (The cursor-based drop of the parent did this too: a guard.)"""

    shown = _transcripts(
        [
            _line("Done."),
            _line("Old reply unplayed tail."),
            [AgentPlaybackEvent("Old", cut=True)],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Fresh response."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Fresh"),
                AgentWordEvent("response."),
            ],
        ]
    )

    assert shown == ["Done.", " Old", "", " Fresh response."]


def test_a_line_that_starts_speaking_twice_still_lets_a_later_self_cut_drop():
    """Speaking starts are not lines: a stalled line starts twice and sends
    one caption."""

    shown = _transcripts(
        [
            _line("Done. ", "Done."),
            [AgentStartedSpeakingEvent(), _audio_ticks(1)],
            [AgentPlaybackEvent("Done.", cut=False)],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("La XEWRD9, su ida y vuelta."),
                TranscriptEvent(" No admite cambios."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("La"),
            ],
            [AgentPlaybackEvent("La", cut=True)],
            _line("Perdona, me ha fallado."),
            [_audio_ticks(1), AgentWordEvent("me"), AgentWordEvent("ha")],
        ]
    )

    assert shown == ["Done.", "", "", "", " La", " Perdona,", " me ha"]


def test_a_self_cut_spelling_a_code_otherwise_after_a_text_less_cut_drops():
    shown = _transcripts(
        [
            [AgentLLMStartedEvent(), AgentStartedSpeakingEvent(), _audio_ticks(1)],
            [_audio_ticks(1), SpeechEvent("speech_started")],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("X E W R D 9, su ida y vuelta."),
                TranscriptEvent(" No admite cambios."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("X"),
            ],
            [AgentPlaybackEvent("XEWRD9", cut=True)],
            _line("Perdona, me ha fallado."),
            [_audio_ticks(1), AgentWordEvent("me"), AgentWordEvent("ha")],
        ]
    )

    assert shown == ["", "", "X", "EWRD9", " Perdona,", " me ha"]


def test_a_ghost_text_while_stale_does_not_make_a_text_less_cut_owe():
    shown = _transcripts(
        [
            [SpeechEvent("speech_started")],
            [TranscriptEvent("Ghost.")],
            [
                AgentLLMStartedEvent(),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Hello"),
            ],
            [SpeechEvent("speech_started")],
            _line("Fresh response continues."),
            [AgentPlaybackEvent("Hello", cut=True)],
            [_audio_ticks(1), AgentWordEvent("response"), AgentWordEvent("continues.")],
        ]
    )

    assert shown == ["", "", "", "", "Fresh", "", " response continues."]


def _spoken(text):
    return [
        AgentLLMStartedEvent(),
        TranscriptEvent(text),
        AgentStartedSpeakingEvent(),
        _audio_ticks(1),
        *[AgentWordEvent(word) for word in text.split()],
    ]


def test_a_late_caption_spelled_otherwise_keeps_the_speaking_reply():
    shown = _transcripts(
        [
            _spoken("Code X E W R D 9."),
            _line("Fresh reply continues."),
            [AgentPlaybackEvent("Code XEWRD9", cut=True)],
            _REST,
        ]
    )

    assert shown[-1] == " reply continues."


def test_a_late_caption_parting_at_an_apostrophe_keeps_the_speaking_reply():
    shown = _transcripts(
        [
            _spoken("I’m done."),
            _line("Fresh reply continues."),
            [AgentPlaybackEvent("I'm done.", cut=True)],
            _REST,
        ]
    )

    assert shown == ["I’m done.", " Fresh", "", " reply continues."]


def test_a_late_caption_is_not_placed_in_a_later_reply_repeating_it():
    """The earlier line's caption parts from its own text at the apostrophe;
    the later reply holds it whole, but not at its start."""

    shown = _transcripts(
        [
            _spoken("I’m done."),
            _line("Well, I'm done. Fresh reply continues."),
            [AgentPlaybackEvent("I'm done.", cut=True)],
            [
                _audio_ticks(1),
                AgentWordEvent("I'm"),
                AgentWordEvent("done."),
                AgentWordEvent("Fresh"),
            ],
        ]
    )

    assert shown == ["I’m done.", " Well,", "", " I'm done. Fresh"]


def test_a_late_caption_found_inside_a_later_reply_does_not_cut_it():
    shown = _transcripts(
        [
            _line("Okay."),
            _line("Okay, I booked it. Anything else?", "Okay,"),
            [AgentPlaybackEvent("Okay", cut=True)],
            [_audio_ticks(1), AgentWordEvent("I"), AgentWordEvent("booked")],
        ]
    )

    assert shown[-1] == " I booked"


@pytest.mark.parametrize(
    "caption, shown_by_caption",
    [("La X E W R D 9", " X E W R D 9"), ("La XEWRD9 eh", " XEWRD9 eh")],
)
def test_a_self_cut_spelled_otherwise_after_a_line_without_caption_cuts_its_reply(
    caption, shown_by_caption
):
    """ "Done." sent no caption; the self-cut caption opens like the next
    reply, so it is placed there and cuts that reply's rest."""

    shown = _transcripts(
        [
            _line("Done."),
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("La XEWRD9, su ida y vuelta."),
                TranscriptEvent(" No admite cambios."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("La"),
            ],
            [AgentPlaybackEvent(caption, cut=True)],
            _line("Perdona, me ha fallado."),
            [_audio_ticks(1), AgentWordEvent("me"), AgentWordEvent("ha")],
        ]
    )

    assert shown == ["Done.", " La", shown_by_caption, " Perdona,", " me ha"]


def test_a_code_spelled_otherwise_after_a_line_without_caption_cuts_its_reply():
    shown = _transcripts(
        [
            _line("Done."),
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Your code is AB123."),
                TranscriptEvent(" Unplayed tail."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Your"),
            ],
            [AgentPlaybackEvent("Your code is A B one two three", cut=True)],
            _spoken("Fresh reply continues."),
        ]
    )

    assert shown[-1] == " Fresh reply continues."


def test_a_caption_after_a_line_without_caption_shows_its_own_line():
    shown = _transcripts(
        [
            _line("Done."),
            _line("Fresh reply continues."),
            [AgentPlaybackEvent("Fresh reply continues.", cut=False)],
        ]
    )

    assert shown == ["Done.", " Fresh", " reply continues."]


@pytest.mark.parametrize(
    "spoken, reply, caption, cut_tick",
    [
        ("Done.", "Different reply unplayed tail.", "Different", ""),
        ("The flight is booked.", "Thanks reply unplayed tail.", "Thanks", ""),
        ("I booked it.", "I can help unplayed tail.", "I can", " can"),
    ],
)
def test_a_self_cut_sharing_a_short_opening_with_a_line_without_caption_cuts_its_own(
    spoken, reply, caption, cut_tick
):
    """The earlier line sent no caption and shares only a letter or a word
    with the self-cut caption, which the later reply holds whole: that reply
    is the one cut, and the next reply shows."""

    shown = _transcripts(
        [
            _spoken(spoken),
            _line(reply),
            [AgentPlaybackEvent(caption, cut=True)],
            _spoken("Fresh response."),
        ]
    )

    assert shown[-2:] == [cut_tick, " Fresh response."]


def test_a_self_cut_spelled_otherwise_sharing_a_letter_with_a_line_without_caption():
    """ "Listo." shares only "L" with "La X E W R D 9": with the cursor in the
    next reply, that is no evidence, and the caption is placed there."""

    shown = _transcripts(
        [
            _spoken("Listo."),
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("La XEWRD9, su ida y vuelta."),
                TranscriptEvent(" No admite cambios."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("La"),
            ],
            [AgentPlaybackEvent("La X E W R D 9", cut=True)],
            _line("Perdona, me ha fallado."),
            [_audio_ticks(1), AgentWordEvent("me"), AgentWordEvent("ha")],
        ]
    )

    assert shown == ["Listo.", " La", " X E W R D 9", " Perdona,", " me ha"]


def test_a_late_caption_placed_nowhere_still_takes_only_its_own_line():
    """ "I'm done." matches no line ("I’m" is spelled otherwise and shares no
    first word), but it is still the oldest line's caption: "Second."'s caption
    then finds its own line and the speaking reply keeps its text."""

    shown = _transcripts(
        [
            _spoken("I’m done."),
            _spoken("Second."),
            _line("Fresh reply continues."),
            [AgentPlaybackEvent("I'm done.", cut=True)],
            [AgentPlaybackEvent("Second.", cut=True)],
            _REST,
        ]
    )

    assert shown[-1] == " reply continues."


def test_a_code_spelled_otherwise_late_leaves_the_next_line_s_caption_its_line():
    """ "XEWRD9" shares only "x" with its line, so it is placed nowhere; the
    next line's start stays where that reply starts, so its self-cut caption
    still finds it and drops its unplayed rest."""

    shown = _transcripts(
        [
            _spoken("X E W R D 9."),
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Fresh reply continues."),
                TranscriptEvent(" Unplayed tail."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Fresh"),
            ],
            [AgentPlaybackEvent("XEWRD9", cut=True)],
            _REST,
            [AgentPlaybackEvent("Fresh reply continues.", cut=True)],
            _spoken("Next one."),
        ]
    )

    assert shown[-1] == " Next one."


def test_a_self_cut_spelled_otherwise_over_a_resumed_reply_cuts_its_rest():
    """One line spoken over two LLM runs (resumed after a tool call); its
    caption spells the code otherwise and runs into the second run."""

    shown = _transcripts(
        [
            [
                *_line("Let me check."),
                AgentWordEvent("me"),
                AgentWordEvent("check."),
            ],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Code ABC-123 is set."),
                _audio_ticks(1),
                AgentWordEvent("Code"),
                AgentWordEvent("ABC-123"),
            ],
            [AgentPlaybackEvent("Let me check. Code ABC123", cut=True)],
            _line("Perdona, me ha fallado."),
            [_audio_ticks(1), AgentWordEvent("me"), AgentWordEvent("ha")],
        ]
    )

    assert shown[-2:] == [" Perdona,", " me ha"]


def test_a_late_caption_sharing_a_letter_with_the_next_reply_does_not_run_into_it():
    shown = _transcripts(
        [
            _spoken("Done."),
            _line("Fresh reply continues."),
            [AgentPlaybackEvent("Done. Fine", cut=True)],
            _REST,
        ]
    )

    assert shown[-1] == " reply continues."


@pytest.mark.parametrize("cut", [False, True])
def test_a_caption_s_unsent_tail_stays_out_of_a_next_reply_not_yet_heard(cut):
    """The unsent "for Monday." opens like the next reply's " For your
    return": that reply has not been heard, so the caption is not over it."""

    transcript = _SpokenTranscript()
    transcript.begin_reply()
    transcript.add_text("Your flight is booked")
    for word in ("Your", "flight", "is", "booked"):
        transcript.add_word(word)
    transcript.begin_reply()
    transcript.add_text(" For your return, call us.")
    end = transcript.add_playback("Your flight is booked for Monday.")
    if cut:
        transcript.drop_unplayed_reply(end)

    assert transcript.text.endswith("For your return, call us.")
    assert transcript.text.startswith("Your flight is booked for Monday.")


def test_a_late_caption_word_only_starting_like_the_next_reply_does_not_run_into_it():
    """ "There" starts like the speaking reply's "The" but is another word."""

    shown = _transcripts(
        [
            _spoken("Done."),
            _line("The reply continues."),
            [AgentPlaybackEvent("Done. There", cut=True)],
            _REST,
        ]
    )

    assert shown[-1] == " reply continues."


def test_a_self_cut_spelling_a_resumed_run_s_first_word_otherwise_cuts_nothing():
    """A known limit: "ABC123" against the resumed run's "ABC-123" could be
    an earlier line's caption, so the run is not cut and the next reply
    waits for its caption, as before self-cuts were handled."""

    shown = _transcripts(
        [
            [
                *_line("Let me check."),
                AgentWordEvent("me"),
                AgentWordEvent("check."),
            ],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("ABC-123 is set."),
                _audio_ticks(1),
                AgentWordEvent("ABC-123"),
                AgentWordEvent("is"),
            ],
            [AgentPlaybackEvent("Let me check. ABC123 is", cut=True)],
            _line("Perdona, me ha fallado."),
            [_audio_ticks(1), AgentWordEvent("me"), AgentWordEvent("ha")],
        ]
    )

    assert shown[-2:] == ["", ""]


def test_a_late_caption_found_whole_into_a_word_of_the_next_reply_does_not_cut_it():
    """ "Done. The" is found whole in "Done. There reply...", ending inside
    "There": it does not run on into that reply."""

    shown = _transcripts(
        [
            _spoken("Done."),
            _line("There reply continues."),
            [AgentPlaybackEvent("Done. The", cut=True)],
            _REST,
        ]
    )

    assert shown[-1] == " reply continues."


@pytest.mark.parametrize("caption", ["Done. The reply con", "Done. The rep"])
def test_a_late_caption_ending_inside_a_word_of_the_speaking_reply_is_placed_nowhere(
    caption,
):
    """Shared whole up to inside "continues" or "reply": neither the whole
    nor the aligned placement takes it, and the speaking reply keeps its
    text."""

    shown = _transcripts(
        [
            _spoken("Done."),
            _line("The reply continues."),
            [AgentPlaybackEvent(caption, cut=True)],
            _REST,
        ]
    )

    assert shown[-1] == " reply continues."


def test_a_self_cut_ending_inside_a_word_of_a_resumed_run_cuts_nothing_of_it():
    """A known limit: "Let me check. Your fli" ends inside a word of the
    resumed run, like an earlier line's late caption would, so it is placed
    nowhere and the next reply waits for its caption, as on #758."""

    shown = _transcripts(
        [
            [
                *_line("Let me check."),
                AgentWordEvent("me"),
                AgentWordEvent("check."),
            ],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Your flight is booked."),
                _audio_ticks(1),
                AgentWordEvent("Your"),
                AgentWordEvent("flight"),
            ],
            [AgentPlaybackEvent("Let me check. Your fli", cut=True)],
            _line("Perdona, me ha fallado."),
            [_audio_ticks(1), AgentWordEvent("me"), AgentWordEvent("ha")],
        ]
    )

    assert shown[-2:] == ["", ""]
