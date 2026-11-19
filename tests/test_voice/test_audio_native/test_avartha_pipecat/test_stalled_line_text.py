"""Lines that played but never showed text: words spoken before their
sentence's text came, and a line the agent cut itself."""

import pytest

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
