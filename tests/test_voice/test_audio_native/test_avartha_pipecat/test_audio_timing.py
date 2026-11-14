"""Avartha word-end timing over WebRTC (``avartha-audio-timing/1``)."""

import asyncio
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("aiortc")

from av import AudioFrame  # noqa: E402
from av.audio.resampler import AudioResampler  # noqa: E402

from tau2.voice.audio_native.avartha_pipecat.audio_timing import (  # noqa: E402
    AudioClockBinding,
    WordEndTime,
    parse_audio_timeline,
    parse_word_end,
    unwrap_rtp,
)
from tau2.voice.audio_native.avartha_pipecat.discrete_time_adapter import (  # noqa: E402
    DiscreteTimeAvarthaPipecatAdapter,
)
from tau2.voice.audio_native.avartha_pipecat.provider import (  # noqa: E402
    AgentAudioEvent,
    AgentAudioTimelineEvent,
    AgentLLMStartedEvent,
    AgentPlaybackEvent,
    AgentStartedSpeakingEvent,
    AgentWordEvent,
    AvarthaPipecatProvider,
    SpeechEvent,
    TranscriptEvent,
)
from tau2.voice.audio_native.avartha_pipecat.webrtc import (  # noqa: E402
    _rtp_origin,
    _RtpPosition,
)

from .test_adapter import FakeProvider, _transcripts  # noqa: E402

STREAM = "73dfe9ec-e3b4-4d42-8f74-1076e94be983"
TIMELINE = {
    "type": "audio_timeline",
    "protocol": "avartha-audio-timing/1",
    "stream_id": STREAM,
    "sample_rate_hz": 48000,
    "rtp_timestamp": 1000,
    "media_sample_offset": -312,
}
BINDING = AudioClockBinding(STREAM, 1000, -312)
# The first audio event starts at RTP 1000; each is 20 ms, 960 RTP ticks.
FIRST_RTP = 1000
TICK_RTP = 960


def _audio(index, *, rtp=True, decoded_from=None):
    """The ``index``-th 20 ms of agent audio (index 0 starts at FIRST_RTP)."""

    start = FIRST_RTP + index * TICK_RTP
    return AgentAudioEvent(
        b"\x01\x00" * 480,
        24000,
        rtp_start=start if rtp else None,
        rtp_decoded_from=(start if decoded_from is None else decoded_from)
        if rtp
        else None,
    )


def _word(text, end_rtp, stream=STREAM):
    """A word whose sound ends at RTP ``end_rtp``, timed on BINDING's clock."""

    offset = end_rtp - BINDING.rtp_timestamp + BINDING.media_sample_offset
    return AgentWordEvent(text, WordEndTime(stream, offset))


# --- wire parsing -----------------------------------------------------------


def test_an_audio_timeline_binding_is_parsed_and_validated():
    assert parse_audio_timeline(TIMELINE) == BINDING
    for field, bad in [
        ("protocol", "avartha-audio-timing/2"),
        ("stream_id", "not-a-uuid"),
        ("sample_rate_hz", 24000),
        ("rtp_timestamp", -1),
        ("rtp_timestamp", 1 << 32),
        ("rtp_timestamp", True),
        ("rtp_timestamp", 1.5),
        ("sample_rate_hz", 48000.0),
        ("media_sample_offset", 1 << 53),
        ("media_sample_offset", None),
    ]:
        assert parse_audio_timeline({**TIMELINE, field: bad}) is None, field


def test_a_word_end_is_parsed_and_validated():
    audio_time = {
        "stream_id": STREAM,
        "sample_offset": 38400,
        "sample_rate_hz": 48000,
        "boundary": "word_end",
    }
    assert parse_word_end({"text": "arrive", "audio_time": audio_time}) == (
        WordEndTime(STREAM, 38400)
    )
    assert parse_word_end({"text": "arrive"}) is None
    for field, bad in [
        ("boundary", "word_start"),
        ("sample_rate_hz", 24000),
        ("stream_id", 7),
        ("sample_offset", "38400"),
        ("sample_offset", False),
        ("sample_rate_hz", 48000.0),
    ]:
        data = {"text": "arrive", "audio_time": {**audio_time, field: bad}}
        assert parse_word_end(data) is None, field


def test_word_ends_map_to_rtp_timestamps_across_the_wrap():
    binding = AudioClockBinding(STREAM, (1 << 32) - 100, -312)
    # Source sample -312 is the binding's packet; 412 samples later wraps.
    assert binding.rtp_timestamp_of(-312) == (1 << 32) - 100
    assert binding.rtp_timestamp_of(100) == 312
    near = (1 << 32) - 50
    assert unwrap_rtp(312, near) == (1 << 32) + 312
    assert unwrap_rtp((1 << 32) - 60, near) == (1 << 32) - 60
    assert unwrap_rtp(5, 10 + (3 << 32)) == 5 + (3 << 32)


def test_the_provider_routes_timelines_and_timed_words():
    provider = AvarthaPipecatProvider.__new__(AvarthaPipecatProvider)
    events = []
    provider._queue = SimpleNamespace(put_nowait=events.append)
    provider._route_message(
        {"label": "rtvi-ai", "type": "server-message", "data": TIMELINE}
    )
    provider._route_message(
        {
            "label": "rtvi-ai",
            "type": "server-message",
            "data": {**TIMELINE, "sample_rate_hz": 24000},
        }
    )
    provider._route_message(
        {
            "label": "rtvi-ai",
            "type": "bot-tts-text",
            "data": {
                "text": "arrive",
                "audio_time": {
                    "stream_id": STREAM,
                    "sample_offset": 38400,
                    "sample_rate_hz": 48000,
                    "boundary": "word_end",
                },
            },
        }
    )
    provider._route_message(
        {"label": "rtvi-ai", "type": "bot-tts-text", "data": {"text": "now"}}
    )

    assert events == [
        AgentAudioTimelineEvent(BINDING),
        AgentWordEvent("arrive", WordEndTime(STREAM, 38400)),
        AgentWordEvent("now"),
    ]


# --- receiver: decoded frames keep their RTP position -----------------------


def _decoded(pts, value=1000):
    """A decoded 20 ms Opus frame as aiortc delivers it (48 kHz stereo)."""

    frame = AudioFrame.from_ndarray(
        np.full((1, 960 * 2), value, dtype=np.int16), format="s16", layout="stereo"
    )
    frame.sample_rate = 48000
    frame.pts = pts
    frame.time_base = Fraction(1, 48000)
    return frame


def test_resampled_audio_keeps_its_rtp_position_and_loss_restarts_decoding():
    origin = (1 << 32) - 1500  # wraps within the call
    position = _RtpPosition(origin)
    resampler = AudioResampler(format="s16", layout="mono", rate=24000)
    spans = []
    for pts in [0, 960, 1920, 4800]:  # 2880..4800 lost
        frame = _decoded(pts)
        position.decoded(frame)
        for converted in resampler.resample(frame):
            spans.append(
                (position.start_of(converted), converted.samples, position.decoded_from)
            )

    # PyAV's resampler carries pts through its delay: output sample k of a
    # frame at pts p is input pts 2 * (p + k).
    assert spans[0] == (origin, 464, origin)
    assert spans[1] == (origin + 928, 480, origin)
    assert spans[3][0] == origin + 4800 - 32
    assert spans[3][2] == origin + 4800  # decoding restarted after the loss


def test_a_receiver_without_an_origin_gives_no_rtp_position():
    position = _RtpPosition(None)
    frame = _decoded(0)
    position.decoded(frame)
    assert position.start_of(frame) is None
    assert position.decoded_from is None

    other_rate = _decoded(0)
    other_rate.sample_rate = 16000
    position = _RtpPosition(5)
    position.decoded(other_rate)
    assert position.start_of(other_rate) is None


def test_the_origin_is_read_from_the_tracks_own_receiver():
    track = object()
    mapper = SimpleNamespace(_origin=123456)
    receivers = [
        SimpleNamespace(track=object(), _RTCRtpReceiver__timestamp_mapper=mapper),
        SimpleNamespace(
            track=track,
            _RTCRtpReceiver__timestamp_mapper=SimpleNamespace(_origin=777),
        ),
    ]
    peer = SimpleNamespace(getReceivers=lambda: receivers)
    assert _rtp_origin(peer, track) == 777
    receivers[1] = SimpleNamespace(track=track)
    assert _rtp_origin(peer, track) is None


# --- adapter: a cut keeps exactly the words whose sound played --------------


@pytest.mark.parametrize(
    "world_ends_in_tick, expected",
    [
        # T171 A: "world." ended in tick 2's audio, which played; its word
        # event arrived late, in the cut tick.
        (2, "Hello world."),
        # T171 B: "world." ended in the cut tick's audio, which the cut
        # discarded; the same word event, on time.
        (3, "Hello"),
    ],
)
def test_t171_schedules_get_the_text_whose_sound_played(world_ends_in_tick, expected):
    """Untimed, both schedules reach the adapter as the same events and get
    the same answer, right only in B (T171). Timed on the RTP clock, each word
    is placed where its sound ends."""

    world_end = FIRST_RTP + (world_ends_in_tick - 1) * TICK_RTP + 500
    shown = _transcripts(
        [
            [
                AgentAudioTimelineEvent(BINDING),
                TranscriptEvent("Hello world."),
                _audio(0),
                _word("Hello", FIRST_RTP + 400),
            ],
            [_audio(1)],
            [_audio(2), _word("world.", world_end), SpeechEvent("speech_started")],
            [],
        ]
    )

    assert "".join(shown).strip() == expected


@pytest.mark.parametrize(
    "missing", ["binding", "receiver timestamps", "word time", "epoch"]
)
def test_without_timing_the_t171_schedules_keep_todays_answer(missing):
    """No binding, a receiver that keeps no timestamps, an untimed word or a
    word on another clock epoch: the word is placed at its arrival, as before."""

    timed = missing != "receiver timestamps"
    stream = "00000000-0000-4000-8000-000000000000" if missing == "epoch" else STREAM

    def word(text, end):
        if missing == "word time":
            return AgentWordEvent(text)
        return _word(text, end, stream)

    first = [TranscriptEvent("Hello world."), _audio(0, rtp=timed)]
    if missing != "binding":
        first.insert(0, AgentAudioTimelineEvent(BINDING))
    shown = _transcripts(
        [
            [*first, word("Hello", FIRST_RTP + 400)],
            [_audio(1, rtp=timed)],
            [
                _audio(2, rtp=timed),
                word("world.", FIRST_RTP + TICK_RTP + 500),
                SpeechEvent("speech_started"),
            ],
            [],
        ]
    )

    assert shown == ["Hello", "", "", ""]


def test_a_word_reported_before_its_audio_airs_with_that_audio():
    """Untimed, a word shows as soon as the audio received when it arrives
    has played: here on the first tick, 20 ms before its sound."""

    shown = _transcripts(
        [
            [
                AgentAudioTimelineEvent(BINDING),
                TranscriptEvent("Hello world."),
                _audio(0),
                _word("Hello", FIRST_RTP + 400),
                _word("world.", FIRST_RTP + TICK_RTP + 500),
            ],
            [_audio(1)],
            [SpeechEvent("speech_started")],
        ]
    )

    assert shown == ["Hello", " world.", ""]


def test_a_cut_drops_a_word_whose_audio_never_arrived():
    shown = _transcripts(
        [
            [
                AgentAudioTimelineEvent(BINDING),
                TranscriptEvent("Hello world."),
                _audio(0),
                _word("Hello", FIRST_RTP + 400),
                _word("world.", FIRST_RTP + 5 * TICK_RTP),
            ],
            [_audio(1)],
            [SpeechEvent("speech_started")],
            [],
        ]
    )

    assert "".join(shown) == "Hello"


def test_a_caption_places_a_word_whose_audio_never_arrived_at_its_arrival():
    """The utterance ended without the word's audio (lost at its end): the
    played caption is the evidence, as untimed."""

    shown = _transcripts(
        [
            [
                AgentAudioTimelineEvent(BINDING),
                TranscriptEvent("Hello world."),
                _audio(0),
                _word("Hello", FIRST_RTP + 400),
                _word("world.", FIRST_RTP + 5 * TICK_RTP),
            ],
            [AgentPlaybackEvent("Hello world.", cut=False)],
            [],
        ]
    )

    assert "".join(shown) == "Hello world."


@pytest.mark.parametrize(
    "decoded_from, expected",
    [
        # 20 ms of digital silence the receiver dropped between the two
        # chunks: "world." ended in it, before the second chunk played.
        (FIRST_RTP + TICK_RTP, "Hello world."),
        # Those 20 ms were lost: where "world." ended is unknown, so it is
        # placed at its arrival, as untimed, and the cut drops it.
        (FIRST_RTP + 2 * TICK_RTP, "Hello"),
    ],
)
def test_a_word_ending_in_dropped_silence_or_lost_media(decoded_from, expected):
    shown = _transcripts(
        [
            [
                AgentAudioTimelineEvent(BINDING),
                TranscriptEvent("Hello world."),
                _audio(0),
                _word("Hello", FIRST_RTP + 400),
            ],
            [_audio(2, decoded_from=decoded_from)],
            [
                _word("world.", FIRST_RTP + TICK_RTP + 500),
                _audio(3),
                SpeechEvent("speech_started"),
            ],
            [],
        ]
    )

    assert "".join(shown).strip() == expected


def test_a_second_clock_binding_stops_word_timing():
    other = AudioClockBinding("00000000-0000-4000-8000-000000000000", 9, 0)
    shown = _transcripts(
        [
            [
                AgentAudioTimelineEvent(BINDING),
                TranscriptEvent("Hello world."),
                _audio(0),
                _word("Hello", FIRST_RTP + 400),
            ],
            [_audio(1), AgentAudioTimelineEvent(other)],
            [
                _audio(2),
                _word("world.", FIRST_RTP + TICK_RTP + 500),
                SpeechEvent("speech_started"),
            ],
            [],
        ]
    )

    assert "".join(shown) == "Hello"


def test_word_times_follow_the_rtp_clock_across_its_wrap():
    first = (1 << 32) - 1500
    binding = AudioClockBinding(STREAM, first, 0)

    def audio(index):
        start = first + index * TICK_RTP
        return AgentAudioEvent(
            b"\x01\x00" * 480, 24000, rtp_start=start, rtp_decoded_from=first
        )

    def word(text, end):
        return AgentWordEvent(text, WordEndTime(STREAM, end - first))

    shown = _transcripts(
        [
            [
                AgentAudioTimelineEvent(binding),
                TranscriptEvent("Hello world."),
                audio(0),
                word("Hello", first + 400),
            ],
            [audio(1)],
            # world. ends at RTP 2**32 + 260, which is 260 on the wire.
            [
                audio(2),
                word("world.", first + TICK_RTP + 800),
                SpeechEvent("speech_started"),
            ],
            [],
        ]
    )

    assert "".join(shown).strip() == "Hello world."


def test_the_held_last_millisecond_counts_as_played_for_timed_words_too():
    """The adapter holds an utterance's last millisecond until its caption,
    and every mark counts it as played. A timed word ending in it is no
    different, or a reply's last word would be lost to a barge-in that keeps
    it when untimed (architect review r4)."""

    shown = _transcripts(
        [
            [
                AgentAudioTimelineEvent(BINDING),
                TranscriptEvent("Hello world."),
                _audio(0),
                _word("Hello", FIRST_RTP + 400),
            ],
            [_audio(1), _word("world.", FIRST_RTP + TICK_RTP + 955)],
            [SpeechEvent("speech_started")],
            [],
        ]
    )

    assert "".join(shown).strip() == "Hello world."


def test_a_word_reported_after_the_cut_is_dropped():
    """Avartha can report a word after the barge-in that cut it. The adapter
    ignores the cut reply's late events, timed or not, so such a word is
    lost even if it played (a conservative miss, never an extra word)."""

    shown = _transcripts(
        [
            [
                AgentAudioTimelineEvent(BINDING),
                TranscriptEvent("Hello world."),
                _audio(0),
                _word("Hello", FIRST_RTP + 400),
            ],
            [_audio(1)],
            [
                SpeechEvent("speech_started"),
                _word("world.", FIRST_RTP + TICK_RTP + 500),
            ],
            [],
        ]
    )

    assert "".join(shown).strip() == "Hello"


def test_heard_text_kept_at_a_cut_precedes_the_next_reply():
    shown = _transcripts(
        [
            [
                AgentAudioTimelineEvent(BINDING),
                TranscriptEvent("Hello world."),
                _audio(0),
                _word("Hello", FIRST_RTP + 400),
            ],
            [_audio(1)],
            [
                _word("world.", FIRST_RTP + TICK_RTP + 500),
                SpeechEvent("speech_started"),
            ],
            [
                AgentStartedSpeakingEvent(),
                AgentLLMStartedEvent(),
                TranscriptEvent("Sure."),
            ],
            [_audio(10), _word("Sure.", FIRST_RTP + 10 * TICK_RTP + 400)],
        ]
    )

    assert shown == ["Hello", "", " world.", "", " Sure."]


def _ticks(events_by_tick, rtp_silence=None):
    """The adapter's TickResults; ``rtp_silence`` as the receiver reports it."""

    provider = FakeProvider(events_by_tick)
    provider.rtp_silence = rtp_silence
    adapter = DiscreteTimeAvarthaPipecatAdapter(tick_duration_ms=20, provider=provider)
    adapter.connect("policy", [])
    try:
        return [
            adapter.run_tick(b"\xff" * 160, tick_number=tick)
            for tick in range(1, len(events_by_tick) + 1)
        ]
    finally:
        adapter.disconnect()


def test_two_cut_events_in_one_tick_keep_the_heard_words():
    """Avartha can send both user-started-speaking and bot-interrupted in one
    tick. The second cut used to cut back to the text shown so far, losing the
    words the first had kept (pr-review r4)."""

    shown = _transcripts(
        [
            [
                AgentAudioTimelineEvent(BINDING),
                TranscriptEvent("Hello world."),
                _audio(0),
                _word("Hello", FIRST_RTP + 400),
            ],
            [_audio(1), _audio(2), _word("world.", FIRST_RTP + TICK_RTP + 500)],
            [SpeechEvent("speech_started"), SpeechEvent("interrupted")],
            [],
        ]
    )

    assert "".join(shown).strip() == "Hello world."


def test_an_untimed_replacement_word_waits_for_its_audio_after_a_cut():
    """Untimed calls are unchanged: the replacement reply's word arriving in
    the cut tick is not shown on that silent tick (pr-review r4)."""

    shown = _transcripts(
        [
            [
                TranscriptEvent("Hello world."),
                _audio(0, rtp=False),
                AgentWordEvent("Hello"),
            ],
            [
                SpeechEvent("speech_started"),
                AgentLLMStartedEvent(),
                AgentStartedSpeakingEvent(),
                TranscriptEvent("Sorry."),
                AgentWordEvent("Sorry."),
            ],
            [_audio(5, rtp=False)],
        ]
    )

    assert shown == ["Hello", "", " Sorry."]


def test_a_word_ending_in_trailing_silence_is_kept_at_a_cut():
    """The reply's last word ends in the digital silence after its audio,
    which the receiver drops. Untimed it is kept; timed, it waited for audio
    that never came, and the cut dropped it (pr-review and architect r4). The
    receiver reports the silence it decoded after the last audio."""

    last_end = FIRST_RTP + 2 * TICK_RTP
    events = [
        [
            AgentAudioTimelineEvent(BINDING),
            TranscriptEvent("Hello world."),
            _audio(0),
            _word("Hello", FIRST_RTP + 400),
        ],
        [_audio(1), _word("world.", last_end + 300)],
        [],
        [SpeechEvent("speech_started")],
        [],
    ]

    def shown(rtp_silence):
        text = [t.proportional_transcript for t in _ticks(events, rtp_silence)]
        return "".join(text).strip()

    assert shown((last_end, last_end + 960)) == "Hello world."
    # Silence decoded only up to before the word's end, or after audio the
    # adapter has not received: unknown, so the cut drops the word.
    assert shown((last_end, last_end + 200)) == "Hello"
    assert shown((last_end + 960, last_end + 1920)) == "Hello"
    assert shown(None) == "Hello"


def test_heard_words_on_a_cut_that_discards_no_audio_join_their_utterance():
    """The reply had played all its audio and waited for its caption, so the
    cut discarded nothing; the kept word's tick still carries the reply's
    utterance (architect review r4)."""

    ticks = _ticks(
        [
            [
                AgentAudioTimelineEvent(BINDING),
                TranscriptEvent("Hello world."),
                _audio(0),
                _word("Hello", FIRST_RTP + 400),
            ],
            [_audio(1)],
            [],
            [
                _word("world.", FIRST_RTP + TICK_RTP + 500),
                SpeechEvent("speech_started"),
            ],
        ]
    )

    assert ticks[3].proportional_transcript == " world."
    assert ticks[3].agent_audio_chunks == [(b"", ticks[0].agent_audio_chunks[0][1])]


def test_a_word_end_far_beyond_the_audio_is_not_on_this_clock():
    """A minute past the audio received: a wrong binding or rewritten
    timestamps, so the word is placed at its arrival, with tick 1's audio."""

    shown = _transcripts(
        [
            [
                AgentAudioTimelineEvent(BINDING),
                TranscriptEvent("Hello world."),
                _audio(0),
                _word("Hello", FIRST_RTP + 400),
                _word("world.", FIRST_RTP + 61 * 48000),
            ],
            [_audio(1)],
            [],
        ]
    )

    assert shown == ["Hello world.", "", ""]


def test_a_second_binding_places_waiting_words_at_their_arrival():
    other = AudioClockBinding("00000000-0000-4000-8000-000000000000", 9, 0)
    shown = _transcripts(
        [
            [
                AgentAudioTimelineEvent(BINDING),
                TranscriptEvent("Hello world."),
                _audio(0),
                _word("Hello", FIRST_RTP + 400),
                _word("world.", FIRST_RTP + 5 * TICK_RTP),
            ],
            [_audio(1), AgentAudioTimelineEvent(other)],
            [],
        ]
    )

    assert shown == ["Hello", " world.", ""]


# --- receiver canaries against the real aiortc -------------------------------


def test_aiortc_keeps_the_rtp_origin_where_the_receiver_reads_it():
    """The timed path rests on aiortc's private TimestampMapper; if an upgrade
    moves it, this fails instead of timing silently switching off."""

    from aiortc.rtcrtpreceiver import RTCRtpReceiver

    transport = SimpleNamespace(state="new", _register_rtp_receiver=lambda *a: None)
    receiver = RTCRtpReceiver("audio", transport)
    mapper = receiver._RTCRtpReceiver__timestamp_mapper
    assert mapper.map((1 << 32) - 10) == 0
    assert mapper.map(950) == 960  # across the wrap
    track = object()
    receiver._track = track
    peer = SimpleNamespace(getReceivers=lambda: [receiver])
    assert _rtp_origin(peer, track) % (1 << 32) == (1 << 32) - 10


@pytest.mark.parametrize("origin, skipped", [(12345, 0), ((1 << 32) - 96, 0), (777, 7)])
def test_the_real_opus_chain_puts_source_samples_where_the_binding_says(
    origin, skipped
):
    """aiortc's Opus encoder (the sender, bound as svara binds it), its RTP
    timestamp mapper and decoder (the receiver), then this receiver's RTP
    position through the 24 kHz resampler: a noise burst at source sample s
    decodes at RTP binding.rtp_timestamp_of(s), within a 24 kHz sample.
    Packets the receiver never got (``skipped``) move its origin, not the
    result."""

    from aiortc.codecs.opus import OpusDecoder, OpusEncoder
    from aiortc.jitterbuffer import JitterFrame
    from aiortc.rtcrtpreceiver import TimestampMapper

    rate = 48000
    rng = np.random.default_rng(1)
    source = np.zeros(rate * 2, dtype=np.int16)
    burst = int(rate * 0.7)
    source[burst : burst + rate // 5] = (rng.standard_normal(rate // 5) * 6000).astype(
        np.int16
    )
    encoder = OpusEncoder()
    packets = []
    step = rate // 50
    for start in range(0, len(source), step):
        frame = AudioFrame.from_ndarray(
            source[start : start + step].reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate, frame.pts = rate, start
        frame.time_base = Fraction(1, rate)
        payloads, timestamp = encoder.encode(frame)
        packets += [((origin + timestamp) % (1 << 32), p) for p in payloads]
    binding = AudioClockBinding(STREAM, packets[0][0], encoder.first_packet_pts)

    mapper, decoder = TimestampMapper(), OpusDecoder()
    resampler = AudioResampler(format="s16", layout="mono", rate=24000)
    position = None
    samples = {}
    for timestamp, payload in packets[skipped:]:
        encoded = JitterFrame(data=payload, timestamp=mapper.map(timestamp))
        for frame in decoder.decode(encoded):
            if position is None:
                position = _RtpPosition(mapper._origin)
            position.decoded(frame)
            for converted in resampler.resample(frame):
                start = position.start_of(converted)
                for k, value in enumerate(converted.to_ndarray().reshape(-1)):
                    samples[start + 2 * k] = value

    keys = sorted(samples)
    decoded = np.array([samples[k] for k in keys], dtype=float)
    template = source[burst : burst + 4800 : 2].astype(float)
    measured = keys[int(np.argmax(np.correlate(decoded, template, mode="valid")))]
    expected = unwrap_rtp(binding.rtp_timestamp_of(burst), keys[0])
    assert abs(measured - expected) <= 2  # one 24 kHz sample


def test_the_receiver_reports_silence_it_drops_after_its_audio():
    from tau2.voice.audio_native.avartha_pipecat import webrtc

    frames = [
        _decoded(0),
        _decoded(960),
        _decoded(1920, value=0),
        _decoded(2880, value=0),
        _decoded(3840, value=0),
    ]
    provider = webrtc.AvarthaWebRTCProvider.__new__(webrtc.AvarthaWebRTCProvider)
    provider._closing = False
    provider._queue = asyncio.Queue()

    class Track:
        async def recv(self):
            if frames:
                return frames.pop(0)
            provider._closing = True
            raise webrtc.MediaStreamError

    track = Track()
    receiver = SimpleNamespace(
        track=track, _RTCRtpReceiver__timestamp_mapper=SimpleNamespace(_origin=5000)
    )
    peer = SimpleNamespace(getReceivers=lambda: [receiver])
    asyncio.run(provider._receive_audio(track, peer))

    events = []
    while not provider._queue.empty():
        events.append(provider._queue.get_nowait())
    starts = [e.rtp_start for e in events]
    assert starts == [
        5000 + sum(len(x.audio) for x in events[:i]) for i in range(len(events))
    ]
    audio_end = starts[-1] + len(events[-1].audio)
    # The silence after the audio runs, contiguous, to the last frame decoded.
    assert provider.rtp_silence[0] == audio_end
    assert provider.rtp_silence[1] > audio_end + 960


def test_a_first_frame_off_the_mapped_origin_gives_no_timing():
    position = _RtpPosition(1000)
    frame = _decoded(960)
    position.decoded(frame)
    assert position.start_of(frame) is None


def test_text_a_cut_keeps_stays_in_the_transcript_text():
    """Kept words are part of what aired: the next reply's first sentence
    still joins them with a space."""

    shown = _transcripts(
        [
            [
                AgentAudioTimelineEvent(BINDING),
                TranscriptEvent("Hello world."),
                _audio(0),
                _audio(1),
            ],
            [],
            [
                _word("Hello", FIRST_RTP + 400),
                _word("world.", FIRST_RTP + 900),
                SpeechEvent("speech_started"),
            ],
            [
                AgentLLMStartedEvent(),
                AgentStartedSpeakingEvent(),
                TranscriptEvent("Sure."),
                _audio(5),
                _word("Sure.", FIRST_RTP + 5 * TICK_RTP + 400),
            ],
            [],
        ]
    )

    assert shown[2] == "Hello world."  # kept at the cut
    assert "".join(shown) == "Hello world. Sure."


def test_audio_rewinding_the_rtp_clock_stops_timing_from_it():
    """Audio that starts before the audio already received is not on a clock
    the adapter can follow: a word timed into it is placed at its arrival, not
    in the earlier audio with the same timestamps."""

    shown = _transcripts(
        [
            [
                AgentAudioTimelineEvent(BINDING),
                TranscriptEvent("Hello world."),
                _audio(0),
                _word("Hello", FIRST_RTP + 400),
            ],
            [_audio(1)],
            [_audio(0), _word("world.", FIRST_RTP + 900)],
            [_audio(3)],
            [],
        ]
    )

    assert shown == ["Hello", "", "", " world.", ""]
