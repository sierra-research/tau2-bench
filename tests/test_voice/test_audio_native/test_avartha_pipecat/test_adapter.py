import asyncio
from collections import deque

import pytest

from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.avartha_pipecat.config import AvarthaPipecatConfig
from tau2.voice.audio_native.avartha_pipecat.discrete_time_adapter import (
    DiscreteTimeAvarthaPipecatAdapter,
)
from tau2.voice.audio_native.avartha_pipecat.provider import (
    AgentAudioEvent,
    AgentLLMStartedEvent,
    AgentLLMTextEvent,
    AgentPlaybackEvent,
    AgentStartedSpeakingEvent,
    AgentWordEvent,
    AvarthaBootError,
    AvarthaPipecatProvider,
    AvarthaProtocolError,
    SpeechEvent,
    ToolCallEvent,
    TranscriptEvent,
)

AGENT_URL = "https://platform.example/inference/serverless/pipecat/v1/public/agent-1"


class FakeProvider:
    def __init__(self, events_by_tick=(), boot_failures=0):
        self.config = AvarthaPipecatConfig(base_url=AGENT_URL)
        self.events_by_tick = deque(events_by_tick)
        self.boot_failures = boot_failures
        self.connect_attempts = 0
        self.connect_prompts = []
        self.sent_audio = []
        self.tool_results = []
        self.registered = None
        self.session_id = "session-test"
        self.is_connected = False
        self.reset_count = 0
        # No RTP positions (see AvarthaPipecatProvider.rtp_silence).
        self.rtp_silence = None

    async def connect(self, system_prompt=""):
        self.connect_attempts += 1
        self.connect_prompts.append(system_prompt)
        self.is_connected = True
        if self.connect_attempts <= self.boot_failures:
            raise AvarthaBootError("cold worker")

    async def register_tools(self, schemas):
        self.registered = schemas

    async def send_audio(self, audio, sample_rate):
        self.sent_audio.append((audio, sample_rate))

    async def send_tool_result(
        self,
        call_id,
        result,
        request_response=True,
        is_error=False,
    ):
        self.tool_results.append((call_id, result, request_response, is_error))

    async def receive_events_for_duration(self, _duration):
        return self.events_by_tick.popleft() if self.events_by_tick else []

    def drain_pending_events(self):
        return []

    async def disconnect(self):
        self.is_connected = False

    def reset_for_retry(self):
        self.reset_count += 1
        self.is_connected = False


def make_lookup_tool():
    def lookup(order_id: str) -> str:
        """Look up an order.

        Args:
            order_id: Order identifier.
        """
        return order_id

    return Tool(lookup)


def test_factory_constructs_avartha_for_a_hosted_agent():
    config = AvarthaPipecatConfig(base_url=AGENT_URL)

    adapter, model = create_adapter(
        "avartha_pipecat", tick_duration_ms=20, avartha_config=config
    )

    assert isinstance(adapter, DiscreteTimeAvarthaPipecatAdapter)
    assert model == "avartha"
    assert adapter.model == "avartha"
    assert adapter.avartha_config is config


def test_the_hosted_agent_url_is_required():
    with pytest.raises(ValueError, match="hosted agent's URL"):
        create_adapter("avartha_pipecat", tick_duration_ms=20)


def test_ticks_keep_to_the_wall_clock():
    adapter = DiscreteTimeAvarthaPipecatAdapter(
        tick_duration_ms=20,
        avartha_config=AvarthaPipecatConfig(base_url=AGENT_URL),
    )

    assert adapter.realtime_pacing is True


def test_reasoning_effort_is_rejected():
    with pytest.raises(ValueError, match="configured on the hosted agent"):
        DiscreteTimeAvarthaPipecatAdapter(
            tick_duration_ms=20,
            reasoning_effort="high",
            avartha_config=AvarthaPipecatConfig(base_url=AGENT_URL),
        )


def test_audio_transcript_tool_and_result_round_trip():
    provider = FakeProvider(
        [
            [
                TranscriptEvent("Order shipped"),
                AgentAudioEvent(b"\x00\x00" * 480, 24000),
                AgentWordEvent("Order"),
                AgentWordEvent("shipped"),
                _played("Order shipped"),
                ToolCallEvent("call-1", "lookup", {"order_id": "A-1"}),
            ],
            [],
        ]
    )
    adapter = DiscreteTimeAvarthaPipecatAdapter(
        tick_duration_ms=20,
        provider=provider,
    )
    adapter.connect("system policy", [make_lookup_tool()])

    try:
        tick = adapter.run_tick(b"\xff" * 160, tick_number=1)
        schemas = provider.registered
        assert provider.connect_prompts == ["system policy"]
        assert schemas[0]["name"] == "lookup"
        assert schemas[0]["inputSchema"]["properties"]["order_id"]["type"] == "string"
        assert provider.sent_audio[0][1] == 16000
        assert len(provider.sent_audio[0][0]) > 160
        assert tick.agent_audio_bytes == 160
        assert tick.proportional_transcript == "Order shipped"
        assert tick.tool_calls[0].model_dump() == {
            "id": "call-1",
            "name": "lookup",
            "arguments": {"order_id": "A-1"},
            "requestor": "assistant",
        }

        adapter.send_tool_result(
            "call-1",
            "shipped",
            request_response=False,
            is_error=True,
        )
        adapter.run_tick(b"\xff" * 160, tick_number=2)
        assert provider.tool_results == [("call-1", "shipped", False, True)]
    finally:
        adapter.disconnect()


def test_boot_failure_retries_before_registering_tools():
    provider = FakeProvider(boot_failures=1)
    config = AvarthaPipecatConfig(base_url=AGENT_URL, boot_attempts=2)
    adapter = DiscreteTimeAvarthaPipecatAdapter(
        tick_duration_ms=20,
        avartha_config=config,
        provider=provider,
    )

    adapter.connect("policy", [])
    try:
        assert provider.connect_attempts == 2
        assert provider.reset_count == 1
        assert provider.connect_prompts == ["policy", "policy"]
        assert provider.registered == []
    finally:
        adapter.disconnect()


def test_user_speech_truncates_same_tick_audio():
    provider = FakeProvider(
        [
            [
                AgentAudioEvent(b"\x00\x00" * 480, 24000),
                SpeechEvent("speech_started"),
            ]
        ]
    )
    adapter = DiscreteTimeAvarthaPipecatAdapter(
        tick_duration_ms=20,
        provider=provider,
    )
    adapter.connect("policy", [])
    try:
        tick = adapter.run_tick(b"\xff" * 160)
        assert tick.was_truncated is True
        assert tick.agent_audio_bytes == 0
        assert tick.truncated_audio_bytes == 160
    finally:
        adapter.disconnect()


def test_interruption_discards_transcript_accounting_before_next_response():
    provider = FakeProvider(
        [
            [
                TranscriptEvent("one two three four five"),
                AgentAudioEvent(b"\x00\x00" * (480 * 5), 24000),
            ],
            [SpeechEvent("speech_started")],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("You are welcome"),
                AgentStartedSpeakingEvent(),
                AgentAudioEvent(b"\x00\x00" * 480, 24000),
                _played("You are welcome"),
            ],
        ]
    )
    adapter = DiscreteTimeAvarthaPipecatAdapter(
        tick_duration_ms=20,
        provider=provider,
    )
    adapter.connect("policy", [])
    try:
        first = adapter.run_tick(b"\xff" * 160)
        interrupted = adapter.run_tick(b"\xff" * 160)
        next_response = adapter.run_tick(b"\xff" * 160)

        assert first.agent_audio_bytes == 160
        assert interrupted.was_truncated is True
        assert interrupted.truncated_audio_bytes == 640
        assert next_response.proportional_transcript == "You are welcome"
    finally:
        adapter.disconnect()


def test_bot_interruption_discards_buffered_and_late_cancelled_output():
    provider = FakeProvider(
        [
            [
                TranscriptEvent("This cancelled response must disappear"),
                AgentAudioEvent(b"\x00\x00" * (480 * 5), 24000),
            ],
            [
                SpeechEvent("interrupted"),
                AgentAudioEvent(b"\x00\x00" * 480, 24000),
                TranscriptEvent("cancelled tail"),
            ],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Fresh response"),
                AgentStartedSpeakingEvent(),
                AgentAudioEvent(b"\x00\x00" * 480, 24000),
                _played("Fresh response"),
            ],
        ]
    )
    adapter = DiscreteTimeAvarthaPipecatAdapter(
        tick_duration_ms=20,
        provider=provider,
    )

    async def exercise():
        first = await adapter._async_run_tick(b"\xff" * 160, 0)
        interrupted = await adapter._async_run_tick(b"\xff" * 160, 1)
        next_response = await adapter._async_run_tick(b"\xff" * 160, 2)
        return first, interrupted, next_response

    first, interrupted, next_response = asyncio.run(exercise())

    assert first.agent_audio_bytes == 160
    assert interrupted.vad_events == ["interrupted"]
    assert interrupted.was_truncated is True
    assert interrupted.agent_audio_bytes == 0
    assert interrupted.proportional_transcript == ""
    assert interrupted.truncated_audio_bytes == 800
    assert next_response.agent_audio_bytes == 160
    assert next_response.proportional_transcript == "Fresh response"


def _played(caption):
    return AgentPlaybackEvent(caption, cut=False)


# 20 ms of 24 kHz PCM16 agent audio: one 160-byte telephony tick.
def _audio_ticks(n):
    return AgentAudioEvent(b"\x00\x00" * 480 * n, 24000)


def _transcripts(events_by_tick):
    provider = FakeProvider(events_by_tick)
    adapter = DiscreteTimeAvarthaPipecatAdapter(tick_duration_ms=20, provider=provider)
    adapter.connect("policy", [])
    try:
        return [
            adapter.run_tick(b"\xff" * 160, tick_number=tick).proportional_transcript
            for tick in range(1, len(events_by_tick) + 1)
        ]
    finally:
        adapter.disconnect()


def test_sentence_text_waits_for_the_audio_that_speaks_it():
    """Avartha sends a sentence before its audio. Releasing it on arrival
    showed "...spelled out letter by letter?" on the first 20 ms of the line,
    and the simulated caller answered 1.6 s into it."""

    shown = _transcripts(
        [
            [
                TranscriptEvent("Got it."),
                TranscriptEvent("Please spell your name?"),
                _audio_ticks(1),
                AgentWordEvent("Got"),
            ],
            [_audio_ticks(1), AgentWordEvent("it.")],
            [_audio_ticks(1), AgentWordEvent("Please")],
            [_audio_ticks(1), AgentWordEvent("spell"), AgentWordEvent("your")],
            [
                _audio_ticks(1),
                AgentWordEvent("name?"),
                _played("Got it. Please spell your name?"),
            ],
        ]
    )

    assert shown == ["Got", " it.", " Please", " spell your", " name?"]


def test_text_follows_playback_when_audio_arrives_faster_than_it_plays():
    """tau2 plays one tick of audio per tick however much has arrived, so
    words that came with queued audio wait for that audio to be played."""

    shown = _transcripts(
        [
            [
                TranscriptEvent("one two three"),
                _audio_ticks(1),
                AgentWordEvent("one"),
                _audio_ticks(1),
                AgentWordEvent("two"),
                _audio_ticks(1),
                AgentWordEvent("three"),
            ],
            [],
            [],
        ]
    )

    assert shown == ["one", " two", " three"]


def test_played_caption_releases_words_the_alignment_dropped():
    """ElevenLabs alignment splits and drops words ("Wha" + "t", no "with
    today"); the shown text is still the sentence text, whole."""

    shown = _transcripts(
        [
            [
                TranscriptEvent("What can I help you with today?"),
                _audio_ticks(1),
                AgentWordEvent("Wha"),
                AgentWordEvent("t"),
            ],
            [_audio_ticks(1), AgentWordEvent("help"), AgentWordEvent(",")],
            [_audio_ticks(1), _played("What can I help you with today?")],
        ]
    )

    # "Wha" + "t" join into "What", the sentence's next word.
    assert shown == ["What", " can I help", " you with today?"]


def test_a_split_word_keeps_text_with_the_audio_before_a_barge_in():
    """The alignment aired "T-shirt" as "T-shir" + "t". Text used to stop at " T",
    and a barge-in then discarded the rest of the line although it had played."""

    shown = _transcripts(
        [
            [
                TranscriptEvent("We have 12 T-shirt styles, and 10 are in stock."),
                _audio_ticks(1),
                AgentWordEvent("We"),
                AgentWordEvent("have"),
                AgentWordEvent("12"),
            ],
            [_audio_ticks(1), AgentWordEvent("T-shir"), AgentWordEvent("t")],
            [_audio_ticks(1), AgentWordEvent("styles,"), AgentWordEvent("and")],
            [_audio_ticks(1), AgentWordEvent("10"), AgentWordEvent("are")],
            [SpeechEvent("speech_started")],
        ]
    )

    assert shown == ["We have 12", " T-shirt", " styles, and", " 10 are", ""]


def test_a_short_word_split_in_two_still_advances_to_the_amount():
    """The alignment aired "at" as "a" + "t". The text used to stay at "Camera"
    until the line ended, after "$481.50." had played."""

    shown = _transcripts(
        [
            [
                TranscriptEvent("The most expensive is the Action Camera at $481.50."),
                _audio_ticks(1),
                AgentWordEvent("The"),
                AgentWordEvent("most"),
                AgentWordEvent("expensive"),
                AgentWordEvent("is"),
                AgentWordEvent("the"),
            ],
            [_audio_ticks(1), AgentWordEvent("Action"), AgentWordEvent("Camera")],
            [_audio_ticks(1), AgentWordEvent("a"), AgentWordEvent("t")],
            [_audio_ticks(1), AgentWordEvent("$481.50.")],
        ]
    )

    assert shown == [
        "The most expensive is the",
        " Action Camera",
        " at",
        " $481.50.",
    ]


def test_a_piece_its_next_word_does_not_complete_is_dropped():
    """A held piece joins only the very next word; otherwise the next word
    is matched alone."""

    shown = _transcripts(
        [
            [
                TranscriptEvent("What can I help you with today?"),
                _audio_ticks(1),
                AgentWordEvent("Wha"),
            ],
            [_audio_ticks(1), AgentWordEvent("can")],
        ]
    )

    assert shown == ["", "What can"]


def test_barge_in_discards_the_unheard_rest_of_the_line():
    shown = _transcripts(
        [
            [
                TranscriptEvent("Thanks. Now the long question?"),
                _audio_ticks(1),
                AgentWordEvent("Thanks."),
                _audio_ticks(2),
                AgentWordEvent("Now"),
            ],
            [SpeechEvent("speech_started")],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Go ahead."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                _played("Go ahead."),
            ],
        ]
    )

    assert shown == ["Thanks.", "", " Go ahead."]


def test_line_end_releases_only_what_the_caption_says_played():
    """The next response's sentence can be queued before this
    utterance ends. Its end must not release it."""

    shown = _transcripts(
        [
            [
                TranscriptEvent("One moment."),
                _audio_ticks(1),
                AgentWordEvent("One"),
            ],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Your order has shipped."),
                _played("One moment."),
            ],
            [_audio_ticks(1)],
            [AgentStartedSpeakingEvent(), _audio_ticks(1), AgentWordEvent("Your")],
        ]
    )

    # The caption's text rides the 1 ms held back at the end of "One moment."
    assert shown == ["One", " moment.", "", " Your"]


def test_line_end_after_a_fully_spoken_sentence_holds_the_queued_reply():
    shown = _transcripts(
        [
            [TranscriptEvent("One moment."), _audio_ticks(1), AgentWordEvent("One")],
            [_audio_ticks(1), AgentWordEvent("moment.")],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Your order has shipped."),
                _played("One moment."),
            ],
        ]
    )

    assert shown == ["One", " moment.", ""]


def test_a_late_caption_still_delivers_the_line_end_with_audio():
    """The caption can arrive after the line's audio has all played.
    Text on a tick without audio is dropped from the caller's history, so the
    end of the line must come with audio, not on a silent tick."""

    provider = FakeProvider(
        [
            [
                TranscriptEvent("Thanks for waiting."),
                _audio_ticks(1),
                AgentWordEvent("Thanks"),
            ],
            [],
            [_played("Thanks for waiting.")],
        ]
    )
    adapter = DiscreteTimeAvarthaPipecatAdapter(tick_duration_ms=20, provider=provider)
    adapter.connect("policy", [])
    try:
        ticks = [adapter.run_tick(b"\xff" * 160, tick_number=i) for i in (1, 2, 3)]
    finally:
        adapter.disconnect()

    assert [t.proportional_transcript for t in ticks] == ["Thanks", "", " for waiting."]
    assert [t.agent_audio_bytes for t in ticks] == [152, 0, 8]


def test_a_word_never_skips_ahead_into_an_unaired_sentence():
    """ "Let me" can be the next sentence's words or a false match;
    either way the sentence in between is not proven aired, so it waits for
    the caption."""

    shown = _transcripts(
        [
            [
                TranscriptEvent("Okay."),
                TranscriptEvent("Let me check that order for you."),
                _audio_ticks(1),
            ],
            [_audio_ticks(1), AgentWordEvent("Let"), AgentWordEvent("me")],
            [_audio_ticks(1), _played("Okay. Let me check that order for you.")],
        ]
    )

    assert shown == ["", "", "Okay. Let me check that order for you."]


def test_decimal_split_by_the_sentence_splitter_is_rejoined():
    """Avartha sends "$13." and "46 back." as two sentences. Rejoining
    them with a space showed "$13. 46", and the caption "$13.46" then no
    longer matched the text."""

    shown = _transcripts(
        [
            [
                TranscriptEvent("You get $13."),
                TranscriptEvent("46 back."),
                TranscriptEvent(" Anything else?"),
                _audio_ticks(1),
            ],
            [_played("You get $13.46 back. Anything else?")],
        ]
    )

    assert shown == ["", "You get $13.46 back. Anything else?"]


def _llm(*tokens):
    return [AgentLLMTextEvent(t) for t in tokens]


def test_a_cut_replys_unsent_fragment_does_not_open_the_next_reply():
    """Seen live: Pipecat keeps the cut reply's unsent " First I need to
    verify your" and sends it glued to the next reply's first sentence."""

    shown = _transcripts(
        [
            [
                AgentLLMStartedEvent(),
                *_llm("I can help", " with that exchange."),
                TranscriptEvent("I can help with that exchange."),
                *_llm(" First I need", " to verify your"),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                SpeechEvent("interrupted"),
            ],
            [
                AgentLLMStartedEvent(),
                *_llm("Hi, sorry", " about that."),
                TranscriptEvent(" First I need to verify yourHi, sorry about that."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("Hi,"),
                AgentWordEvent("sorry"),
            ],
            [_audio_ticks(1), _played("Hi, sorry about that.")],
        ]
    )

    assert shown == ["", "Hi, sorry", " about that."]


def test_an_unpunctuated_tail_is_shown_from_the_caption_without_duplication():
    """A reply ending without sentence punctuation never gets its tail sent
    as a sentence; the caption says it aired, so it is shown once."""

    shown = _transcripts(
        [
            [
                AgentLLMStartedEvent(),
                *_llm("It shipped."),
                TranscriptEvent("It shipped."),
                *_llm(" Anything else"),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("It"),
                AgentWordEvent("shipped."),
            ],
            [_audio_ticks(1), _played("It shipped. Anything else")],
            [
                AgentLLMStartedEvent(),
                *_llm("Bye."),
                TranscriptEvent(" Anything elseBye."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                _played("Bye."),
            ],
        ]
    )

    assert shown == ["It shipped.", " Anything else", " Bye."]


def test_a_short_word_does_not_match_inside_a_later_word():
    shown = _transcripts(
        [
            [
                TranscriptEvent("If you want, I can wait."),
                _audio_ticks(1),
                AgentWordEvent("I"),
            ],
            [_audio_ticks(1), AgentWordEvent("If")],
        ]
    )

    assert shown == ["", "If"]


def test_korean_syllables_and_attached_particles_advance_the_text():
    """Korean words can arrive a syllable at a time. Only two-syllable words
    matched, and "218,840" never matched with "원" attached, so a line's text
    waited for its caption and a barge-in discarded the rest although it had
    played."""

    shown = _transcripts(
        [
            [
                TranscriptEvent("함께 확인해 볼게요."),
                TranscriptEvent(" 차액은 218,840원입니다."),
                _audio_ticks(1),
                *[AgentWordEvent(s) for s in ("함", "께", "확", "인", "해")],
            ],
            [_audio_ticks(1), *[AgentWordEvent(s) for s in ("볼", "게", "요.")]],
            [_audio_ticks(1), *[AgentWordEvent(s) for s in ("차", "액", "은")]],
            [
                _audio_ticks(1),
                *[AgentWordEvent(s) for s in ("218,840", "원", "입", "니", "다.")],
            ],
        ]
    )

    assert shown == ["함께 확인해", " 볼게요.", " 차액은", " 218,840원입니다."]


def test_hindi_syllables_advance_the_text_even_out_of_order():
    """Hindi words arrive a syllable at a time, sometimes with a word's last
    syllable after the next word's first ("छो", "सी", "बा", "टी", "त"). Only
    "ए" + "क" ever joined into a word, so a line's text stopped at "एक" until
    its caption, and a barge-in before then discarded the rest although it
    had played."""

    shown = _transcripts(
        [
            [
                TranscriptEvent("एक छोटी सी बात — आपने जो ऑर्डर ID बताया, वह नहीं मिला।"),
                _audio_ticks(1),
                *[AgentWordEvent(s) for s in ("ए", "क", "छो", "सी", "बा", "टी", "त")],
            ],
            [
                _audio_ticks(1),
                *[
                    AgentWordEvent(s)
                    for s in ("आ", "प", "ने", "जो", "ऑ", "र्ड", "र", "ID")
                ],
            ],
            [_audio_ticks(1), *[AgentWordEvent(s) for s in ("ब", "ता", "या,")]],
        ]
    )

    assert shown == ["एक छोटी सी बात", " — आपने जो ऑर्डर ID", " बताया,"]


def test_chinese_digits_between_full_width_commas_advance_the_text():
    """No space follows "，", so "1，1，6，" never ended as a whole word and
    the line waited for its caption."""

    shown = _transcripts(
        [
            [
                TranscriptEvent("邮编是：1，1，6，0，7，8。如果不对，请再念一遍。"),
                _audio_ticks(1),
                *[AgentWordEvent(s) for s in ("邮", "编", "是")],
            ],
            [_audio_ticks(1), AgentWordEvent("1，1，6，"), AgentWordEvent("0，7，8。")],
            [_audio_ticks(1), *[AgentWordEvent(s) for s in ("如", "果", "不", "对，")]],
        ]
    )

    assert shown == ["邮编是", "：1，1，6，0，7，8。", "如果不对，"]


def test_html_escaped_text_is_shown_unescaped():
    """Seen live: "I&amp;#x27;m" in bot-transcription and the caption."""

    shown = _transcripts(
        [
            [
                TranscriptEvent("I&amp;#x27;m on it."),
                _audio_ticks(1),
                AgentWordEvent("I&#x27;m"),
            ],
            [_audio_ticks(1), _played("I&amp;#x27;m on it.")],
        ]
    )

    assert shown == ["I'm", " on it."]


def test_without_evidence_of_airing_nothing_is_shown():
    """Audio alone does not say which text it spoke, and the next
    reply may already be queued. With no word and no caption, text waits."""

    shown = _transcripts(
        [
            [TranscriptEvent("One moment."), _audio_ticks(1)],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("Your order shipped."),
                _audio_ticks(1),
            ],
            [_audio_ticks(1)],
        ]
    )

    assert shown == ["", "", ""]


def test_a_caption_with_no_sentence_behind_it_is_shown_as_played():
    """Speech that no LLM sentence produced still reaches the caller."""

    shown = _transcripts(
        [
            [_audio_ticks(1)],
            [_played("Please hold.")],
        ]
    )

    assert shown == ["", "Please hold."]


def test_replacement_audio_in_the_same_tick_as_the_cut_is_kept():
    """A cut followed by the new reply inside one tick used to drop
    the reply's audio and text along with the cut-off tail."""

    provider = FakeProvider(
        [
            [TranscriptEvent("Old line."), _audio_ticks(1)],
            [
                _audio_ticks(1),
                SpeechEvent("interrupted"),
                _audio_ticks(1),  # cut-off tail still in flight
                AgentLLMStartedEvent(),
                TranscriptEvent("New line."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                AgentWordEvent("New"),
                AgentWordEvent("line."),
                _played("New line."),
            ],
            [],
        ]
    )
    adapter = DiscreteTimeAvarthaPipecatAdapter(tick_duration_ms=20, provider=provider)
    adapter.connect("policy", [])
    try:
        ticks = [adapter.run_tick(b"\xff" * 160, tick_number=i) for i in (1, 2, 3)]
    finally:
        adapter.disconnect()

    cut, replacement = ticks[1], ticks[2]
    assert cut.was_truncated is True
    assert cut.agent_audio_bytes == 0
    assert cut.truncated_audio_bytes == 328  # incl. 1 ms held back in tick 1
    assert replacement.agent_audio_bytes == 160
    assert replacement.proportional_transcript == "New line."


def test_cut_off_audio_arriving_after_the_cut_tick_stays_discarded():
    provider = FakeProvider(
        [
            [TranscriptEvent("Old line."), _audio_ticks(1)],
            [SpeechEvent("speech_started")],
            [_audio_ticks(1), TranscriptEvent("old tail")],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("New."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                _played("New."),
            ],
        ]
    )
    adapter = DiscreteTimeAvarthaPipecatAdapter(tick_duration_ms=20, provider=provider)
    adapter.connect("policy", [])
    try:
        ticks = [adapter.run_tick(b"\xff" * 160, tick_number=i) for i in (1, 2, 3, 4)]
    finally:
        adapter.disconnect()

    assert ticks[2].agent_audio_bytes == 0
    assert ticks[2].truncated_audio_bytes == 160
    assert ticks[3].agent_audio_bytes == 160
    assert ticks[3].proportional_transcript == "New."


def _ticks(events_by_tick):
    provider = FakeProvider(events_by_tick)
    adapter = DiscreteTimeAvarthaPipecatAdapter(tick_duration_ms=20, provider=provider)
    adapter.connect("policy", [])
    try:
        return [
            adapter.run_tick(b"\xff" * 160, tick_number=tick)
            for tick in range(1, len(events_by_tick) + 1)
        ]
    finally:
        adapter.disconnect()


def test_a_pause_inside_an_utterance_keeps_its_id_and_the_caption_ends_it():
    """With one ID for every utterance, tau2 split a line at each silent tick,
    so judges saw "I couldn't find a reservation under that" while its clip's
    audio went on to say "code"."""

    ticks = _ticks(
        [
            [TranscriptEvent("Let me check."), _audio_ticks(1), AgentWordEvent("Let")],
            [],  # the TTS falls behind: a silent tick inside the line
            [_audio_ticks(1), AgentWordEvent("check."), _played("Let me check.")],
            [],
            [
                AgentStartedSpeakingEvent(),
                TranscriptEvent("Found it."),
                _audio_ticks(1),
                _played("Found it."),
            ],
        ]
    )

    assert [t.item_ids for t in ticks] == [
        ["avartha-1"],
        ["avartha-1"],
        ["avartha-1"],
        ["avartha-1"],  # its last 1 ms, held for the caption
        ["avartha-2"],
    ]
    assert [t.proportional_transcript for t in ticks] == [
        "Let",
        "",
        " me check.",
        "",
        " Found it.",
    ]


def test_a_cut_starts_a_new_utterance():
    ticks = _ticks(
        [
            [TranscriptEvent("Old line, long."), _audio_ticks(3)],
            [SpeechEvent("speech_started")],
            [
                AgentLLMStartedEvent(),
                TranscriptEvent("New."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
                _played("New."),
            ],
        ]
    )

    # The cut discarded the old line's queued audio, so its silent tick belongs
    # to that line and carries the truncation; the next reply opens a new one.
    assert [t.item_ids for t in ticks] == [
        ["avartha-1"],
        ["avartha-1"],
        ["avartha-2"],
    ]
    assert [t.was_truncated for t in ticks] == [False, True, False]


def test_caller_speech_after_a_fully_played_reply_does_not_mark_it_cut():
    """A reply stays open until its played caption, which can come late. Caller
    speech after its last audio discards only the 1 ms held tail: marking that
    cut told judges a complete reply was cut off."""

    ticks = _ticks(
        [
            [TranscriptEvent("Done."), _audio_ticks(1), AgentWordEvent("Done.")],
            [],
            [SpeechEvent("speech_started")],
        ]
    )

    assert ticks[-1].was_truncated
    assert ticks[-1].item_ids == []


def test_replacement_audio_in_the_cut_tick_keeps_both_utterances_apart():
    ticks = _ticks(
        [
            [TranscriptEvent("Old line, long."), _audio_ticks(3)],
            [
                SpeechEvent("speech_started"),
                AgentLLMStartedEvent(),
                TranscriptEvent("New."),
                AgentStartedSpeakingEvent(),
                _audio_ticks(1),
            ],
            [_played("New.")],
        ]
    )

    assert [t.item_ids for t in ticks] == [
        ["avartha-1"],
        ["avartha-1"],
        ["avartha-2"],
    ]
    assert ticks[1].was_truncated and not ticks[2].was_truncated


def test_the_first_cut_in_a_tick_names_the_utterance():
    """A second cut in the same tick, after a new reply's audio arrived, must not
    move the truncation to that reply, which played nothing."""

    ticks = _ticks(
        [
            [TranscriptEvent("Old line, long."), _audio_ticks(3)],
            [
                SpeechEvent("speech_started"),
                AgentLLMStartedEvent(),
                AgentStartedSpeakingEvent(),
                _audio_ticks(2),
                SpeechEvent("interrupted"),
            ],
        ]
    )

    assert ticks[1].item_ids == ["avartha-1"]
    assert ticks[1].was_truncated


def test_a_cut_before_the_ticks_streamed_audio_still_names_the_reply():
    """A reply streamed at playback pace has only its held tail queued when a
    cut opens a tick; the rest of its audio arrives after the cut, stale."""

    ticks = _ticks(
        [
            [TranscriptEvent("Old line, long."), _audio_ticks(1)],
            [_audio_ticks(1)],
            [
                SpeechEvent("speech_started"),
                SpeechEvent("interrupted"),
                _audio_ticks(1),
            ],
        ]
    )

    assert ticks[-1].item_ids == ["avartha-1"]
    assert ticks[-1].was_truncated


def test_stale_audio_after_cutting_a_finished_reply_does_not_mark_it_cut():
    """A reply whose audio stopped a tick before the cut only waits for its
    played caption. Audio arriving after the cut in the same tick did not
    belong to it, so the reply must not get a cut-off marker."""

    ticks = _ticks(
        [
            [TranscriptEvent("Done."), _audio_ticks(1), AgentWordEvent("Done.")],
            [],
            [SpeechEvent("speech_started"), _audio_ticks(1)],
        ]
    )

    assert ticks[-1].was_truncated
    assert ticks[-1].item_ids == []


def test_stale_audio_after_a_cut_in_a_tts_pause_does_not_mark_the_reply():
    ticks = _ticks(
        [
            [TranscriptEvent("Old line, long."), _audio_ticks(1)],
            [],
            [],
            [SpeechEvent("interrupted"), _audio_ticks(1)],
        ]
    )

    assert ticks[-1].was_truncated
    assert ticks[-1].item_ids == []


def test_a_cut_in_the_tick_after_streamed_audio_names_the_reply():
    """The reply's first audio arrived the tick before the cut, so it was still
    streaming when cut, even with only its held tail queued."""

    ticks = _ticks(
        [
            [TranscriptEvent("Old line, long."), _audio_ticks(1)],
            [SpeechEvent("speech_started"), _audio_ticks(1)],
        ]
    )

    assert ticks[-1].item_ids == ["avartha-1"]
    assert ticks[-1].was_truncated


def test_a_replacement_cut_does_not_mark_a_completed_previous_reply():
    ticks = _ticks(
        [
            [TranscriptEvent("Done."), _audio_ticks(1), AgentWordEvent("Done.")],
            [],
            [
                SpeechEvent("speech_started"),
                AgentLLMStartedEvent(),
                AgentStartedSpeakingEvent(),
                SpeechEvent("interrupted"),
                _audio_ticks(1),
            ],
        ]
    )

    assert ticks[-1].was_truncated
    assert ticks[-1].item_ids == []


def test_caller_speech_with_no_open_utterance_labels_nothing():
    ticks = _ticks(
        [
            [TranscriptEvent("Done."), _audio_ticks(1), _played("Done.")],
            [],
            [SpeechEvent("speech_started")],
        ]
    )

    assert [t.item_ids for t in ticks] == [["avartha-1"], [], []]


def test_non_telephony_audio_format_is_rejected():
    """Audio is always converted to 8 kHz u-law, so a PCM format
    would label u-law bytes as PCM and size ticks for the wrong rate."""

    from tau2.data_model.audio import AudioEncoding, AudioFormat

    with pytest.raises(ValueError, match="telephony"):
        DiscreteTimeAvarthaPipecatAdapter(
            tick_duration_ms=20,
            audio_format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=16000, channels=1
            ),
            avartha_config=AvarthaPipecatConfig(base_url=AGENT_URL),
        )


def test_avartha_events_count_as_provider_activity():
    """The stall detector looked for a ``type`` the events lacked,
    so a call streaming audio was reported inactive every 30 s."""

    provider = FakeProvider([[_audio_ticks(1)], []])
    adapter = DiscreteTimeAvarthaPipecatAdapter(tick_duration_ms=20, provider=provider)
    adapter.connect("policy", [])
    try:
        busy = adapter.run_tick(b"\xff" * 160, tick_number=1)
        idle = adapter.run_tick(b"\xff" * 160, tick_number=2)
    finally:
        adapter.disconnect()

    assert busy.has_provider_activity is True
    assert idle.has_provider_activity is False


def test_shutdown_delivers_results_of_tools_already_executed():
    """Results are sent at the start of the next tick. When the call
    ends first, disconnect used to drop them unsent."""

    provider = FakeProvider([[]])
    adapter = DiscreteTimeAvarthaPipecatAdapter(tick_duration_ms=20, provider=provider)
    adapter.connect("policy", [])
    adapter.run_tick(b"\xff" * 160, tick_number=1)
    adapter.send_tool_result("call-9", "transferred")
    adapter.disconnect()

    assert provider.tool_results == [("call-9", "transferred", True, False)]


def test_protocol_error_aborts_connection_without_boot_retries():
    class IncompatibleProvider(FakeProvider):
        async def register_tools(self, schemas):
            raise AvarthaProtocolError("Incompatible Avartha tool protocol")

    provider = IncompatibleProvider()
    adapter = DiscreteTimeAvarthaPipecatAdapter(tick_duration_ms=20, provider=provider)
    with pytest.raises(AvarthaProtocolError, match="Incompatible Avartha"):
        adapter.connect("policy", [])

    assert provider.connect_attempts == 1
    assert not provider.is_connected
    assert not adapter.is_connected
    assert provider.sent_audio == []


def test_failed_audio_send_drains_receive_task_before_loop_stops():
    class FailingProvider(FakeProvider):
        receive_started = None
        receive_cancelled = False

        async def send_audio(self, audio, sample_rate):
            await self.receive_started.wait()
            raise RuntimeError("socket closed")

        async def receive_events_for_duration(self, duration):
            self.receive_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.receive_cancelled = True

    async def exercise():
        provider = FailingProvider()
        provider.receive_started = asyncio.Event()
        adapter = DiscreteTimeAvarthaPipecatAdapter(
            tick_duration_ms=20, provider=provider
        )
        with pytest.raises(RuntimeError, match="socket closed"):
            await adapter._async_run_tick(b"\xff" * 160, 0)
        assert provider.receive_cancelled
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(exercise())


@pytest.mark.parametrize("deadline_lag", [0.0, 1.0])
@pytest.mark.parametrize("cut", [False, True])
def test_tick_includes_events_queued_while_sender_finishes(deadline_lag, cut):
    """A closed receive window must not hide events queued before send completion."""

    class WindowProvider(AvarthaPipecatProvider):
        def __init__(self):
            super().__init__(AvarthaPipecatConfig(base_url=AGENT_URL))
            self.window_closed = asyncio.Event()
            self.sent = False
            self.before = [AgentStartedSpeakingEvent()]
            self.after = [
                AgentAudioEvent(b"\xe8\x03" * 480, 24000),
                AgentPlaybackEvent(text="", cut=False),
                ToolCallEvent("call-window", "lookup", {"order_id": "A"}),
            ]
            if cut:
                self.after.extend(
                    [
                        SpeechEvent(kind="interrupted"),
                        AgentStartedSpeakingEvent(),
                        AgentAudioEvent(b"\xd0\x07" * 480, 24000),
                        AgentPlaybackEvent(text="", cut=False),
                    ]
                )

        async def receive_events_for_duration(self, duration):
            # A barrier controls the race, independent of scheduler timing.
            self.window_closed.set()
            return list(self.before) if not self.sent else []

        async def send_audio(self, audio_data, sample_rate):
            await self.window_closed.wait()
            if not self.sent:
                self.sent = True
                for event in self.after:
                    self._queue.put_nowait(event)

    async def check():
        provider = WindowProvider()
        adapter = DiscreteTimeAvarthaPipecatAdapter(
            tick_duration_ms=20, provider=provider
        )
        adapter._next_tick_start = asyncio.get_running_loop().time() - deadline_lag
        first = await adapter._async_run_tick(b"\xff" * 160, 0)
        second = await adapter._async_run_tick(b"\xff" * 160, 1)
        assert first.events == provider.before + provider.after
        assert len(first.agent_audio_data) == (0 if cut else 160)
        assert first.was_truncated is cut
        assert first.tool_calls[0].id == "call-window"
        assert provider._queue.empty()
        assert len(second.agent_audio_data) == (160 if cut else 0)
        assert not second.tool_calls

    asyncio.run(check())


def test_pending_event_drain_is_fifo_and_nonblocking():
    provider = AvarthaPipecatProvider(AvarthaPipecatConfig(base_url=AGENT_URL))
    events = [
        AgentAudioEvent(b"\xe8\x03" * 480, 24000),
        SpeechEvent(kind="interrupted"),
        ToolCallEvent("call-cut", "lookup", {}),
        AgentStartedSpeakingEvent(),
    ]
    for event in events:
        provider._queue.put_nowait(event)
    assert provider.drain_pending_events() == events
    assert provider.drain_pending_events() == []


def test_cancelled_tick_drains_tasks_without_consuming_pending_events():
    class BlockingProvider(FakeProvider):
        def __init__(self):
            super().__init__()
            self.sending = asyncio.Event()
            self.receiving = asyncio.Event()
            self.stopped = []
            self.events_drained = False

        async def send_audio(self, audio, sample_rate):
            self.sending.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.stopped.append("send")

        async def receive_events_for_duration(self, duration):
            self.receiving.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.stopped.append("receive")

        def drain_pending_events(self):
            self.events_drained = True
            return []

    async def check():
        provider = BlockingProvider()
        adapter = DiscreteTimeAvarthaPipecatAdapter(
            tick_duration_ms=20, provider=provider
        )
        task = asyncio.create_task(adapter._async_run_tick(b"\xff" * 160, 0))
        await asyncio.wait_for(provider.sending.wait(), 1)
        await asyncio.wait_for(provider.receiving.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sorted(provider.stopped) == ["receive", "send"]
        assert not provider.events_drained

    asyncio.run(check())
