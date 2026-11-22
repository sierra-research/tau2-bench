"""The OpenAI adapter must report how much of an interrupted item was played.

``conversation.item.truncate`` takes ``audio_end_ms`` measured from the start of
the item. These tests drive the adapter tick by tick with a scripted provider and
check the value sent when the user barges in.
"""

import asyncio
import base64

import pytest

from tau2.voice.audio_native.openai.discrete_time_adapter import (
    DiscreteTimeOpenAIAdapter,
)
from tau2.voice.audio_native.openai.events import (
    AudioDeltaEvent,
    SpeechStartedEvent,
)
from tau2.voice.audio_native.tick_result import TickResult

TICK_MS = 200
# Telephony mu-law 8 kHz is 8000 bytes per second.
BYTES_PER_MS = 8
BYTES_PER_TICK = TICK_MS * BYTES_PER_MS


class ScriptedProvider:
    """Replays one list of events per tick and records truncate calls."""

    def __init__(self, ticks):
        self._ticks = list(ticks)
        self.truncates = []

    async def send_audio(self, audio):
        pass

    async def send_tool_result(self, call_id, result, request_response=True):
        pass

    async def receive_events_for_duration(self, duration):
        return self._ticks.pop(0) if self._ticks else []

    async def truncate_item(self, item_id, content_index, audio_end_ms):
        self.truncates.append((item_id, content_index, audio_end_ms))


def audio(item_id, num_bytes):
    return AudioDeltaEvent(
        type="response.output_audio.delta",
        item_id=item_id,
        delta=base64.b64encode(b"\x01" * num_bytes).decode(),
    )


def speech_started(audio_start_ms):
    return SpeechStartedEvent(
        type="input_audio_buffer.speech_started", audio_start_ms=audio_start_ms
    )


def run_ticks(ticks):
    provider = ScriptedProvider(ticks)
    adapter = DiscreteTimeOpenAIAdapter(
        tick_duration_ms=TICK_MS, send_audio_instant=True, provider=provider
    )
    user_audio = b"\xff" * BYTES_PER_TICK

    async def drive():
        for tick_number in range(len(ticks)):
            await adapter._async_run_tick(user_audio, tick_number)

    asyncio.run(drive())
    return provider.truncates


def test_barge_in_during_first_tick_reports_audio_played_so_far():
    # The agent speaks for the whole tick. The user cuts in 50 ms into it.
    truncates = run_ticks([[audio("item_a", BYTES_PER_TICK), speech_started(50)]])

    assert truncates == [("item_a", 0, 50)]


def test_barge_in_reports_audio_from_earlier_ticks_too():
    # One full tick of item_a was already played, then the user cuts in 100 ms
    # into the second tick. The item has played for 200 + 100 ms in total.
    truncates = run_ticks(
        [
            [audio("item_a", BYTES_PER_TICK)],
            [audio("item_a", BYTES_PER_TICK), speech_started(TICK_MS + 100)],
        ]
    )

    assert truncates == [("item_a", 0, 300)]


def test_barge_in_counts_audio_that_was_buffered_from_an_earlier_tick():
    # The server sent two ticks of audio at once. Only one tick is played, the
    # rest is buffered and played during the next tick. The user then cuts in
    # 100 ms into that second tick.
    truncates = run_ticks(
        [
            [audio("item_a", 2 * BYTES_PER_TICK)],
            [speech_started(TICK_MS + 100)],
        ]
    )

    assert truncates == [("item_a", 0, 300)]


def test_barge_in_only_counts_the_interrupted_item():
    # item_a fills the first 100 ms of the tick and item_b the next 100 ms.
    # The user cuts in at 150 ms, so item_b has played for 50 ms.
    truncates = run_ticks(
        [
            [
                audio("item_a", 100 * BYTES_PER_MS),
                audio("item_b", 100 * BYTES_PER_MS),
                speech_started(150),
            ]
        ]
    )

    assert truncates == [("item_b", 0, 50)]


def test_barge_in_without_agent_audio_sends_no_truncate():
    assert run_ticks([[speech_started(50)]]) == []


@pytest.mark.parametrize("cut_ms", [0, 199, 200])
def test_barge_in_never_reports_more_than_the_tick_allows(cut_ms):
    truncates = run_ticks([[audio("item_a", BYTES_PER_TICK), speech_started(cut_ms)]])

    assert truncates == [("item_a", 0, cut_ms)]


def test_barge_in_on_an_item_that_spans_three_ticks():
    # Two full ticks of item_a were played, then the user cuts in 50 ms into
    # the third. The item has played for 200 + 200 + 50 ms in total.
    truncates = run_ticks(
        [
            [audio("item_a", BYTES_PER_TICK)],
            [audio("item_a", BYTES_PER_TICK)],
            [audio("item_a", BYTES_PER_TICK), speech_started(2 * TICK_MS + 50)],
        ]
    )

    assert truncates == [("item_a", 0, 450)]


def new_tick(**kwargs):
    return TickResult(
        tick_number=0,
        audio_sent_bytes=BYTES_PER_TICK,
        audio_sent_duration_ms=TICK_MS,
        bytes_per_tick=BYTES_PER_TICK,
        bytes_per_second=BYTES_PER_MS * 1000,
        **kwargs,
    )


def make_truncated_tick(chunks, audio_start_ms, tick_start_ms=0):
    tick = new_tick(silence_byte=b"\xff")
    tick.cumulative_user_audio_at_tick_start_ms = tick_start_ms
    tick.agent_audio_chunks = chunks
    tick.truncate_agent_audio(
        item_id=chunks[-1][1],
        audio_start_ms=audio_start_ms,
        cumulative_user_audio_at_tick_start_ms=tick_start_ms,
        bytes_per_tick=BYTES_PER_TICK,
    )
    return tick


def test_played_item_bytes_stop_at_the_interruption_and_ignore_other_items():
    chunks = [
        (b"a" * 400, "item_a"),
        (b"b" * 600, "item_b"),
        (b"b" * 600, "item_b"),
    ]
    # 100 ms is 800 bytes: all of item_a's 400 and 400 of item_b's bytes.
    tick = make_truncated_tick(chunks, audio_start_ms=100)

    assert tick.get_played_item_audio_bytes("item_a") == 400
    assert tick.get_played_item_audio_bytes("item_b") == 400
    assert tick.get_played_item_audio_bytes("item_c") == 0


def test_played_item_bytes_are_relative_to_the_start_of_the_tick():
    tick = make_truncated_tick(
        [(b"a" * BYTES_PER_TICK, "item_a")], audio_start_ms=1050, tick_start_ms=1000
    )

    assert tick.get_played_item_audio_bytes("item_a") == 50 * BYTES_PER_MS


def test_played_item_bytes_need_a_recorded_interruption():
    tick = new_tick()

    with pytest.raises(ValueError, match="No interruption"):
        tick.get_played_item_audio_bytes("item_a")


def test_padded_played_audio_is_unchanged_by_the_interruption_helper():
    tick = make_truncated_tick([(b"a" * BYTES_PER_TICK, "item_a")], audio_start_ms=50)

    played = tick.get_played_agent_audio()

    assert len(played) == BYTES_PER_TICK
    assert played[: 50 * BYTES_PER_MS] == b"a" * (50 * BYTES_PER_MS)
    assert played[50 * BYTES_PER_MS :] == b"\xff" * (150 * BYTES_PER_MS)
