"""The smallest adapter's tick lifecycle on scripted Hydra events.

Each tick runs the shared ``_async_run_tick`` template with one scripted batch
of Hydra events, so audio buffering, barge-in, tool calls, usage and the
transcript a tick hands to the caller are checked exactly. Agent audio is
8 kHz PCM16 here, so 200 ms of it becomes exactly 1600 u-law bytes.
"""

from tau2.data_model.message import ToolCall
from tau2.voice.audio_native.smallest import DiscreteTimeSmallestAdapter
from tau2.voice.audio_native.smallest.processing import parse_tool_arguments

from .fake_hydra import (
    SILENCE_TICK,
    TICK_BYTES,
    TICK_MS,
    ScriptedProvider,
    audio,
    audio_done,
    barge_in,
    caption,
    created,
    done,
    run_ticks,
    speech_started,
    tool_call,
)

SENTENCE = "I can change that flight for you."  # 33 characters, 2.2 s at 15/s


def adapter_for(
    batches, cps=None
) -> tuple[DiscreteTimeSmallestAdapter, ScriptedProvider]:
    provider = ScriptedProvider(batches)
    adapter = DiscreteTimeSmallestAdapter(
        tick_duration_ms=TICK_MS,
        send_audio_instant=True,
        provider=provider,
        transcript_reveal_cps=cps,
    )
    return adapter, provider


def texts(ticks) -> list[str]:
    return [t.proportional_transcript for t in ticks]


def test_caption_ahead_of_its_audio_is_revealed_at_the_speaking_rate():
    # The caption arrives with the first 200 ms of its audio, the rest of the
    # audio in real time: no text before audio, 3 characters per tick.
    batches = [[created("r1"), caption("r1", SENTENCE)]]
    batches += [[audio("r1", 200)] for _ in range(11)]
    adapter, _ = adapter_for(batches)
    ticks = run_ticks(adapter, 12)
    assert ticks[0].proportional_transcript == ""  # caption but no audio yet
    assert texts(ticks[1:]) == [SENTENCE[i : i + 3] for i in range(0, 33, 3)]
    for tick in ticks[1:]:
        assert tick.agent_audio_bytes == TICK_BYTES
        assert tick.item_ids == ["r1"]


def test_cap_off_restores_the_upstream_reveal():
    batches = [[created("r1"), caption("r1", SENTENCE), audio("r1", 200)]]
    batches += [[audio("r1", 200)]]
    adapter, _ = adapter_for(batches, cps=0)
    ticks = run_ticks(adapter, 2)
    assert texts(ticks) == [SENTENCE, ""]  # whole sentence at the first audio


def test_completed_reply_releases_its_tail_on_a_tick_without_audio():
    # A voice faster than the cap: 33 characters in 1 s. Hydra's
    # response.done arrives one tick after the last audio has played.
    batches = [[created("r1"), caption("r1", SENTENCE), audio("r1", 200)]]
    batches += [[audio("r1", 200)] for _ in range(4)]
    batches += [[audio_done("r1"), done("r1")]]
    adapter, _ = adapter_for(batches)
    ticks = run_ticks(adapter, 6)
    assert "".join(texts(ticks[:5])) == SENTENCE[:15]
    last = ticks[5]
    assert last.proportional_transcript == SENTENCE[15:]
    assert last.agent_audio_bytes == 0 and last.item_ids == ["r1"]
    assert last.get_played_agent_audio() == SILENCE_TICK


def test_completed_reply_releases_its_tail_with_its_last_audio():
    batches = [[created("r1"), caption("r1", SENTENCE), audio("r1", 200)]]
    batches += [[audio("r1", 200)] for _ in range(3)]
    batches += [[audio("r1", 200), audio_done("r1"), done("r1")], []]
    adapter, _ = adapter_for(batches)
    ticks = run_ticks(adapter, 6)
    assert texts(ticks) == ["I c", "an ", "cha", "nge", SENTENCE[12:], ""]


def test_older_reply_tail_comes_before_the_next_reply_text():
    batches = [[created("r1"), caption("r1", "Done."), audio("r1", 200)]]
    batches += [[audio_done("r1"), done("r1"), created("r2"), caption("r2", "Next?")]]
    batches[1].append(audio("r2", 200))
    adapter, _ = adapter_for(batches)
    ticks = run_ticks(adapter, 2)
    assert texts(ticks) == ["Don", "e. Nex"]
    assert ticks[1].item_ids == ["r1", "r2"]


def test_barge_in_after_generation_ran_ahead_never_shows_the_unspoken_rest():
    # Hydra streamed the whole 2.2 s reply and said completed at once; the
    # caller speaks during the third tick of playback.
    batches = [
        [created("r1"), caption("r1", SENTENCE), audio("r1", 2200)],
        [audio_done("r1"), done("r1")],
        [speech_started(), barge_in()],
        [],
    ]
    adapter, _ = adapter_for(batches)
    ticks = run_ticks(adapter, 4)
    assert texts(ticks) == ["I c", "an ", "cha", ""]
    cut = ticks[2]
    assert cut.was_truncated and cut.vad_events == ["speech_started", "interrupted"]
    assert cut.agent_audio_bytes == TICK_BYTES
    assert cut.truncated_audio_bytes == 11 * TICK_BYTES - 3 * TICK_BYTES
    assert ticks[3].agent_audio_bytes == 0 and adapter._buffered_agent_audio == []


def test_barge_in_while_the_reply_streams_skips_its_later_audio():
    batches = [
        [created("r1"), caption("r1", SENTENCE), audio("r1", 200)],
        [audio("r1", 200), barge_in()],
        [audio("r1", 200), audio_done("r1"), done("r1", "cancelled")],
        [created("r2"), caption("r2", "Sure."), audio("r2", 200)],
    ]
    adapter, _ = adapter_for(batches)
    ticks = run_ticks(adapter, 4)
    assert ticks[1].was_truncated and ticks[1].skip_item_id == "r1"
    stale = ticks[2]
    assert stale.agent_audio_bytes == 0 and stale.truncated_audio_bytes == TICK_BYTES
    assert stale.proportional_transcript == ""
    assert adapter._utterance_transcripts["r1"].truncated
    assert ticks[3].item_ids == ["r2"] and ticks[3].proportional_transcript == "Sur"


def test_reply_cut_mid_stream_stays_cut_when_hydra_later_says_completed():
    # The cut tick plays every byte received so far; Hydra then reports the
    # reply completed (its generation had finished). The rest was never
    # spoken, so it must never be shown.
    batches = [
        [created("r1"), caption("r1", SENTENCE), audio("r1", 200)],
        [audio("r1", 200), barge_in()],
        [audio("r1", 200), audio_done("r1"), done("r1")],
        [],
    ]
    adapter, _ = adapter_for(batches)
    ticks = run_ticks(adapter, 4)
    assert "".join(texts(ticks)) == SENTENCE[:6]
    tracker = adapter._utterance_transcripts["r1"]
    assert tracker.truncated and not tracker.completed
    assert tracker.audio_fully_played  # all received audio played, yet no tail


def test_caller_speech_alone_cuts_nothing_while_hydra_keeps_the_floor():
    # The caller starts talking during the reply; Hydra does not commit a
    # barge-in (it holds the floor), so its audio and caption keep playing.
    batches = [
        [created("r1"), caption("r1", SENTENCE), audio("r1", 800)],
        [speech_started()],
        [],
        [audio_done("r1"), done("r1")],
    ]
    adapter, _ = adapter_for(batches)
    ticks = run_ticks(adapter, 4)
    assert ticks[1].vad_events == ["speech_started"]
    assert not any(t.was_truncated for t in ticks)
    assert all(t.agent_audio_bytes == TICK_BYTES for t in ticks)
    assert sum(t.truncated_audio_bytes for t in ticks) == 0
    assert not adapter._utterance_transcripts["r1"].truncated


def test_the_cut_waits_for_hydra_barge_in_after_caller_speech():
    # Caller speech at tick 1, Hydra commits the barge-in at tick 3: the agent
    # keeps talking until then and is cut on the barge-in tick only.
    batches = [
        [created("r1"), caption("r1", SENTENCE), audio("r1", 1200)],
        [speech_started()],
        [],
        [barge_in()],
        [],
    ]
    adapter, _ = adapter_for(batches)
    ticks = run_ticks(adapter, 5)
    assert [t.was_truncated for t in ticks] == [False, False, False, True, False]
    assert all(t.agent_audio_bytes == TICK_BYTES for t in ticks[:4])
    assert ticks[3].vad_events == ["interrupted"]
    assert ticks[3].truncated_audio_bytes == 2 * TICK_BYTES
    assert ticks[4].agent_audio_bytes == 0


def test_hydra_barge_in_event_cuts_audio_queued_after_speech_began():
    batches = [
        [speech_started()],  # nothing playing yet: nothing to cut
        [created("r1"), caption("r1", SENTENCE), audio("r1", 1000)],
        [barge_in()],
        [],
    ]
    adapter, _ = adapter_for(batches)
    ticks = run_ticks(adapter, 4)
    assert not ticks[0].was_truncated and ticks[0].vad_events == ["speech_started"]
    assert ticks[1].agent_audio_bytes == TICK_BYTES
    assert ticks[2].was_truncated and ticks[2].vad_events == ["interrupted"]
    assert ticks[2].truncated_audio_bytes == 3 * TICK_BYTES
    assert ticks[3].agent_audio_bytes == 0


def test_tool_call_is_relayed_and_its_result_returned_next_tick():
    args = '{"order_id": "#W123"}'
    batches = [[created("r1"), tool_call("r1", "call_1", "get_order_status", args)]]
    adapter, provider = adapter_for(batches + [[], []])
    first = run_ticks(adapter, 1)[0]
    assert first.tool_calls == [
        ToolCall(id="call_1", name="get_order_status", arguments={"order_id": "#W123"})
    ]
    adapter.send_tool_result("call_1", "shipped", request_response=True)
    run_ticks(adapter, 1)
    assert provider.tool_results == [("call_1", "shipped", True)]
    adapter.send_tool_result("call_2", "a", request_response=False)
    adapter.send_tool_result("call_3", "b", request_response=True)
    run_ticks(adapter, 1)
    assert provider.tool_results[1:] == [("call_2", "a", False), ("call_3", "b", True)]


def test_tool_arguments_that_are_not_a_json_object_become_empty():
    assert parse_tool_arguments('{"a": 1}') == {"a": 1}
    assert parse_tool_arguments({"a": 1}) == {"a": 1}
    for bad in (None, "", "not json", "[1, 2]", '"text"', 7):
        assert parse_tool_arguments(bad) == {}


def test_the_provider_records_no_usage():
    """A usage block in response.done (if a server sent one) is not recorded: the provider reports no usage."""
    usage = {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}
    adapter, _ = adapter_for([[created("r1"), done("r1", usage=usage)]])
    tick = run_ticks(adapter, 1)[0]
    assert tick.usage_records == [] and adapter.get_usage_records() == []


def test_caller_audio_is_sent_as_16khz_pcm16():
    adapter, provider = adapter_for([[]])
    run_ticks(adapter, 1, user_audio=SILENCE_TICK)
    (sent,) = provider.sent_audio  # send_audio_instant: one frame per tick
    assert 6390 <= len(sent) <= 6400  # 200 ms of 16 kHz PCM16 (resampler start)
