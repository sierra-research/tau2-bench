"""Parity with runtime/evaluator/judge views recorded before evaluator consolidation.

Only generated wall-clock message timestamps are excluded from the golden file;
all transcript text, ordering, metadata, spans and reviewer rendering are exact.
"""

import json
from pathlib import Path

from tau2.data_model.message import (
    AssistantMessage,
    Tick,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason


def sample_simulation():
    def agent(text, ids=None, **kwargs):
        return AssistantMessage(
            role="assistant",
            content=text,
            utterance_ids=ids,
            contains_speech=bool(text),
            **kwargs,
        )

    def user(text, ids=None):
        return UserMessage(
            role="user", content=text, utterance_ids=ids, contains_speech=bool(text)
        )

    gold = '<message uuid="a" active="0"><chunk id=0>Gold</chunk><chunk id=1> lost</chunk></message>'
    call = ToolCall(id="lookup", name="lookup", arguments={})
    tool = ToolMessage(role="tool", id="lookup", content="found", requestor="assistant")
    chunks = [
        (None, user("Question", ["q"])),
        (agent("Raw", ["a"], audio_script_gold=gold), None),
        (agent(" tail", ["a", "b"]), user("interrupt", ["i"])),
        (agent("", ["b"]), None),
        (agent("unknown one"), user("earlier")),
        (agent("unknown two"), user(" later")),
        (None, None),
        (agent("After", ["z"]), user("still talking", ["i"])),
        (agent(None, tool_calls=[call]), None),
        (agent("Done", ["end"], raw_data={"was_truncated": True}), None),
    ]
    ticks = [
        Tick(
            tick_id=10 + 2 * i,
            timestamp=f"2026-01-01T00:00:{i:02d}",
            agent_chunk=a,
            user_chunk=u,
            tick_duration_seconds=0.2,
        )
        for i, (a, u) in enumerate(chunks)
    ]
    ticks[8].agent_tool_calls = [call]
    ticks[8].agent_tool_results = [tool]
    ticks[8].user_tool_calls = [ToolCall(id="u", name="user_lookup", arguments={})]
    ticks[8].user_tool_results = [
        ToolMessage(role="tool", id="u", content="ok", requestor="user")
    ]
    return SimulationRun(
        id="parity",
        task_id="task",
        start_time="start",
        end_time="end",
        duration=1,
        trial=0,
        termination_reason=TerminationReason.USER_STOP,
        ticks=ticks,
    )


def snapshot():
    from tau2.utils.transcript.views import (
        agent_messages,
        agent_utterance_tick_spans,
        build_agent_turns,
        extract_spoken_turns,
        speech_messages,
        tool_messages,
    )

    sim = sample_simulation()

    def dump(items):
        return [item.model_dump(mode="json", exclude={"timestamp"}) for item in items]

    from dataclasses import asdict

    from tau2.metrics.voice_interaction_metrics import (
        extract_agent_segments,
        extract_user_segments,
    )
    from tau2.utils.display import MarkdownDisplay

    runtime, asr = edge_snapshots()
    return {
        "flattened_ticks": dump([m for t in sim.ticks for m in t.get_all_messages()]),
        "runtime_pending": runtime,
        "asr": asr,
        "markdown": MarkdownDisplay.display_ticks_consolidated(sim.ticks),
        "user_visible_markdown": MarkdownDisplay.display_ticks_consolidated(
            sim.ticks, user_visible_only=True
        ),
        "agent_segments": [asdict(x) for x in extract_agent_segments(sim.ticks)],
        "user_segments": [asdict(x) for x in extract_user_segments(sim.ticks)],
        "raw_turns": dump(extract_spoken_turns(sim)),
        "delivered_turns": dump(build_agent_turns(sim)),
        "audio_spans": [list(span) for span in agent_utterance_tick_spans(sim)],
        "agent_messages": dump(agent_messages(sim.ticks)),
        "tool_messages": dump(tool_messages(sim.ticks)),
        "speech_messages": dump(speech_messages(sim.ticks)),
    }


def test_reconstruction_matches_recorded_pre_refactor_views():
    expected = json.loads(
        (Path(__file__).parent / "fixtures/transcript_views.json").read_text()
    )
    assert snapshot() == expected


def test_absent_ids_have_explicit_view_semantics():
    from tau2.utils.transcript.views import agent_messages, extract_spoken_turns

    sim = sample_simulation()
    # Delivery/audio indexing treats absent IDs as separate utterances; raw call
    # transcripts join adjacent unidentified chunks, without changing caller turns.
    assert "unknown oneunknown two" in [t.text for t in extract_spoken_turns(sim)]
    assert {"unknown one", "unknown two"} <= {
        m.content for m in agent_messages(sim.ticks)
    }


def edge_snapshots():
    from tau2.agent.base.streaming import StreamingState
    from tau2.data_model.transcript import ParticipantTick
    from tau2.voice.utils.transcript_utils import compute_proportional_user_transcripts

    call = ToolCall(id="lookup", name="lookup", arguments={})
    state = StreamingState(
        ticks=[
            ParticipantTick(
                tick_id=0,
                timestamp="fixed",
                self_chunk=AssistantMessage(
                    role="assistant", tool_calls=[call], contains_speech=False
                ),
            ),
            ParticipantTick(
                tick_id=1,
                timestamp="fixed",
                other_chunk=UserMessage(
                    role="user", content="Caller ", contains_speech=True
                ),
                env_chunk=ToolMessage(
                    role="tool", id="lookup", content="found", requestor="assistant"
                ),
            ),
            ParticipantTick(
                tick_id=2,
                timestamp="fixed",
                self_chunk=AssistantMessage(
                    role="assistant",
                    content="Answer",
                    contains_speech=True,
                    audio_content="YQA=",
                    audio_script_gold="Whole unheard answer",
                ),
            ),
        ],
        input_turn_taking_buffer=[
            UserMessage(role="user", content="Wait", contains_speech=True)
        ],
    )
    before = state.model_dump_json()
    runtime = [
        m.model_dump(mode="json", exclude={"timestamp"})
        for m in state.get_linearized_messages(
            include_pending_input=True, indicate_current_incomplete=True
        )
    ]
    assert state.model_dump_json() == before
    ticks = []
    for i, content in enumerate(["Gold one", " more", "", "Gold two", " end"]):
        raw = {
            "cumulative_user_audio_at_tick_start_ms": i * 200,
            "audio_sent_duration_ms": 200,
        }
        if i == 0:
            raw["events"] = [
                {
                    "type": "input_audio_buffer.speech_started",
                    "item_id": "a",
                    "audio_start_ms": 0,
                },
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "item_id": "a",
                    "audio_end_ms": 400,
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "a",
                    "transcript": "Heard first",
                },
                {
                    "type": "input_audio_buffer.speech_started",
                    "item_id": "b",
                    "audio_start_ms": 600,
                },
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "item_id": "b",
                    "audio_end_ms": 1000,
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "b",
                    "transcript": "Heard second",
                },
            ]
        ticks.append(
            Tick(
                tick_id=i,
                timestamp="fixed",
                user_chunk=UserMessage(role="user", content=content),
                agent_chunk=AssistantMessage(
                    role="assistant", content="", raw_data=raw
                ),
            )
        )
    compute_proportional_user_transcripts(ticks)
    return runtime, [t.user_transcript for t in ticks]


def test_livekit_uses_shared_proportional_text_accounting(monkeypatch):
    from tau2.voice.audio_native.livekit import discrete_time_adapter as livekit
    from tau2.voice.audio_native.tick_result import (
        UtteranceTranscript,
        get_proportional_transcript,
    )

    seen = []

    def shared(chunks, transcripts):
        seen.append(chunks)
        return get_proportional_transcript(chunks, transcripts)

    monkeypatch.setattr(livekit, "get_proportional_transcript", shared)
    adapter = object.__new__(livekit.LiveKitCascadedAdapter)
    adapter._utterance_transcripts = {
        "a": UtteranceTranscript(
            item_id="a", audio_bytes_received=4, transcript_received="abcd"
        ),
        "zero": UtteranceTranscript(item_id="zero", transcript_received="not played"),
    }
    assert adapter._get_proportional_transcript([]) == ""
    assert (
        adapter._get_proportional_transcript(
            [(b"1", "a"), (b"2", "a"), (b"x", "zero"), (b"x", None)]
        )
        == "ab"
    )
    assert adapter._get_proportional_transcript([(b"123456", "a")]) == "cd"
    assert adapter._get_proportional_transcript([(b"x", "a")]) == ""
    assert len(seen) == 4
    assert adapter._utterance_transcripts["a"].audio_bytes_played == 9
