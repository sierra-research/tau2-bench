import asyncio
import json
import unittest

from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.apriel.discrete_time_adapter import (
    DiscreteTimeAprielAdapter,
)
from tau2.voice.audio_native.apriel.provider import AprielRealtimeProvider
from tau2.voice.audio_native.apriel.tick_runner import TickRunner
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript

try:
    from tau2.voice.audio_native.apriel.events import parse_realtime_event
except ImportError:
    from tau2.voice.audio_native.openai.events import parse_realtime_event

import base64


def event(kind, **kw):
    return parse_realtime_event({"type": kind, **kw})


def audio(item="old", response="r1", n=800, **kw):
    return event(
        "response.output_audio.delta",
        item_id=item,
        response_id=response,
        delta=base64.b64encode(b"x" * n).decode(),
        **kw,
    )


def result(start=1000):
    return TickResult(
        tick_number=1,
        audio_sent_bytes=1600,
        audio_sent_duration_ms=200,
        bytes_per_tick=1600,
        bytes_per_second=8000,
        cumulative_user_audio_at_tick_start_ms=start,
    )


class FakeProvider:
    def __init__(self):
        self.truncated = []
        self.sent = []
        self.events = []

    async def truncate_item(self, **kw):
        self.truncated.append(kw)

    async def send_tool_result(self, call_id, result, request_response=True):
        self.sent.append(("result", call_id, result, request_response))

    async def request_response(self):
        self.sent.append(("continue",))

    async def send_audio(self, data):
        self.sent.append(("audio", data))

    async def receive_events(self):
        while self.events:
            yield self.events.pop(0)
        yield event("timeout")


def runner(buffer_complete=False, **kw):
    return TickRunner(
        provider=FakeProvider(),
        tick_duration_ms=200,
        bytes_per_tick=1600,
        send_audio_instant=True,
        buffer_until_complete=buffer_complete,
        audio_format=TELEPHONY_AUDIO_FORMAT,
        **kw,
    )


class Wire:
    def __init__(self, clocks=None):
        self.sent = []
        self.clocks = clocks or {"playback_clock": "media", "turn_clock": "media"}

    async def send(self, data):
        self.sent.append(json.loads(data))

    async def recv(self):
        return json.dumps(
            {"type": "session.updated", "session": {"apriel": self.clocks}}
        )

    async def close(self):
        pass


class Provider(AprielRealtimeProvider):
    @property
    def is_connected(self):
        return self.ws is not None


class ProviderContracts(unittest.IsolatedAsyncioTestCase):
    async def test_session_requests_and_verifies_media_clocks(self):
        p = Provider(
            api_key="offline-dummy", base_url="ws://unused", trace_enabled=True
        )
        p.ws = Wire()
        await p.configure_session("test", [], None, "audio")
        self.assertEqual(
            p.ws.sent[0]["session"].get("apriel"),
            {
                "playback_clock": "media",
                "turn_clock": "media",
                "extensions": True,
                "vad_level": True,
                "trace_vllm_payloads": True,
            },
        )
        await p.send_audio(b"abc")
        self.assertEqual(p.effective_session["apriel"]["turn_clock"], "media")

    async def test_session_fails_closed_without_effective_media_clock(self):
        for clocks in (
            {"playback_clock": "media"},
            {"playback_clock": "wall", "turn_clock": "media"},
        ):
            p = Provider(api_key="offline-dummy", base_url="ws://unused")
            p.ws = Wire(clocks)
            with self.assertRaisesRegex(RuntimeError, "clock"):
                await p.configure_session("test", [], None, "audio")
            with self.assertRaises(RuntimeError):
                await p.send_audio(b"abc")

    async def test_disconnect_revokes_configuration(self):
        p = Provider(api_key="offline-dummy", base_url="ws://unused")
        p.ws = Wire()
        await p.configure_session("test", [], None, "audio")
        await p.disconnect()
        p.ws = Wire()
        with self.assertRaises(RuntimeError):
            await p.send_audio(b"x")

    async def test_decode_preserves_ownership(self):
        for kind in ("response.created", "response.done", "response.cancelled"):
            with self.subTest(kind=kind):
                e = event(kind, response={"id": "r1", "status": "completed"})
                self.assertEqual(getattr(e, "response_id", None), "r1")
        for kind in (
            "response.output_audio.done",
            "response.output_audio_transcript.done",
            "response.function_call_arguments.done",
            "output_audio_buffer.cleared",
        ):
            with self.subTest(kind=kind):
                e = event(
                    kind,
                    response_id="r1",
                    item_id="i1",
                    call_id="c1",
                    name="lookup",
                    arguments="{}",
                )
                self.assertEqual(getattr(e, "response_id", None), "r1")
                self.assertEqual(getattr(e, "item_id", None), "i1")


class AudioContracts(unittest.IsolatedAsyncioTestCase):
    async def test_authoritative_clear_removes_streaming_and_partial_audio(self):
        for size in (800, 1600, 9600):
            for complete in (False, True):
                tr, res = runner(complete), result()
                await tr._process_event(res, audio(n=size))
                await tr._process_event(
                    res, event("output_audio_buffer.cleared", response_id="r1")
                )
                self.assertEqual(
                    res.agent_audio_bytes
                    + sum(len(a) for a, _ in tr.buffered_agent_audio)
                    + sum(p.total_audio_bytes for p in tr.pending_utterances.values()),
                    0,
                )
                await tr._process_event(res, audio(n=800))
                self.assertEqual(res.agent_audio_bytes, 0)

    async def test_clear_before_audio_fences_old_response_and_stale_clear_preserves_new(
        self,
    ):
        for onset in (
            "input_audio_buffer.speech_started",
            "output_audio_buffer.cleared",
        ):
            tr, res = runner(), result()
            await tr._process_event(
                res, event("response.created", response={"id": "r1"})
            )
            await tr._process_event(
                res, event(onset, response_id="r1", audio_start_ms=1000)
            )
            await tr._process_event(
                res, event("response.created", response={"id": "r2"})
            )
            await tr._process_event(res, audio("new", "r2"))
            await tr._process_event(res, audio("old", "r1"))
            await tr._process_event(
                res, event("output_audio_buffer.cleared", response_id="r1")
            )
            self.assertEqual(res.agent_audio_chunks, [(b"x" * 800, "new")])

    async def test_played_cursor_uses_tick_prefix_not_total_backlog(self):
        for backlog in (200, 400, 1000, 2000):
            for onset in (0, 50, 100, 199, 200):
                tr, res = runner(), result()
                res.agent_audio_chunks = [(b"x" * (backlog * 8), "old")]
                tr.utterance_transcripts["old"] = UtteranceTranscript(
                    item_id="old", audio_bytes_played=8000
                )
                self.assertEqual(
                    tr._calculate_item_audio_played_ms(
                        item_id="old", result=res, audio_start_ms=1000 + onset
                    ),
                    1000 + onset,
                )

    async def test_interruption_targets_playing_item_and_keeps_only_played_prefix(self):
        for first_ms in (50, 100, 200, 500):
            for onset in (25, 75, 150):
                tr, res = runner(), result()
                res.agent_audio_chunks = [
                    (b"a" * (first_ms * 8), "first"),
                    (b"b" * 8000, "second"),
                ]
                await tr._process_event(
                    res,
                    event(
                        "input_audio_buffer.speech_started", audio_start_ms=1000 + onset
                    ),
                )
                target = "first" if onset < first_ms else "second"
                self.assertEqual(
                    tr.provider.truncated[0],
                    dict(
                        item_id=target,
                        content_index=0,
                        audio_end_ms=onset if target == "first" else onset - first_ms,
                    ),
                )
                self.assertEqual(res.agent_audio_bytes, onset * 8)

    async def test_transcript_advances_only_for_played_prefix(self):
        tr, res = runner(), result()
        await tr._process_event(res, audio(n=16000))
        await tr._process_event(
            res,
            event(
                "response.output_audio_transcript.delta",
                response_id="r1",
                item_id="old",
                delta="one two three four five six seven eight nine ten ",
            ),
        )
        tr.utterance_transcripts["old"].audio_bytes_played = 8000
        await tr._process_event(
            res, event("input_audio_buffer.speech_started", audio_start_ms=1100)
        )
        tr._buffer_excess_audio(res)
        tr._get_proportional_transcript(res)
        self.assertEqual(tr.utterance_transcripts["old"].audio_bytes_played, 8800)
        self.assertEqual(tr.provider.truncated[0]["audio_end_ms"], 1100)

    async def test_duplicate_audio_event_does_not_play_twice(self):
        tr, res = runner(), result()
        e = audio(event_id="same")
        await tr._process_event(res, e)
        await tr._process_event(res, e)
        self.assertEqual(res.agent_audio_bytes, 800)


class ToolContracts(unittest.IsolatedAsyncioTestCase):
    async def test_tools_require_completed_response_and_valid_object_arguments(self):
        for status in ("completed", "cancelled", "failed", "incomplete"):
            for args in ("{}", '{"x":1}', "broken", "[]", "null", ""):
                tr, res = runner(), result()
                await tr._process_event(
                    res,
                    event(
                        "response.function_call_arguments.done",
                        response_id="r1",
                        call_id="c1",
                        name="lookup",
                        arguments=args,
                    ),
                )
                await tr._process_event(
                    res, event("response.done", response={"id": "r1", "status": status})
                )
                self.assertEqual(
                    len(res.tool_calls),
                    int(status == "completed" and args in ("{}", '{"x":1}')),
                )

    async def test_duplicate_calls_and_terminal_events_dispatch_once(self):
        tr, res = runner(), result()
        e = event(
            "response.function_call_arguments.done",
            response_id="r1",
            call_id="c1",
            name="lookup",
            arguments="{}",
        )
        for _ in range(3):
            await tr._process_event(res, e)
        for _ in range(2):
            await tr._process_event(
                res,
                event("response.done", response={"id": "r1", "status": "completed"}),
            )
        await tr._process_event(res, e)
        await tr._process_event(
            res, event("response.done", response={"id": "r1", "status": "completed"})
        )
        self.assertEqual([c.id for c in res.tool_calls], ["c1"])

    async def test_stale_terminal_only_finalizes_its_own_calls(self):
        tr, res = runner(), result()
        for rid in ("r1", "r2"):
            await tr._process_event(
                res, event("response.created", response={"id": rid})
            )
            await tr._process_event(
                res,
                event(
                    "response.function_call_arguments.done",
                    response_id=rid,
                    call_id=rid,
                    name="lookup",
                    arguments="{}",
                ),
            )
        await tr._process_event(
            res, event("response.done", response={"id": "r1", "status": "failed"})
        )
        self.assertTrue(tr.response_active)
        await tr._process_event(
            res, event("response.done", response={"id": "r2", "status": "completed"})
        )
        self.assertEqual([c.id for c in res.tool_calls], ["r2"])

    async def test_audio_clear_preserves_completed_tool_declarations(self):
        tr, res = runner(), result()
        await tr._process_event(res, event("response.created", response={"id": "r1"}))
        await tr._process_event(
            res,
            event(
                "response.function_call_arguments.done",
                response_id="r1",
                call_id="c1",
                name="lookup",
                arguments="{}",
            ),
        )
        await tr._process_event(
            res, event("output_audio_buffer.cleared", response_id="r1")
        )
        await tr._process_event(
            res, event("response.done", response={"id": "r1", "status": "completed"})
        )
        self.assertEqual([c.id for c in res.tool_calls], ["c1"])


class BoundsAndTemplateContracts(unittest.IsolatedAsyncioTestCase):
    def adapter(self):
        adapter = object.__new__(DiscreteTimeAprielAdapter)
        DiscreteTimeAdapter.__init__(
            adapter, tick_duration_ms=200, send_audio_instant=True
        )
        adapter._provider = FakeProvider()
        adapter._tick_runner = runner()
        adapter._tick_runner.provider = adapter._provider
        adapter._connected = False
        return adapter

    async def test_audio_overflow_fails_closed(self):
        for complete in (False, True):
            tr, res = runner(complete), result()
            tr.max_buffer_bytes = 1000
            await tr._process_event(res, audio(n=800))
            with self.assertRaisesRegex(RuntimeError, "limit"):
                await tr._process_event(res, audio(n=800))
            self.assertLessEqual(
                res.agent_audio_bytes
                + sum(p.total_audio_bytes for p in tr.pending_utterances.values()),
                1000,
            )

    async def test_response_fences_are_bounded_without_forgetting_stale_ids(self):
        tr, res = runner(), result()
        tr.max_state_entries = 2
        for rid in ("r1", "r2"):
            await tr._process_event(
                res, event("response.created", response={"id": rid})
            )
            await tr._process_event(
                res, event("output_audio_buffer.cleared", response_id=rid)
            )
        with self.assertRaisesRegex(RuntimeError, "limit"):
            await tr._process_event(
                res, event("response.created", response={"id": "r3"})
            )
        await tr._process_event(res, audio("old", "r1"))
        self.assertEqual(res.agent_audio_bytes, 0)

    async def test_adapter_uses_base_template_with_fifo_and_transcript(self):
        adapter = self.adapter()
        adapter._provider.events = [
            event("response.created", response={"id": "r1"}),
            audio(n=4800),
            event(
                "response.output_audio_transcript.done",
                response_id="r1",
                item_id="old",
                transcript="one two three four five six",
            ),
        ]
        outputs = []
        for tick in range(3):
            outputs.append(await adapter._async_run_tick(b"\xff" * 1600, tick))
        self.assertEqual([r.agent_audio_bytes for r in outputs], [1600] * 3)
        self.assertEqual(adapter._cumulative_user_audio_ms, 600)
        self.assertEqual(adapter._current_tick_number, 2)
        self.assertEqual(adapter._utterance_transcripts["old"].audio_bytes_played, 4800)
        self.assertEqual(
            "".join(r.proportional_transcript for r in outputs).split(),
            "one two three four five six".split(),
        )

    async def test_adapter_partial_tick_transcript_matches_audio(self):
        adapter = self.adapter()
        adapter._provider.events = [
            event("response.created", response={"id": "r1"}),
            audio(n=16000),
            event(
                "response.output_audio_transcript.done",
                response_id="r1",
                item_id="old",
                transcript="one two three four five six",
            ),
            event("input_audio_buffer.speech_started", audio_start_ms=100),
        ]
        res = await adapter._async_run_tick(b"\xff" * 1600, 0)
        self.assertEqual(res.agent_audio_bytes, 800)
        self.assertEqual(adapter._utterance_transcripts["old"].audio_bytes_played, 800)
        self.assertEqual(adapter._buffered_agent_audio, [])
        self.assertEqual(adapter._provider.truncated[0]["audio_end_ms"], 100)

    async def test_complete_batch_results_continue_exactly_once_after_audio_clear(self):
        adapter = self.adapter()
        tr, res = adapter._tick_runner, result()
        for cid in ("a", "b"):
            await tr._process_event(
                res,
                event(
                    "response.function_call_arguments.done",
                    response_id="r1",
                    call_id=cid,
                    name="lookup",
                    arguments="{}",
                ),
            )
        await tr._process_event(
            res, event("response.done", response={"id": "r1", "status": "completed"})
        )
        adapter.send_tool_result("a", "A")
        await adapter._flush_pending_tool_results()
        self.assertEqual(adapter._provider.sent, [])
        await tr._process_event(
            res, event("output_audio_buffer.cleared", response_id="r1")
        )
        adapter.send_tool_result("a", "A")
        adapter.send_tool_result("b", "B")
        await adapter._flush_pending_tool_results()
        await adapter._flush_pending_tool_results()
        adapter.send_tool_result("a", "A")
        await adapter._flush_pending_tool_results()
        self.assertEqual(
            adapter._provider.sent,
            [("result", "a", "A", False), ("result", "b", "B", False), ("continue",)],
        )

    async def test_clear_buffers_resets_runner_and_tool_state(self):
        adapter = self.adapter()
        await adapter._tick_runner._process_event(result(), audio())
        adapter.clear_buffers()
        self.assertEqual(adapter._tick_runner.utterance_transcripts, {})
        self.assertEqual(adapter._tick_runner.pending_utterances, {})


class OrderingContracts(unittest.IsolatedAsyncioTestCase):
    adapter = BoundsAndTemplateContracts.adapter

    async def test_clear_after_speech_keeps_already_played_tick_prefix(self):
        adapter = self.adapter()
        adapter._provider.events = [
            event("response.created", response={"id": "r1"}),
            audio(n=16000),
            event("input_audio_buffer.speech_started", audio_start_ms=100),
            event("output_audio_buffer.cleared", response_id="r1"),
        ]
        res = await adapter._async_run_tick(b"\xff" * 1600, 0)
        self.assertEqual(res.agent_audio_bytes, 800)

    async def test_new_response_after_partial_interruption_plays_next_tick(self):
        adapter = self.adapter()
        adapter._provider.events = [
            event("response.created", response={"id": "r1"}),
            audio(n=16000),
            event("input_audio_buffer.speech_started", audio_start_ms=100),
            event("response.created", response={"id": "r2"}),
            audio("new", "r2", 1600),
        ]
        first = await adapter._async_run_tick(b"\xff" * 1600, 0)
        second = await adapter._async_run_tick(b"\xff" * 1600, 1)
        self.assertEqual(first.agent_audio_chunks, [(b"x" * 800, "old")])
        self.assertEqual(second.agent_audio_chunks, [(b"x" * 1600, "new")])

    async def test_buffer_complete_respects_fifo_when_later_item_completes_first(self):
        tr, res = runner(True), result()
        await tr._process_event(res, audio("first", "r1"))
        await tr._process_event(res, audio("second", "r1"))
        for kind in (
            "response.output_audio.done",
            "response.output_audio_transcript.done",
        ):
            await tr._process_event(
                res, event(kind, item_id="second", response_id="r1")
            )
        self.assertEqual(res.agent_audio_chunks, [])
        for kind in (
            "response.output_audio.done",
            "response.output_audio_transcript.done",
        ):
            await tr._process_event(res, event(kind, item_id="first", response_id="r1"))
        self.assertEqual(
            [item for _, item in res.agent_audio_chunks], ["first", "second"]
        )

    async def test_duplicate_created_for_old_response_does_not_replace_active(self):
        tr, res = runner(), result()
        for rid in ("r1", "r2", "r1"):
            await tr._process_event(
                res, event("response.created", response={"id": rid})
            )
        self.assertEqual(tr.active_response_id, "r2")

    async def test_completed_tool_arguments_reject_nonfinite_json(self):
        tr, res = runner(), result()
        await tr._process_event(
            res,
            event(
                "response.function_call_arguments.done",
                response_id="r1",
                call_id="c1",
                name="lookup",
                arguments='{"x":NaN}',
            ),
        )
        await tr._process_event(
            res, event("response.done", response={"id": "r1", "status": "completed"})
        )
        self.assertEqual(res.tool_calls, [])

    async def test_configure_failure_closes_connection(self):
        adapter = self.adapter()

        class FailedProvider(Provider):
            async def connect(self):
                self.ws = Wire({"playback_clock": "wall"})

        adapter._provider = FailedProvider(
            api_key="offline-dummy", base_url="ws://unused"
        )
        with self.assertRaises(RuntimeError):
            await adapter._async_connect("test", [], None, "audio")
        self.assertIsNone(adapter.provider.ws)


class ResourceContracts(unittest.IsolatedAsyncioTestCase):
    adapter = BoundsAndTemplateContracts.adapter

    async def test_empty_audio_events_do_not_accumulate_fragments(self):
        tr, res = runner(True), result()
        for _ in range(20):
            await tr._process_event(res, audio(n=0))
        self.assertEqual(
            sum(len(p.audio_chunks) for p in tr.pending_utterances.values()), 0
        )

    async def test_tool_state_limit_is_session_wide(self):
        tr, res = runner(), result()
        tr.max_state_entries = 2
        for rid, cid in (("r1", "a"), ("r2", "b")):
            await tr._process_event(
                res,
                event(
                    "response.function_call_arguments.done",
                    response_id=rid,
                    call_id=cid,
                    name="lookup",
                    arguments="{}",
                ),
            )
        with self.assertRaisesRegex(RuntimeError, "limit"):
            await tr._process_event(
                res,
                event(
                    "response.function_call_arguments.done",
                    response_id="r2",
                    call_id="c",
                    name="lookup",
                    arguments="{}",
                ),
            )

    async def test_tool_result_size_and_pending_queue_are_bounded(self):
        adapter = self.adapter()
        adapter._tick_runner.max_text_chars = 4
        with self.assertRaisesRegex(RuntimeError, "limit"):
            adapter.send_tool_result("a", "12345")
        adapter._tick_runner.max_state_entries = 2
        adapter.send_tool_result("a", "1")
        adapter.send_tool_result("b", "2")
        with self.assertRaisesRegex(RuntimeError, "limit"):
            adapter.send_tool_result("c", "3")

    async def test_tick_disconnect_cancels_send_task(self):
        tr = runner()
        cancelled = asyncio.Event()

        class BrokenProvider(FakeProvider):
            async def send_audio(self, data):
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

            async def receive_events(self):
                raise RuntimeError("disconnected")
                yield

        tr.provider = BrokenProvider()
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            await tr.execute_tick(b"x", result(), asyncio.get_running_loop().time())
        self.assertTrue(cancelled.is_set())

    async def test_terminal_fences_audio_and_preserves_dispatched_results(self):
        tr, res = runner(), result()
        await tr._process_event(
            res,
            event(
                "response.function_call_arguments.done",
                response_id="r1",
                call_id="a",
                name="lookup",
                arguments="{}",
            ),
        )
        await tr._process_event(
            res, event("response.done", response={"id": "r1", "status": "completed"})
        )
        await tr._process_event(res, event("response.cancelled", response_id="r1"))
        await tr._process_event(res, audio())
        await tr.flush_tool_results([("a", "A", True, False)])
        self.assertEqual(res.agent_audio_bytes, 0)
        self.assertEqual(tr.provider.sent, [("result", "a", "A", False), ("continue",)])

    async def test_tool_continuation_wire_retains_full_history(self):
        p = Provider(api_key="offline-dummy", base_url="ws://unused")
        p.ws = Wire()
        await p.send_tool_result("a", "A", False)
        await p.send_tool_result("b", "B", False)
        await p.request_response()
        self.assertEqual(
            [e["type"] for e in p.ws.sent],
            ["conversation.item.create", "conversation.item.create", "response.create"],
        )
        self.assertEqual(p.ws.sent[-1], {"type": "response.create"})


class DuplicateCompletionContracts(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_item_done_does_not_block_next_utterance(self):
        tr, res = runner(True), result()
        await tr._process_event(res, audio("first", "r1"))
        for kind in (
            "response.output_audio.done",
            "response.output_audio_transcript.done",
        ):
            await tr._process_event(res, event(kind, item_id="first", response_id="r1"))
        await tr._process_event(
            res, event("response.output_audio.done", item_id="first", response_id="r1")
        )
        await tr._process_event(res, audio("second", "r1"))
        for kind in (
            "response.output_audio.done",
            "response.output_audio_transcript.done",
        ):
            await tr._process_event(
                res, event(kind, item_id="second", response_id="r1")
            )
        self.assertEqual(
            [item for _, item in res.agent_audio_chunks], ["first", "second"]
        )
        self.assertEqual(tr.pending_utterances, {})

    async def test_audio_after_item_done_is_fenced(self):
        tr, res = runner(), result()
        await tr._process_event(res, audio())
        await tr._process_event(
            res, event("response.output_audio.done", item_id="old", response_id="r1")
        )
        await tr._process_event(res, audio())
        self.assertEqual(res.agent_audio_bytes, 800)

    async def test_manual_commit_without_provider_support_preserves_compatibility(self):
        tr = runner(manual_commit=True)
        await tr.execute_tick(b"x", result(), asyncio.get_running_loop().time())
        self.assertEqual(tr.provider.sent, [("audio", b"x")])

    async def test_interrupted_unplayed_complete_buffer_truncates_at_zero(self):
        tr, res = runner(True), result()
        await tr._process_event(res, event("response.created", response={"id": "r1"}))
        await tr._process_event(res, audio())
        await tr._process_event(
            res, event("input_audio_buffer.speech_started", audio_start_ms=1100)
        )
        self.assertEqual(
            tr.provider.truncated,
            [{"item_id": "old", "content_index": 0, "audio_end_ms": 0}],
        )


class NamespaceCompatibilityContracts(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_old_namespace_echo_with_both_media_clocks(self):
        class OldWire(Wire):
            async def recv(self):
                return json.dumps(
                    {"type": "session.updated", "session": {"aurion": self.clocks}}
                )

        p = Provider(api_key="offline-dummy", base_url="ws://unused")
        p.ws = OldWire()
        await p.configure_session("test", [], None, "audio")
        await p.send_audio(b"x")
        self.assertIn("apriel", p.ws.sent[0]["session"])
        self.assertNotIn("aurion", p.ws.sent[0]["session"])

    async def test_accepts_equal_aliases_and_rejects_conflicting_echoes(self):
        class AliasWire(Wire):
            def __init__(self, conflict):
                super().__init__()
                self.conflict = conflict

            async def recv(self):
                return json.dumps(
                    {
                        "type": "session.updated",
                        "session": {
                            "apriel": self.clocks,
                            "aurion": dict(self.clocks, turn_clock="wall")
                            if self.conflict
                            else self.clocks,
                        },
                    }
                )

        for conflict in (False, True):
            p = Provider(api_key="offline-dummy", base_url="ws://unused")
            p.ws = AliasWire(conflict)
            if conflict:
                with self.assertRaisesRegex(RuntimeError, "conflict"):
                    await p.configure_session("test", [], None, "audio")
            else:
                await p.configure_session("test", [], None, "audio")


class UnscopedClearContracts(unittest.IsolatedAsyncioTestCase):
    async def test_clear_without_id_removes_completed_response_backlog(self):
        tr, res = runner(), result()
        await tr._process_event(res, event("response.created", response={"id": "r1"}))
        await tr._process_event(res, audio(n=9600))
        await tr._process_event(
            res, event("response.done", response={"id": "r1", "status": "completed"})
        )
        await tr._process_event(res, event("output_audio_buffer.cleared"))
        self.assertEqual(res.agent_audio_bytes, 0)
        self.assertEqual(tr.utterance_transcripts, {})

    async def test_ambiguous_unscoped_clear_fails_without_erasing_new_audio(self):
        tr, res = runner(), result()
        await tr._process_event(res, audio("old", "r1"))
        await tr._process_event(res, audio("new", "r2"))
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            await tr._process_event(res, event("output_audio_buffer.cleared"))
        self.assertEqual([i for _, i in res.agent_audio_chunks], ["old", "new"])


class ClearedHeadFifoContracts(unittest.IsolatedAsyncioTestCase):
    adapter = BoundsAndTemplateContracts.adapter

    async def test_clear_or_cancel_head_releases_completed_newer_response(self):
        for terminal in ("output_audio_buffer.cleared", "response.cancelled"):
            with self.subTest(terminal=terminal):
                adapter = self.adapter()
                adapter._tick_runner.buffer_until_complete = True
                adapter._provider.events = [
                    audio("old", "r1", 800),
                    audio("new", "r2", 1600),
                    event(
                        "response.output_audio.done", response_id="r2", item_id="new"
                    ),
                    event(
                        "response.output_audio_transcript.done",
                        response_id="r2",
                        item_id="new",
                        transcript="new response",
                    ),
                    event(
                        "response.done", response={"id": "r2", "status": "completed"}
                    ),
                    event(terminal, response_id="r1"),
                    audio("old", "r1", 800),
                ]
                first = await adapter._async_run_tick(b"\xff" * 1600, 0)
                self.assertEqual(first.agent_audio_chunks, [(b"x" * 1600, "new")])
                self.assertEqual(first.proportional_transcript, "new response")
                self.assertEqual(first.truncated_audio_bytes, 1600)
                self.assertEqual(adapter._tick_runner.pending_utterances, {})
                self.assertEqual(adapter._tick_runner.item_responses["new"], "r2")
                self.assertEqual(
                    adapter._tick_runner.responses["r2"].terminal, "completed"
                )
                self.assertEqual(
                    adapter._utterance_transcripts["new"].audio_bytes_played, 1600
                )
                self.assertNotIn("old", adapter._utterance_transcripts)
                second = await adapter._async_run_tick(b"\xff" * 1600, 1)
                self.assertEqual(second.agent_audio_bytes, 0)
                self.assertEqual(second.proportional_transcript, "")

    async def test_clearing_head_drains_only_complete_fifo_prefix(self):
        tr, res = runner(True), result()
        for item, rid in (
            ("old", "r1"),
            ("ready", "r2"),
            ("waiting", "r3"),
            ("later", "r4"),
        ):
            await tr._process_event(res, audio(item, rid, 800))
        for item, rid in (("ready", "r2"), ("later", "r4")):
            for kind in (
                "response.output_audio.done",
                "response.output_audio_transcript.done",
            ):
                await tr._process_event(res, event(kind, item_id=item, response_id=rid))
            await tr._process_event(
                res, event("response.done", response={"id": rid, "status": "completed"})
            )
        await tr._process_event(
            res, event("output_audio_buffer.cleared", response_id="r1")
        )
        self.assertEqual(res.agent_audio_chunks, [(b"x" * 800, "ready")])
        self.assertEqual(list(tr.pending_utterances), ["waiting", "later"])
        self.assertEqual(res.truncated_audio_bytes, 800)
        for kind in (
            "response.output_audio.done",
            "response.output_audio_transcript.done",
        ):
            await tr._process_event(
                res, event(kind, item_id="waiting", response_id="r3")
            )
        self.assertEqual(
            [item for _, item in res.agent_audio_chunks], ["ready", "waiting", "later"]
        )
        self.assertEqual(tr.pending_utterances, {})


class UnexecutedToolSettlementContracts(unittest.IsolatedAsyncioTestCase):
    async def test_noncompleted_response_settles_all_published_calls_without_execution(
        self,
    ):
        for status in ("cancelled", "failed", "incomplete"):
            with self.subTest(status=status):
                tr, res = runner(), result()
                for cid, arguments in (("valid", "{}"), ("invalid", "[]")):
                    await tr._process_event(
                        res,
                        event(
                            "response.function_call_arguments.done",
                            response_id="r1",
                            call_id=cid,
                            name="lookup",
                            arguments=arguments,
                        ),
                    )
                done = event("response.done", response={"id": "r1", "status": status})
                await tr._process_event(res, done)
                await tr._process_event(res, done)
                self.assertEqual(res.tool_calls, [])
                await tr.flush_tool_results([])
                await tr.flush_tool_results([])
                self.assertEqual(
                    [entry[1] for entry in tr.provider.sent], ["valid", "invalid"]
                )
                for entry in tr.provider.sent:
                    self.assertEqual(entry[0], "result")
                    self.assertFalse(entry[3])
                    outcome = json.loads(entry[2])
                    self.assertIs(outcome["executed"], False)
                    self.assertEqual(outcome["status"], status)
                    self.assertIn("not executed", outcome["error"])

    async def test_completed_invalid_and_conflicting_calls_settle_and_continue_once(
        self,
    ):
        tr, res = runner(), result()
        for cid, arguments in (
            ("invalid", "[]"),
            ("conflict", "{}"),
            ("conflict", '{"x":1}'),
        ):
            await tr._process_event(
                res,
                event(
                    "response.function_call_arguments.done",
                    response_id="r1",
                    call_id=cid,
                    name="lookup",
                    arguments=arguments,
                ),
            )
        await tr._process_event(
            res, event("response.done", response={"id": "r1", "status": "completed"})
        )
        self.assertEqual(res.tool_calls, [])
        await tr.flush_tool_results([])
        await tr.flush_tool_results([])
        self.assertEqual(
            [entry[:2] for entry in tr.provider.sent],
            [("result", "invalid"), ("result", "conflict"), ("continue",)],
        )
        for entry in tr.provider.sent[:-1]:
            outcome = json.loads(entry[2])
            self.assertIs(outcome["executed"], False)
            self.assertEqual(outcome["status"], "rejected")

    async def test_rejections_wait_for_valid_sibling_result_before_batch_continuation(
        self,
    ):
        tr, res = runner(), result()
        for cid, arguments in (("invalid", "[]"), ("valid", "{}")):
            await tr._process_event(
                res,
                event(
                    "response.function_call_arguments.done",
                    response_id="r1",
                    call_id=cid,
                    name="lookup",
                    arguments=arguments,
                ),
            )
        await tr._process_event(
            res, event("response.done", response={"id": "r1", "status": "completed"})
        )
        self.assertEqual([call.id for call in res.tool_calls], ["valid"])
        await tr.flush_tool_results([])
        self.assertEqual(tr.provider.sent, [])
        await tr._process_event(
            res, event("output_audio_buffer.cleared", response_id="r1")
        )
        await tr.flush_tool_results([("valid", "actual tool result", True, False)])
        await tr.flush_tool_results([("valid", "actual tool result", True, False)])
        self.assertEqual(
            [entry[:2] for entry in tr.provider.sent],
            [("result", "invalid"), ("result", "valid"), ("continue",)],
        )
        self.assertEqual(tr.provider.sent[1][2], "actual tool result")


class MultipleActiveResponseFenceContracts(unittest.IsolatedAsyncioTestCase):
    async def test_completed_filler_cannot_hide_silent_open_main_from_speech_fence(
        self,
    ):
        tr, res = runner(), result()
        await tr._process_event(res, event("response.created", response={"id": "main"}))
        await tr._process_event(
            res, event("response.created", response={"id": "filler"})
        )
        await tr._process_event(
            res,
            event("response.done", response={"id": "filler", "status": "completed"}),
        )
        main_remains_active = tr.response_active
        await tr._process_event(
            res, event("input_audio_buffer.speech_started", audio_start_ms=1000)
        )
        await tr._process_event(res, event("response.created", response={"id": "next"}))
        await tr._process_event(res, audio("old-main", "main", 800))
        await tr._process_event(res, audio("new-answer", "next", 1600))
        self.assertEqual(res.agent_audio_chunks, [(b"x" * 1600, "new-answer")])
        self.assertTrue(tr.responses["main"].audio_cleared)
        self.assertTrue(main_remains_active)


class AmbiguousSettlementContracts(unittest.IsolatedAsyncioTestCase):
    async def test_automatic_settlement_rejects_reused_call_id_before_any_batch_send(
        self,
    ):
        tr, res = runner(), result()
        await tr._process_event(
            res,
            event(
                "response.function_call_arguments.done",
                response_id="r1",
                call_id="shared",
                name="lookup",
                arguments="{}",
            ),
        )
        await tr._process_event(
            res, event("response.done", response={"id": "r1", "status": "completed"})
        )
        self.assertEqual([call.id for call in res.tool_calls], ["shared"])
        for cid in ("unique-first", "shared"):
            await tr._process_event(
                res,
                event(
                    "response.function_call_arguments.done",
                    response_id="r2",
                    call_id=cid,
                    name="lookup",
                    arguments="{}",
                ),
            )
        await tr._process_event(
            res, event("response.done", response={"id": "r2", "status": "cancelled"})
        )
        with self.assertRaisesRegex(RuntimeError, "ambiguous.*response"):
            await tr.flush_tool_results([])
        self.assertEqual(tr.provider.sent, [])
        self.assertTrue(tr.responses["r1"].dispatched)
        self.assertEqual(tr.responses["r1"].results, {})
