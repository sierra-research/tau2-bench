"""CPU-only native adapter contract tests; no model calls or paid services."""

import asyncio
import base64
import json
import unittest
from unittest.mock import AsyncMock, patch

import websockets

from tau2.config import TELEPHONY_ULAW_SILENCE
from tau2.environment.tool import Tool
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.launa import (
    DiscreteTimeLaunaAdapter,
    LaunaRealtimeWSProvider,
)
from tau2.voice.audio_native.launa.config import validate_endpoint
from tau2.voice.audio_native.launa.provider import _NoRedirectConnect
from tau2.voice.audio_native.openai.events import (
    AudioDeltaEvent,
    AudioTranscriptDeltaEvent,
    AudioTranscriptDoneEvent,
    FunctionCallArgumentsDoneEvent,
    ResponseCancelledEvent,
    SpeechStartedEvent,
)
from tau2.voice.audio_native.tick_result import TickResult


def lookup(ids: list[str]) -> str:
    """Read records.

    Args:
        ids: Stable IDs returned previously.
    """
    return json.dumps(ids)


class LaunaTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.received = []
        self.headers = []
        self.connections = []

        async def handler(ws):
            self.headers.append(ws.request.headers.get("Authorization"))
            self.connections.append(ws)
            async for raw in ws:
                message = json.loads(raw)
                self.received.append(message)
                if message["type"] == "session.update":
                    await ws.send(json.dumps({"type": "latency.ping", "id": "init"}))
                elif message == {"type": "latency.pong", "id": "init"}:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "session.created",
                                "session": {"id": "session-local"},
                            }
                        )
                    )

        self.server = await websockets.serve(handler, "127.0.0.1", 0)
        self.url = "ws://127.0.0.1:%s/ws" % self.server.sockets[0].getsockname()[1]
        self.provider = LaunaRealtimeWSProvider(
            endpoint=self.url, api_key="test-token", allow_insecure_local=True
        )
        with patch(
            "tau2.voice.audio_native.launa.provider._NoRedirectConnect",
            wraps=_NoRedirectConnect,
        ) as connect:
            await self.provider.connect()
            self.application_headers = connect.call_args.kwargs["additional_headers"]
        self.tools = [Tool(lookup)]
        await self.provider.configure_session(
            "POLICY\n  exact whitespace\n", self.tools, None
        )

    async def asyncTearDown(self):
        await self.provider.disconnect()
        self.server.close()
        await self.server.wait_closed()


class TestProviderConfiguration(LaunaTestCase):
    async def test_handshake_and_schema(self):
        self.assertEqual(self.headers, ["Bearer test-token"])
        self.assertEqual(
            self.application_headers, {"Authorization": "Bearer test-token"}
        )
        expected_tools = [
            {k: t[k] for k in ("name", "description", "parameters")}
            for t in self.provider._format_tools_for_api(self.tools)
        ]
        self.assertEqual(
            self.received[0],
            {
                "type": "session.update",
                "instructions": "POLICY\n  exact whitespace\n",
                "tools": expected_tools,
                "audio": {"input": {"type": "audio/pcm", "sample_rate_hz": 24000}},
            },
        )
        self.assertEqual(self.received[1], {"type": "latency.pong", "id": "init"})
        self.assertEqual(self.provider.session_id, "session-local")


class TestProviderConnection(LaunaTestCase):
    async def test_unexpected_close_is_error(self):
        stream = self.provider.receive_events()
        pending = asyncio.create_task(anext(stream))
        await self.connections[0].close()
        try:
            with self.assertRaises(RuntimeError):
                await pending
        finally:
            await stream.aclose()

    async def test_redirect_does_not_forward_credentials(self):
        def redirect(connection, request):
            response = connection.respond(302, "Moved")
            response.headers["Location"] = self.url
            return response

        async def unused_handler(ws):
            self.fail("Redirect must not open a socket")

        async with websockets.serve(
            unused_handler, "127.0.0.1", 0, process_request=redirect
        ) as server:
            url = "ws://127.0.0.1:%s/ws" % server.sockets[0].getsockname()[1]
            with self.assertRaises(websockets.InvalidStatus):
                await _NoRedirectConnect(
                    url, additional_headers={"Authorization": "Bearer secret"}
                )
        self.assertEqual(self.headers, ["Bearer test-token"])


class TestProviderAudioSend(LaunaTestCase):
    async def test_input_resampling_state_survives_interrupt(self):
        first = DiscreteTimeLaunaAdapter(100, provider=self.provider)
        control = DiscreteTimeLaunaAdapter(100, provider=self.provider)
        audio = bytes(range(256)) * 4
        self.assertEqual(
            first._converter.convert_input(audio),
            control._converter.convert_input(audio),
        )
        result = TickResult(
            tick_number=1,
            audio_sent_bytes=0,
            audio_sent_duration_ms=0,
            bytes_per_tick=first.bytes_per_tick,
        )
        await first._process_event(
            result,
            SpeechStartedEvent(type="input_audio_buffer.speech_started", item_id="u"),
        )
        await first._process_event(
            result, ResponseCancelledEvent(type="response.cancelled")
        )
        self.assertEqual(
            first._converter.convert_input(audio),
            control._converter.convert_input(audio),
        )


class TestProviderAudioReceive(LaunaTestCase):
    @staticmethod
    def audio(item_id, pcm=b"\x10\x20" * 4800):
        return AudioDeltaEvent(
            type="response.output_audio.delta",
            item_id=item_id,
            delta=base64.b64encode(pcm).decode(),
        )

    async def run_event_ticks(self, batches):
        adapter = DiscreteTimeLaunaAdapter(
            50, provider=self.provider, send_audio_instant=True
        )
        results = []
        with (
            patch.object(self.provider, "send_audio", new_callable=AsyncMock),
            patch.object(
                self.provider,
                "receive_events_for_duration",
                new=AsyncMock(side_effect=batches),
            ),
        ):
            for tick in range(len(batches)):
                results.append(
                    await adapter._async_run_tick(TELEPHONY_ULAW_SILENCE * 400, tick)
                )
        return adapter, results

    async def test_cancellation_alone_stops_buffered_playback_and_late_audio(self):
        new_pcm = b"\x20\x30" * 4800
        adapter, ticks = await self.run_event_ticks(
            [
                [
                    self.audio("old"),
                    AudioTranscriptDoneEvent(
                        type="response.output_audio_transcript.done",
                        item_id="old",
                        transcript="The cancelled response must not keep playing.",
                    ),
                ],
                [ResponseCancelledEvent(type="response.cancelled")],
                [self.audio("old")],
                [self.audio("new", new_pcm), self.audio("old")],
                [],
            ]
        )
        self.assertNotEqual(
            ticks[0].get_played_agent_audio(), TELEPHONY_ULAW_SILENCE * 400
        )
        for tick in ticks[1:3]:
            self.assertEqual(
                tick.get_played_agent_audio(), TELEPHONY_ULAW_SILENCE * 400
            )
            self.assertEqual(tick.agent_audio_bytes, 0)
            self.assertEqual(tick.proportional_transcript, "")
        self.assertTrue(ticks[1].was_truncated)
        self.assertGreater(ticks[1].truncated_audio_bytes, 0)
        expected = StreamingTelephonyConverter(
            input_sample_rate=24000, output_sample_rate=24000
        ).convert_output(new_pcm)
        self.assertEqual(ticks[3].get_played_agent_audio(), expected[:400])
        self.assertEqual(ticks[4].get_played_agent_audio(), expected[400:800])
        self.assertTrue(all(item == "new" for _, item in adapter._buffered_agent_audio))

    async def test_both_interruption_event_orders_stop_playback(self):
        speech = SpeechStartedEvent(
            type="input_audio_buffer.speech_started", item_id="u"
        )
        cancel = ResponseCancelledEvent(type="response.cancelled")
        for events in ([speech, cancel], [cancel, speech], [cancel, cancel]):
            with self.subTest(events=[event.type for event in events]):
                adapter, ticks = await self.run_event_ticks(
                    [[self.audio("old")], events, [self.audio("old")], []]
                )
                for tick in ticks[1:]:
                    self.assertEqual(
                        tick.get_played_agent_audio(), TELEPHONY_ULAW_SILENCE * 400
                    )
                self.assertEqual(adapter._buffered_agent_audio, [])

    async def test_cancel_remembers_item_even_when_playback_buffer_is_empty(self):
        _, ticks = await self.run_event_ticks(
            [
                [self.audio("old", b"\x10\x20" * 600)],
                [ResponseCancelledEvent(type="response.cancelled")],
                [self.audio("old")],
            ]
        )
        self.assertLess(ticks[0].agent_audio_bytes, 400)
        self.assertEqual(
            ticks[2].get_played_agent_audio(), TELEPHONY_ULAW_SILENCE * 400
        )

    async def test_cancel_discards_audio_received_in_the_same_tick(self):
        adapter, ticks = await self.run_event_ticks(
            [
                [self.audio("old"), ResponseCancelledEvent(type="response.cancelled")],
                [self.audio("old")],
            ]
        )
        for tick in ticks:
            self.assertEqual(tick.agent_audio_bytes, 0)
            self.assertEqual(
                tick.get_played_agent_audio(), TELEPHONY_ULAW_SILENCE * 400
            )
        self.assertTrue(ticks[0].was_truncated)
        self.assertEqual(adapter._buffered_agent_audio, [])

    async def test_cancel_preserves_preceding_speech_start_timing(self):
        speech = SpeechStartedEvent(
            type="input_audio_buffer.speech_started", item_id="u", audio_start_ms=75
        )
        _, expected = await self.run_event_ticks([[self.audio("old")], [speech], []])
        adapter, actual = await self.run_event_ticks(
            [
                [self.audio("old")],
                [speech, ResponseCancelledEvent(type="response.cancelled")],
                [],
            ]
        )
        self.assertEqual(
            actual[1].get_played_agent_audio(), expected[1].get_played_agent_audio()
        )
        self.assertEqual(actual[1].interruption_audio_start_ms, 75)
        self.assertEqual(
            actual[2].get_played_agent_audio(), TELEPHONY_ULAW_SILENCE * 400
        )
        self.assertEqual(adapter._buffered_agent_audio, [])

    async def test_native_tick_lifecycle(self):
        adapter = DiscreteTimeLaunaAdapter(
            100, provider=self.provider, send_audio_instant=True
        )
        await self.connections[0].send(
            json.dumps(
                {
                    "type": "response.output_audio.delta",
                    "item_id": "a",
                    "delta": base64.b64encode(bytes(9600)).decode(),
                }
            )
        )
        await self.connections[0].send(
            json.dumps(
                {
                    "type": "response.output_audio_transcript.done",
                    "item_id": "a",
                    "transcript": "Result confirmed.",
                }
            )
        )
        await self.connections[0].send(
            json.dumps(
                {
                    "type": "agent-turn.tool-call.fired",
                    "tool_call_id": "c",
                    "name": "lookup",
                    "arguments": '{"ids":["A"]}',
                }
            )
        )
        result = await adapter._async_run_tick(bytes([255]) * 800, 1)
        self.assertEqual(result.tick_number, 1)
        self.assertEqual(result.audio_sent_duration_ms, 100)
        self.assertEqual(len(result.get_played_agent_audio()), 800)
        self.assertEqual(result.tool_calls[0].arguments, {"ids": ["A"]})
        self.assertTrue(adapter._buffered_agent_audio)
        sent = [m for m in self.received if m["type"] == "input_audio_buffer.append"]
        self.assertTrue(sent)
        self.assertGreater(len(base64.b64decode(sent[0]["audio"])), 800)


class TestProviderTranscription(LaunaTestCase):
    async def test_final_transcript_not_duplicated(self):
        adapter = DiscreteTimeLaunaAdapter(100, provider=self.provider)
        result = TickResult(tick_number=1, audio_sent_bytes=0, audio_sent_duration_ms=0)
        final = AudioTranscriptDoneEvent(
            type="response.output_audio_transcript.done",
            item_id="a",
            transcript="Hello.",
        )
        await adapter._process_event(result, final)
        await adapter._process_event(result, final)
        self.assertEqual(
            adapter._utterance_transcripts["a"].transcript_received, "Hello."
        )
        await adapter._process_event(
            result,
            AudioTranscriptDeltaEvent(
                type="response.output_audio_transcript.delta", item_id="b", delta="Hi."
            ),
        )
        await adapter._process_event(
            result,
            AudioTranscriptDoneEvent(
                type="response.output_audio_transcript.done",
                item_id="b",
                transcript="Hi.",
            ),
        )
        self.assertEqual(adapter._utterance_transcripts["b"].transcript_received, "Hi.")


class TestProviderToolFlow(LaunaTestCase):
    async def test_audio_and_tool_result_wire(self):
        audio = bytes(range(128)) * 2
        await self.provider.send_audio(audio)
        await self.provider.send_tool_result("call-1", ' { "found": true } ')
        await self.provider.truncate_item("item-1", 0, 20)
        await asyncio.sleep(0.02)
        self.assertEqual(
            self.received[2:],
            [
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio).decode(),
                },
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": "call-1",
                        "output": ' { "found": true } ',
                    },
                },
            ],
        )

    async def test_ping_tool_and_interrupt_events(self):
        for payload in [
            {"type": "latency.ping", "id": 42},
            {
                "type": "agent-turn.tool-call.fired",
                "tool_call_id": "c1",
                "name": "lookup",
                "arguments": '{"ids":["A"]}',
            },
            {"type": "agent-turn.interrupted"},
        ]:
            await self.connections[0].send(json.dumps(payload))
        stream = self.provider.receive_events()
        try:
            first = await anext(stream)
            second = await anext(stream)
            self.assertIsInstance(first, FunctionCallArgumentsDoneEvent)
            self.assertEqual(first.arguments, '{"ids":["A"]}')
            self.assertIsInstance(second, ResponseCancelledEvent)
        finally:
            await stream.aclose()
        await asyncio.sleep(0.02)
        self.assertIn({"type": "latency.pong", "id": 42}, self.received)


class SecurityTests(unittest.TestCase):
    def test_endpoint_guards(self):
        self.assertEqual(
            validate_endpoint("wss://example.test/ws"), "wss://example.test/ws"
        )
        for value in [
            "ws://example.test/ws",
            "wss://u:p@example.test/ws",
            "wss://example.test/ws?k=v",
            "wss://example.test/ws#x",
            "",
        ]:
            with self.assertRaises(ValueError):
                validate_endpoint(value)

    def test_bad_keys_rejected(self):
        for key in ["bad key", "\x00", "x" * 4097]:
            with self.assertRaises(ValueError):
                LaunaRealtimeWSProvider(endpoint="wss://example.test/ws", api_key=key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
