"""Offline tests for Apriel realtime provider behavior.

These tests exercise the parts of the Apriel integration that tau relies on
without calling the hosted model: environment configuration, WebSocket
handshake/session payloads, provider error handling, and tick-runner event
processing.
"""

import asyncio
import base64
import json
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

import websockets

from tau2.config import TELEPHONY_ULAW_SILENCE
from tau2.environment.tool import Tool
from tau2.voice.audio_native.apriel import AprielRealtimeProvider
from tau2.voice.audio_native.apriel.tick_runner import TickRunner
from tau2.voice.audio_native.openai.events import (
    AudioDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    SpeechStartedEvent,
)
from tau2.voice.audio_native.tick_result import TickResult


def lookup(ids: list[str]) -> str:
    """Read records.

    Args:
        ids: Stable IDs returned previously.
    """
    return json.dumps(ids)


class TestProviderEnvironment(unittest.TestCase):
    def test_constructor_overrides_connection_settings(self):
        with patch.dict(os.environ, {}, clear=True):
            provider = AprielRealtimeProvider(
                api_key="constructor-key",
                model="Apriel-Voice-Test",
                base_url="wss://constructor.example/realtime",
                websocket_max_size=12345,
            )

        self.assertEqual(provider.api_key, "constructor-key")
        self.assertEqual(provider.model, "Apriel-Voice-Test")
        self.assertEqual(provider.base_url, "wss://constructor.example/realtime")
        self.assertEqual(provider._websocket_max_size, 12345)

    def test_provider_requires_base_url(self):
        with patch.dict(os.environ, {"APRIEL_API_KEY": "test-key"}, clear=True):
            with self.assertRaisesRegex(ValueError, "APRIEL_REALTIME_BASE_URL"):
                AprielRealtimeProvider()

    def test_provider_requires_websocket_base_url(self):
        with patch.dict(
            os.environ,
            {
                "APRIEL_REALTIME_BASE_URL": "https://apriel.example/realtime",
                "APRIEL_API_KEY": "test-key",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must start with ws:// or wss://"):
                AprielRealtimeProvider()

    def test_provider_requires_api_key(self):
        with patch.dict(
            os.environ,
            {"APRIEL_REALTIME_BASE_URL": "wss://apriel.example/realtime"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "APRIEL_API_KEY"):
                AprielRealtimeProvider()


class AprielProviderTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.received = []
        self.headers = []
        self.connections = []

        async def handler(ws):
            self.headers.append(ws.request.headers.get("Authorization"))
            self.connections.append(ws)
            await ws.send(
                json.dumps(
                    {
                        "type": "session.created",
                        "session": {"id": "session-local"},
                    }
                )
            )
            async for raw in ws:
                message = json.loads(raw)
                self.received.append(message)
                if message["type"] == "session.update":
                    await ws.send(json.dumps({"type": "session.updated"}))

        self.server = await websockets.serve(handler, "127.0.0.1", 0)
        self.url = "ws://127.0.0.1:%s/v1/realtime" % (
            self.server.sockets[0].getsockname()[1]
        )
        self.env = patch.dict(
            os.environ,
            {
                "APRIEL_REALTIME_BASE_URL": self.url,
                "APRIEL_API_KEY": "test-token",
            },
            clear=False,
        )
        self.env.start()
        self.provider = AprielRealtimeProvider(model="Apriel-Voice-Test")
        self.tools = [Tool(lookup)]
        await self.provider.connect()
        await self.provider.configure_session(
            "POLICY\n  exact whitespace\n",
            self.tools,
            vad_config=None,
            modality="audio",
        )

    async def asyncTearDown(self):
        await self.provider.disconnect()
        self.env.stop()
        self.server.close()
        await self.server.wait_closed()


class TestProviderConfiguration(AprielProviderTestCase):
    async def test_handshake_and_schema(self):
        self.assertEqual(self.headers, ["Bearer test-token"])
        expected_tools = self.provider._format_tools_for_api(self.tools)
        self.assertEqual(
            self.received[0],
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": "POLICY\n  exact whitespace\n",
                    "output_modalities": ["audio", "text"],
                    "tools": expected_tools,
                    "tool_choice": "auto",
                    "enable_thinking": True,
                    "pre_tool_speech": True,
                    "temperature": 1.0,
                    "max_output_tokens": 8192,
                    "apriel": {
                        "extensions": True,
                        "vad_level": True,
                        "trace_vllm_payloads": True,
                    },
                    "audio": {
                        "input": {
                            "format": "g711_ulaw",
                            "noise_reduction": {"type": "near_field"},
                        },
                        "output": {"format": "g711_ulaw"},
                    },
                },
            },
        )
        self.assertEqual(self.provider.session_id, "session-local")

    async def test_constructor_overrides_session_settings(self):
        provider = AprielRealtimeProvider(
            model="Apriel-Voice-Test",
            enable_thinking=False,
            pre_tool_speech=False,
            temperature=0.7,
            max_tokens=1234,
            reasoning_effort="high",
            trace_enabled=False,
        )

        self.assertEqual(provider.reasoning_effort, "high")
        self.assertEqual(
            provider._session_parameters,
            {
                "enable_thinking": False,
                "pre_tool_speech": False,
                "temperature": 0.7,
                "max_output_tokens": 1234,
            },
        )
        self.assertNotIn("apriel", provider._session_parameters)


class TestProviderConnection(AprielProviderTestCase):
    async def test_unexpected_close_is_error(self):
        stream = self.provider.receive_events()
        pending = asyncio.create_task(anext(stream))
        await self.connections[0].close()
        try:
            with self.assertRaises(RuntimeError):
                await pending
        finally:
            await stream.aclose()


class TestTickRunnerContract(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def audio(item_id: str, data: bytes = b"\x10\x20" * 600) -> AudioDeltaEvent:
        return AudioDeltaEvent(
            type="response.output_audio.delta",
            item_id=item_id,
            delta=base64.b64encode(data).decode(),
        )

    def make_runner(self) -> TickRunner:
        provider = Mock(spec=AprielRealtimeProvider)
        provider.truncate_item = AsyncMock()
        return TickRunner(
            provider=provider,
            tick_duration_ms=50,
            bytes_per_tick=400,
            send_audio_instant=True,
            buffer_until_complete=False,
        )

    @staticmethod
    def tick_result(**kwargs) -> TickResult:
        defaults = {
            "tick_number": 1,
            "audio_sent_bytes": 0,
            "audio_sent_duration_ms": 0,
            "bytes_per_tick": 400,
        }
        defaults.update(kwargs)
        return TickResult(**defaults)

    async def test_tool_calls_are_gated_until_response_done(self):
        runner = self.make_runner()
        result = self.tick_result()
        event = FunctionCallArgumentsDoneEvent(
            type="response.function_call_arguments.done",
            call_id="call-1",
            name="lookup",
            arguments='{"ids": ["A"]}',
        )

        await runner._process_event(result, event)
        self.assertEqual(result.tool_calls, [])

        await runner._process_event(result, ResponseDoneEvent(type="response.done"))
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "lookup")
        self.assertEqual(result.tool_calls[0].arguments, {"ids": ["A"]})

    async def test_speech_started_truncates_buffered_audio(self):
        runner = self.make_runner()
        runner.buffered_agent_audio = [(b"\x10" * 400, "item-1")]
        result = self.tick_result(
            bytes_per_second=8000,
            cumulative_user_audio_at_tick_start_ms=0,
        )

        await runner._process_event(
            result,
            SpeechStartedEvent(
                type="input_audio_buffer.speech_started",
                item_id="user-item",
                audio_start_ms=25,
            ),
        )

        self.assertEqual(runner.buffered_agent_audio, [])
        self.assertTrue(result.was_truncated)
        runner.provider.truncate_item.assert_awaited_once()

    async def test_late_audio_for_skipped_item_is_silenced(self):
        runner = self.make_runner()
        result = self.tick_result(
            bytes_per_second=8000,
            skip_item_id="old-item",
        )

        await runner._process_event(result, self.audio("old-item"))

        self.assertEqual(result.agent_audio_bytes, 0)
        self.assertEqual(result.truncated_audio_bytes, 1200)
        self.assertEqual(
            result.get_played_agent_audio(),
            TELEPHONY_ULAW_SILENCE * 400,
        )

    async def test_response_created_and_done_track_lifecycle(self):
        runner = self.make_runner()
        result = self.tick_result()

        await runner._process_event(
            result, ResponseCreatedEvent(type="response.created")
        )
        self.assertTrue(runner.response_active)

        await runner._process_event(result, ResponseDoneEvent(type="response.done"))
        self.assertFalse(runner.response_active)
