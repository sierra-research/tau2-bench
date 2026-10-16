import asyncio
from urllib.parse import parse_qs, urlsplit

import pytest

from tau2.voice.audio_native.ava_zero_pipecat import provider as provider_module
from tau2.voice.audio_native.ava_zero_pipecat.config import AvaZeroPipecatConfig
from tau2.voice.audio_native.ava_zero_pipecat.provider import (
    HARNESS_TOOLS_PROTOCOL,
    AgentLLMStartedEvent,
    AgentPlaybackEvent,
    AgentStartedSpeakingEvent,
    AgentWordEvent,
    AvaZeroBootError,
    AvaZeroPipecatProvider,
    AvaZeroProtocolError,
    ProviderErrorEvent,
    SpeechEvent,
    ToolCallEvent,
    TranscriptEvent,
    build_websocket_url,
)


def test_websocket_url_carries_agent_session_and_token(monkeypatch):
    monkeypatch.setenv("AVA_ZERO_TOKEN", "secret")
    config = AvaZeroPipecatConfig(
        base_url="https://voice.example/root?existing=yes",
        agent="conductor-v2",
        websocket_token_env="AVA_ZERO_TOKEN",
    )

    parts = urlsplit(build_websocket_url(config, session_id="session_1"))
    assert (parts.scheme, parts.netloc, parts.path) == (
        "wss",
        "voice.example",
        "/root/ws-client",
    )
    assert parse_qs(parts.query) == {
        "existing": ["yes"],
        "agent": ["conductor-v2"],
        "session_id": ["session_1"],
        "token": ["secret"],
    }


def test_unsupported_config_fields_are_rejected():
    with pytest.raises(ValueError):
        AvaZeroPipecatConfig(session_id="preconfigured")


def test_connect_log_redacts_websocket_authentication(monkeypatch):
    placeholder_token = "placeholder-websocket-secret"
    monkeypatch.setenv("AVA_ZERO_TOKEN", placeholder_token)
    messages = []

    class WebSocket:
        async def close(self):
            return None

    async def websocket_connect(*_args, **_kwargs):
        return WebSocket()

    async def send_message(_message):
        return None

    async def read_loop():
        return None

    async def await_boot():
        return None

    async def create_session(_system_prompt):
        return "session-1"

    provider = AvaZeroPipecatProvider(
        AvaZeroPipecatConfig(
            base_url="wss://voice.example?api_key=base-secret",
            websocket_token_env="AVA_ZERO_TOKEN",
        )
    )
    monkeypatch.setattr(provider, "_create_session", create_session)
    provider._message_label = "rtvi-ai"
    provider._protocol_version = "2.0.0"
    monkeypatch.setattr(provider, "_load_protocol", lambda: None)
    monkeypatch.setattr(provider, "_send_message", send_message)
    monkeypatch.setattr(provider, "_read_loop", read_loop)
    monkeypatch.setattr(provider, "_await_boot", await_boot)
    monkeypatch.setattr(provider_module.websockets, "connect", websocket_connect)
    monkeypatch.setattr(provider_module.logger, "info", messages.append)

    async def exercise():
        await provider.connect()
        await provider.disconnect(request_cost_report=False)

    asyncio.run(exercise())

    logged = "\n".join(messages)
    assert placeholder_token not in logged
    assert "base-secret" not in logged
    assert "token=REDACTED" in logged
    assert "api_key=REDACTED" in logged


def test_websocket_url_requires_named_token(monkeypatch):
    monkeypatch.delenv("MISSING_AVA_ZERO_TOKEN", raising=False)
    with pytest.raises(ValueError, match="not set"):
        build_websocket_url(
            AvaZeroPipecatConfig(websocket_token_env="MISSING_AVA_ZERO_TOKEN")
        )


def test_connect_resets_state_from_the_completed_session(monkeypatch):
    class WebSocket:
        async def close(self):
            return None

    async def websocket_connect(*_args, **_kwargs):
        return WebSocket()

    async def read_loop():
        return None

    async def create_session(_system_prompt):
        return "public-session"

    provider = AvaZeroPipecatProvider(
        AvaZeroPipecatConfig(
            base_url="wss://voice.example",
            boot_timeout_seconds=0.01,
        )
    )
    monkeypatch.setattr(provider, "_create_session", create_session)
    provider._message_label = "rtvi-ai"
    provider._protocol_version = "2.0.0"
    connection_number = 0

    async def send_message(_message):
        if connection_number == 1:
            provider._note_boot({"ready": True})

    async def connect():
        nonlocal connection_number
        connection_number += 1
        await provider.connect()

    monkeypatch.setattr(provider, "_load_protocol", lambda: None)
    monkeypatch.setattr(provider, "_send_message", send_message)
    monkeypatch.setattr(provider, "_read_loop", read_loop)
    monkeypatch.setattr(provider_module.websockets, "connect", websocket_connect)

    async def exercise():
        await connect()
        provider.session_id = "completed-session"
        provider.cost_report = {"total_cost_usd": 0.25}
        provider._queue.put_nowait(TranscriptEvent("stale"))
        await provider.disconnect(request_cost_report=False)

        assert provider.session_id == "completed-session"
        assert provider.cost_report == {"total_cost_usd": 0.25}

        with pytest.raises(AvaZeroBootError, match="never settled"):
            await connect()
        assert provider.session_id is None
        assert provider.cost_report is None
        assert provider._queue.empty()
        await provider.disconnect(request_cost_report=False)

    asyncio.run(exercise())


def test_route_message_tracks_handshakes_metadata_and_events():
    provider = AvaZeroPipecatProvider()
    provider._message_label = "rtvi-ai"

    def server(data):
        provider._route_message(
            {"label": "rtvi-ai", "type": "server-message", "data": data}
        )

    server({"type": "conductor_session", "sid": "sid-1"})
    server({"type": "session_boot", "ready": False, "stages": {}})
    assert not provider._boot_settled.is_set()
    server({"type": "session_boot", "ready": True, "stages": {}})
    server({"type": "tools_registered", "protocol": HARNESS_TOOLS_PROTOCOL, "count": 2})
    server(
        {
            "type": "tool_call",
            "id": "call-1",
            "name": "lookup",
            "arguments": {"id": "A-1"},
        }
    )
    server(
        {
            "type": "cost_report",
            "data": {"currency": "USD", "total_cost_usd": 0.25},
        }
    )
    provider._route_message(
        {
            "label": "rtvi-ai",
            "type": "bot-transcription",
            "data": {"text": "It shipped."},
        }
    )
    provider._route_message({"label": "rtvi-ai", "type": "bot-llm-started"})
    provider._route_message({"label": "rtvi-ai", "type": "bot-started-speaking"})
    provider._route_message(
        {"label": "rtvi-ai", "type": "bot-tts-text", "data": {"text": "It"}}
    )
    server(
        {
            "type": "assistant_playback",
            "commit_id": "c1",
            "delivery_status": "cut",
            "text": "It",
        }
    )
    provider._route_message(
        {"label": "rtvi-ai", "type": "user-started-speaking", "data": {}}
    )
    provider._route_message({"label": "rtvi-ai", "type": "bot-interrupted", "data": {}})

    assert provider.session_id == "sid-1"
    assert provider.boot_ready is True
    assert provider._boot_settled.is_set()
    assert provider._registered.is_set()
    assert provider.cost_report["total_cost_usd"] == 0.25
    assert provider._cost_received.is_set()
    events = [provider._queue.get_nowait() for _ in range(8)]
    assert events == [
        ToolCallEvent("call-1", "lookup", {"id": "A-1"}),
        TranscriptEvent("It shipped."),
        AgentLLMStartedEvent(),
        AgentStartedSpeakingEvent(),
        AgentWordEvent("It"),
        AgentPlaybackEvent("It", cut=True),
        SpeechEvent("speech_started"),
        SpeechEvent("interrupted"),
    ]


def test_registration_error_settles_waiter_and_preserves_message():
    provider = AvaZeroPipecatProvider()
    provider._message_label = "rtvi-ai"
    provider._route_message(
        {
            "label": "rtvi-ai",
            "type": "server-message",
            "data": {
                "type": "tools_registration_error",
                "protocol": HARNESS_TOOLS_PROTOCOL,
                "error": {"code": "invalid_registration", "message": "bad schema"},
            },
        }
    )

    assert provider._registered.is_set()
    assert (
        str(provider._registration_error)
        == "Ava Zero rejected register_tools: bad schema"
    )


@pytest.mark.parametrize("protocol", [None, "avartha-harness-tools/2", ""])
@pytest.mark.parametrize(
    "message_type", ["tools_registered", "tools_registration_error"]
)
def test_registration_rejects_incompatible_server_immediately(protocol, message_type):
    async def exercise():
        provider = AvaZeroPipecatProvider()
        provider._message_label = "rtvi-ai"
        # A previous successful registration must not satisfy this handshake.
        provider._registered.set()

        async def client_message(*args):
            provider._route_message(
                {
                    "label": "rtvi-ai",
                    "type": "server-message",
                    "data": {
                        "type": message_type,
                        "protocol": protocol,
                        "count": 0,
                    },
                }
            )

        provider._client_message = client_message
        with pytest.raises(
            AvaZeroProtocolError, match="Incompatible Ava Zero tool protocol"
        ) as error:
            await asyncio.wait_for(provider.register_tools([]), timeout=0.1)
        assert HARNESS_TOOLS_PROTOCOL in str(error.value)
        assert "--ava-zero-base-url" in str(error.value)
        assert (
            "missing (legacy tool format)" if protocol is None else repr(protocol)
        ) in str(error.value)

    asyncio.run(exercise())


def test_registration_waits_for_a_new_acknowledgement():
    async def exercise():
        provider = AvaZeroPipecatProvider(
            AvaZeroPipecatConfig(tool_registration_timeout_seconds=0.01)
        )
        provider._registered.set()

        async def client_message(*args):
            pass  # The current server never acknowledges registration.

        provider._client_message = client_message
        with pytest.raises(RuntimeError, match="did not acknowledge"):
            await provider.register_tools([])

    asyncio.run(exercise())


def test_create_session_installs_policy_outside_tool_registration(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"id": "public-session"}

    class Client:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return Response()

    monkeypatch.setattr(provider_module.httpx, "AsyncClient", Client)
    provider = AvaZeroPipecatProvider(
        AvaZeroPipecatConfig(base_url="ws://voice.example", agent="conductor-v2")
    )

    session_id = asyncio.run(provider._create_session("Retail policy"))

    assert session_id == "public-session"
    assert captured["url"] == "http://voice.example/api/sessions"
    assert captured["json"]["agent"] == {
        "base_agent_id": "conductor-v2",
        "prompt": "Retail policy",
        "tools": [],
        "capabilities": [],
        "chat_artifacts_enabled": False,
    }


def test_tool_registration_and_results_use_versioned_harness_mcp():
    async def exercise():
        provider = AvaZeroPipecatProvider(
            AvaZeroPipecatConfig(tool_registration_timeout_seconds=0.1)
        )
        provider._message_label = "rtvi-ai"
        messages = []

        async def client_message(message_type, data):
            messages.append((message_type, data))
            if message_type == "register_tools":
                provider._route_message(
                    {
                        "label": "rtvi-ai",
                        "type": "server-message",
                        "data": {
                            "type": "tools_registered",
                            "protocol": HARNESS_TOOLS_PROTOCOL,
                            "server": "harness",
                            "count": len(data["tools"]),
                            "names": [tool["name"] for tool in data["tools"]],
                        },
                    }
                )

        provider._client_message = client_message
        tools = [
            {
                "name": "lookup",
                "description": "Look up an order.",
                "inputSchema": {"type": "object", "additionalProperties": False},
            }
        ]
        await provider.register_tools(tools)
        await provider.send_tool_result(
            "call-1",
            '{"status":"shipped"}',
            request_response=False,
            is_error=True,
        )
        return messages, tools

    messages, tools = asyncio.run(exercise())

    assert messages == [
        (
            "register_tools",
            {"protocol": HARNESS_TOOLS_PROTOCOL, "tools": tools},
        ),
        (
            "tool_result",
            {
                "protocol": HARNESS_TOOLS_PROTOCOL,
                "id": "call-1",
                "result": {
                    "content": [{"type": "text", "text": '{"status":"shipped"}'}],
                    "isError": True,
                },
            },
        ),
    ]


def test_boot_error_only_settles_when_terminal():
    provider = AvaZeroPipecatProvider()
    provider._message_label = "rtvi-ai"
    provider._route_message(
        {
            "label": "rtvi-ai",
            "type": "server-message",
            "data": {
                "type": "session_boot",
                "ready": False,
                "error": "Tools failed",
            },
        }
    )

    assert provider.boot_ready is False
    assert provider.boot_error == "Tools failed"
    assert provider._boot_settled.is_set()


@pytest.mark.parametrize("stage", ["boot", "registration", "before_registration"])
def test_fatal_rtvi_error_wakes_handshakes_with_original_cause(stage):
    async def exercise():
        provider = AvaZeroPipecatProvider()
        provider._message_label = "rtvi-ai"
        detail = "Daily model limit reached: 1,000,000 uncached input tokens/day."

        def fail():
            provider._route_message(
                {
                    "label": "rtvi-ai",
                    "type": "error",
                    "data": {
                        "error": detail,
                        "fatal": True,
                    },
                }
            )

        async def client_message(*args):
            fail()

        provider._client_message = client_message
        if stage == "boot":
            asyncio.get_running_loop().call_soon(fail)
            operation = provider._await_boot()
        else:
            if stage == "before_registration":
                fail()
            operation = provider.register_tools([])
        with pytest.raises(RuntimeError, match="Daily model limit reached"):
            await asyncio.wait_for(operation, timeout=0.2)
        event = provider._queue.get_nowait()
        assert isinstance(event, ProviderErrorEvent)
        assert detail in event.message
        with pytest.raises(RuntimeError, match="Daily model limit reached"):
            await provider.send_audio(b"audio", 16000)
        provider.reset_for_retry()
        assert provider._session_error is None
        assert provider._queue.empty()

    asyncio.run(exercise())


def test_clean_remote_close_wakes_registration_instead_of_timing_out():
    async def exercise():
        provider = AvaZeroPipecatProvider()

        class ClosedSocket:
            async def __aiter__(self):
                if False:
                    yield

        provider._ws = ClosedSocket()

        async def client_message(*args):
            await provider._read_loop()

        provider._client_message = client_message
        with pytest.raises(RuntimeError, match="WebSocket closed before disconnect"):
            await asyncio.wait_for(provider.register_tools([]), timeout=0.2)

    asyncio.run(exercise())


def test_reliable_outputs_are_acknowledged_and_routed_once():
    """Ava Zero resends each output every second until it is acknowledged."""

    def server_message(data):
        return {"label": "rtvi-ai", "type": "server-message", "data": data}

    playback = server_message(
        {
            "type": "assistant_playback",
            "protocol": "avartha-anchor-output/1",
            "commit_id": "u3",
            "delivery_status": "complete",
            "text": "Hi, how can I help you today?",
            "output_event_id": "playback-final:u3",
            "output_sequence": 17,
        }
    )
    artifact = server_message(
        {
            "type": "anchor_output",
            "output_event_id": "artifact:a1",
            "output_sequence": 18,
        }
    )
    started = {"label": "rtvi-ai", "type": "bot-started-speaking"}

    async def exercise():
        provider = AvaZeroPipecatProvider()
        provider._load_protocol()
        messages = [playback, started, playback, artifact, playback, started]

        class Serializer:
            async def deserialize(self, raw):
                return provider._input_transport_frame(message=messages[raw[0]])

        class Socket:
            async def __aiter__(self):
                for index in range(len(messages)):
                    yield bytes([index])

        sent = []

        async def client_message(message_type, data):
            sent.append((message_type, data))

        provider._serializer = Serializer()
        provider._client_message = client_message

        async def read_events():
            provider._ws = Socket()
            await provider._read_loop()
            events = []
            while not provider._queue.empty():
                events.append(provider._queue.get_nowait())
            return [e for e in events if not isinstance(e, ProviderErrorEvent)]

        first = await read_events()
        acked = list(sent)
        provider.reset_for_retry()
        provider._client_message = client_message
        again = await read_events()
        return first, acked, again

    first, acked, again = asyncio.run(exercise())

    heard = AgentPlaybackEvent(text="Hi, how can I help you today?", cut=False)
    assert first == [heard, AgentStartedSpeakingEvent(), AgentStartedSpeakingEvent()]
    # Every copy is acknowledged in case an earlier ack was lost; the artifact
    # never is, since the simulated caller never sees it.
    assert acked == [("output_ack", {"event_id": "playback-final:u3"})] * 3
    # A new session has not delivered anything yet.
    assert again == first


def test_pinned_pipecat_protobuf_round_trip():
    async def round_trip():
        provider = AvaZeroPipecatProvider()
        provider._load_protocol()
        outgoing = provider._output_audio_frame(
            audio=b"\x00\x00" * 160,
            sample_rate=16000,
            num_channels=1,
        )
        wire = await provider._serializer.serialize(outgoing)
        incoming = await provider._serializer.deserialize(wire)
        return provider, incoming

    provider, incoming = asyncio.run(round_trip())

    assert provider._message_label == "rtvi-ai"
    assert provider._protocol_version == "2.0.0"
