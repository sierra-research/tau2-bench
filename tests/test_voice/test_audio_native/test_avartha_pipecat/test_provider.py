import asyncio

import pytest

from tau2.voice.audio_native.avartha_pipecat.config import AvarthaPipecatConfig
from tau2.voice.audio_native.avartha_pipecat.provider import (
    HARNESS_TOOLS_PROTOCOL,
    AgentLLMStartedEvent,
    AgentPlaybackEvent,
    AgentStartedSpeakingEvent,
    AgentWordEvent,
    AvarthaPipecatProvider,
    AvarthaProtocolError,
    ProviderErrorEvent,
    SpeechEvent,
    ToolCallEvent,
    TranscriptEvent,
)

AGENT_URL = "https://platform.example/inference/serverless/pipecat/v1/public/agent-1"


def make_provider(**overrides):
    return AvarthaPipecatProvider(AvarthaPipecatConfig(base_url=AGENT_URL, **overrides))


async def deliver(provider, message):
    """Route one RTVI message the way a transport's reader does."""

    if await provider._accept_output(message):
        provider._route_message(message)


def test_unsupported_config_fields_are_rejected():
    with pytest.raises(ValueError):
        AvarthaPipecatConfig(base_url=AGENT_URL, session_id="preconfigured")


def test_route_message_tracks_handshakes_metadata_and_events():
    provider = make_provider()

    def server(data):
        provider._route_message(
            {"label": "rtvi-ai", "type": "server-message", "data": data}
        )

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

    assert provider.boot_ready is True
    assert provider._boot_settled.is_set()
    assert provider._registered.is_set()
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
    provider = make_provider()
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
        == "Avartha rejected register_tools: bad schema"
    )


@pytest.mark.parametrize("protocol", [None, "avartha-harness-tools/2", ""])
@pytest.mark.parametrize(
    "message_type", ["tools_registered", "tools_registration_error"]
)
def test_registration_rejects_incompatible_server_immediately(protocol, message_type):
    async def exercise():
        provider = make_provider()
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
            AvarthaProtocolError, match="Incompatible Avartha tool protocol"
        ) as error:
            await asyncio.wait_for(provider.register_tools([]), timeout=0.1)
        assert HARNESS_TOOLS_PROTOCOL in str(error.value)
        assert "--avartha-base-url" in str(error.value)
        assert (
            "missing (legacy tool format)" if protocol is None else repr(protocol)
        ) in str(error.value)

    asyncio.run(exercise())


def test_registration_waits_for_a_new_acknowledgement():
    async def exercise():
        provider = make_provider(tool_registration_timeout_seconds=0.01)
        provider._registered.set()

        async def client_message(*args):
            pass  # The agent never acknowledges registration.

        provider._client_message = client_message
        with pytest.raises(RuntimeError, match="did not acknowledge"):
            await provider.register_tools([])

    asyncio.run(exercise())


def test_tool_registration_and_results_use_versioned_harness_mcp():
    async def exercise():
        provider = make_provider(tool_registration_timeout_seconds=0.1)
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
    provider = make_provider()
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
        provider = make_provider()
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
            await provider.register_tools([])
        provider.reset_for_retry()
        assert provider._session_error is None
        assert provider._queue.empty()

    asyncio.run(exercise())


def test_reliable_outputs_are_acknowledged_and_routed_once():
    """Avartha resends each output every second until it is acknowledged."""

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
        provider = make_provider()
        messages = [playback, started, playback, artifact, playback, started]
        sent = []

        async def client_message(message_type, data):
            sent.append((message_type, data))

        provider._client_message = client_message

        async def read_events():
            for message in messages:
                await deliver(provider, message)
            events = []
            while not provider._queue.empty():
                events.append(provider._queue.get_nowait())
            return events

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
