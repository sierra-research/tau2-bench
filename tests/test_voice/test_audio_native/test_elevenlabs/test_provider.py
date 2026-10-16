import asyncio
import json
from collections import deque

import pytest
from websockets.protocol import State

from tau2.voice.audio_native.elevenlabs.provider import (
    ElevenLabsAgentsProvider,
    parse_elevenlabs_audio_format,
)


class FakeWebSocket:
    def __init__(self, incoming=()):
        self.state = State.OPEN
        self.sent = []
        self.incoming = deque(json.dumps(event) for event in incoming)

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def recv(self):
        return self.incoming.popleft()

    async def close(self):
        self.state = State.CLOSED


def test_requires_an_elevenlabs_agent_or_speech_engine_id(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="beginning with 'agent_' or 'seng_'"):
        ElevenLabsAgentsProvider(agent_id="gpt-realtime-2")

    assert ElevenLabsAgentsProvider(agent_id="seng_external").agent_id == (
        "seng_external"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("pcm_16000", ("pcm", 16000)),
        ("pcm_44100", ("pcm", 44100)),
        ("ulaw_8000", ("ulaw", 8000)),
    ],
)
def test_parse_audio_format(value, expected):
    assert parse_elevenlabs_audio_format(value) == expected


@pytest.mark.parametrize(
    "value", ["mp3_44100", "pcm_12345", "ulaw_16000", "pcm", "pcm_zero", ""]
)
def test_reject_unsupported_audio_format(value):
    with pytest.raises(ValueError, match="Unsupported ElevenLabs audio format"):
        parse_elevenlabs_audio_format(value)


def test_initiation_payload_overrides_prompt_and_waits_for_user():
    payload = ElevenLabsAgentsProvider._build_initiation_payload("domain policy")

    agent = payload["conversation_config_override"]["agent"]
    assert payload["type"] == "conversation_initiation_client_data"
    assert agent["first_message"] == ""
    assert agent["prompt"]["prompt"] == "domain policy"


def test_connect_sends_overrides_and_records_negotiated_formats(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    websocket = FakeWebSocket(
        [
            {
                "type": "conversation_initiation_metadata",
                "conversation_initiation_metadata_event": {
                    "conversation_id": "conv_123",
                    "agent_output_audio_format": "pcm_44100",
                    "user_input_audio_format": "pcm_16000",
                },
            }
        ]
    )

    async def fake_connect(_url):
        return websocket

    monkeypatch.setattr(
        "tau2.voice.audio_native.elevenlabs.provider.websockets.connect",
        fake_connect,
    )
    provider = ElevenLabsAgentsProvider(agent_id="agent_test")

    asyncio.run(provider.connect("domain policy"))

    assert provider.session_id == "conv_123"
    assert provider.user_input_audio_format == "pcm_16000"
    assert provider.agent_output_audio_format == "pcm_44100"
    assert websocket.sent == [provider._build_initiation_payload("domain policy")]


def test_connect_closes_socket_when_override_is_rejected(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    websocket = FakeWebSocket(
        [
            {
                "type": "client_error",
                "error_event": {
                    "code": 400,
                    "error_name": "override_not_allowed",
                    "message": "Prompt override is disabled",
                },
            }
        ]
    )

    async def fake_connect(_url):
        return websocket

    monkeypatch.setattr(
        "tau2.voice.audio_native.elevenlabs.provider.websockets.connect",
        fake_connect,
    )
    provider = ElevenLabsAgentsProvider(agent_id="agent_test")

    with pytest.raises(RuntimeError, match="Enable System prompt and First message"):
        asyncio.run(provider.connect("domain policy"))

    assert provider.ws is None
    assert websocket.state == State.CLOSED


def test_send_audio_and_client_tool_result(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    provider = ElevenLabsAgentsProvider(agent_id="agent_test")
    provider.ws = FakeWebSocket()

    asyncio.run(provider.send_audio(b"hello"))
    asyncio.run(provider.send_tool_result("call_1", "done", is_error=True))

    assert provider.ws.sent[0] == {"user_audio_chunk": "aGVsbG8="}
    assert provider.ws.sent[1] == {
        "type": "client_tool_result",
        "tool_call_id": "call_1",
        "result": "done",
        "is_error": True,
    }


def test_private_agent_url_uses_xi_api_key(monkeypatch):
    request = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"signed_url": "wss://signed.example/conversation?token=test"}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, params, headers):
            request.update(url=url, params=params, headers=headers)
            return FakeResponse()

    monkeypatch.setattr(
        "tau2.voice.audio_native.elevenlabs.provider.httpx.AsyncClient",
        lambda **_kwargs: FakeClient(),
    )
    provider = ElevenLabsAgentsProvider(
        agent_id="agent_private",
        api_key="secret-test-key",
    )

    url = asyncio.run(provider._get_websocket_url())

    assert url == "wss://signed.example/conversation?token=test"
    assert request["params"] == {"agent_id": "agent_private"}
    assert request["headers"] == {"xi-api-key": "secret-test-key"}


def test_public_agent_url_does_not_require_api_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    provider = ElevenLabsAgentsProvider(agent_id="agent_public")

    url = asyncio.run(provider._get_websocket_url())

    assert url.endswith("?agent_id=agent_public")
