import asyncio
import json
import time

import numpy as np
import pytest

pytest.importorskip("aiortc")

import tau2.voice.audio_native.avartha_pipecat.webrtc as webrtc_module  # noqa: E402
from tau2.voice.audio_native.adapter import create_adapter  # noqa: E402
from tau2.voice.audio_native.avartha_pipecat.config import (  # noqa: E402
    AvarthaPipecatConfig,
)
from tau2.voice.audio_native.avartha_pipecat.discrete_time_adapter import (  # noqa: E402
    DiscreteTimeAvarthaPipecatAdapter,
)
from tau2.voice.audio_native.avartha_pipecat.provider import (  # noqa: E402
    HARNESS_TOOLS_PROTOCOL,
    ProviderErrorEvent,
    ToolCallEvent,
)
from tau2.voice.audio_native.avartha_pipecat.webrtc import (  # noqa: E402
    AvarthaWebRTCProvider,
    is_digital_silence,
    webrtc_start_body,
)

AGENT_URL = "https://platform.example/inference/serverless/pipecat/v1/public/agent-1"


def webrtc_config(**overrides):
    return AvarthaPipecatConfig(base_url=AGENT_URL, **overrides)


def test_adapter_calls_the_hosted_agent():
    adapter = DiscreteTimeAvarthaPipecatAdapter(
        tick_duration_ms=200, avartha_config=webrtc_config()
    )

    assert isinstance(adapter.provider, AvarthaWebRTCProvider)
    assert adapter.provider.url == AGENT_URL


def test_start_body_opts_into_caller_tools_and_carries_the_policy():
    assert webrtc_start_body("Retail policy") == {
        "transport": "webrtc",
        "body": {
            "caller_tools": True,
            "conversation_config_override": {
                "agent": {"prompt": {"prompt": "Retail policy"}}
            },
        },
    }
    # A blank prompt is invalid on the server; keep the published one.
    assert webrtc_start_body("  ") == {
        "transport": "webrtc",
        "body": {"caller_tools": True},
    }


def test_a_call_language_sets_the_agents_asr_and_leaves_the_greeting_to_tau2():
    """tau2 gives the caller the language pack's greeting as the agent's
    opening text, so the agent speaks no welcome of its own and listens in the
    call's language, as tau2 does for ElevenLabs agents."""

    assert webrtc_start_body("Retail policy", "zh-CN") == {
        "transport": "webrtc",
        "body": {
            "caller_tools": True,
            "conversation_config_override": {
                "agent": {
                    "prompt": {"prompt": "Retail policy"},
                    "first_message": "",
                },
                "asr": {"language": "zh"},
            },
        },
    }
    adapter, _ = create_adapter(
        "avartha_pipecat",
        tick_duration_ms=200,
        avartha_config=webrtc_config(),
        language="ko",
    )
    assert adapter.provider.language == "ko"


def test_only_digital_silence_is_dropped_from_agent_audio():
    assert is_digital_silence(np.zeros(480, dtype=np.int16).tobytes())
    assert is_digital_silence(np.full(480, -8, dtype=np.int16).tobytes())
    quiet_speech = np.zeros(480, dtype=np.int16)
    quiet_speech[100] = 40
    assert not is_digital_silence(quiet_speech.tobytes())


def test_avartha_api_key_is_the_bearer_token(monkeypatch):
    provider = AvarthaWebRTCProvider(webrtc_config())
    monkeypatch.delenv("AVARTHA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="AVARTHA_API_KEY"):
        provider._headers()

    monkeypatch.setenv("AVARTHA_API_KEY", "key_test")
    assert provider._headers() == {"Authorization": "Bearer key_test"}


class FakeChannel:
    def __init__(self):
        self.handlers = {}
        self.sent = []
        self.readyState = "connecting"

    def on(self, event):
        def register(handler):
            self.handlers[event] = handler
            return handler

        return register

    def deliver(self, message):
        self.handlers["message"](json.dumps(message))

    def send(self, text):
        message = json.loads(text)
        self.sent.append(message)
        data = message.get("data") or {}
        if message["type"] == "client-ready":
            self.deliver(
                {
                    "label": "rtvi-ai",
                    "type": "server-message",
                    "data": {"type": "session_boot", "ready": True},
                }
            )
        elif message["type"] == "disconnect-bot":
            self.readyState = "closed"
            self.handlers["close"]()
        elif data.get("t") == "register_tools":
            self.deliver(
                {
                    "label": "rtvi-ai",
                    "type": "server-message",
                    "data": {
                        "type": "tools_registered",
                        "protocol": HARNESS_TOOLS_PROTOCOL,
                    },
                }
            )


class FakePeer:
    def __init__(self, configuration):
        self.ice_servers = configuration.iceServers
        self.channel = FakeChannel()
        self.handlers = {}
        self.tracks = []
        self.connectionState = "new"
        self.localDescription = None
        self.closed = False

    def addTrack(self, track):
        self.tracks.append(track)

    def createDataChannel(self, label, ordered):
        assert label == "chat"
        assert ordered
        return self.channel

    def on(self, event):
        def register(handler):
            self.handlers[event] = handler
            return handler

        return register

    async def createOffer(self):
        return webrtc_module.RTCSessionDescription(sdp="offer-sdp", type="offer")

    async def setLocalDescription(self, description):
        self.localDescription = description

    async def setRemoteDescription(self, description):
        assert description.sdp == "answer-sdp"
        self.channel.readyState = "open"
        self.channel.handlers["open"]()

    async def close(self):
        self.closed = True
        self.connectionState = "closed"


def test_call_starts_negotiates_boots_and_exchanges_tools(monkeypatch):
    posts = []
    peers = []

    class Response:
        is_success = True

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class Client:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, headers, json):
            posts.append((url, headers, json))
            if url.endswith("/start"):
                return Response(
                    {
                        "sessionId": "conv-1",
                        "iceConfig": {
                            "iceServers": [
                                {
                                    "urls": ["turn:relay.example:3478"],
                                    "username": "u",
                                    "credential": "c",
                                }
                            ]
                        },
                    }
                )
            return Response({"sdp": "answer-sdp", "type": "answer", "pc_id": "conv-1"})

    def make_peer(configuration):
        peers.append(FakePeer(configuration))
        return peers[-1]

    monkeypatch.setenv("AVARTHA_API_KEY", "key_test")
    monkeypatch.setattr(webrtc_module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(webrtc_module, "RTCPeerConnection", make_peer)

    async def exercise():
        provider = AvarthaWebRTCProvider(AvarthaPipecatConfig(base_url=AGENT_URL + "/"))
        await provider.connect(system_prompt="Retail policy")
        await provider.register_tools([{"name": "get_order", "inputSchema": {}}])
        channel = peers[0].channel
        channel.deliver(
            {
                "label": "rtvi-ai",
                "type": "server-message",
                "data": {
                    "type": "tool_call",
                    "id": "tb-1",
                    "name": "get_order",
                    "arguments": {"order_id": "#W1"},
                },
            }
        )
        events = await provider.receive_events_for_duration(0.05)
        await provider.send_tool_result("tb-1", '{"status": "pending"}')
        started = time.monotonic()
        await provider.disconnect()
        # The server ended the call on disconnect-bot, so the hang-up did
        # not wait out its grace period.
        assert time.monotonic() - started < webrtc_module._HANGUP_GRACE_SECS
        return provider, events

    provider, events = asyncio.run(exercise())

    (start_url, start_headers, start_body), (offer_url, _, offer_body) = posts
    assert start_url == AGENT_URL + "/start"
    assert start_headers == {"Authorization": "Bearer key_test"}
    assert start_body == webrtc_start_body("Retail policy")
    assert offer_url == AGENT_URL + "/sessions/conv-1/api/offer"
    assert offer_body == {
        "sdp": "offer-sdp",
        "type": "offer",
        "pc_id": None,
        "restart_pc": False,
    }
    assert peers[0].ice_servers[0].urls == ["turn:relay.example:3478"]
    assert provider.session_id == "conv-1"
    assert events == [
        ToolCallEvent(id="tb-1", name="get_order", arguments={"order_id": "#W1"})
    ]
    sent = peers[0].channel.sent
    assert [m["type"] for m in sent] == [
        "client-ready",
        "client-message",
        "client-message",
        "disconnect-bot",
    ]
    assert sent[0]["data"]["about"] == {"library": "tau2"}
    assert sent[1]["data"]["t"] == "register_tools"
    assert sent[2]["data"] == {
        "t": "tool_result",
        "d": {
            "protocol": HARNESS_TOOLS_PROTOCOL,
            "id": "tb-1",
            "result": {
                "content": [{"type": "text", "text": '{"status": "pending"}'}],
                "isError": False,
            },
        },
    }
    assert peers[0].closed
    assert not provider.is_connected


def test_channel_loss_mid_call_fails_the_session():
    async def exercise():
        provider = AvarthaWebRTCProvider(webrtc_config())
        provider._reset_session_state()
        channel = FakeChannel()
        peer = FakePeer(webrtc_module.RTCConfiguration(iceServers=[]))
        provider._peer, provider._channel = peer, channel
        provider._wire_peer(peer, asyncio.Event())
        channel.handlers["close"]()
        return provider, await provider.receive_events_for_duration(0.01)

    provider, events = asyncio.run(exercise())

    assert events == [
        ProviderErrorEvent("Avartha data channel closed before disconnect")
    ]
    with pytest.raises(RuntimeError, match="closed before disconnect"):
        asyncio.run(provider.send_audio(b"\x00\x00", 16000))
