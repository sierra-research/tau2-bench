"""Tests for AAIVoiceAgentProvider.connect() handshake and async I/O.

The AAI host accepts the socket immediately but only signals "agent
initialized" via a `config`-type frame sent after the client's config
message. connect() must not hang forever if that frame never arrives, and
must not silently drop other frames (events or audio) that show up while
it's waiting.

Covers: (a) a completed handshake marks the provider connected and records
the sent config message; (b) an early non-config frame is buffered (not
dropped) and surfaced by the next receive_events_for_duration(); (c) a
stalled connection with no config frame raises a clear error rather than
hanging; (d) send_audio/send_tool_result send the expected raw/JSON frames;
(e) receive_events_for_duration() maps a binary frame to AAIAudioChunkEvent.
"""

import asyncio
import json

import pytest
from websockets.protocol import State

from tau2.voice.audio_native.aai.events import AAIAudioChunkEvent
from tau2.voice.audio_native.aai.provider import AAIVoiceAgentProvider


class FakeWebSocket:
    """Minimal async fake matching the websockets client surface we rely on."""

    def __init__(self, frames):
        self.state = State.OPEN
        self._frames = list(frames)
        self.sent = []

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        if not self._frames:
            # Simulate a stalled connection: nothing to hand back before the
            # caller's own per-frame timeout should fire.
            await asyncio.sleep(10)
        return self._frames.pop(0)

    async def close(self):
        self.state = State.CLOSED


def _provider():
    return AAIVoiceAgentProvider(ws_url="ws://localhost:3000/websocket")


def _patch_connect(monkeypatch, fake_ws):
    """Make `websockets.connect(url)` (awaited directly, not as a context
    manager) resolve to the given fake websocket."""

    async def fake_connect(url, *args, **kwargs):
        return fake_ws

    monkeypatch.setattr(
        "tau2.voice.audio_native.aai.provider.websockets.connect", fake_connect
    )


def test_connect_completes_on_config_frame(monkeypatch):
    provider = _provider()
    config_frame = json.dumps({"type": "config"})
    fake_ws = FakeWebSocket(frames=[config_frame])
    _patch_connect(monkeypatch, fake_ws)

    asyncio.run(provider.connect())

    assert provider.is_connected
    assert len(fake_ws.sent) == 1
    sent = json.loads(fake_ws.sent[0])
    assert sent["type"] == "config"
    assert provider._buffered_events == []


def test_connect_buffers_early_frame_for_next_receive(monkeypatch):
    provider = _provider()
    early_frame = json.dumps({"type": "agent_transcript", "text": "hello"})
    config_frame = json.dumps({"type": "config"})
    fake_ws = FakeWebSocket(frames=[early_frame, config_frame])
    _patch_connect(monkeypatch, fake_ws)

    asyncio.run(provider.connect())

    # The early agent_transcript frame was parsed and buffered, not dropped.
    assert len(provider._buffered_events) == 1
    assert provider._buffered_events[0].type == "agent_transcript"

    # The next receive_events_for_duration() surfaces it first, then clears.
    provider.ws = FakeWebSocket(frames=[])
    events = asyncio.run(provider.receive_events_for_duration(0.02))
    assert len(events) == 1
    assert events[0].type == "agent_transcript"
    assert provider._buffered_events == []


def test_connect_raises_when_host_never_sends_config(monkeypatch):
    # Shrink the timeout so the test doesn't actually wait 10s.
    monkeypatch.setattr(
        "tau2.voice.audio_native.aai.provider.DEFAULT_AAI_CONFIG_FRAME_TIMEOUT",
        0.05,
    )
    provider = _provider()
    fake_ws = FakeWebSocket(frames=[])  # host never sends a config frame
    _patch_connect(monkeypatch, fake_ws)

    # No config frame arrives -> clear error, not a silent hang.
    with pytest.raises(RuntimeError, match="did not initialize"):
        asyncio.run(provider.connect())


def test_send_audio_sends_raw_bytes_frame():
    provider = _provider()
    fake_ws = FakeWebSocket(frames=[])
    provider.ws = fake_ws

    asyncio.run(provider.send_audio(b"\x00\x01\x02\x03"))

    assert fake_ws.sent[-1] == b"\x00\x01\x02\x03"
    assert isinstance(fake_ws.sent[-1], bytes)


def test_send_tool_result_sends_expected_json():
    provider = _provider()
    fake_ws = FakeWebSocket(frames=[])
    provider.ws = fake_ws

    asyncio.run(provider.send_tool_result("c1", "{}"))

    sent = json.loads(fake_ws.sent[-1])
    assert sent == {"type": "tool_result", "toolCallId": "c1", "result": "{}"}


def test_receive_events_for_duration_maps_binary_frame_to_audio_chunk():
    provider = _provider()
    fake_ws = FakeWebSocket(frames=[b"\x01\x02\x03\x04"])
    provider.ws = fake_ws

    events = asyncio.run(provider.receive_events_for_duration(0.02))

    assert len(events) == 1
    assert isinstance(events[0], AAIAudioChunkEvent)
    assert events[0].pcm16 == b"\x01\x02\x03\x04"
