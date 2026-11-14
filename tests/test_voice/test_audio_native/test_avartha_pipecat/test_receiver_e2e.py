"""Offline RTP -> WebRTC receiver -> ticks -> recorded Generation reference.

Use real Opus packets/decoder and aiortc's TimestampMapper, without ICE or
sockets. Explicit queue barriers control media/data delivery; no wall-clock
race or sleep determines when a caption reaches a playback tick.
"""

import asyncio
import audioop
import base64
import json
import wave
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("aiortc")

from aiortc.codecs.opus import OpusDecoder, OpusEncoder  # noqa: E402
from aiortc.jitterbuffer import JitterFrame  # noqa: E402
from aiortc.rtcrtpreceiver import TimestampMapper  # noqa: E402
from aiortc.rtp import RtpPacket  # noqa: E402
from av import AudioFrame  # noqa: E402

from tau2.agent.discrete_time_audio_native_agent import (  # noqa: E402
    DiscreteTimeAudioNativeAgent,
)
from tau2.data_model.message import Tick  # noqa: E402
from tau2.data_model.simulation import SimulationRun, TerminationReason  # noqa: E402
from tau2.judges.run_evidence import load_audio_clips  # noqa: E402
from tau2.voice.audio_native.avartha_pipecat.config import (  # noqa: E402
    AvarthaPipecatConfig,
)
from tau2.voice.audio_native.avartha_pipecat.discrete_time_adapter import (  # noqa: E402
    DiscreteTimeAvarthaPipecatAdapter,
)
from tau2.voice.audio_native.avartha_pipecat.webrtc import (  # noqa: E402
    AvarthaWebRTCProvider,
)

STREAM = "73dfe9ec-e3b4-4d42-8f74-1076e94be983"
ORIGIN = 918273
RATE = 48000


def _message(kind, data=None):
    return {"label": "rtvi-ai", "type": kind, "data": data or {}}


def _word(text, sample_offset):
    return _message(
        "bot-tts-text",
        {
            "text": text,
            "audio_time": {
                "stream_id": STREAM,
                "sample_offset": sample_offset,
                "sample_rate_hz": RATE,
                "boundary": "word_end",
            },
        },
    )


class _DeliveryQueue(asyncio.Queue):
    """The next get acknowledges completion of processing the preceding item."""

    def __init__(self):
        super().__init__()
        self.idle = asyncio.Event()

    async def get(self):
        self.idle.set()
        return await super().get()

    async def deliver(self, item):
        self.idle.clear()
        self.put_nowait(item)
        await asyncio.wait_for(self.idle.wait(), 1)


class _Emitter:
    def __init__(self):
        self.handlers = {}

    def on(self, event):
        def register(fn):
            self.handlers[event] = fn
            return fn

        return register


class _OfflineReceiver(AvarthaWebRTCProvider):
    """Replace the call handshake and receive window, retaining media/message code."""

    def __init__(self, schedule):
        super().__init__(AvarthaPipecatConfig(base_url="https://agent.example"))
        self.schedule = iter(schedule)
        encoder = OpusEncoder()
        self.packets = []
        rng = np.random.default_rng(12)
        # Non-silent PCM avoids digital-silence suppression in this placement test.
        for start in range(0, RATE // 5, 960):
            pcm = (rng.standard_normal(960) * 4000).astype(np.int16)
            frame = AudioFrame.from_ndarray(pcm.reshape(1, -1), "s16", "mono")
            frame.sample_rate, frame.pts = RATE, start
            frame.time_base = Fraction(1, RATE)
            payloads, timestamp = encoder.encode(frame)
            for payload in payloads:
                packet = RtpPacket(payload_type=111, sequence_number=len(self.packets))
                packet.ssrc, packet.timestamp = 77, (ORIGIN + timestamp) % (1 << 32)
                packet.payload = payload
                self.packets.append(packet.serialize())
        self.binding = _message(
            "server-message",
            {
                "type": "audio_timeline",
                "protocol": "avartha-audio-timing/1",
                "stream_id": STREAM,
                "sample_rate_hz": RATE,
                "rtp_timestamp": RtpPacket.parse(self.packets[0]).timestamp,
                "media_sample_offset": encoder.first_packet_pts,
            },
        )

    async def connect(self, system_prompt=""):
        self._reset_session_state()
        self._inbox = _DeliveryQueue()
        self.frames = _DeliveryQueue()
        self.mapper, self.decoder = TimestampMapper(), OpusDecoder()
        track = SimpleNamespace(kind="audio", recv=self.frames.get)
        self.receiver = SimpleNamespace(
            track=track, _RTCRtpReceiver__timestamp_mapper=self.mapper
        )
        peer = _Emitter()
        peer.getReceivers = lambda: [self.receiver]

        async def close():
            pass

        peer.close = close
        self._peer = peer
        self._channel = _Emitter()
        # Avoid the real disconnect grace period; no external peer exists.
        self._channel.readyState = "closed"
        self._wire_peer(peer, asyncio.Event())
        self._reader = asyncio.create_task(self._read_loop())
        peer.handlers["track"](track)
        await self._inbox.idle.wait()
        await self.frames.idle.wait()

    async def register_tools(self, schemas):
        assert schemas == []

    async def send_audio(self, audio_data, sample_rate):
        pass  # Caller input is irrelevant to synthetic agent output.

    async def receive_events_for_duration(self, duration):
        for action in next(self.schedule, []):
            if action == "restart-receiver":
                # A repeated RTP packet alone is interpreted as wraparound by
                # aiortc. Restart its receive mapping to model a decoded track
                # rewinding in-place, without injecting adapter audio events.
                self.mapper = TimestampMapper()
                self.receiver._RTCRtpReceiver__timestamp_mapper = self.mapper
                self.decoder = OpusDecoder()
            elif isinstance(action, int):
                packet = RtpPacket.parse(self.packets[action])
                encoded = JitterFrame(
                    data=packet.payload, timestamp=self.mapper.map(packet.timestamp)
                )
                for frame in self.decoder.decode(encoded):
                    await self.frames.deliver(frame)
            else:
                message = self.binding if action == "binding" else action
                # Enter through the actual data-channel callback and read loop.
                self._inbox.idle.clear()
                self._channel.handlers["message"](json.dumps(message))
                await asyncio.wait_for(self._inbox.idle.wait(), 1)
        assert self._session_error is None
        return self.drain_pending_events()


def _run(schedule, tmp_path):
    provider = _OfflineReceiver(schedule)
    adapter = DiscreteTimeAvarthaPipecatAdapter(tick_duration_ms=20, provider=provider)
    agent = DiscreteTimeAudioNativeAgent(
        tools=[],
        domain_policy="policy",
        adapter=adapter,
        provider="avartha_pipecat",
        tick_duration_ms=20,
    )
    ticks = []
    try:
        adapter.connect("policy", [])
        for index in range(len(schedule)):
            result = adapter.run_tick(b"\xff" * 160, tick_number=index + 1)
            chunk = agent._create_response(result, [])
            ticks.append(
                Tick(
                    tick_id=index,
                    timestamp=chunk.timestamp,
                    tick_duration_seconds=0.02,
                    agent_chunk=chunk,
                )
            )
        # Write exactly the played audio as the saved simulation's agent channel.
        stereo = []
        for tick in ticks:
            pcm = audioop.ulaw2lin(base64.b64decode(tick.agent_chunk.audio_content), 2)
            samples = np.frombuffer(pcm, dtype="<i2")
            stereo.append(np.column_stack((np.zeros_like(samples), samples)))
        path = tmp_path / "call.wav"
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(np.concatenate(stereo).astype("<i2").tobytes())
        simulation = SimulationRun(
            id="receiver-test",
            task_id="test",
            ticks=ticks,
            start_time=ticks[0].timestamp,
            end_time=ticks[-1].timestamp,
            duration=len(ticks) * 0.02,
            mode="full_duplex",
            termination_reason=TerminationReason.MAX_STEPS,
        )
        # Exercise the exact saved-audio/reference extraction used by Generation.
        clips = load_audio_clips(path, simulation)
        return [tick.agent_chunk.content or "" for tick in ticks], [
            clip.reference for clip in clips
        ]
    finally:
        try:
            if adapter._bg_loop.is_running:
                adapter._bg_loop.run_coroutine(provider.disconnect(), timeout=2)
        finally:
            adapter.disconnect()
        assert not provider._audio_tasks or all(t.done() for t in provider._audio_tasks)


def _start(timed=True):
    return (["binding"] if timed else []) + [
        _message("bot-transcription", {"text": "One two three."})
    ]


def test_early_and_late_words_follow_played_audio(tmp_path):
    shown, references = _run(
        [
            _start() + [0, _word("One", 400), _word("two", 2200), 1, 2, 3, 4],
            [],
            [],
            [],
            [_word("three.", 3100)],
            [],
            [],
            [],
        ],
        tmp_path,
    )
    assert shown[:5] == ["One", "", " two", "", " three."]
    assert references == ["One two three."]


@pytest.mark.parametrize("late_first_word", [False, True])
def test_barge_in_keeps_only_words_ending_before_playback_cut(
    tmp_path, late_first_word
):
    early = [] if late_first_word else [_word("One", 400), _word("two", 2200)]
    late = [_word("One", 400), _word("two", 2200)] if late_first_word else []
    shown, references = _run(
        [
            _start() + [0, *early, 1, 2, 3, 4],
            [],
            [*late, _word("three.", 3100), _message("user-started-speaking")],
            [_word("three.", 3100)],
            [],
        ],
        tmp_path,
    )
    assert shown[0] == ("" if late_first_word else "One")
    assert "".join(shown).strip() == "One"
    assert references == ["One"]


def test_without_timeline_retains_arrival_placement(tmp_path):
    shown, references = _run(
        [
            _start(False) + [0, _word("One", 400), _word("two", 2200), 1, 2, 3, 4],
            [],
            [],
            [],
            [_word("three.", 3100)],
            [],
            [],
            [],
        ],
        tmp_path,
    )
    assert shown[:5] == ["One two", "", "", "", " three."]
    assert references == ["One two three."]


def test_rtp_rewind_uses_arrival_for_the_rest_of_the_call(tmp_path):
    shown, references = _run(
        [
            _start() + [0, 1, _word("One", 400)],
            ["restart-receiver", 0, _word("two", 8000)],
            [2, 3, 4, _word("three.", 9000)],
            [],
            [],
            [],
            [],
        ],
        tmp_path,
    )
    assert shown == ["One", "", " two", "", "", " three.", ""]
    assert references == ["One two three."]
