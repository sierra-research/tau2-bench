"""OpenAI Live V3 over WebRTC, with server-owned Responses delegation."""

import asyncio
import audioop
import base64
import contextlib
import json
import time
from fractions import Fraction
from pathlib import Path
from typing import AsyncGenerator

import aiohttp
import numpy as np
from aiortc import (
    AudioStreamTrack,
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError
from aiortc.rtcrtpparameters import RTCRtcpFeedback
from aiortc.rtcrtpreceiver import NackGenerator
from aiortc.sdp import SessionDescription
from av import AudioFrame
from av.audio.resampler import AudioResampler
from loguru import logger

from tau2.config import DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE
from tau2.data_model.audio import AudioEncoding, AudioFormat
from tau2.environment.tool import Tool
from tau2.voice.audio_native.openai.events import (
    AudioTranscriptDeltaEvent,
    BaseRealtimeEvent,
    TimeoutEvent,
    parse_realtime_event,
)
from tau2.voice.audio_native.openai.live_config import LiveConfig
from tau2.voice.audio_native.openai.provider import (
    OpenAIRealtimeProvider,
    OpenAIVADConfig,
)


class LiveTranscriptDelta(AudioTranscriptDeltaEvent):
    """A native transcript delta located on the received PCM timeline."""

    audio_position_bytes: int
    start_ms: int
    end_ms: int


class _LiveInputAudioTrack(AudioStreamTrack):
    """Clock queued PCM at 20 ms intervals; never gate it on transcript events."""

    def __init__(self) -> None:
        super().__init__()
        self._buffer = bytearray()
        self._resample_state = None
        self._timestamp = 0
        self._started_at: float | None = None
        self.sample_rate = 48_000
        self.samples_per_frame = 960
        self.max_schedule_lag_ms = 0.0
        self.max_buffer_ms = 0.0
        self.frames_sent = 0
        self.frames_late_100ms = 0

    def append(
        self, audio: bytes, sample_rate: int = DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE
    ) -> None:
        if not audio:
            return
        resampled, self._resample_state = audioop.ratecv(
            audio, 2, 1, sample_rate, self.sample_rate, self._resample_state
        )
        self._buffer.extend(resampled)
        self.max_buffer_ms = max(
            self.max_buffer_ms, len(self._buffer) * 500 / self.sample_rate
        )

    async def recv(self) -> AudioFrame:
        loop = asyncio.get_running_loop()
        if self._started_at is None:
            self._started_at = loop.time()
        target = self._started_at + self._timestamp / self.sample_rate
        await asyncio.sleep(max(0.0, target - loop.time()))
        lag_ms = (loop.time() - target) * 1000
        self.max_schedule_lag_ms = max(self.max_schedule_lag_ms, lag_ms)
        self.frames_sent += 1
        self.frames_late_100ms += lag_ms > 100
        size = self.samples_per_frame * 2
        audio = bytes(self._buffer[:size]).ljust(size, b"\x00")
        del self._buffer[:size]
        frame = AudioFrame.from_ndarray(
            np.frombuffer(audio, dtype=np.int16).reshape(1, self.samples_per_frame),
            format="s16",
            layout="mono",
        )
        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)
        frame.pts = self._timestamp
        self._timestamp += self.samples_per_frame
        return frame


class OpenAILiveProvider(OpenAIRealtimeProvider):
    """Keep media continuous while exposing the existing agent event contract.

    V3 tool continuations are client-triggered. A response is continued only
    after its complete batch of function calls has been returned. Speech and
    transcript arrival never trigger a response or truncate the media track.
    """

    def __init__(
        self,
        *,
        model: str,
        config: LiveConfig,
        reasoning_effort: str | None = None,
        api_key: str | None = None,
        trace_path: Path | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key, model=model, reasoning_effort=reasoning_effort
        )
        self.config = config
        self.trace_path = trace_path
        self._trace = None
        self.session_id: str | None = None
        self._peer: RTCPeerConnection | None = None
        self._channel = None
        self._input_track: _LiveInputAudioTrack | None = None
        self._channel_open = asyncio.Event()
        self._started = asyncio.Event()
        self._closed = asyncio.Event()
        self._closing = False
        self._events: asyncio.Queue[dict] = asyncio.Queue()
        self._received_audio_bytes = 0
        self._received_audio_gap_ms = 0.0
        self._received_audio_overlap_ms = 0.0
        self._audio_tasks: set[asyncio.Task] = set()
        self._response_ids: dict[str, str] = {}
        self._tool_calls: dict[str, set[str]] = {}
        self._pending: dict[str, set[str]] = {}
        self._returned: set[str] = set()
        self._continued: set[str] = set()
        self._surfaced: set[str] = set()
        self.delegation_count = 0

    def _record(self, direction: str, event: dict) -> None:
        if self._trace is not None:
            self._trace.write(
                json.dumps(
                    {"timestamp": time.time(), "direction": direction, "event": event}
                )
                + "\n"
            )

    def _send_event(self, event: dict) -> None:
        if self._channel is None or self._channel.readyState != "open":
            raise ConnectionError("Live data channel is not open")
        self._channel.send(json.dumps(event))
        self._record("sent", event)

    def _accept_event(self, event: dict) -> None:
        self._record("received", event)
        if event["type"] == "session.started":
            self._started.set()
        elif event["type"] == "session.closed":
            self._closed.set()
        elif event["type"] == "session.output_transcript.delta":
            # Capture at arrival, not when a delayed tick drains the event queue.
            event = dict(event, audio_position_bytes=self._received_audio_bytes)
            self._record("transcript_ingress", event)
        self._events.put_nowait(event)

    @property
    def is_connected(self) -> bool:
        return self._peer is not None and self._peer.connectionState not in {
            "closed",
            "failed",
        }

    async def connect(self) -> None:
        if self._peer is not None or self._closing:
            raise RuntimeError("Create a new Live provider for each session")
        peer = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        self._peer = peer
        self._input_track = _LiveInputAudioTrack()
        peer.addTrack(self._input_track)
        # aiortc 1.15 has no public audio NACK switch. Enable its existing
        # recovery mechanism on this receiver without changing other providers.
        peer.getReceivers()[0]._RTCRtpReceiver__nack_generator = NackGenerator()
        self._channel = peer.createDataChannel("")

        @self._channel.on("open")
        def on_open() -> None:
            self._channel_open.set()

        @self._channel.on("message")
        def on_message(message: str | bytes) -> None:
            self._accept_event(json.loads(message))

        @peer.on("connectionstatechange")
        def on_state() -> None:
            if not self._closing and peer.connectionState in {"failed", "closed"}:
                self._events.put_nowait(
                    {
                        "type": "error",
                        "error": {
                            "code": "webrtc_connection_closed",
                            "message": peer.connectionState,
                        },
                    }
                )

        @peer.on("track")
        def on_track(track: AudioStreamTrack) -> None:
            if track.kind == "audio":
                task = asyncio.create_task(self._receive_audio(track))
                self._audio_tasks.add(task)
                task.add_done_callback(on_audio_finished)

        def on_audio_finished(task: asyncio.Task) -> None:
            self._audio_tasks.discard(task)
            if not task.cancelled() and (error := task.exception()) is not None:
                self._events.put_nowait(
                    {
                        "type": "error",
                        "error": {
                            "code": "webrtc_audio_failed",
                            "message": str(error),
                        },
                    }
                )

    async def _receive_audio(self, track: AudioStreamTrack) -> None:
        resampler = AudioResampler(
            format="s16", layout="mono", rate=DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE
        )
        expected_audio_time = None
        while not self._closing:
            try:
                frame = await track.recv()
            except MediaStreamError:
                if not self._closing:
                    raise ConnectionError("Live remote audio track ended") from None
                return
            audio_time = frame.pts * frame.time_base
            if expected_audio_time is not None:
                gap_ms = float(audio_time - expected_audio_time) * 1000
                self._received_audio_gap_ms += max(0, gap_ms)
                self._received_audio_overlap_ms += max(0, -gap_ms)
            expected_audio_time = audio_time + Fraction(
                frame.samples, frame.sample_rate
            )
            for converted in resampler.resample(frame):
                pcm = (
                    converted.to_ndarray()
                    .reshape(-1)
                    .astype(np.int16, copy=False)
                    .tobytes()
                )
                self._received_audio_bytes += len(pcm)
                self._events.put_nowait(
                    {
                        "type": "live.media.audio",
                        "audio": base64.b64encode(pcm).decode("ascii"),
                    }
                )

    def _build_session(self, system_prompt: str, tools: list[Tool]) -> dict:
        responses = {
            "model": self.config.backend_model,
            "instructions": self.config.render_backend_prompt(system_prompt),
            "tools": self._format_tools_for_api(tools),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }
        if self.reasoning_effort is not None:
            responses["reasoning"] = {"effort": self.reasoning_effort}
        return {
            "model": self.model,
            "instructions": self.config.render_frontend_prompt(system_prompt),
            "audio": {"output": {"voice": self.config.voice}},
            "delegation": {"type": "responses", "responses": responses},
        }

    async def _create_offer(self) -> RTCSessionDescription:
        offer = await self._peer.createOffer()
        description = SessionDescription.parse(offer.sdp)
        for media in description.media:
            if media.kind == "audio":
                for codec in media.rtp.codecs:
                    codec.rtcpFeedback.append(RTCRtcpFeedback(type="nack"))
        return RTCSessionDescription(sdp=str(description), type=offer.type)

    async def configure_session(
        self,
        system_prompt: str,
        tools: list[Tool],
        vad_config: OpenAIVADConfig,
        modality: str = "audio",
        audio_format: AudioFormat | None = None,
    ) -> None:
        if modality != "audio":
            raise ValueError("OpenAI Live requires audio modality")
        if self._peer is None:
            raise RuntimeError("Call connect before configuring Live")
        self._audio_format = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE,
        )
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace = self.trace_path.open("a", encoding="utf-8", buffering=1)
        try:
            session = self._build_session(system_prompt, tools)
            self._record("configuration", session)
            offer = await self._create_offer()
            await self._peer.setLocalDescription(offer)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Alpha": "quicksilver=v3",
            }
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=25)
            ) as client:
                async with client.post(
                    "https://api.openai.com/v1/live/sessions",
                    headers=headers,
                    json={
                        "session": session,
                        "transport": {
                            "type": "webrtc",
                            "sdp": self._peer.localDescription.sdp,
                        },
                    },
                ) as response:
                    if not 200 <= response.status < 300:
                        raise RuntimeError(
                            f"Live session creation failed (HTTP {response.status}): {(await response.text())[:600]}"
                        )
                    created = await response.json()
            self.session_id = created["session"]["id"]
            self._record("created", {"session": created["session"]})
            await self._peer.setRemoteDescription(
                RTCSessionDescription(sdp=created["transport"]["sdp"], type="answer")
            )
            await asyncio.wait_for(self._channel_open.wait(), timeout=20)
            await asyncio.wait_for(self._started.wait(), timeout=20)
        except BaseException:
            await self.disconnect()
            raise

    async def send_audio(self, audio_data: bytes) -> None:
        if not self.is_connected or self._input_track is None:
            raise ConnectionError("Live audio track is not connected")
        self._input_track.append(audio_data)

    async def send_tool_result(
        self, call_id: str, result: str, request_response: bool = True
    ) -> None:
        self._send_event(
            {
                "type": "response.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result,
                },
            }
        )
        self._returned.add(call_id)
        if request_response:
            self._continue_tool_batches()

    def _continue_tool_batches(self) -> None:
        for response_id, required in list(self._pending.items()):
            if required <= self._returned:
                self._send_event({"type": "response.create"})
                self._continued.add(response_id)
                del self._pending[response_id]

    def _normalize_event(self, data: dict) -> list[BaseRealtimeEvent]:
        event_type = data["type"]
        if event_type == "live.media.audio":
            return [
                parse_realtime_event(
                    {
                        "type": "response.output_audio.delta",
                        "delta": data["audio"],
                    }
                )
            ]
        if event_type == "session.output_transcript.delta":
            return [
                LiveTranscriptDelta(
                    type="response.output_audio_transcript.delta",
                    delta=data["delta"],
                    audio_position_bytes=data["audio_position_bytes"],
                    start_ms=data["start_ms"],
                    end_ms=data["end_ms"],
                )
            ]
        if event_type == "session.delegation.created":
            self.delegation_count += 1
        elif event_type == "session.closed" and not self._closing:
            raise ConnectionError(
                f"Live session closed during conversation: {self.session_id}"
            )
        elif event_type == "error":
            raise RuntimeError(f"Live API error: {data['error']}")
        elif event_type == "response.event":
            delegation_id = data["delegation_id"]
            event = data["event"]
            inner_type = event["type"]
            if inner_type == "response.created":
                self._response_ids[delegation_id] = event["response"]["id"]
            elif (
                inner_type == "response.output_item.done"
                and event["item"]["type"] == "function_call"
            ):
                item = event["item"]
                response_id = self._response_ids[delegation_id]
                self._tool_calls.setdefault(response_id, set()).add(item["call_id"])
                if item["call_id"] not in self._surfaced:
                    self._surfaced.add(item["call_id"])
                    return [
                        parse_realtime_event(
                            {
                                "type": "response.function_call_arguments.done",
                                "call_id": item["call_id"],
                                "name": item["name"],
                                "arguments": item["arguments"],
                            }
                        )
                    ]
            elif inner_type == "response.completed":
                response_id = event["response"]["id"]
                # Live compacts response.output; item.done events define the batch.
                required = self._tool_calls.pop(response_id, set())
                if required and response_id not in self._continued:
                    self._pending[response_id] = required
                    self._continue_tool_batches()
            elif inner_type in {"response.failed", "response.incomplete", "error"}:
                raise RuntimeError(f"Live backend failed: {event}")
        return [parse_realtime_event(data)]

    async def receive_events(self) -> AsyncGenerator[BaseRealtimeEvent, None]:
        while self.is_connected:
            try:
                event = await asyncio.wait_for(self._events.get(), timeout=0.01)
            except TimeoutError:
                yield TimeoutEvent(type="timeout")
                continue
            for normalized in self._normalize_event(event):
                yield normalized

    async def disconnect(self) -> None:
        self._closing = True
        try:
            if self._started.is_set() and not self._closed.is_set():
                try:
                    self._send_event({"type": "session.close"})
                    await asyncio.wait_for(self._closed.wait(), timeout=5)
                except (TimeoutError, ConnectionError):
                    # Missing close acknowledgement must not invalidate a scored task.
                    logger.warning("Live close not confirmed: {}", self.session_id)
            for task in list(self._audio_tasks):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if self._peer is not None:
                try:
                    stats = await self._peer.getStats()
                    for stat in stats.values():
                        if stat.type == "inbound-rtp" and stat.kind == "audio":
                            self._record(
                                "media_rtp",
                                {
                                    "packets_received": stat.packetsReceived,
                                    "packets_lost": stat.packetsLost,
                                    "jitter_samples": stat.jitter,
                                },
                            )
                        elif stat.type == "remote-inbound-rtp" and stat.kind == "audio":
                            self._record(
                                "media_rtp_remote",
                                {
                                    "packets_received": stat.packetsReceived,
                                    "packets_lost": stat.packetsLost,
                                    "jitter_samples": stat.jitter,
                                    "round_trip_time_s": stat.roundTripTime,
                                },
                            )
                finally:
                    await self._peer.close()
                    self._peer = None
        finally:
            if self._input_track is not None:
                self._record(
                    "media",
                    {
                        "max_schedule_lag_ms": self._input_track.max_schedule_lag_ms,
                        "max_buffer_ms": self._input_track.max_buffer_ms,
                        "frames_sent": self._input_track.frames_sent,
                        "frames_late_100ms": self._input_track.frames_late_100ms,
                        "received_audio_gap_ms": self._received_audio_gap_ms,
                        "received_audio_overlap_ms": self._received_audio_overlap_ms,
                    },
                )
            if self._trace is not None:
                self._trace.close()
                self._trace = None
