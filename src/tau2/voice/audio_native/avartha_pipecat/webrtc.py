"""Call a hosted Avartha agent over Pipecat SmallWebRTC.

A hosted agent is served through Pipecat's agent-server protocol:
``POST {agent_url}/start`` admits a call, ``POST
{agent_url}/sessions/{id}/api/offer`` opens one Opus audio track plus a data
channel, and RTVI runs on that channel. The RTVI session itself (boot verdict,
``avartha-harness-tools/1`` tool exchange, routing) is
:class:`AvarthaPipecatProvider`; this module carries it over the call.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from fractions import Fraction
from typing import Any, Optional

import httpx
import numpy as np
from loguru import logger

try:
    from aiortc import (
        AudioStreamTrack,
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
    )
    from aiortc.mediastreams import MediaStreamError
    from av import AudioFrame
    from av.audio.resampler import AudioResampler
except ImportError as exc:  # pragma: no cover - depends on installed extras
    raise ImportError(
        "The avartha_pipecat provider requires aiortc. "
        "Install tau2 with the live extra: uv sync --extra live"
    ) from exc

from tau2.config import (
    DEFAULT_AVARTHA_INPUT_SAMPLE_RATE,
    DEFAULT_AVARTHA_OUTPUT_SAMPLE_RATE,
)
from tau2.voice.audio_native.avartha_pipecat.config import AvarthaPipecatConfig
from tau2.voice.audio_native.avartha_pipecat.provider import (
    RTVI_MESSAGE_LABEL,
    RTVI_PROTOCOL_VERSION,
    AgentAudioEvent,
    AvarthaPipecatProvider,
)

# The hosted agent's API key, sent as a bearer token to /start.
AVARTHA_API_KEY_ENV = "AVARTHA_API_KEY"
# Received frames whose peak is at or below this are the digital silence the
# agent's track sends between utterances. They are dropped, because the
# adapter treats any agent audio as speech.
_SILENCE_PEAK = 8
# How long a hang-up waits for the server to end the call before closing the
# peer anyway. Closing at once can drop the queued disconnect-bot, and the
# server then keeps the call open until its own timeout ends it.
_HANGUP_GRACE_SECS = 2.0


class _CallerAudioTrack(AudioStreamTrack):
    """Play queued caller PCM as 20 ms frames on the wall clock.

    tau2 hands over a tick of audio at once; WebRTC needs it paced. A frame
    with no queued audio is sent as silence, as a phone line would.
    """

    def __init__(self, sample_rate: int) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.samples_per_frame = sample_rate // 50
        self._buffer = bytearray()
        self._timestamp = 0
        self._started_at: Optional[float] = None

    def append(self, audio: bytes) -> None:
        self._buffer.extend(audio)

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError
        loop = asyncio.get_running_loop()
        if self._started_at is None:
            self._started_at = loop.time()
        target = self._started_at + self._timestamp / self.sample_rate
        await asyncio.sleep(max(0.0, target - loop.time()))
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


def webrtc_start_body(
    system_prompt: str, language: Optional[str] = None
) -> dict[str, Any]:
    """The ``/start`` body: caller-executed tools, the domain policy and, for a
    call in a known language, the agent's speech recognition language."""

    body: dict[str, Any] = {"caller_tools": True}
    override: dict[str, Any] = {}
    if system_prompt.strip():
        override["agent"] = {"prompt": {"prompt": system_prompt}}
    if language:
        # tau2 gives the caller the language pack's greeting as the agent's
        # opening text, so the agent speaks no welcome of its own (as tau2 does
        # for ElevenLabs agents) and listens in the call's language.
        override.setdefault("agent", {})["first_message"] = ""
        override["asr"] = {"language": language.split("-")[0].lower()}
    if override:
        body["conversation_config_override"] = override
    return {"transport": "webrtc", "body": body}


def ice_servers(ice_config: Any) -> list[RTCIceServer]:
    """Convert the ``/start`` response's ``iceConfig`` for aiortc."""

    servers = ice_config.get("iceServers") if isinstance(ice_config, dict) else None
    return [
        RTCIceServer(
            urls=server["urls"],
            username=server.get("username"),
            credential=server.get("credential"),
        )
        for server in servers or []
    ]


def is_digital_silence(pcm: bytes) -> bool:
    """Whether a PCM16 frame carries no agent speech."""

    samples = np.frombuffer(pcm, dtype=np.int16)
    return samples.size == 0 or int(np.abs(samples.astype(np.int32)).max()) <= (
        _SILENCE_PEAK
    )


class AvarthaWebRTCProvider(AvarthaPipecatProvider):
    """Exchange audio, transcripts and tool calls with a hosted agent."""

    def __init__(self, config: AvarthaPipecatConfig, language: Optional[str] = None):
        super().__init__(config)
        self.url = config.base_url.rstrip("/")
        self.language = language
        self._peer: Optional[RTCPeerConnection] = None
        self._channel: Any = None
        self._track: Optional[_CallerAudioTrack] = None
        self._inbox: asyncio.Queue[Any] = asyncio.Queue()
        self._audio_tasks: set[asyncio.Task[None]] = set()
        self._closing = False

    @property
    def is_connected(self) -> bool:
        """Whether the provider currently owns a peer connection."""

        return self._peer is not None

    def _headers(self) -> dict[str, str]:
        api_key = os.getenv(AVARTHA_API_KEY_ENV)
        if not api_key:
            raise ValueError(
                f"Set {AVARTHA_API_KEY_ENV} to call a hosted Avartha agent."
            )
        return {"Authorization": f"Bearer {api_key}"}

    async def connect(self, system_prompt: str = "") -> None:
        """Start a call, open its media and wait for a healthy boot."""

        if self.is_connected:
            raise RuntimeError("Avartha WebRTC call is already connected")
        self._reset_session_state()
        headers = self._headers()
        async with httpx.AsyncClient(
            timeout=self.config.connect_timeout_seconds
        ) as client:
            response = await client.post(
                f"{self.url}/start",
                headers=headers,
                json=webrtc_start_body(system_prompt, self.language),
            )
            _raise_for_status(response, "start")
            started = response.json()
            self.session_id = started["sessionId"]
            logger.info("Avartha WebRTC session id={}", self.session_id)

            peer = RTCPeerConnection(
                configuration=RTCConfiguration(
                    iceServers=ice_servers(started.get("iceConfig"))
                )
            )
            self._peer = peer
            self._track = _CallerAudioTrack(DEFAULT_AVARTHA_INPUT_SAMPLE_RATE)
            peer.addTrack(self._track)
            channel_open = asyncio.Event()
            self._channel = peer.createDataChannel("chat", ordered=True)
            self._wire_peer(peer, channel_open)

            # aiortc gathers every candidate before returning; the server
            # takes no trickled candidates.
            await peer.setLocalDescription(await peer.createOffer())
            response = await client.post(
                f"{self.url}/sessions/{self.session_id}/api/offer",
                headers=headers,
                json={
                    "sdp": peer.localDescription.sdp,
                    "type": "offer",
                    "pc_id": None,
                    "restart_pc": False,
                },
            )
            _raise_for_status(response, "offer")
            answer = response.json()
        await peer.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )
        await asyncio.wait_for(
            channel_open.wait(), timeout=self.config.connect_timeout_seconds
        )
        self._reader = asyncio.create_task(
            self._read_loop(), name="tau2-avartha-webrtc-reader"
        )
        await self._send_message(
            {
                "label": RTVI_MESSAGE_LABEL,
                "type": "client-ready",
                "id": "tau2-0",
                "data": {
                    "version": RTVI_PROTOCOL_VERSION,
                    "about": {"library": "tau2"},
                },
            }
        )
        logger.info(f"Avartha WebRTC call connected and client-ready ({self.url})")
        await self._await_boot()

    def _wire_peer(self, peer: RTCPeerConnection, channel_open: asyncio.Event) -> None:
        channel = self._channel

        @channel.on("open")
        def on_open() -> None:
            channel_open.set()

        @channel.on("message")
        def on_message(message: str | bytes) -> None:
            self._inbox.put_nowait(message)

        @channel.on("close")
        def on_close() -> None:
            if not self._closing:
                self._fail_session("Avartha data channel closed before disconnect")

        @peer.on("connectionstatechange")
        def on_state() -> None:
            if not self._closing and peer.connectionState in {"failed", "closed"}:
                self._fail_session(f"Avartha WebRTC connection {peer.connectionState}")

        @peer.on("track")
        def on_track(track: Any) -> None:
            if track.kind != "audio":
                return
            task = asyncio.create_task(self._receive_audio(track))
            self._audio_tasks.add(task)
            task.add_done_callback(self._audio_tasks.discard)

    async def _receive_audio(self, track: Any) -> None:
        try:
            resampler = AudioResampler(
                format="s16", layout="mono", rate=DEFAULT_AVARTHA_OUTPUT_SAMPLE_RATE
            )
            while not self._closing:
                try:
                    frame = await track.recv()
                except MediaStreamError:
                    if not self._closing:
                        self._fail_session(
                            "Avartha audio track ended before disconnect"
                        )
                    return
                for converted in resampler.resample(frame):
                    pcm = converted.to_ndarray().reshape(-1).astype(np.int16).tobytes()
                    if not is_digital_silence(pcm):
                        await self._queue.put(
                            AgentAudioEvent(
                                audio=pcm,
                                sample_rate=DEFAULT_AVARTHA_OUTPUT_SAMPLE_RATE,
                            )
                        )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closing:
                self._fail_session(f"Avartha audio receive failed: {exc}")

    async def send_audio(self, audio_data: bytes, sample_rate: int) -> None:
        """Queue one mono PCM16 chunk on the caller track."""

        if self._session_error:
            raise self._session_error
        if not audio_data or self._track is None:
            return
        if sample_rate != self._track.sample_rate:
            raise ValueError(
                f"Avartha WebRTC caller audio is {self._track.sample_rate} Hz, "
                f"got {sample_rate} Hz"
            )
        self._track.append(audio_data)

    async def disconnect(self) -> None:
        """Hang up and close the peer connection and reader."""

        if self._peer is None:
            return
        self._closing = True
        peer, self._peer = self._peer, None
        channel = self._channel
        if channel is not None and channel.readyState == "open":
            hung_up = asyncio.Event()

            @channel.on("close")
            def on_hang_up() -> None:
                hung_up.set()

            with contextlib.suppress(Exception):
                channel.send(
                    json.dumps(
                        {
                            "label": RTVI_MESSAGE_LABEL,
                            "type": "disconnect-bot",
                            "id": "tau2-bye",
                        }
                    )
                )
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(hung_up.wait(), _HANGUP_GRACE_SECS)
        tasks = list(self._audio_tasks)
        if self._reader is not None:
            tasks.append(self._reader)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._reader = None
        try:
            await peer.close()
        except Exception as exc:
            logger.debug(f"Avartha WebRTC peer close failed: {exc!r}")

    def _reset_session_state(self) -> None:
        super()._reset_session_state()
        self._peer = None
        self._channel = None
        self._track = None
        self._inbox = asyncio.Queue()
        self._audio_tasks = set()
        self._closing = False

    async def _send_message(self, message: dict[str, Any]) -> None:
        if self._session_error:
            raise self._session_error
        if self._channel is None or self._channel.readyState != "open":
            raise RuntimeError("Avartha data channel is not open")
        self._channel.send(json.dumps(message))

    async def _read_loop(self) -> None:
        while True:
            raw = await self._inbox.get()
            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                continue  # keepalives and other non-RTVI text
            if isinstance(message, dict) and message.get("type") == "error-response":
                logger.warning("Avartha rejected a client message: {}", message)
                continue
            if await self._accept_output(message):
                self._route_message(message)


def _raise_for_status(response: httpx.Response, step: str) -> None:
    if response.is_success:
        return
    raise RuntimeError(
        f"Avartha WebRTC {step} failed (HTTP {response.status_code}): "
        f"{response.text[:600]}"
    )
