"""Discrete-time audio native adapter for the GPT-Live API (alpha).

Provides a tick-based interface for the GPT-Live full-duplex voice model
using the shared DiscreteTimeAdapter template method.

GPT-Live-specific behavior:

- Audio conversion: telephony (8kHz μ-law) ↔ 24kHz PCM16 in BOTH directions
  (the Live WebSocket surface only accepts/produces 24kHz PCM16LE mono).

- Silence gating: contrary to the alpha guide's "gap = omitted silence"
  description, the API streams output_audio.delta continuously in real time,
  including silence. An RMS hysteresis gate forwards audio only while the
  model is audibly speaking (see GPTLIVE_SILENCE_GATE_* in config.py).

- Synthetic utterance IDs: output_audio.delta events carry no item_id (only
  a server-timeline range), but the framework's proportional-transcript
  machinery is keyed on item IDs. The adapter maintains a synthetic
  "live_utt_N" id that both audio deltas and output_transcript.added
  fragments feed into. The id is rotated when the silence gate closes and
  on interruption, so undelivered transcript from a finished or interrupted
  utterance is not attributed to the next one.

- Synthesized barge-in: GPT-Live is full-duplex and emits no VAD events;
  the model yields on its own when the user speaks over it. The framework
  still needs an interruption signal, so the adapter emits "speech_started"
  and truncates locally when an input_transcript.added fragment's timeline
  range overlaps the agent's ongoing speech (see _is_barge_in). There is no
  server-side truncate call — the model's own timeline already reflects
  what the user heard.

Usage:
    adapter = DiscreteTimeGPTLiveAdapter(tick_duration_ms=200)
    adapter.connect(system_prompt, tools, vad_config=None)

    for tick in range(max_ticks):
        result = adapter.run_tick(user_audio_bytes, tick_number=tick)

    adapter.disconnect()
"""

import asyncio
import audioop
import base64
import json
from typing import Any, List, Optional

from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
    DEFAULT_GPTLIVE_SAMPLE_RATE,
    DEFAULT_GPTLIVE_VOICE,
    GPTLIVE_SILENCE_GATE_CLOSE_RMS,
    GPTLIVE_SILENCE_GATE_OPEN_RMS,
)
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.gptlive.events import (
    DelegationCreatedEvent,
    ErrorEvent,
    InputTranscriptAddedEvent,
    OutputAudioDeltaEvent,
    OutputTranscriptAddedEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
    SessionClosedEvent,
    TimeoutEvent,
)
from tau2.voice.audio_native.gptlive.provider import GPTLiveProvider
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript

# GPT-Live input is 24kHz PCM16 mono = 48000 bytes/second
_GPTLIVE_INPUT_BYTES_PER_SECOND = DEFAULT_GPTLIVE_SAMPLE_RATE * 2

# Barge-in detection thresholds (all on the cumulative user-audio clock).
# The agent must have been speaking this long before the user started for the
# user's speech to count as an interruption — otherwise it's the tail of the
# user's own turn overlapping the agent's fast full-duplex response onset.
_MIN_AGENT_SPEECH_BEFORE_BARGE_IN_MS = 300
# The user speech must have started no later than this after the agent's
# audio stopped — otherwise it's normal turn-taking, not an interruption.
_BARGE_IN_OVERLAP_MARGIN_MS = 200


class DiscreteTimeGPTLiveAdapter(DiscreteTimeAdapter):
    """Adapter for discrete-time full-duplex simulation with the GPT-Live API.

    Runs an async event loop in a background thread to communicate with the
    Live API, while exposing a synchronous tick interface.

    Audio conversion is handled automatically:
    - Input: telephony (8kHz μ-law) → GPT-Live (24kHz PCM16)
    - Output: GPT-Live (24kHz PCM16) → telephony (8kHz μ-law)
    """

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = False,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        provider: Optional[GPTLiveProvider] = None,
        voice: str = DEFAULT_GPTLIVE_VOICE,
    ):
        """Initialize the discrete-time GPT-Live adapter.

        Args:
            tick_duration_ms: Duration of each tick in milliseconds. Must be > 0.
            send_audio_instant: If True, send audio in one call per tick.
                If False, send in 20ms chunks with sleeps (VoIP-style).
            model: Live model slug. Defaults to the provider default.
            reasoning_effort: Optional reasoning effort forwarded to the
                Responses delegation backend model.
            provider: Optional provider instance. Created lazily if not given.
            voice: Output voice. Default: marin.
        """
        super().__init__(tick_duration_ms, send_audio_instant=send_audio_instant)

        self._chunk_size = int(
            _GPTLIVE_INPUT_BYTES_PER_SECOND * self._voip_interval_ms / 1000
        )
        self.voice = voice

        if model is not None and provider is not None:
            raise ValueError("model and provider cannot be provided together")

        self.model = model
        self.reasoning_effort = reasoning_effort

        self._audio_converter = StreamingTelephonyConverter(
            input_sample_rate=DEFAULT_GPTLIVE_SAMPLE_RATE,
            output_sample_rate=DEFAULT_GPTLIVE_SAMPLE_RATE,
        )

        # Synthetic utterance id (output_audio.delta has no item_id).
        self._utterance_seq = 1

        # Silence gate state: True while the model is audibly speaking.
        self._gate_open = False

        # Barge-in bookkeeping on the cumulative user-audio clock. Input
        # transcript fragments lag real speech by ~600-1000ms (ASR latency)
        # and the full-duplex model stops speaking quickly when interrupted,
        # so overlap must be checked against the timeline, not against the
        # current tick's audio.
        self._agent_speech_started_user_ms: Optional[int] = None
        self._last_agent_audio_user_ms: Optional[int] = None

        # Function-call announcements (call_id/name) keyed by output item id.
        # response.function_call_arguments.done does not carry call_id/name in
        # practice; response.output_item.added does. Join on item_id.
        self._function_calls_by_item_id: dict[str, tuple[str, str]] = {}

        self._provider = provider
        self._owns_provider = provider is None

        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False

    @property
    def provider(self) -> GPTLiveProvider:
        """Get the provider, creating it if needed."""
        if self._provider is None:
            self._provider = GPTLiveProvider(
                model=self.model,
                voice=self.voice,
                reasoning_effort=self.reasoning_effort,
            )
        return self._provider

    @property
    def _utterance_id(self) -> str:
        """Current synthetic utterance id for audio/transcript correlation."""
        return f"live_utt_{self._utterance_seq}"

    @property
    def is_connected(self) -> bool:
        return self._connected and self._bg_loop.is_running

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any = None,
        modality: str = "audio",
    ) -> None:
        """Connect to the GPT-Live API and configure the session.

        Args:
            system_prompt: System prompt for the agent (also passed to the
                Responses delegation backend that executes tools).
            tools: Tools the agent can use (via Responses delegation).
            vad_config: Ignored — GPT-Live is full-duplex and has no VAD.
            modality: Must be "audio".
        """
        if self._connected:
            logger.warning("Already connected, disconnecting first")
            self.disconnect()

        self._bg_loop.start()

        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools, vad_config, modality),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info(
                f"DiscreteTimeGPTLiveAdapter connected to GPT-Live API "
                f"(tick={self.tick_duration_ms}ms, bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as e:
            logger.error(f"DiscreteTimeGPTLiveAdapter failed to connect: {e}")
            self._bg_loop.stop()
            raise RuntimeError(f"Failed to connect to GPT-Live API: {e}") from e

    async def _async_connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any,
        modality: str,
    ) -> None:
        await self.provider.connect()
        await self.provider.configure_session(
            system_prompt=system_prompt,
            tools=tools,
            vad_config=vad_config,
            modality=modality,
        )

    def disconnect(self) -> None:
        """Disconnect from the API and clean up resources."""
        if not self._connected:
            return

        if self._bg_loop.is_running:
            try:
                self._bg_loop.run_coroutine(
                    self._async_disconnect(),
                    timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
                )
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

        self._bg_loop.stop()
        self._connected = False
        self._tick_count = 0
        self._cumulative_user_audio_ms = 0
        self.clear_buffers()
        self._function_calls_by_item_id.clear()
        self._audio_converter.reset()
        self._gate_open = False
        self._agent_speech_started_user_ms = None
        self._last_agent_audio_user_ms = None
        logger.info("DiscreteTimeGPTLiveAdapter disconnected")

    async def _async_disconnect(self) -> None:
        if self._owns_provider and self._provider is not None:
            await self.provider.disconnect()

    def run_tick(
        self, user_audio: bytes, tick_number: Optional[int] = None
    ) -> TickResult:
        """Run one tick of the simulation.

        Args:
            user_audio: User audio bytes in telephony format (8kHz μ-law).
            tick_number: Optional tick number for logging.

        Returns:
            TickResult with audio in telephony format (8kHz μ-law).
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to GPT-Live API. Call connect() first.")

        if tick_number is None:
            tick_number = self._tick_count
        self._tick_count = tick_number + 1

        try:
            return self._bg_loop.run_coroutine(
                self._async_run_tick(user_audio, tick_number),
                timeout=self.tick_duration_ms / 1000
                + DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
            )
        except Exception as e:
            logger.error(f"Error in run_tick (tick={tick_number}): {e}")
            raise

    async def _flush_pending_tool_results(self) -> None:
        """Send pending tool results via delegation.function_call_output.create."""
        for (
            call_id,
            result_str,
            request_response,
            _is_error,
        ) in self._pending_tool_results:
            await self.provider.send_tool_result(call_id, result_str, request_response)
        self._pending_tool_results.clear()

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        """GPT-Live-specific: convert audio, send, receive events, process."""
        live_audio = self._audio_converter.convert_input(user_audio)

        async def receive_events():
            elapsed_so_far = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, (self.tick_duration_ms / 1000) - elapsed_so_far)
            return await self.provider.receive_events_for_duration(remaining)

        _, events = await asyncio.gather(
            self._send_audio_chunked(
                live_audio, self.provider.send_audio, self._chunk_size
            ),
            receive_events(),
        )

        for event in events:
            self._process_event(result, event)

    def _get_utterance_transcript(self) -> UtteranceTranscript:
        """Get or create the tracker for the current synthetic utterance."""
        item_id = self._utterance_id
        if item_id not in self._utterance_transcripts:
            self._utterance_transcripts[item_id] = UtteranceTranscript(item_id=item_id)
        return self._utterance_transcripts[item_id]

    def _process_event(self, result: TickResult, event: Any) -> None:
        """Process a GPT-Live event."""
        result.events.append(event)

        if isinstance(event, OutputAudioDeltaEvent):
            if not event.audio:
                return
            live_audio = base64.b64decode(event.audio)

            # Silence gate: GPT-Live streams audio continuously (including
            # silence) in real time and emits no speech events. Only forward
            # audio while the model is audibly speaking, otherwise the
            # framework thinks the agent never stops talking. Thresholds and
            # calibration notes live in config.py.
            now_user_ms = result.cumulative_user_audio_at_tick_start_ms + int(
                result.audio_sent_duration_ms
            )
            rms = audioop.rms(live_audio, 2) if live_audio else 0
            if not self._gate_open:
                if rms < GPTLIVE_SILENCE_GATE_OPEN_RMS:
                    return  # true silence or noise floor between utterances
                self._gate_open = True
                self._agent_speech_started_user_ms = now_user_ms
            elif rms < GPTLIVE_SILENCE_GATE_CLOSE_RMS:
                # Utterance ended: close the gate and rotate the synthetic
                # utterance id so the next response gets a fresh tracker.
                # (Speech start/last timestamps persist for barge-in checks.)
                self._gate_open = False
                self._utterance_seq += 1
                return

            telephony_audio = self._audio_converter.convert_output(live_audio)
            if not telephony_audio:
                return

            result.agent_audio_chunks.append((telephony_audio, self._utterance_id))
            self._get_utterance_transcript().add_audio(len(telephony_audio))
            # Audible agent audio forwarded this tick plays until ~end of tick
            # (on the cumulative user-audio clock).
            self._last_agent_audio_user_ms = now_user_ms

        elif isinstance(event, OutputTranscriptAddedEvent):
            if event.text:
                self._get_utterance_transcript().add_transcript(event.text)

        elif isinstance(event, InputTranscriptAddedEvent):
            # No VAD events exist on this API; an input transcript fragment is
            # our evidence that the user is speaking. If the fragment's time
            # range overlaps the agent's ongoing speech, it's a barge-in.
            logger.debug(f"Input transcript fragment: {event.text!r}")
            if not result.was_truncated and self._is_barge_in(event):
                self._handle_barge_in(result, event)

        elif isinstance(event, ResponseOutputItemAddedEvent):
            if event.item_type == "function_call" and event.item_id and event.call_id:
                self._function_calls_by_item_id[event.item_id] = (
                    event.call_id,
                    event.name or "",
                )

        elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            # call_id/name are absent on this event in practice; recover them
            # from the response.output_item.added announcement via item_id.
            call_id, name = event.call_id, event.name
            if (not call_id or not name) and event.item_id:
                call_id, name = self._function_calls_by_item_id.get(
                    event.item_id, (call_id, name)
                )
            if call_id and name:
                try:
                    arguments = json.loads(event.arguments) if event.arguments else {}
                except json.JSONDecodeError:
                    arguments = {}

                tool_call = ToolCall(
                    id=call_id,
                    name=name,
                    arguments=arguments,
                )
                result.tool_calls.append(tool_call)
                logger.debug(f"Tool call detected: {name}({call_id})")
            else:
                logger.warning(
                    f"Function call arguments done without resolvable call_id/name "
                    f"(item_id={event.item_id})"
                )

        elif isinstance(event, DelegationCreatedEvent):
            logger.debug(
                f"Delegation created (target={event.target}, "
                f"response_id={event.response_id}): {event.text!r}"
            )

        elif isinstance(event, ErrorEvent):
            logger.error(
                f"GPT-Live error: {event.code}: {event.message} "
                f"(param={event.param}, event_id={event.event_id})"
            )

        elif isinstance(event, SessionClosedEvent):
            logger.warning(f"GPT-Live session closed by server (reason={event.reason})")

        elif isinstance(event, TimeoutEvent):
            pass

        else:
            logger.debug(f"Event {type(event).__name__} received")

    def _is_barge_in(self, event: InputTranscriptAddedEvent) -> bool:
        """Check if a user speech fragment overlaps the agent's speech.

        The fragment's start_ms is on the same clock as the cumulative user
        audio we've sent, so it can be compared against when the agent's
        current utterance started and when its audio last played:

        - The agent must have started speaking sufficiently BEFORE the user
          fragment (otherwise the "overlap" is just the model's fast
          full-duplex response onset landing on the tail of the user's turn).
        - The agent's audio must still be playing, or have stopped only just
          before the user fragment started (the model yields quickly when
          interrupted, and ASR fragments arrive with delay).
        """
        if self._agent_speech_started_user_ms is None:
            return False
        if self._last_agent_audio_user_ms is None or event.start_ms is None:
            return False
        agent_spoke_long_enough = (
            event.start_ms - self._agent_speech_started_user_ms
            >= _MIN_AGENT_SPEECH_BEFORE_BARGE_IN_MS
        )
        overlaps_agent_audio = (
            event.start_ms
            <= self._last_agent_audio_user_ms + _BARGE_IN_OVERLAP_MARGIN_MS
        )
        return agent_spoke_long_enough and overlaps_agent_audio

    def _handle_barge_in(
        self, result: TickResult, event: InputTranscriptAddedEvent
    ) -> None:
        """Synthesize an interruption when the user speaks over the agent.

        Emits a "speech_started" vad event, discards locally buffered agent
        audio, and marks the tick truncated. The GPT-Live model yields on its
        own (full-duplex) and its server timeline already reflects the barge-in,
        so unlike the Realtime API there is no truncate call to send.
        """
        logger.debug(
            f"Barge-in synthesized from input transcript at {event.start_ms}ms"
        )
        result.vad_events.append("speech_started")

        if self._buffered_agent_audio:
            buffered_bytes = sum(len(c[0]) for c in self._buffered_agent_audio)
            result.truncated_audio_bytes += buffered_bytes
            self._buffered_agent_audio.clear()

        # event.start_ms is on the server input timeline, which tracks the
        # cumulative user audio clock; fall back to "now" if missing.
        audio_start_ms = (
            event.start_ms
            if event.start_ms is not None
            else result.cumulative_user_audio_at_tick_start_ms
        )
        # No item id: deltas are unattributed, so per-item skipping is
        # meaningless. Local buffer clearing + the model yielding is enough.
        result.truncate_agent_audio(
            item_id=None,
            audio_start_ms=audio_start_ms,
            cumulative_user_audio_at_tick_start_ms=result.cumulative_user_audio_at_tick_start_ms,
            bytes_per_tick=result.bytes_per_tick,
        )

        # Fresh resample state, closed gate, and a fresh utterance id for
        # whatever the model says next, so the interrupted utterance's
        # undelivered transcript isn't attributed to the new response.
        # Clearing the speech timestamps stops the same interruption from
        # re-triggering on the user's subsequent transcript fragments.
        self._audio_converter.reset()
        self._gate_open = False
        self._utterance_seq += 1
        self._agent_speech_started_user_ms = None
        self._last_agent_audio_user_ms = None
