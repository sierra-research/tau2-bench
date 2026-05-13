"""Discrete-time adapter for Inworld Realtime API.

Provides a tick-based interface for the Inworld Realtime API with audio
conversion between tau2's telephony format (8 kHz μ-law) and Inworld's
required PCM16 @ 24 kHz.

Audio format notes:
- Telephony: 8 kHz μ-law (8000 bytes/sec)
- Inworld input/output: 24 kHz PCM16 (48000 bytes/sec)

Inworld-specific behavior: on user interruption (SpeechStartedEvent) the
adapter sends ``response.cancel`` to the server (no per-item truncate
handshake like OpenAI requires).
"""

import asyncio
import base64
import json
from typing import Any, List, Optional

from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
    DEFAULT_INWORLD_OUTPUT_SAMPLE_RATE,
)
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.inworld.events import (
    InworldAudioDeltaEvent,
    InworldAudioDoneEvent,
    InworldAudioTranscriptDeltaEvent,
    InworldAudioTranscriptDoneEvent,
    InworldErrorEvent,
    InworldFunctionCallArgumentsDoneEvent,
    InworldInputTranscriptionCompletedEvent,
    InworldResponseDoneEvent,
    InworldResponseOutputItemAddedEvent,
    InworldSpeechStartedEvent,
    InworldSpeechStoppedEvent,
    InworldTimeoutEvent,
)
from tau2.voice.audio_native.inworld.provider import (
    InworldRealtimeProvider,
    InworldVADConfig,
)
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript

# Bytes/sec for the chunk size used when send_audio_instant=False
INWORLD_INPUT_BYTES_PER_SECOND = DEFAULT_INWORLD_OUTPUT_SAMPLE_RATE * 2


class DiscreteTimeInworldAdapter(DiscreteTimeAdapter):
    """Adapter for discrete-time full-duplex simulation with Inworld Realtime API.

    The orchestrator hands us telephony bytes (8 kHz μ-law); we transcode to
    Inworld's PCM16 @ 24 kHz before sending and back to telephony on the way
    out. ``bytes_per_tick`` stays in telephony units so the base class's
    capping/buffering logic is unchanged.
    """

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = False,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        provider: Optional[InworldRealtimeProvider] = None,
        voice: Optional[str] = None,
    ):
        """Initialize the discrete-time Inworld adapter.

        Args:
            tick_duration_ms: Duration of each tick in milliseconds. Must be > 0.
            send_audio_instant: If True, send audio in one call per tick.
                If False, send in VoIP-style chunks.
            model: LLM backbone (e.g., "openai/gpt-4.1-mini"). Defaults from
                config / env.
            reasoning_effort: Not supported by Inworld. Must be None.
            provider: Optional pre-built provider instance.
            voice: TTS voice name. Defaults from config / env.
        """
        if reasoning_effort is not None:
            raise ValueError(
                f"Inworld provider does not support reasoning_effort "
                f"(got '{reasoning_effort}')"
            )
        super().__init__(tick_duration_ms, send_audio_instant=send_audio_instant)

        if model is not None and provider is not None:
            raise ValueError("model and provider cannot be provided together")

        self.model = model
        self.voice = voice

        # Send-side chunk size, in Inworld input bytes (PCM16 @ 24 kHz)
        self._chunk_size = int(
            INWORLD_INPUT_BYTES_PER_SECOND * self._voip_interval_ms / 1000
        )

        # Telephony ↔ Inworld PCM16 (24 kHz both directions)
        self._audio_converter = StreamingTelephonyConverter(
            input_sample_rate=DEFAULT_INWORLD_OUTPUT_SAMPLE_RATE,
            output_sample_rate=DEFAULT_INWORLD_OUTPUT_SAMPLE_RATE,
        )

        self._provider = provider
        self._owns_provider = provider is None

        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False

        # Inworld emits function-call name in response.output_item.added (under
        # item.name) but only call_id + arguments in
        # response.function_call_arguments.done. Cache name keyed by call_id so
        # the .done handler can look it up.
        self._function_call_names: dict[str, str] = {}

    @property
    def provider(self) -> InworldRealtimeProvider:
        if self._provider is None:
            self._provider = InworldRealtimeProvider(
                model=self.model,
                voice=self.voice,
            )
        return self._provider

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
        if self._connected:
            logger.warning("Already connected, disconnecting first")
            self.disconnect()

        if vad_config is None:
            vad_config = InworldVADConfig()

        self._bg_loop.start()

        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools, vad_config, modality),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info(
                f"DiscreteTimeInworldAdapter connected to Inworld Realtime API "
                f"(tick={self.tick_duration_ms}ms, bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as e:
            logger.error(f"DiscreteTimeInworldAdapter failed to connect: {e}")
            self._bg_loop.stop()
            raise RuntimeError(f"Failed to connect to Inworld Realtime API: {e}") from e

    async def _async_connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: InworldVADConfig,
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
        self._audio_converter.reset()
        self._function_call_names.clear()
        logger.info("DiscreteTimeInworldAdapter disconnected")

    async def _async_disconnect(self) -> None:
        if self._owns_provider and self._provider is not None:
            await self.provider.disconnect()

    def run_tick(
        self, user_audio: bytes, tick_number: Optional[int] = None
    ) -> TickResult:
        if not self.is_connected:
            raise RuntimeError(
                "Not connected to Inworld Realtime API. Call connect() first."
            )

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
        """Inworld-specific: convert audio, send, receive events, process."""
        inworld_audio = self._audio_converter.convert_input(user_audio)

        async def receive_events():
            elapsed_so_far = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, (self.tick_duration_ms / 1000) - elapsed_so_far)
            return await self.provider.receive_events_for_duration(remaining)

        _, events = await asyncio.gather(
            self._send_audio_chunked(
                inworld_audio, self.provider.send_audio, self._chunk_size
            ),
            receive_events(),
        )

        for event in events:
            await self._process_event(result, event)

    async def _process_event(self, result: TickResult, event: Any) -> None:
        """Process an Inworld Realtime event."""
        result.events.append(event)

        if isinstance(event, InworldAudioDeltaEvent):
            item_id = event.item_id or self._current_item_id

            # Skip audio from a truncated item, but still account for the
            # discarded bytes in telephony units.
            if result.skip_item_id is not None and item_id == result.skip_item_id:
                if event.delta:
                    inworld_bytes = base64.b64decode(event.delta)
                    telephony_bytes = self._audio_converter.convert_output(
                        inworld_bytes
                    )
                    result.truncated_audio_bytes += len(telephony_bytes)
                return

            if event.delta:
                inworld_bytes = base64.b64decode(event.delta)
                telephony_bytes = self._audio_converter.convert_output(inworld_bytes)
                if telephony_bytes:
                    result.agent_audio_chunks.append((telephony_bytes, item_id))

                if item_id:
                    self._current_item_id = item_id
                    if item_id not in self._utterance_transcripts:
                        self._utterance_transcripts[item_id] = UtteranceTranscript(
                            item_id=item_id
                        )
                    self._utterance_transcripts[item_id].add_audio(len(telephony_bytes))

        elif isinstance(event, InworldAudioTranscriptDeltaEvent):
            item_id = event.item_id or self._current_item_id
            if item_id and event.delta:
                if item_id not in self._utterance_transcripts:
                    self._utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                self._utterance_transcripts[item_id].add_transcript(event.delta)

        elif isinstance(event, InworldSpeechStartedEvent):
            logger.debug("Inworld VAD: speech started — interruption")
            result.vad_events.append("speech_started")

            # Clear any audio buffered for the next tick.
            if self._buffered_agent_audio:
                buffered_bytes = sum(len(c[0]) for c in self._buffered_agent_audio)
                result.truncated_audio_bytes += buffered_bytes
                self._buffered_agent_audio.clear()

            # Reset converter state so any subsequent audio decodes cleanly.
            self._audio_converter.reset()

            result.was_truncated = True
            result.skip_item_id = self._current_item_id

            # Inworld's `interrupt_response: True` makes the server auto-cancel
            # on speech_started, but sending response.cancel here is harmless
            # and ensures any in-flight response stops if interrupt_response
            # is later disabled.
            try:
                await self.provider.cancel_response()
            except Exception as e:
                logger.debug(f"Inworld response.cancel failed (non-fatal): {e}")

        elif isinstance(event, InworldSpeechStoppedEvent):
            logger.debug("Inworld VAD: speech stopped")
            result.vad_events.append("speech_stopped")

        elif isinstance(event, InworldResponseOutputItemAddedEvent):
            # Inworld puts function-call name here; the .done event only has
            # call_id + arguments. Cache name keyed by call_id for later lookup.
            item = event.item or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                name = item.get("name")
                if call_id and name:
                    self._function_call_names[call_id] = name

        elif isinstance(event, InworldFunctionCallArgumentsDoneEvent):
            try:
                arguments = json.loads(event.arguments) if event.arguments else {}
            except json.JSONDecodeError:
                arguments = {}

            call_id = event.call_id or ""
            # Resolve name: prefer the event field, fall back to cached name
            # from the output_item.added event (Inworld's normal path).
            name = event.name or self._function_call_names.get(call_id, "")

            tool_call = ToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
            )
            result.tool_calls.append(tool_call)
            logger.debug(f"Tool call detected: {name}({call_id})")

        elif isinstance(event, InworldInputTranscriptionCompletedEvent):
            logger.debug(f"Input transcription: {event.transcript}")

        elif isinstance(event, InworldResponseDoneEvent):
            logger.debug("Response done (turn complete)")

        elif isinstance(event, InworldAudioDoneEvent):
            logger.debug(f"Audio done for item {event.item_id}")

        elif isinstance(event, InworldAudioTranscriptDoneEvent):
            logger.debug(f"Transcript done for item {event.item_id}")

        elif isinstance(event, InworldErrorEvent):
            logger.error(f"Inworld error: {event.message or event.error}")

        elif isinstance(event, InworldTimeoutEvent):
            pass

        else:
            logger.debug(f"Event {type(event).__name__} received")
