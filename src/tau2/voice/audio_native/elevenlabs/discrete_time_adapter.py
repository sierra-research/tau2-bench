"""Discrete-time adapter for hosted ElevenLabs voice agents."""

import asyncio
from typing import Any, List, Optional

from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
    ELEVENLABS_AGENT_ID_PREFIXES,
)
from tau2.data_model.message import ToolCall
from tau2.data_model.usage import UsageRecord
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.elevenlabs.events import (
    ElevenLabsAgentResponseCompleteEvent,
    ElevenLabsAgentResponseCorrectionEvent,
    ElevenLabsAgentResponseEvent,
    ElevenLabsAudioEvent,
    ElevenLabsClientErrorEvent,
    ElevenLabsClientToolCallEvent,
    ElevenLabsInterruptionEvent,
    ElevenLabsPingEvent,
    ElevenLabsTimeoutEvent,
    ElevenLabsUserTranscriptEvent,
)
from tau2.voice.audio_native.elevenlabs.provider import (
    ElevenLabsAgentsProvider,
    parse_elevenlabs_audio_format,
)
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript


class DiscreteTimeElevenLabsAdapter(DiscreteTimeAdapter):
    """Bridge an ElevenLabs Agents conversation into tau2 ticks.

    The generic ``model`` value is the hosted ElevenLabs ``agent_id``. The
    agent's configured client tools must match the tau2 domain tools by name
    and JSON schema, and must be configured to wait for a client response.
    """

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = True,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        provider: Optional[ElevenLabsAgentsProvider] = None,
    ):
        if reasoning_effort is not None:
            raise ValueError(
                "ElevenLabs agent reasoning is configured on the hosted agent; "
                "--reasoning-effort is not supported"
            )

        agent_id = (model or getattr(provider, "agent_id", "")).strip()
        if not agent_id.startswith(ELEVENLABS_AGENT_ID_PREFIXES):
            raise ValueError(
                "ElevenLabs requires an agent ID beginning with 'agent_' or "
                "'seng_'. Pass it with --audio-native-model <agent_id>."
            )

        super().__init__(tick_duration_ms, send_audio_instant=send_audio_instant)
        self.model = agent_id
        self.agent_id = agent_id
        self.reasoning_effort = reasoning_effort
        self._provider = provider
        self._owns_provider = provider is None
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False
        self._audio_converter: Optional[StreamingTelephonyConverter] = None
        self._input_is_ulaw = False
        self._output_is_ulaw = False
        self._chunk_size = 0
        self._pending_response_text: dict[str, str] = {}
        self._items_with_alignment: set[str] = set()
        self._configured_tool_names: set[str] = set()
        self._last_interrupt_event_id = -1
        self._session_counter = 0

    @property
    def provider(self) -> ElevenLabsAgentsProvider:
        """Get the provider, creating it lazily."""
        if self._provider is None:
            self._provider = ElevenLabsAgentsProvider(agent_id=self.agent_id)
        return self._provider

    @property
    def is_connected(self) -> bool:
        """Whether the adapter and its background loop are active."""
        return self._connected and self._bg_loop.is_running

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any = None,
        modality: str = "audio",
    ) -> None:
        """Start a hosted ElevenLabs conversation.

        VAD, LLM, ASR, TTS, voice, and client-tool definitions are owned by the
        ElevenLabs agent. The agent must allow prompt and first-message
        overrides so tau2 can inject the domain policy and wait for the user.
        """
        if modality != "audio":
            raise ValueError("ElevenLabs Agents currently supports audio modality only")
        if self._connected:
            self.disconnect()

        self._configured_tool_names = {
            tool.openai_schema["function"]["name"] for tool in tools
        }
        self._bg_loop.start()
        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            self._session_counter += 1
            logger.info(
                "DiscreteTimeElevenLabsAdapter connected "
                f"(agent_id={self.agent_id}, tick={self.tick_duration_ms}ms)"
            )
        except Exception as error:
            self._bg_loop.stop()
            raise RuntimeError(
                f"Failed to connect to ElevenLabs agent {self.agent_id}: {error}"
            ) from error

    async def _async_connect(self, system_prompt: str) -> None:
        await self.provider.connect(system_prompt=system_prompt)

        input_format = self.provider.user_input_audio_format
        output_format = self.provider.agent_output_audio_format
        if input_format is None or output_format is None:
            raise RuntimeError("ElevenLabs did not report negotiated audio formats")

        input_encoding, input_rate = parse_elevenlabs_audio_format(input_format)
        output_encoding, output_rate = parse_elevenlabs_audio_format(output_format)
        self._input_is_ulaw = input_encoding == "ulaw"
        self._output_is_ulaw = output_encoding == "ulaw"
        self._audio_converter = StreamingTelephonyConverter(
            input_sample_rate=input_rate,
            output_sample_rate=output_rate,
        )
        input_sample_width = 1 if self._input_is_ulaw else 2
        self._chunk_size = int(
            input_rate * input_sample_width * self._voip_interval_ms / 1000
        )

    def disconnect(self) -> None:
        """End the conversation and reset adapter state."""
        if not self._connected:
            return

        usage = self._make_audio_usage_record()
        if usage is not None:
            self._usage_records.append(usage)

        if self._bg_loop.is_running:
            try:
                self._bg_loop.run_coroutine(
                    self._async_disconnect(),
                    timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
                )
            except Exception as error:
                logger.warning(f"Error disconnecting ElevenLabs agent: {error}")

        self._bg_loop.stop()
        self._connected = False
        self._tick_count = 0
        self._cumulative_user_audio_ms = 0
        self.clear_buffers()
        self._pending_response_text.clear()
        self._items_with_alignment.clear()
        self._last_interrupt_event_id = -1
        if self._audio_converter is not None:
            self._audio_converter.reset()
        logger.info("DiscreteTimeElevenLabsAdapter disconnected")

    async def _async_disconnect(self) -> None:
        if self._owns_provider and self._provider is not None:
            await self.provider.disconnect()

    def _make_audio_usage_record(self) -> Optional[UsageRecord]:
        if self._cumulative_user_audio_ms <= 0:
            return None
        return UsageRecord(
            provider="elevenlabs",
            model=self.agent_id,
            component="realtime",
            semantics="cumulative",
            scope_id=self.provider.session_id
            or f"audio-session-{self._session_counter}",
            audio_input_seconds=self._cumulative_user_audio_ms / 1000,
        )

    def get_usage_records(self) -> List[UsageRecord]:
        """Return ledger records plus the current conversation-duration meter."""
        records = super().get_usage_records()
        live_audio = self._make_audio_usage_record()
        if live_audio is not None:
            records.append(live_audio)
        return records

    def run_tick(
        self,
        user_audio: bytes,
        tick_number: Optional[int] = None,
    ) -> TickResult:
        """Exchange one tick of audio and events."""
        if not self.is_connected:
            raise RuntimeError(
                "Not connected to ElevenLabs Agents API. Call connect() first."
            )
        if tick_number is None:
            tick_number = self._tick_count
        self._tick_count = tick_number + 1
        return self._bg_loop.run_coroutine(
            self._async_run_tick(user_audio, tick_number),
            timeout=self.tick_duration_ms / 1000
            + DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
        )

    async def _flush_pending_tool_results(self) -> None:
        """Send queued client-tool results before this tick's audio."""
        for call_id, result, _request_response, is_error in self._pending_tool_results:
            await self.provider.send_tool_result(
                call_id=call_id,
                result=result,
                is_error=is_error,
            )
        self._pending_tool_results.clear()

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        """Convert audio, exchange WebSocket events, and populate the tick."""
        if self._audio_converter is None:
            raise RuntimeError("ElevenLabs audio converter is not configured")

        provider_audio = (
            user_audio
            if self._input_is_ulaw
            else self._audio_converter.convert_input(user_audio)
        )

        async def receive_events():
            elapsed = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, self.tick_duration_ms / 1000 - elapsed)
            return await self.provider.receive_events_for_duration(remaining)

        _, events = await asyncio.gather(
            self._send_audio_chunked(
                provider_audio,
                self.provider.send_audio,
                self._chunk_size,
            ),
            receive_events(),
        )
        for event in events:
            self._process_event(result, event)

    def _transcript(self, item_id: str) -> UtteranceTranscript:
        if item_id not in self._utterance_transcripts:
            self._utterance_transcripts[item_id] = UtteranceTranscript(item_id=item_id)
        return self._utterance_transcripts[item_id]

    def _process_event(self, result: TickResult, event: Any) -> None:
        result.events.append(event)

        if isinstance(event, ElevenLabsAudioEvent):
            item_id = str(event.event_id)
            self._current_item_id = item_id
            if (
                event.event_id <= self._last_interrupt_event_id
                or result.skip_item_id == item_id
            ):
                result.truncated_audio_bytes += self._estimated_telephony_bytes(
                    event.audio_data
                )
                return

            if self._output_is_ulaw:
                audio = event.audio_data
            else:
                if self._audio_converter is None:
                    raise RuntimeError("ElevenLabs audio converter is not configured")
                audio = self._audio_converter.convert_output(event.audio_data)
            if audio:
                result.agent_audio_chunks.append((audio, item_id))
                self._transcript(item_id).add_audio(len(audio))

            alignment_chars = (event.alignment or {}).get("chars") or []
            if alignment_chars:
                self._items_with_alignment.add(item_id)
                self._transcript(item_id).add_transcript("".join(alignment_chars))
            elif (
                item_id not in self._items_with_alignment
                and not self._transcript(item_id).transcript_received
                and item_id in self._pending_response_text
            ):
                self._transcript(item_id).add_transcript(
                    self._pending_response_text[item_id]
                )

        elif isinstance(event, ElevenLabsAgentResponseEvent):
            item_id = str(event.event_id)
            self._current_item_id = item_id
            self._pending_response_text[item_id] = event.transcript
            transcript = self._utterance_transcripts.get(item_id)
            if (
                transcript is not None
                and item_id not in self._items_with_alignment
                and not transcript.transcript_received
            ):
                transcript.add_transcript(event.transcript)

        elif isinstance(event, ElevenLabsAgentResponseCorrectionEvent):
            item_id = str(event.event_id)
            transcript = self._transcript(item_id)
            transcript.transcript_received = event.corrected_transcript
            transcript.transcript_chars_shown = min(
                transcript.transcript_chars_shown,
                len(event.corrected_transcript),
            )
            self._pending_response_text[item_id] = event.corrected_transcript

        elif isinstance(event, ElevenLabsInterruptionEvent):
            item_id = str(event.event_id)
            self._last_interrupt_event_id = max(
                self._last_interrupt_event_id, event.event_id
            )
            result.vad_events.append("interrupted")
            kept_chunks = []
            for chunk, chunk_item_id in result.agent_audio_chunks:
                if chunk_item_id == item_id:
                    result.truncated_audio_bytes += len(chunk)
                else:
                    kept_chunks.append((chunk, chunk_item_id))
            result.agent_audio_chunks = kept_chunks
            self._buffered_agent_audio.clear()
            result.was_truncated = True
            result.skip_item_id = item_id
            if self._audio_converter is not None:
                self._audio_converter.reset()

        elif isinstance(event, ElevenLabsClientToolCallEvent):
            if (
                self._configured_tool_names
                and event.tool_name not in self._configured_tool_names
            ):
                logger.warning(
                    f"ElevenLabs agent called unconfigured tau2 tool "
                    f"{event.tool_name!r}; expected one of "
                    f"{sorted(self._configured_tool_names)}"
                )
            if not event.expects_response:
                logger.warning(
                    f"ElevenLabs client tool {event.tool_name!r} is configured "
                    "without a response; tau2 environment tools should wait for one"
                )
            result.tool_calls.append(
                ToolCall(
                    id=event.tool_call_id,
                    name=event.tool_name,
                    arguments=event.parameters,
                )
            )

        elif isinstance(event, ElevenLabsUserTranscriptEvent):
            logger.debug(f"ElevenLabs user transcript: {event.transcript}")

        elif isinstance(event, ElevenLabsAgentResponseCompleteEvent):
            logger.debug(f"ElevenLabs response complete: {event.event_id}")

        elif isinstance(event, ElevenLabsClientErrorEvent):
            logger.error(
                "ElevenLabs client error: "
                f"{event.error_name or event.code}: {event.message}"
            )

        elif isinstance(event, (ElevenLabsPingEvent, ElevenLabsTimeoutEvent)):
            pass

        else:
            logger.debug(f"ElevenLabs event {type(event).__name__} received")

    def _estimated_telephony_bytes(self, provider_audio: bytes) -> int:
        if self._output_is_ulaw:
            return len(provider_audio)
        output_format = self.provider.agent_output_audio_format
        if output_format is None:
            return 0
        _, output_rate = parse_elevenlabs_audio_format(output_format)
        return int(
            len(provider_audio) * self.audio_format.bytes_per_second / (output_rate * 2)
        )
