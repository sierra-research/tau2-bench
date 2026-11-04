"""Native Tau adapter for a persistent Launa voice session."""

import base64
from typing import List, Optional

from tau2.config import LAUNA_SAMPLE_RATE
from tau2.data_model.audio import AudioFormat
from tau2.environment.tool import Tool
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.openai.discrete_time_adapter import (
    DiscreteTimeOpenAIAdapter,
)
from tau2.voice.audio_native.openai.events import (
    AudioDeltaEvent,
    AudioTranscriptDoneEvent,
    BaseRealtimeEvent,
    ResponseCancelledEvent,
    SpeechStartedEvent,
)
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript

from .provider import LaunaRealtimeWSProvider


class DiscreteTimeLaunaAdapter(DiscreteTimeOpenAIAdapter):
    """Stream every simulator audio tick through the hosted voice service.

    The adapter deliberately ignores simulator transcript/final-chunk metadata.
    The hosted voice service owns turn-taking, conversation history, response
    audio, interruption handling, and post-tool continuation in one session.
    """

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = False,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        provider: Optional[LaunaRealtimeWSProvider] = None,
        audio_format: Optional[AudioFormat] = None,
    ) -> None:
        super().__init__(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=send_audio_instant,
            model=model,
            reasoning_effort=reasoning_effort,
            provider=provider,
            audio_format=audio_format,
        )
        self._converter = StreamingTelephonyConverter(
            input_sample_rate=LAUNA_SAMPLE_RATE,
            output_sample_rate=LAUNA_SAMPLE_RATE,
        )
        self._output_converter = StreamingTelephonyConverter(
            input_sample_rate=LAUNA_SAMPLE_RATE,
            output_sample_rate=LAUNA_SAMPLE_RATE,
        )
        self._chunk_size = int(LAUNA_SAMPLE_RATE * 2 * self._voip_interval_ms / 1_000)
        self._cancelled_audio_items: set[str] = set()
        self._last_audio_item_id: Optional[str] = None

    @property
    def provider(self) -> LaunaRealtimeWSProvider:
        if self._provider is None:
            self._provider = LaunaRealtimeWSProvider(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
            )
        if not isinstance(self._provider, LaunaRealtimeWSProvider):
            raise TypeError("Expected a Launa provider")
        return self._provider

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: object = None,
        modality: str = "audio",
    ) -> None:
        self._converter.reset()
        self._output_converter.reset()
        self._cancelled_audio_items.clear()
        self._last_audio_item_id = None
        super().connect(system_prompt, tools, vad_config, modality)

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        provider_audio = self._converter.convert_input(user_audio)
        await super()._execute_tick(
            provider_audio,
            tick_number,
            result,
            tick_start,
        )

    async def _process_event(
        self, result: TickResult, event: BaseRealtimeEvent
    ) -> None:
        if isinstance(event, AudioDeltaEvent):
            provider_audio = base64.b64decode(event.delta)
            if event.item_id in self._cancelled_audio_items:
                result.events.append(event)
                result.truncated_audio_bytes += int(
                    len(provider_audio)
                    * self.audio_format.bytes_per_second
                    / (LAUNA_SAMPLE_RATE * 2)
                )
                return
            if event.item_id:
                self._last_audio_item_id = event.item_id
            telephony_audio = self._output_converter.convert_output(provider_audio)
            event = event.model_copy(
                update={"delta": base64.b64encode(telephony_audio).decode("ascii")}
            )

        if isinstance(event, AudioTranscriptDoneEvent) and event.item_id:
            transcript = self._utterance_transcripts.setdefault(
                event.item_id,
                UtteranceTranscript(item_id=event.item_id),
            )
            # the hosted voice service emits only the final audible transcript, not deltas.
            # Avoid appending it twice if the protocol later adds deltas.
            if not transcript.transcript_received:
                transcript.add_transcript(event.transcript)

        await super()._process_event(result, event)

        if isinstance(event, ResponseCancelledEvent):
            # The cancellation signal must stop playback even if no speech-start
            # event arrived. Remember item IDs so late chunks cannot resume it.
            chunks = result.agent_audio_chunks + self._buffered_agent_audio
            self._cancelled_audio_items.update(
                item_id for _, item_id in chunks if item_id is not None
            )
            if self._last_audio_item_id:
                self._cancelled_audio_items.add(self._last_audio_item_id)
            buffered_bytes = sum(len(data) for data, _ in self._buffered_agent_audio)
            self._buffered_agent_audio.clear()
            if not result.was_truncated:
                discarded = result.truncated_audio_bytes + result.agent_audio_bytes
                result.truncate_agent_audio(
                    item_id=self._last_audio_item_id,
                    audio_start_ms=result.cumulative_user_audio_at_tick_start_ms,
                    cumulative_user_audio_at_tick_start_ms=(
                        result.cumulative_user_audio_at_tick_start_ms
                    ),
                    bytes_per_tick=result.bytes_per_tick,
                )
                result.agent_audio_chunks.clear()
                result.truncated_audio_bytes = discarded
                result.vad_events.append("interrupted")
            # Preserve a preceding speech-start event's more precise timing.
            result.truncated_audio_bytes += buffered_bytes
            self._output_converter.reset()

        if isinstance(event, SpeechStartedEvent):
            if result.skip_item_id:
                self._cancelled_audio_items.add(result.skip_item_id)
            self._output_converter.reset()

    def disconnect(self) -> None:
        super().disconnect()
        self._converter.reset()
        self._output_converter.reset()
        self._cancelled_audio_items.clear()
        self._last_audio_item_id = None
