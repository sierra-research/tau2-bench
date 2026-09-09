"""Connect Live media to Sierra's voice simulator tick interface."""

import audioop
import base64
import hashlib
from collections import deque
from pathlib import Path

from tau2.config import DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioFormat
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.openai.discrete_time_adapter import (
    DiscreteTimeOpenAIAdapter,
)
from tau2.voice.audio_native.openai.events import AudioDeltaEvent
from tau2.voice.audio_native.openai.live_config import LiveConfig
from tau2.voice.audio_native.openai.live_provider import (
    LiveTranscriptDelta,
    OpenAILiveProvider,
)
from tau2.voice.audio_native.tick_result import TickResult


class DiscreteTimeOpenAILiveAdapter(DiscreteTimeOpenAIAdapter):
    def __init__(
        self,
        *,
        tick_duration_ms: int,
        model: str,
        config: LiveConfig,
        reasoning_effort: str | None = None,
        send_audio_instant: bool = False,
        audio_format: AudioFormat | None = None,
        trace_path: Path | None = None,
    ) -> None:
        if audio_format is not None and audio_format != TELEPHONY_AUDIO_FORMAT:
            raise ValueError("The Live adapter expects 8 kHz mu-law telephony audio")
        super().__init__(
            tick_duration_ms=tick_duration_ms,
            # Queue the tick once; the WebRTC track already clocks its 20 ms frames.
            send_audio_instant=True,
            provider=OpenAILiveProvider(
                model=model,
                config=config,
                reasoning_effort=reasoning_effort,
                trace_path=trace_path,
            ),
            audio_format=audio_format,
        )
        self.model = model
        self.realtime_pacing = True
        # One encoder runs at a time; leave a second slot for DNS/setup work.
        self._bg_loop = BackgroundAsyncLoop(executor_workers=2)
        self._live_config = config
        self._live_reasoning_effort = reasoning_effort
        self._trace_path = trace_path
        self._owns_provider = True
        self._transcript_deltas: deque[LiveTranscriptDelta] = deque()
        self._played_audio_bytes = 0
        self._received_audio_bytes = 0
        self._received_audio_hash = hashlib.sha256()
        self._played_audio_hash = hashlib.sha256()
        self._max_buffered_audio_bytes = 0
        self._converter = StreamingTelephonyConverter(
            input_sample_rate=DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE,
            output_sample_rate=DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE,
        )
        self._chunk_size = int(
            DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE * 2 * self._voip_interval_ms / 1000
        )

    def connect(self, system_prompt, tools, vad_config=None, modality="audio") -> None:
        if self.is_connected:
            self.disconnect()
        if self.provider._closing:
            self._provider = OpenAILiveProvider(
                model=self.model,
                config=self._live_config,
                reasoning_effort=self._live_reasoning_effort,
                trace_path=self._trace_path,
            )
        super().connect(system_prompt, tools, vad_config, modality)

    async def _execute_tick(
        self, user_audio: bytes, tick_number: int, result: TickResult, tick_start: float
    ) -> None:
        await super()._execute_tick(
            self._converter.convert_input(user_audio), tick_number, result, tick_start
        )

    async def _process_event(self, result: TickResult, event: object) -> None:
        if isinstance(event, LiveTranscriptDelta):
            result.events.append(event)
            self._transcript_deltas.append(event)
            return
        if isinstance(event, AudioDeltaEvent):
            audio = self._converter.convert_output(base64.b64decode(event.delta))
            self._received_audio_bytes += len(audio)
            self._received_audio_hash.update(audio)
            event = event.model_copy(
                update={"delta": base64.b64encode(audio).decode("ascii")}
            )
        await super()._process_event(result, event)

    def run_tick(self, user_audio: bytes, tick_number: int | None = None) -> TickResult:
        result = super().run_tick(user_audio, tick_number)
        self._played_audio_bytes += len(result.agent_audio_data)
        self._played_audio_hash.update(result.agent_audio_data)
        self._max_buffered_audio_bytes = max(
            self._max_buffered_audio_bytes,
            self._received_audio_bytes - self._played_audio_bytes,
        )
        # Compare sample durations across PCM16 ingress and mu-law playback.
        played_pcm_bytes = (
            self._played_audio_bytes
            * DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE
            * 2
            // self.audio_format.bytes_per_second
        )
        text = []
        while (
            self._transcript_deltas
            and self._transcript_deltas[0].audio_position_bytes <= played_pcm_bytes
        ):
            text.append(self._transcript_deltas.popleft().delta)
        result.proportional_transcript = "".join(text)
        # WebRTC also transports silence. Classify the delivered audio, without
        # deleting quiet frames or changing the shared tick buffering lifecycle.
        pcm = audioop.ulaw2lin(result.agent_audio_data, 2)
        result.contains_speech = bool(pcm) and audioop.max(pcm, 2) > 32
        return result

    async def _async_disconnect(self) -> None:
        delivered_and_pending = self._played_audio_hash.copy()
        buffered_bytes = 0
        for audio, _ in self._buffered_agent_audio:
            delivered_and_pending.update(audio)
            buffered_bytes += len(audio)
        self.provider._record(
            "audio_delivery",
            {
                "received_bytes": self._received_audio_bytes,
                "played_bytes": self._played_audio_bytes,
                "buffered_bytes": buffered_bytes,
                "audio_preserved_in_order": (
                    delivered_and_pending.digest() == self._received_audio_hash.digest()
                    and self._played_audio_bytes + buffered_bytes
                    == self._received_audio_bytes
                ),
                "max_buffer_ms": self._max_buffered_audio_bytes
                * 1000
                / self.audio_format.bytes_per_second,
            },
        )
        await super()._async_disconnect()

    def disconnect(self) -> None:
        super().disconnect()
        self._converter.reset()

    def clear_buffers(self) -> None:
        super().clear_buffers()
        self._transcript_deltas.clear()
        self._played_audio_bytes = 0
        self._received_audio_bytes = 0
        self._received_audio_hash = hashlib.sha256()
        self._played_audio_hash = hashlib.sha256()
        self._max_buffered_audio_bytes = 0
