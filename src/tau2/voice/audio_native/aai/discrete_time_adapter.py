"""Discrete-time adapter for the AAI (AssemblyAI voice-agent host) provider.

Mirrors the AssemblyAI adapter's lifecycle/tick structure, with one key
difference: the aai host speaks PCM16 over its WebSocket (16kHz in, 24kHz
out), while the adapter's external interface stays tau2 telephony mu-law/8k.
Audio conversion therefore happens at this boundary:

- Send path: mu-law 8k (external) -> PCM16 -> resample to 16kHz -> provider.
- Receive path: PCM16 24k (from provider) -> resample to 8kHz -> mu-law.

aai events don't carry a reply/turn id, so a synthetic running turn id
(``turn-N``) is maintained locally and incremented on ``reply_done``.
"""

import asyncio
from typing import Any, List, Optional

from loguru import logger

from tau2.config import (
    DEFAULT_AAI_INPUT_SAMPLE_RATE,
    DEFAULT_AAI_OUTPUT_SAMPLE_RATE,
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
)
from tau2.data_model.audio import (
    TELEPHONY_AUDIO_FORMAT,
    AudioData,
    AudioEncoding,
    AudioFormat,
)
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.aai.events import (
    AAIAgentTranscriptEvent,
    AAIAudioChunkEvent,
    AAIAudioDoneEvent,
    AAIErrorEvent,
    AAIReplyDoneEvent,
    AAISpeechStartedEvent,
    AAISpeechStoppedEvent,
    AAITimeoutEvent,
    AAIToolCallEvent,
)
from tau2.voice.audio_native.aai.provider import AAIVoiceAgentProvider
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript
from tau2.voice.utils.audio_preprocessing import (
    convert_to_pcm16,
    convert_to_ulaw,
    resample_audio,
)

# aai host WebSocket audio rates (shared with AAIVoiceAgentProvider via
# tau2.config so the two sides can't drift).
AAI_SEND_SAMPLE_RATE = DEFAULT_AAI_INPUT_SAMPLE_RATE
AAI_RECEIVE_SAMPLE_RATE = DEFAULT_AAI_OUTPUT_SAMPLE_RATE
AAI_RECEIVE_AUDIO_FORMAT = AudioFormat(
    encoding=AudioEncoding.PCM_S16LE,
    sample_rate=AAI_RECEIVE_SAMPLE_RATE,
    channels=1,
)


class DiscreteTimeAAIAdapter(DiscreteTimeAdapter):
    """Discrete-time adapter for the aai voice-agent host provider."""

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = True,
        reasoning_effort: Optional[str] = None,
        provider: Optional[AAIVoiceAgentProvider] = None,
        system_prompt: str = "",
        tools: tuple = (),
    ):
        if reasoning_effort is not None:
            raise ValueError(
                f"aai provider does not support reasoning_effort "
                f"(got '{reasoning_effort}')"
            )
        # External interface stays tau2 telephony mu-law/8k (base default);
        # PCM16 conversion happens at the WebSocket boundary, not here.
        super().__init__(tick_duration_ms, send_audio_instant=send_audio_instant)
        self._chunk_size = int(AAI_SEND_SAMPLE_RATE * 2 * self._voip_interval_ms / 1000)
        self.system_prompt = system_prompt
        self.tools = tools
        self._provider = provider
        self._owns_provider = provider is None
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False
        self._turn_index = 0
        # Turn ids interrupted by a barge-in (AAISpeechStartedEvent); their
        # reply_done is expected to carry zero audio/transcript, so the
        # loud-failure guard in _process_event skips them.
        self._interrupted_turn_ids: set[str] = set()

    @property
    def provider(self) -> AAIVoiceAgentProvider:
        if self._provider is None:
            self._provider = AAIVoiceAgentProvider(
                system_prompt=self.system_prompt,
                tools=self.tools,
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
        self.system_prompt = system_prompt
        self.tools = tuple(tools)
        self._bg_loop.start()
        try:
            self._bg_loop.run_coroutine(
                self.provider.connect(),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info(
                f"DiscreteTimeAAIAdapter connected "
                f"(tick={self.tick_duration_ms}ms, bytes_per_tick={self.bytes_per_tick})"
            )
        except Exception as e:
            logger.error(f"DiscreteTimeAAIAdapter failed to connect: {e}")
            self._bg_loop.stop()
            raise RuntimeError(f"Failed to connect to aai host: {e}") from e

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
        logger.info("DiscreteTimeAAIAdapter disconnected")

    async def _async_disconnect(self) -> None:
        if self._owns_provider and self._provider is not None:
            await self.provider.disconnect()

    def run_tick(
        self, user_audio: bytes, tick_number: Optional[int] = None
    ) -> TickResult:
        if not self.is_connected:
            raise RuntimeError("Not connected to aai host. Call connect() first.")
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
        # aai's tool_result auto-fires the next reply; no follow-up message.
        for (
            call_id,
            result_str,
            _request_response,
            _is_error,
        ) in self._pending_tool_results:
            await self.provider.send_tool_result(call_id, result_str)
        self._pending_tool_results.clear()

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        pcm16_16k = self._convert_ulaw_to_pcm16_16k(user_audio)

        async def receive_events():
            elapsed = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, (self.tick_duration_ms / 1000) - elapsed)
            return await self.provider.receive_events_for_duration(remaining)

        _, events = await asyncio.gather(
            self._send_audio_chunked(
                pcm16_16k, self.provider.send_audio, self._chunk_size
            ),
            receive_events(),
        )
        for event in events:
            self._process_event(result, event)

    @staticmethod
    def _convert_ulaw_to_pcm16_16k(ulaw_8k: bytes) -> bytes:
        """Convert mu-law 8k user audio to PCM16 16k for the aai host."""
        audio = AudioData(data=ulaw_8k, format=TELEPHONY_AUDIO_FORMAT)
        pcm16 = convert_to_pcm16(audio)
        resampled = resample_audio(pcm16, AAI_SEND_SAMPLE_RATE)
        return resampled.data

    @staticmethod
    def _convert_pcm16_24k_to_ulaw_8k(pcm16_24k: bytes) -> bytes:
        """Convert PCM16 24k agent audio (from the aai host) to mu-law 8k."""
        audio = AudioData(data=pcm16_24k, format=AAI_RECEIVE_AUDIO_FORMAT)
        resampled = resample_audio(audio, 8000)
        ulaw = convert_to_ulaw(resampled)
        return ulaw.data

    def _process_event(self, result: TickResult, event: Any) -> None:
        result.events.append(event)

        if isinstance(event, AAIAgentTranscriptEvent):
            item_id = self._ensure_current_item_id()
            ut = self._utterance_transcripts.setdefault(
                item_id, UtteranceTranscript(item_id=item_id)
            )
            # aai sends the full transcript each time (not deltas); overwrite.
            ut.transcript_received = event.text

        elif isinstance(event, AAIAudioChunkEvent):
            item_id = self._ensure_current_item_id()
            ulaw_bytes = self._convert_pcm16_24k_to_ulaw_8k(event.pcm16)
            if result.skip_item_id is not None and item_id == result.skip_item_id:
                result.truncated_audio_bytes += len(ulaw_bytes)
                return
            if ulaw_bytes:
                result.agent_audio_chunks.append((ulaw_bytes, item_id))
            ut = self._utterance_transcripts.setdefault(
                item_id, UtteranceTranscript(item_id=item_id)
            )
            ut.add_audio(len(ulaw_bytes))

        elif isinstance(event, AAIToolCallEvent):
            result.tool_calls.append(
                ToolCall(
                    id=event.tool_call_id,
                    name=event.tool_name,
                    arguments=event.args,
                )
            )
            logger.debug(f"Tool call detected: {event.tool_name}({event.tool_call_id})")

        elif isinstance(event, AAISpeechStartedEvent):
            logger.debug("Speech started - interruption detected")
            result.vad_events.append("speech_started")
            if self._buffered_agent_audio:
                buffered = sum(len(c[0]) for c in self._buffered_agent_audio)
                result.truncated_audio_bytes += buffered
                self._buffered_agent_audio.clear()
            result.was_truncated = True
            result.skip_item_id = self._current_item_id
            if self._current_item_id is not None:
                self._interrupted_turn_ids.add(self._current_item_id)

        elif isinstance(event, AAISpeechStoppedEvent):
            result.vad_events.append("speech_stopped")

        elif isinstance(event, AAIReplyDoneEvent):
            item_id = self._current_item_id
            was_interrupted = item_id in self._interrupted_turn_ids
            ut = self._utterance_transcripts.get(item_id) if item_id else None
            if ut is not None and not was_interrupted:
                if ut.audio_bytes_received == 0:
                    logger.warning(f"Reply {item_id} completed with no audio")
                if ut.transcript_received == "":
                    logger.warning(
                        f"Reply {item_id} completed with no transcript — "
                        "possible event schema mismatch"
                    )
            self._interrupted_turn_ids.discard(item_id)
            self._turn_index += 1
            self._current_item_id = None

        elif isinstance(event, AAIAudioDoneEvent):
            logger.debug("Audio playback done")

        elif isinstance(event, AAIErrorEvent):
            logger.error(f"aai host error: {event.code} {event.message}")

        elif isinstance(event, AAITimeoutEvent):
            pass

        else:
            logger.debug(f"Event {type(event).__name__} received")

    def _ensure_current_item_id(self) -> str:
        """Return the synthetic turn id for the in-progress turn, creating it
        if this is the first audio/transcript event of the turn."""
        if self._current_item_id is None:
            self._current_item_id = f"turn-{self._turn_index}"
        return self._current_item_id
