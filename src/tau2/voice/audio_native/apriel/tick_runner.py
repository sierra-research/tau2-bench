"""Tick-based simulation utilities for discrete-time audio native interaction.

This module mirrors the behavior of the reference OpenAI tick runner, but targets
Apriel's OpenAI-wire-compatible realtime provider.

Behavioral goals (match reference):
- Proper interruption handling with server-side truncation
- Optional buffer_until_complete mode for utterance-level alignment
- Tool-call gating so tool results cannot race pre-tool audio
- Response completion surfaced only after local playback buffers drain
"""

import asyncio
import base64
import json
from typing import List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field

from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioFormat
from tau2.data_model.message import ToolCall
from tau2.voice.audio_native.apriel.provider import AprielRealtimeProvider
from tau2.voice.audio_native.openai.events import (
    AudioDeltaEvent,
    AudioDoneEvent,
    AudioTranscriptDeltaEvent,
    AudioTranscriptDoneEvent,
    BaseRealtimeEvent,
    FunctionCallArgumentsDoneEvent,
    ResponseCancelledEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
)
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript


class PendingUtterance(BaseModel):
    """Tracks an incomplete utterance waiting for done event.

    Used in buffer_until_complete mode to hold audio/transcript until
    we receive AudioDoneEvent for this item.
    """

    item_id: str
    audio_chunks: List[bytes] = Field(default_factory=list)
    transcript: str = ""
    is_audio_complete: bool = False
    is_transcript_complete: bool = False

    @property
    def is_complete(self) -> bool:
        """Utterance is complete when both audio and transcript are done."""
        return self.is_audio_complete and self.is_transcript_complete

    @property
    def total_audio_bytes(self) -> int:
        return sum(len(chunk) for chunk in self.audio_chunks)


class TickRunner:
    """Manages tick-by-tick simulation with buffered agent audio.

    This class encapsulates the core tick logic that can be used by any consumer
    regardless of how user audio is sourced (file, microphone, user simulator).

    Modes:
    - buffer_until_complete=False (default): Stream audio/text as received,
      use proportional distribution to estimate text per tick.
    - buffer_until_complete=True: Wait until an utterance is complete (AudioDoneEvent)
      before including its audio/text. This guarantees accurate timing and
      text distribution since we know the full utterance length.
    """

    def __init__(
        self,
        provider: AprielRealtimeProvider,
        tick_duration_ms: int,
        bytes_per_tick: int,
        send_audio_instant: bool,
        buffer_until_complete: bool,
        chunk_size: int = 160,
        voip_packet_interval_ms: int = 20,
        audio_format: Optional[AudioFormat] = None,
        fast_forward_mode: bool = False,
        manual_commit: bool = False,
    ):
        if tick_duration_ms <= 0:
            raise ValueError(f"tick_duration_ms must be > 0, got {tick_duration_ms}")
        if bytes_per_tick <= 0:
            raise ValueError(f"bytes_per_tick must be > 0, got {bytes_per_tick}")

        if audio_format is None:
            audio_format = TELEPHONY_AUDIO_FORMAT

        self.provider = provider
        self.tick_duration_ms = tick_duration_ms
        self.bytes_per_tick = bytes_per_tick
        self.send_audio_instant = send_audio_instant
        self.chunk_size = chunk_size
        self.voip_packet_interval_ms = voip_packet_interval_ms
        self.buffer_until_complete = buffer_until_complete
        self.audio_format = audio_format
        self.fast_forward_mode = fast_forward_mode
        self.manual_commit = manual_commit

        # Mutable state - initialized fresh
        self.buffered_agent_audio: List[Tuple[bytes, Optional[str]]] = []
        self.utterance_transcripts: dict[str, UtteranceTranscript] = {}
        self.pending_utterances: dict[str, PendingUtterance] = {}
        self.completed_utterances: List[PendingUtterance] = []

        # Function arguments may finish while pre-tool audio is still streaming.
        # Do not expose calls to the orchestrator until the provider completes
        # that response, otherwise a tool result can start a second response
        # while the first response is still active.
        self.pending_tool_calls: List[ToolCall] = []
        self.response_complete_pending: bool = False

        # Provider response lifecycle is independent of whether any audio delta
        # has reached the local playback buffer yet. Keep it explicit so a VAD
        # onset can cancel generation during that first-delta race window.
        self.response_active: bool = False

        # After requesting cancellation, discard queued deltas until the
        # cancelled response reaches a terminal event. Item IDs may not exist
        # yet when speech_started wins the race, so an item-only fence is not
        # sufficient here.
        self.discard_response_audio: bool = False

        self.skip_item_id: Optional[str] = None
        self.cumulative_user_audio_ms: int = 0

    async def run_tick(self, user_audio: bytes, tick_number: int) -> TickResult:
        bytes_to_send = len(user_audio)

        tick_start = asyncio.get_running_loop().time()
        tick_end = tick_start + (self.tick_duration_ms / 1000)

        result = TickResult(
            tick_number=tick_number,
            audio_sent_bytes=bytes_to_send,
            audio_sent_duration_ms=(bytes_to_send / self.audio_format.bytes_per_second)
            * 1000,
            user_audio_data=user_audio,
            cumulative_user_audio_at_tick_start_ms=self.cumulative_user_audio_ms,
            bytes_per_tick=self.bytes_per_tick,
            bytes_per_second=self.audio_format.bytes_per_second,
        )

        for chunk_data, item_id in self.buffered_agent_audio:
            result.agent_audio_chunks.append((chunk_data, item_id))
        self.buffered_agent_audio.clear()

        result.skip_item_id = self.skip_item_id

        async def send_audio() -> None:
            if self.send_audio_instant:
                await self.provider.send_audio(user_audio)
            else:
                offset = 0
                while offset < len(user_audio):
                    chunk = user_audio[offset : offset + self.chunk_size]
                    await self.provider.send_audio(chunk)
                    offset += len(chunk)
                    await asyncio.sleep(self.voip_packet_interval_ms / 1000)

            if self.manual_commit:
                if hasattr(self.provider, "commit_audio"):
                    await self.provider.commit_audio()  # type: ignore[attr-defined]

        async def receive_events() -> None:
            async for event in self.provider.receive_events():
                current_time = asyncio.get_running_loop().time()

                if current_time >= tick_end:
                    await self._process_event(result, event)
                    break

                await self._process_event(result, event)

                if self.fast_forward_mode:
                    total_audio_bytes = sum(
                        len(data) for data, _ in result.agent_audio_chunks
                    )
                    if total_audio_bytes >= self.bytes_per_tick:
                        break

        await asyncio.gather(send_audio(), receive_events())

        if not self.fast_forward_mode:
            elapsed = asyncio.get_running_loop().time() - tick_start
            remaining_time = (self.tick_duration_ms / 1000) - elapsed
            if remaining_time > 0:
                await asyncio.sleep(remaining_time)

        result.tick_sim_duration_ms = result.audio_sent_duration_ms

        self._buffer_excess_audio(result)
        self._mark_response_complete_if_playback_drained(result)
        result.proportional_transcript = self._get_proportional_transcript(result)
        self.skip_item_id = result.skip_item_id
        self.cumulative_user_audio_ms += int(result.audio_sent_duration_ms)

        return result

    async def _process_event(
        self, result: TickResult, event: BaseRealtimeEvent
    ) -> None:
        result.events.append(event)

        if isinstance(event, SpeechStartedEvent):
            logger.debug(f"Speech started detected at {event.audio_start_ms}ms")
            result.vad_events.append("speech_started")

            has_agent_audio = result.agent_audio_chunks or self.buffered_agent_audio
            if has_agent_audio:
                last_item_id = None
                if result.agent_audio_chunks:
                    last_item_id = result.agent_audio_chunks[-1][1]
                elif self.buffered_agent_audio:
                    last_item_id = self.buffered_agent_audio[-1][1]

                if self.buffered_agent_audio:
                    buffered_bytes = sum(len(c[0]) for c in self.buffered_agent_audio)
                    result.truncated_audio_bytes += buffered_bytes
                    self.buffered_agent_audio.clear()

                if self.buffer_until_complete and self.pending_utterances:
                    for pending in self.pending_utterances.values():
                        result.truncated_audio_bytes += pending.total_audio_bytes
                    self.pending_utterances.clear()

                audio_start_ms = (
                    event.audio_start_ms if event.audio_start_ms is not None else 0
                )
                result.truncate_agent_audio(
                    item_id=last_item_id,
                    audio_start_ms=audio_start_ms,
                    cumulative_user_audio_at_tick_start_ms=result.cumulative_user_audio_at_tick_start_ms,
                    bytes_per_tick=result.bytes_per_tick,
                )

                if last_item_id is not None:
                    audio_end_ms = self._calculate_item_audio_played_ms(
                        item_id=last_item_id,
                        result=result,
                        audio_start_ms=audio_start_ms,
                    )
                    await self.provider.truncate_item(
                        item_id=last_item_id,
                        content_index=0,
                        audio_end_ms=audio_end_ms,
                    )

            if self.response_active and hasattr(self.provider, "cancel_response"):
                try:
                    await self.provider.cancel_response()  # type: ignore[attr-defined]
                finally:
                    self.discard_response_audio = True

        elif isinstance(event, AudioDeltaEvent):
            item_id = getattr(event, "item_id", None)

            if self.discard_response_audio:
                result.truncated_audio_bytes += len(base64.b64decode(event.delta))
                return

            if result.skip_item_id is not None:
                if item_id == result.skip_item_id:
                    result.truncated_audio_bytes += len(base64.b64decode(event.delta))
                    return
                result.skip_item_id = None

            decoded = base64.b64decode(event.delta)

            if self.buffer_until_complete and item_id:
                if item_id not in self.pending_utterances:
                    self.pending_utterances[item_id] = PendingUtterance(item_id=item_id)
                self.pending_utterances[item_id].audio_chunks.append(decoded)
            else:
                result.agent_audio_chunks.append((decoded, item_id))

            if item_id:
                if item_id not in self.utterance_transcripts:
                    self.utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                self.utterance_transcripts[item_id].add_audio(len(decoded))

        elif isinstance(event, AudioTranscriptDeltaEvent):
            item_id = getattr(event, "item_id", None)
            if item_id:
                if item_id not in self.utterance_transcripts:
                    self.utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                self.utterance_transcripts[item_id].add_transcript(event.delta)

                if self.buffer_until_complete:
                    if item_id not in self.pending_utterances:
                        self.pending_utterances[item_id] = PendingUtterance(
                            item_id=item_id
                        )
                    self.pending_utterances[item_id].transcript += event.delta

        elif isinstance(event, AudioDoneEvent):
            logger.debug(f"Audio done event detected for item {event.item_id}")
            item_id = getattr(event, "item_id", None)
            if self.buffer_until_complete and item_id:
                if item_id in self.pending_utterances:
                    self.pending_utterances[item_id].is_audio_complete = True
                    self._check_and_release_utterance(item_id, result)

        elif isinstance(event, AudioTranscriptDoneEvent):
            logger.debug(
                f"Audio transcript done event detected for item {event.item_id}"
            )
            item_id = getattr(event, "item_id", None)
            if self.buffer_until_complete and item_id:
                if item_id in self.pending_utterances:
                    self.pending_utterances[item_id].is_transcript_complete = True
                    self._check_and_release_utterance(item_id, result)

        elif isinstance(event, SpeechStoppedEvent):
            logger.debug(f"Speech stopped detected at {event.audio_end_ms}ms")
            result.vad_events.append("speech_stopped")

        elif isinstance(event, ResponseCreatedEvent):
            self.response_active = True
            self.discard_response_audio = False

        elif isinstance(event, ResponseDoneEvent):
            logger.debug(f"Response done event detected with status {event.status}")
            self.response_active = False
            self.discard_response_audio = False
            response_has_tool_calls = bool(self.pending_tool_calls)
            result.tool_calls.extend(self.pending_tool_calls)
            self.pending_tool_calls.clear()
            self.response_complete_pending = (
                event.status in (None, "completed") and not response_has_tool_calls
            )

        elif isinstance(event, ResponseCancelledEvent):
            logger.debug("Response cancelled; discarding pending tool calls")
            self.response_active = False
            self.discard_response_audio = False
            self.pending_tool_calls.clear()
            self.response_complete_pending = False

        elif isinstance(event, FunctionCallArgumentsDoneEvent):
            if event.call_id and event.name:
                try:
                    arguments = json.loads(event.arguments) if event.arguments else {}
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool arguments: {event.arguments}")
                    arguments = {}

                tool_call = ToolCall(
                    id=event.call_id,
                    name=event.name,
                    arguments=arguments,
                )
                self.pending_tool_calls.append(tool_call)
                logger.debug(
                    f"Tool call pending response completion: "
                    f"{event.name}({event.call_id})"
                )

        else:
            logger.debug(f"Event {event.type} received")

    def _check_and_release_utterance(self, item_id: str, result: TickResult) -> None:
        if item_id not in self.pending_utterances:
            return

        pending = self.pending_utterances[item_id]
        if pending.is_complete:
            for chunk_data in pending.audio_chunks:
                result.agent_audio_chunks.append((chunk_data, item_id))

            del self.pending_utterances[item_id]
            self.completed_utterances.append(pending)

    def _mark_response_complete_if_playback_drained(self, result: TickResult) -> None:
        """Clear pending completion once local playback drains.

        Note: this fork's TickResult does not define an `agent_turn_complete` field,
        so we cannot surface that signal directly on the result object.
        """
        if (
            self.response_complete_pending
            and not self.buffered_agent_audio
            and not self.pending_utterances
        ):
            self.response_complete_pending = False

    def _get_proportional_transcript(self, result: TickResult) -> str:
        if not result.agent_audio_chunks:
            return ""

        audio_by_item: dict[str, int] = {}
        for chunk_data, item_id in result.agent_audio_chunks:
            if item_id:
                audio_by_item[item_id] = audio_by_item.get(item_id, 0) + len(chunk_data)

        transcript_parts = []
        for item_id, audio_bytes in audio_by_item.items():
            if item_id in self.utterance_transcripts:
                ut = self.utterance_transcripts[item_id]
                text = ut.get_transcript_for_audio(audio_bytes)
                if text:
                    transcript_parts.append(text)

        return "".join(transcript_parts)

    def _calculate_item_audio_played_ms(
        self, *, item_id: str, result: TickResult, audio_start_ms: int
    ) -> int:
        previous_ticks_bytes = 0
        if item_id in self.utterance_transcripts:
            previous_ticks_bytes = self.utterance_transcripts[
                item_id
            ].audio_bytes_played

        position_within_tick_ms = (
            audio_start_ms - result.cumulative_user_audio_at_tick_start_ms
        )
        tick_duration_ms = (
            self.bytes_per_tick / self.audio_format.bytes_per_second
        ) * 1000

        position_within_tick_ms = max(0, min(position_within_tick_ms, tick_duration_ms))

        if tick_duration_ms > 0:
            tick_fraction = position_within_tick_ms / tick_duration_ms
        else:
            tick_fraction = 0.0

        item_bytes_in_current_tick = sum(
            len(chunk_data)
            for chunk_data, chunk_item_id in result.agent_audio_chunks
            if chunk_item_id == item_id
        )

        current_tick_played_bytes = int(item_bytes_in_current_tick * tick_fraction)
        total_played_bytes = previous_ticks_bytes + current_tick_played_bytes

        return int((total_played_bytes / self.audio_format.bytes_per_second) * 1000)

    def _buffer_excess_audio(self, result: TickResult) -> None:
        if result.was_truncated:
            if result.interruption_audio_start_ms is not None:
                tick_start_ms = result.cumulative_user_audio_at_tick_start_ms
                position_within_tick_ms = (
                    result.interruption_audio_start_ms - tick_start_ms
                )
                tick_duration_ms = (
                    result.bytes_per_tick / result.bytes_per_second
                ) * 1000
                position_within_tick_ms = max(
                    0, min(position_within_tick_ms, tick_duration_ms)
                )
                max_bytes = int(
                    position_within_tick_ms * result.bytes_per_second / 1000
                )
            else:
                max_bytes = result.bytes_per_tick

            total_bytes = 0
            keep_chunks: List[Tuple[bytes, Optional[str]]] = []
            discarded_bytes = 0

            for chunk_data, item_id in result.agent_audio_chunks:
                if total_bytes + len(chunk_data) <= max_bytes:
                    keep_chunks.append((chunk_data, item_id))
                    total_bytes += len(chunk_data)
                else:
                    space_left = max_bytes - total_bytes
                    if space_left > 0:
                        keep_chunks.append((chunk_data[:space_left], item_id))
                        discarded_bytes += len(chunk_data) - space_left
                    else:
                        discarded_bytes += len(chunk_data)
                    total_bytes = max_bytes

            result.agent_audio_chunks = keep_chunks
            result.truncated_audio_bytes += discarded_bytes
            self.buffered_agent_audio = []
            return

        total_bytes = 0
        keep_chunks_normal: List[Tuple[bytes, Optional[str]]] = []
        buffer_chunks: List[Tuple[bytes, Optional[str]]] = []

        for chunk_data, item_id in result.agent_audio_chunks:
            if total_bytes + len(chunk_data) <= self.bytes_per_tick:
                keep_chunks_normal.append((chunk_data, item_id))
                total_bytes += len(chunk_data)
            else:
                space_left = self.bytes_per_tick - total_bytes
                if space_left > 0:
                    keep_chunks_normal.append((chunk_data[:space_left], item_id))
                    buffer_chunks.append((chunk_data[space_left:], item_id))
                else:
                    buffer_chunks.append((chunk_data, item_id))
                total_bytes = self.bytes_per_tick

        result.agent_audio_chunks = keep_chunks_normal
        self.buffered_agent_audio = buffer_chunks
