"""Tick-based simulation utilities for discrete-time audio native interaction.

This module provides the OpenAI-specific tick runner implementation:
- PendingUtterance: Buffers incomplete utterances in buffer_until_complete mode
- TickRunner: Manages tick-by-tick simulation with audio buffering

UtteranceTranscript and TickResult are now in the shared tick_result module.

These are used by DiscreteTimeAudioNativeAdapter and the demo scripts.
"""

import asyncio
import base64
import json
from typing import List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field

from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioFormat
from tau2.data_model.message import ToolCall
from tau2.voice.audio_native.openai.events import (
    AudioDeltaEvent,
    AudioDoneEvent,
    AudioTranscriptDeltaEvent,
    AudioTranscriptDoneEvent,
    BaseRealtimeEvent,
    FunctionCallArgumentsDoneEvent,
    ResponseDoneEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
)
from tau2.voice.audio_native.openai.provider import OpenAIRealtimeProvider
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
        provider: OpenAIRealtimeProvider,
        tick_duration_ms: int,
        bytes_per_tick: int,
        send_audio_instant: bool,
        buffer_until_complete: bool,
        chunk_size: int = 160,
        voip_packet_interval_ms: int = 20,
        audio_format: Optional[AudioFormat] = None,
        fast_forward_mode: bool = False,
    ):
        """Initialize the tick runner.

        Args:
            provider: The OpenAI Realtime API provider.
            tick_duration_ms: Duration of each tick in milliseconds. Must be > 0.
            bytes_per_tick: Audio bytes per tick. Must be > 0.
            send_audio_instant: If True, send audio in one call (discrete-time mode).
                If False, send in chunks with sleeps (VoIP-style streaming).
            buffer_until_complete: If True, wait for complete utterances before releasing.
            chunk_size: Only used when send_audio_instant=False.
                Chunk size in bytes for VoIP-style streaming (default: 160 = 20ms at 8kHz).
            voip_packet_interval_ms: Only used when send_audio_instant=False.
                Sleep duration between chunks in milliseconds (default: 20ms, standard RTP pacing).
            audio_format: Audio format for byte-to-duration calculations.
                Defaults to telephony (8kHz μ-law).
            fast_forward_mode: If True, exit tick early when we have enough audio
                buffered (>= bytes_per_tick), rather than waiting for wall-clock time.
                This speeds up simulation when the API responds quickly.

        Raises:
            ValueError: If tick_duration_ms or bytes_per_tick is <= 0.
        """
        if tick_duration_ms <= 0:
            raise ValueError(f"tick_duration_ms must be > 0, got {tick_duration_ms}")
        if bytes_per_tick <= 0:
            raise ValueError(f"bytes_per_tick must be > 0, got {bytes_per_tick}")

        # Default to telephony format if not specified
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

        # Mutable state - initialized fresh
        self.buffered_agent_audio: List[Tuple[bytes, Optional[str]]] = []
        self.utterance_transcripts: dict[str, UtteranceTranscript] = {}
        self.pending_utterances: dict[str, PendingUtterance] = {}
        self.completed_utterances: List[PendingUtterance] = []
        self.skip_item_id: Optional[str] = None
        self.cumulative_user_audio_ms: int = 0

    async def run_tick(
        self,
        user_audio: bytes,
        tick_number: int,
    ) -> TickResult:
        """Run a single tick: send user audio and collect events.

        Ensures at least tick_duration_ms of wall-clock time passes.
        Agent audio is capped to tick_duration_ms, excess buffered for next tick.

        Args:
            user_audio: User audio bytes (μ-law) to send this tick.
            tick_number: Which tick this is (1-indexed).

        Returns:
            TickResult with all events collected during this tick.
        """
        bytes_to_send = len(user_audio)

        # Calculate timing - tick must take at least tick_duration_ms (wall-clock)
        tick_start = asyncio.get_running_loop().time()
        tick_end = tick_start + (self.tick_duration_ms / 1000)

        result = TickResult(
            tick_number=tick_number,
            audio_sent_bytes=bytes_to_send,
            audio_sent_duration_ms=(bytes_to_send / self.audio_format.bytes_per_second)
            * 1000,
            user_audio_data=user_audio,
            # Tick-based info for accurate interruption handling
            cumulative_user_audio_at_tick_start_ms=self.cumulative_user_audio_ms,
            bytes_per_tick=self.bytes_per_tick,
            bytes_per_second=self.audio_format.bytes_per_second,
        )

        # Add any buffered agent audio from previous tick
        for chunk_data, item_id in self.buffered_agent_audio:
            result.agent_audio_chunks.append((chunk_data, item_id))
        self.buffered_agent_audio.clear()

        # Carry over skip state from previous tick
        result.skip_item_id = self.skip_item_id

        async def send_audio():
            """Send audio (instant or chunked based on config)."""
            if self.send_audio_instant:
                # Send all at once (fast, for discrete-time simulation)
                await self.provider.send_audio(user_audio)
            else:
                # Send in chunks with sleeps (real-time pacing)
                offset = 0
                while offset < len(user_audio):
                    chunk = user_audio[offset : offset + self.chunk_size]
                    await self.provider.send_audio(chunk)
                    offset += len(chunk)
                    await asyncio.sleep(self.voip_packet_interval_ms / 1000)

        async def receive_events():
            """Receive events until tick time is up or we have enough audio."""
            async for event in self.provider.receive_events():
                current_time = asyncio.get_running_loop().time()

                # Check if tick time is up (always applies)
                if current_time >= tick_end:
                    # Process this last event, then stop
                    await self._process_event(result, event)
                    break

                await self._process_event(result, event)

                # Fast-forward mode: exit early when we have enough audio
                if self.fast_forward_mode:
                    total_audio_bytes = sum(
                        len(data) for data, _ in result.agent_audio_chunks
                    )
                    if total_audio_bytes >= self.bytes_per_tick:
                        # We have a full tick's worth of audio - exit early
                        break

        # Run sending and receiving concurrently
        await asyncio.gather(send_audio(), receive_events())

        # Ensure tick took at least tick_duration_ms (skip in fast-forward mode)
        if not self.fast_forward_mode:
            elapsed = asyncio.get_running_loop().time() - tick_start
            remaining_time = (self.tick_duration_ms / 1000) - elapsed
            if remaining_time > 0:
                await asyncio.sleep(remaining_time)

        # Record simulation timing
        result.tick_sim_duration_ms = result.audio_sent_duration_ms

        # Move excess agent audio to buffer for next tick
        self._buffer_excess_audio(result)

        # Calculate proportional transcript for the audio played this tick
        result.proportional_transcript = self._get_proportional_transcript(result)

        # Update skip state for next tick
        self.skip_item_id = result.skip_item_id

        # Update cumulative user audio tracking (after tick completes)
        self.cumulative_user_audio_ms += int(result.audio_sent_duration_ms)

        return result

    async def _process_event(
        self, result: TickResult, event: BaseRealtimeEvent
    ) -> None:
        """Process an event, handling audio buffering and interruptions."""
        result.events.append(event)
        if isinstance(event, SpeechStartedEvent):
            logger.debug(f"Speech started detected at {event.audio_start_ms}ms")
            result.vad_events.append("speech_started")
            # User interrupted - mark for truncation using tick-based timing
            # Check all sources of agent audio: current chunks, buffered, and pending
            has_agent_audio = (
                result.agent_audio_chunks
                or self.buffered_agent_audio
                or self.pending_utterances
            )
            if has_agent_audio:
                # Determine the item_id being interrupted (priority: current > buffered > pending)
                last_item_id = None
                if result.agent_audio_chunks:
                    last_item_id = result.agent_audio_chunks[-1][1]
                elif self.buffered_agent_audio:
                    last_item_id = self.buffered_agent_audio[-1][1]
                elif self.pending_utterances:
                    # Get the most recent pending utterance
                    last_item_id = list(self.pending_utterances.keys())[-1]

                # Clear buffered audio from previous ticks (it's from interrupted utterance)
                if self.buffered_agent_audio:
                    buffered_bytes = sum(len(c[0]) for c in self.buffered_agent_audio)
                    result.truncated_audio_bytes += buffered_bytes
                    self.buffered_agent_audio.clear()

                # In buffer_until_complete mode, clear ALL pending utterances
                if self.buffer_until_complete and self.pending_utterances:
                    for pending in self.pending_utterances.values():
                        result.truncated_audio_bytes += pending.total_audio_bytes
                    self.pending_utterances.clear()

                # Use audio_start_ms from event for tick-based truncation
                audio_start_ms = (
                    event.audio_start_ms if event.audio_start_ms is not None else 0
                )
                result.truncate_agent_audio(
                    item_id=last_item_id,
                    audio_start_ms=audio_start_ms,
                    cumulative_user_audio_at_tick_start_ms=result.cumulative_user_audio_at_tick_start_ms,
                    bytes_per_tick=result.bytes_per_tick,
                )

                # Send truncation to server so it knows what the user actually heard
                # This is required for WebSocket connections per OpenAI Realtime API docs
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

        elif isinstance(event, AudioDeltaEvent):
            item_id = getattr(event, "item_id", None)

            # Skip audio from truncated item
            if result.skip_item_id is not None:
                if item_id == result.skip_item_id:
                    result.truncated_audio_bytes += len(base64.b64decode(event.delta))
                    return
                else:
                    result.skip_item_id = None

            decoded = base64.b64decode(event.delta)

            if self.buffer_until_complete and item_id:
                # Buffer mode: accumulate in pending until done event
                if item_id not in self.pending_utterances:
                    self.pending_utterances[item_id] = PendingUtterance(item_id=item_id)
                self.pending_utterances[item_id].audio_chunks.append(decoded)
            else:
                # Streaming mode: add directly to result
                result.agent_audio_chunks.append((decoded, item_id))

            # Track audio bytes per utterance for proportional transcript
            if item_id:
                if item_id not in self.utterance_transcripts:
                    self.utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                self.utterance_transcripts[item_id].add_audio(len(decoded))

        elif isinstance(event, AudioTranscriptDeltaEvent):
            # Track transcript per utterance for proportional distribution
            item_id = getattr(event, "item_id", None)
            if item_id:
                if item_id not in self.utterance_transcripts:
                    self.utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                self.utterance_transcripts[item_id].add_transcript(event.delta)

                # In buffer mode, also accumulate transcript in pending
                if self.buffer_until_complete:
                    if item_id not in self.pending_utterances:
                        self.pending_utterances[item_id] = PendingUtterance(
                            item_id=item_id
                        )
                    self.pending_utterances[item_id].transcript += event.delta

        elif isinstance(event, AudioDoneEvent):
            logger.debug(f"Audio done event detected for item {event.item_id}")
            # Audio for this item is complete
            item_id = getattr(event, "item_id", None)
            if self.buffer_until_complete and item_id:
                if item_id in self.pending_utterances:
                    self.pending_utterances[item_id].is_audio_complete = True
                    self._check_and_release_utterance(item_id, result)

        elif isinstance(event, AudioTranscriptDoneEvent):
            logger.debug(
                f"Audio transcript done event detected for item {event.item_id}"
            )
            # Transcript for this item is complete
            item_id = getattr(event, "item_id", None)
            if self.buffer_until_complete and item_id:
                if item_id in self.pending_utterances:
                    self.pending_utterances[item_id].is_transcript_complete = True
                    self._check_and_release_utterance(item_id, result)
        elif isinstance(event, SpeechStoppedEvent):
            logger.debug(f"Speech stopped detected at {event.audio_end_ms}ms")
            result.vad_events.append("speech_stopped")

        elif isinstance(event, ResponseDoneEvent):
            logger.debug(f"Response done event detected with status {event.status}")

        elif isinstance(event, FunctionCallArgumentsDoneEvent):
            # Extract tool call and add to result
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
                result.tool_calls.append(tool_call)
                logger.debug(f"Tool call detected: {event.name}({event.call_id})")

        else:
            logger.debug(f"Event {event.type} received")

    def _check_and_release_utterance(self, item_id: str, result: TickResult) -> None:
        """Check if utterance is complete and release to result if so.

        Used in buffer_until_complete mode to move completed utterances
        from pending to result.agent_audio_chunks.
        """
        if item_id not in self.pending_utterances:
            return

        pending = self.pending_utterances[item_id]
        if pending.is_complete:
            # Move to completed list and add audio to result
            for chunk_data in pending.audio_chunks:
                result.agent_audio_chunks.append((chunk_data, item_id))

            # Clean up
            del self.pending_utterances[item_id]
            self.completed_utterances.append(pending)

    def _get_proportional_transcript(self, result: TickResult) -> str:
        """Get proportional transcript for the audio played this tick.

        For each utterance that has audio in this tick, get the proportional
        amount of transcript text based on how much audio is being "played".
        """
        if not result.agent_audio_chunks:
            return ""

        # Group audio bytes by item_id for this tick
        audio_by_item: dict[str, int] = {}
        for chunk_data, item_id in result.agent_audio_chunks:
            if item_id:
                audio_by_item[item_id] = audio_by_item.get(item_id, 0) + len(chunk_data)

        # Get proportional transcript for each utterance
        transcript_parts = []
        for item_id, audio_bytes in audio_by_item.items():
            if item_id in self.utterance_transcripts:
                ut = self.utterance_transcripts[item_id]
                text = ut.get_transcript_for_audio(audio_bytes)
                if text:
                    transcript_parts.append(text)

        # Join with space between different items' transcripts
        return " ".join(transcript_parts)

    def _calculate_item_audio_played_ms(
        self,
        item_id: str,
        result: TickResult,
        audio_start_ms: int,
    ) -> int:
        """Calculate how much of an item's audio was played before interruption.

        This is used to inform the server how much audio the user actually heard
        when sending a conversation.item.truncate message.

        Args:
            item_id: The ID of the interrupted item.
            result: The current tick result.
            audio_start_ms: The audio_start_ms from SpeechStartedEvent (cumulative
                position in user audio buffer where speech was detected).

        Returns:
            Milliseconds of audio from this item that was played before interruption.
        """
        # Get audio already played in previous (completed) ticks for this item
        previous_ticks_bytes = 0
        if item_id in self.utterance_transcripts:
            previous_ticks_bytes = self.utterance_transcripts[
                item_id
            ].audio_bytes_played

        # Calculate how much of the current tick was "played" before interruption
        # based on timing within the tick
        position_within_tick_ms = (
            audio_start_ms - result.cumulative_user_audio_at_tick_start_ms
        )
        tick_duration_ms = (
            self.bytes_per_tick / self.audio_format.bytes_per_second
        ) * 1000

        # Clamp to valid range
        position_within_tick_ms = max(0, min(position_within_tick_ms, tick_duration_ms))

        # Calculate what fraction of the tick was played
        if tick_duration_ms > 0:
            tick_fraction = position_within_tick_ms / tick_duration_ms
        else:
            tick_fraction = 0.0

        # Count bytes from this item in the current tick's chunks
        item_bytes_in_current_tick = sum(
            len(chunk_data)
            for chunk_data, chunk_item_id in result.agent_audio_chunks
            if chunk_item_id == item_id
        )

        # Estimate how much of the current tick's item audio was played
        current_tick_played_bytes = int(item_bytes_in_current_tick * tick_fraction)

        # Total bytes played for this item
        total_played_bytes = previous_ticks_bytes + current_tick_played_bytes

        # Convert to milliseconds
        audio_end_ms = int(
            (total_played_bytes / self.audio_format.bytes_per_second) * 1000
        )

        return audio_end_ms

    def _buffer_excess_audio(self, result: TickResult) -> None:
        """Move agent audio exceeding tick cap to buffer for next tick.

        If the tick was interrupted (truncated), do NOT buffer any audio -
        it's from the interrupted utterance and should be discarded.
        Audio is trimmed to the interruption point, not bytes_per_tick.
        """
        # If interrupted, trim to interruption point and discard the rest
        if result.was_truncated:
            # Calculate max bytes based on interruption point (same logic as get_played_agent_audio)
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
                # Fallback to bytes_per_tick if no interruption timing info
                max_bytes = result.bytes_per_tick

            # Keep only audio up to interruption point
            total_bytes = 0
            keep_chunks: List[Tuple[bytes, Optional[str]]] = []
            discarded_bytes = 0

            for chunk in result.agent_audio_chunks:
                chunk_data, item_id = chunk
                if total_bytes + len(chunk_data) <= max_bytes:
                    keep_chunks.append(chunk)
                    total_bytes += len(chunk_data)
                else:
                    # Discard excess - it's from interrupted utterance
                    space_left = max_bytes - total_bytes
                    if space_left > 0:
                        keep_chunks.append((chunk_data[:space_left], item_id))
                        discarded_bytes += len(chunk_data) - space_left
                    else:
                        discarded_bytes += len(chunk_data)
                    total_bytes = max_bytes

            result.agent_audio_chunks = keep_chunks
            result.truncated_audio_bytes += discarded_bytes
            # Clear buffer - don't carry interrupted audio
            self.buffered_agent_audio = []
            return

        # Normal case: buffer excess for next tick
        total_bytes = 0
        keep_chunks_normal: List[Tuple[bytes, Optional[str]]] = []
        buffer_chunks: List[Tuple[bytes, Optional[str]]] = []

        for chunk in result.agent_audio_chunks:
            chunk_data, item_id = chunk
            if total_bytes + len(chunk_data) <= self.bytes_per_tick:
                keep_chunks_normal.append(chunk)
                total_bytes += len(chunk_data)
            else:
                # This chunk would exceed cap - split if needed
                space_left = self.bytes_per_tick - total_bytes
                if space_left > 0:
                    keep_chunks_normal.append((chunk_data[:space_left], item_id))
                    buffer_chunks.append((chunk_data[space_left:], item_id))
                else:
                    buffer_chunks.append(chunk)
                total_bytes = self.bytes_per_tick  # Capped

        result.agent_audio_chunks = keep_chunks_normal
        self.buffered_agent_audio = buffer_chunks
