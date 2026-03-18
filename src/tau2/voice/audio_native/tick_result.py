"""Provider-agnostic tick result classes for discrete-time audio native interaction.

This module provides the core data structures for tick-based full-duplex simulation:
- UtteranceTranscript: Tracks audio/text for proportional distribution
- TickResult: Contains all events and audio for a single tick

These classes are provider-agnostic and used by DiscreteTimeAudioNativeAgent
regardless of the underlying provider (OpenAI, Gemini, etc.).
"""

from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from tau2.config import DEFAULT_TELEPHONY_RATE, TELEPHONY_ULAW_SILENCE
from tau2.data_model.message import ToolCall
from tau2.voice.utils.transcript_utils import get_proportional_text


class UtteranceTranscript(BaseModel):
    """Tracks transcript for one utterance, distributing text proportionally.

    As audio streams in, we track total audio bytes and transcript text.
    When playing back, we distribute transcript proportionally across audio.

    This allows showing text at roughly the same rate as speech, even when
    audio and text arrive at different rates from the API.
    """

    model_config = ConfigDict(frozen=False)

    item_id: str
    audio_bytes_received: int = 0
    transcript_received: str = ""
    audio_bytes_played: int = 0
    transcript_chars_shown: int = 0

    def add_audio(self, num_bytes: int) -> None:
        """Track audio bytes received for this utterance."""
        self.audio_bytes_received += num_bytes

    def add_transcript(self, text: str) -> None:
        """Track transcript text received for this utterance."""
        self.transcript_received += text

    def get_transcript_for_audio(self, audio_bytes_to_play: int) -> str:
        """Get proportional transcript text for this audio chunk."""
        self.audio_bytes_played += audio_bytes_to_play

        if self.audio_bytes_received == 0 or not self.transcript_received:
            return ""

        text, self.transcript_chars_shown = get_proportional_text(
            transcript=self.transcript_received,
            total_duration=self.audio_bytes_received,
            audio_played=self.audio_bytes_played,
            start_char=self.transcript_chars_shown,
        )
        return text


class TickResult(BaseModel):
    """Events and audio collected during a single tick.

    This class is provider-agnostic and tracks both RAW and PADDED audio:

    RAW AUDIO (actual audio received from API):
        - agent_audio_chunks: Raw chunks with item IDs
        - agent_audio_data: Property that joins raw chunks
        - agent_audio_bytes: Property for raw byte count

    PADDED AUDIO (for simulation playback):
        - get_played_agent_audio(): Returns exactly bytes_per_tick bytes,
          padded with silence if needed

    Use raw audio to detect if agent actually spoke (agent_audio_bytes > 0).
    Use padded audio for time-aligned simulation output.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Tick identification ---
    tick_number: int = Field(description="1-indexed tick number")

    # --- User audio sent this tick ---
    audio_sent_bytes: int = Field(description="Bytes of user audio sent this tick")
    audio_sent_duration_ms: float = Field(
        description="Duration of user audio sent (ms)"
    )
    user_audio_data: bytes = Field(
        default=b"",
        description="Raw user audio bytes sent this tick",
        exclude=True,  # Exclude from serialization due to large size
    )

    # --- Events received this tick ---
    # Provider-agnostic: stores raw events from any provider
    events: List[Any] = Field(
        default_factory=list,
        description="All API events received during this tick (provider-specific types)",
        exclude=True,  # Exclude from serialization (may contain bytes)
    )

    # --- VAD events (provider-agnostic) ---
    # Normalized VAD event types detected this tick:
    #   - "speech_started": User started speaking (OpenAI/xAI/Nova)
    #   - "speech_stopped": User stopped speaking (OpenAI/xAI/Nova)
    #   - "interrupted": User interrupted agent (Gemini)
    vad_events: List[str] = Field(
        default_factory=list,
        description="VAD event types this tick: 'speech_started', 'speech_stopped', 'interrupted'",
    )

    # --- Tool calls extracted this tick ---
    # Provider-agnostic: adapters populate this from their specific event types
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="Tool calls detected during this tick",
    )

    # --- Raw agent audio (unpadded) ---
    # These store the actual audio received from the API, without padding.
    # Use these to detect if agent actually spoke.
    agent_audio_chunks: List[Tuple[bytes, Optional[str]]] = Field(
        default_factory=list,
        description="Raw audio chunks from agent: (data, item_id)",
        exclude=True,
        # Exclude from serialization due to large size
    )

    # --- Transcript ---
    proportional_transcript: str = Field(
        default="",
        description="Transcript text proportional to audio played this tick",
    )

    # --- Interruption tracking ---
    was_truncated: bool = Field(
        default=False, description="True if agent was interrupted this tick"
    )
    truncated_audio_bytes: int = Field(
        default=0, description="Bytes of agent audio discarded due to interruption"
    )
    skip_item_id: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Item ID to skip audio from after truncation",
    )

    # --- Tick-based interruption info (for accurate timing) ---
    interruption_audio_start_ms: Optional[int] = Field(
        default=None,
        description="Position in cumulative user audio buffer where interruption was detected",
    )
    cumulative_user_audio_at_tick_start_ms: int = Field(
        default=0,
        description="Cumulative user audio sent to API before this tick started (ms)",
    )
    bytes_per_tick: int = Field(
        default=0,
        description="Expected audio bytes per tick. Required for get_played_agent_audio()",
    )
    bytes_per_second: int = Field(
        default=DEFAULT_TELEPHONY_RATE,
        description="Audio bytes per second (sample_rate * bytes_per_sample). "
        "Used for duration calculations. Default is 8000 (8kHz μ-law).",
    )

    # --- Timing info ---
    tick_sim_duration_ms: float = Field(
        default=0.0, description="Simulated audio duration for this tick (ms)"
    )

    # --- Silence byte for padding (provider/format-specific) ---
    silence_byte: bytes = Field(
        default=TELEPHONY_ULAW_SILENCE,
        description="Silence byte value for audio padding. Default is μ-law silence.",
        exclude=True,
    )

    # =========================================================================
    # RAW AUDIO PROPERTIES (unpadded - use for speech detection)
    # =========================================================================

    @property
    def has_provider_activity(self) -> bool:
        """True if any non-timeout events were received this tick.

        Use this to detect provider stalls - if has_provider_activity is False
        for multiple consecutive ticks, the provider may be unresponsive.

        This works across all providers by checking for events with type != "timeout".
        Timeout events are synthetic events indicating no API events arrived during
        the tick's collection window.
        """
        for event in self.events:
            # Check both 'type' and 'event_type' for cross-provider compatibility
            # (most providers use 'type', but Nova uses 'event_type')
            event_type = getattr(event, "type", None) or getattr(
                event, "event_type", None
            )
            if event_type and event_type != "timeout":
                return True
        return False

    @property
    def item_ids(self) -> List[str]:
        """Get unique item_ids that contributed audio to this tick, in order.

        Returns the item_ids (utterance IDs) in the sequence they first appeared,
        with duplicates removed. Multiple audio delta chunks from the same utterance
        will only appear once.
        """
        seen = set()
        result = []
        for _, item_id in self.agent_audio_chunks:
            if item_id and item_id not in seen:
                seen.add(item_id)
                result.append(item_id)
        return result

    @property
    def agent_audio_bytes(self) -> int:
        """Total bytes of raw (unpadded) agent audio received this tick."""
        return sum(len(data) for data, _ in self.agent_audio_chunks)

    @property
    def agent_audio_data(self) -> bytes:
        """Raw (unpadded) agent audio data. Use for speech detection."""
        return b"".join(data for data, _ in self.agent_audio_chunks)

    # =========================================================================
    # PADDED AUDIO METHOD (use for playback/stereo overlay)
    # =========================================================================

    def get_played_agent_audio(self) -> bytes:
        """Get agent audio for this tick, PADDED to exactly bytes_per_tick.

        Always returns exactly bytes_per_tick bytes with appropriate padding:
        - If agent was interrupted: audio up to interruption + silence padding at end
        - If agent started mid-tick: silence padding at beginning + audio
        - Normal case: audio (capped/padded to bytes_per_tick)

        Returns:
            Exactly bytes_per_tick bytes of audio.

        Raises:
            ValueError: If bytes_per_tick is not set (== 0).
        """
        if self.bytes_per_tick <= 0:
            raise ValueError(
                f"bytes_per_tick must be > 0, got {self.bytes_per_tick}. "
                "TickResult must be created with valid bytes_per_tick."
            )

        raw_audio = self.agent_audio_data

        if self.was_truncated and self.interruption_audio_start_ms is not None:
            # Interruption case: return audio up to interruption point, pad at END
            tick_start_ms = self.cumulative_user_audio_at_tick_start_ms
            position_within_tick_ms = self.interruption_audio_start_ms - tick_start_ms

            # Clamp to valid range [0, tick_duration_ms]
            tick_duration_ms = (self.bytes_per_tick / self.bytes_per_second) * 1000
            position_within_tick_ms = max(
                0, min(position_within_tick_ms, tick_duration_ms)
            )

            # Calculate bytes played before interruption
            max_bytes = int(position_within_tick_ms * self.bytes_per_second / 1000)

            # Get audio up to interruption
            result = []
            current_total = 0
            for data, item_id in self.agent_audio_chunks:
                if current_total >= max_bytes:
                    break
                remaining = max_bytes - current_total
                result.append(data[:remaining])
                current_total += len(data[:remaining])
            played_audio = b"".join(result)

            # Pad with silence at END (agent was speaking, then stopped)
            if len(played_audio) < self.bytes_per_tick:
                padding = self.silence_byte * (self.bytes_per_tick - len(played_audio))
                played_audio = played_audio + padding

            return played_audio

        # Normal case (no interruption)
        if len(raw_audio) == 0:
            # No audio this tick - full silence
            return self.silence_byte * self.bytes_per_tick

        if len(raw_audio) >= self.bytes_per_tick:
            # Enough audio - return exactly bytes_per_tick
            return raw_audio[: self.bytes_per_tick]

        # Less audio than tick - need to pad
        # TODO: If we had info about when agent started speaking within tick,
        # we could pad at beginning. For now, assume agent started at tick start.
        padding = self.silence_byte * (self.bytes_per_tick - len(raw_audio))
        return raw_audio + padding

    def truncate_agent_audio(
        self,
        item_id: Optional[str],
        audio_start_ms: int,
        cumulative_user_audio_at_tick_start_ms: int,
        bytes_per_tick: int,
    ) -> int:
        """Mark truncation point using tick-based timing.

        Args:
            item_id: The item ID to skip future audio from.
            audio_start_ms: Cumulative position in user audio buffer where
                speech was detected.
            cumulative_user_audio_at_tick_start_ms: Cumulative user audio sent
                before this tick started.
            bytes_per_tick: Bytes per tick for calculating played audio.

        Returns:
            Bytes that were received but won't be 'played'.
        """
        self.was_truncated = True
        self.skip_item_id = item_id
        self.interruption_audio_start_ms = audio_start_ms
        self.cumulative_user_audio_at_tick_start_ms = (
            cumulative_user_audio_at_tick_start_ms
        )
        self.bytes_per_tick = bytes_per_tick

        # Calculate how much was received vs how much would have been played
        received = self.agent_audio_bytes
        played_bytes = len(self.get_played_agent_audio())
        discarded = max(0, received - played_bytes)
        self.truncated_audio_bytes = discarded
        return discarded

    def summary(self) -> str:
        """Generate a summary of this tick (timing shown separately after pause)."""
        event_counts: dict[str, int] = {}
        for event in self.events:
            event_type = type(event).__name__
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        parts = [f"Tick {self.tick_number}"]
        parts.append(
            f"  Sent: {self.audio_sent_bytes} bytes ({self.audio_sent_duration_ms:.0f}ms)"
        )
        parts.append(f"  Events: {len(self.events)}")
        for event_type, count in sorted(event_counts.items()):
            parts.append(f"    - {event_type}: {count}")
        if self.agent_audio_bytes > 0:
            received_ms = (self.agent_audio_bytes / self.bytes_per_second) * 1000
            played_bytes = len(self.get_played_agent_audio())
            played_ms = (played_bytes / self.bytes_per_second) * 1000
            if self.was_truncated:
                parts.append(
                    f"  Agent audio: {played_bytes} bytes ({played_ms:.0f}ms) played, "
                    f"{self.agent_audio_bytes} bytes ({received_ms:.0f}ms) received"
                )
            else:
                parts.append(
                    f"  Agent audio: {self.agent_audio_bytes} bytes ({received_ms:.0f}ms)"
                )
        if self.was_truncated:
            truncated_ms = (self.truncated_audio_bytes / self.bytes_per_second) * 1000
            parts.append(
                f"  ⚡ INTERRUPTED: {self.truncated_audio_bytes} bytes ({truncated_ms:.0f}ms) discarded"
            )
        # Show proportional transcript text for this tick's audio
        if self.proportional_transcript:
            transcript = self.proportional_transcript
            # Truncate long transcripts for display
            if len(transcript) > 100:
                display_text = transcript[:100] + "..."
            else:
                display_text = transcript
            # Show as a nicely formatted quote
            parts.append(f'  📝 Transcript: "{display_text}"')
        # Show tool calls if any
        if self.tool_calls:
            parts.append(f"  🔧 Tool calls: {len(self.tool_calls)}")
            for tc in self.tool_calls:
                parts.append(f"    - {tc.name}({tc.id})")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Shared helper functions used by multiple discrete-time adapters
# ---------------------------------------------------------------------------


def buffer_excess_audio(
    result: TickResult,
    bytes_per_tick: int,
) -> List[Tuple[bytes, Optional[str]]]:
    """Cap *result.agent_audio_chunks* to *bytes_per_tick* bytes.

    Splits the chunk list into a "keep" portion (written back to
    ``result.agent_audio_chunks``) and an "excess" portion that is returned
    for the caller to buffer until the next tick.

    When ``result.was_truncated`` is True the excess is discarded instead
    (byte count added to ``result.truncated_audio_bytes``) and an empty list
    is returned.
    """
    if result.was_truncated:
        total_bytes = 0
        keep_chunks: List[Tuple[bytes, Optional[str]]] = []
        discarded_bytes = 0

        for chunk_data, item_id in result.agent_audio_chunks:
            if total_bytes + len(chunk_data) <= bytes_per_tick:
                keep_chunks.append((chunk_data, item_id))
                total_bytes += len(chunk_data)
            else:
                space_left = bytes_per_tick - total_bytes
                if space_left > 0:
                    keep_chunks.append((chunk_data[:space_left], item_id))
                    discarded_bytes += len(chunk_data) - space_left
                else:
                    discarded_bytes += len(chunk_data)
                total_bytes = bytes_per_tick

        result.agent_audio_chunks = keep_chunks
        result.truncated_audio_bytes += discarded_bytes
        return []

    # Normal (non-truncated) case: buffer excess for next tick.
    total_bytes = 0
    keep_chunks: List[Tuple[bytes, Optional[str]]] = []
    buffer_chunks: List[Tuple[bytes, Optional[str]]] = []

    for chunk_data, item_id in result.agent_audio_chunks:
        if total_bytes + len(chunk_data) <= bytes_per_tick:
            keep_chunks.append((chunk_data, item_id))
            total_bytes += len(chunk_data)
        else:
            space_left = bytes_per_tick - total_bytes
            if space_left > 0:
                keep_chunks.append((chunk_data[:space_left], item_id))
                buffer_chunks.append((chunk_data[space_left:], item_id))
            else:
                buffer_chunks.append((chunk_data, item_id))
            total_bytes = bytes_per_tick

    result.agent_audio_chunks = keep_chunks
    return buffer_chunks


def get_proportional_transcript(
    agent_audio_chunks: List[Tuple[bytes, Optional[str]]],
    utterance_transcripts: dict[str, UtteranceTranscript],
    item_id_map: Optional[dict[str, str]] = None,
) -> str:
    """Compute proportional transcript for the audio chunks played this tick.

    Args:
        agent_audio_chunks: The ``(data, item_id)`` pairs kept for this tick.
        utterance_transcripts: Mapping of item/content IDs to their
            ``UtteranceTranscript`` trackers.
        item_id_map: Optional mapping from the ID carried on audio chunks to
            the ID used as key in *utterance_transcripts*.  Nova needs this
            because it sends text and audio under different content IDs.
    """
    if not agent_audio_chunks:
        return ""

    audio_by_item: dict[str, int] = {}
    for chunk_data, item_id in agent_audio_chunks:
        if item_id:
            audio_by_item[item_id] = audio_by_item.get(item_id, 0) + len(chunk_data)

    transcript_parts: list[str] = []
    for item_id, audio_bytes in audio_by_item.items():
        lookup_id = item_id_map.get(item_id, item_id) if item_id_map else item_id
        if lookup_id in utterance_transcripts:
            ut = utterance_transcripts[lookup_id]
            text = ut.get_transcript_for_audio(audio_bytes)
            if text:
                transcript_parts.append(text)

    return " ".join(transcript_parts)
