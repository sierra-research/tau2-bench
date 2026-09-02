"""
Voice interaction metrics for full-duplex (voice) simulations.

Computes the τ-voice paper's interaction-quality panel from tick-level
simulation trajectories:

- Latency (lower is better):
    - L_R ``response_latency_mean``: seconds from end of a user turn to the
      start of the agent's response.
    - L_Y ``yield_latency_mean``: seconds from the start of a user
      interruption to the agent stopping.
- Responsiveness (higher is better):
    - R_R ``response_rate``: fraction of user turns that received an agent
      response before the user had to speak again.
    - R_Y ``yield_rate``: fraction of user interruptions where the agent
      yielded within the no-yield window.
- Interrupt (lower is better):
    - I_A ``agent_interruption_rate``: agent-interrupts-user events per user
      turn (``agent_interrupts_count / response_total``).
- Selectivity (higher is better; reported as correct rates = 1 - error rate):
    - S_BC ``backchannel``: agent correctly continued through user
      backchannels ("mm-hmm").
    - S_VT ``vocal_tic``: agent correctly ignored vocal tics (coughs, "um").
    - S_ND ``non_directed``: agent correctly ignored speech not directed at
      it ("hold on", side conversation).

This module is a faithful port of the analysis pipeline used for the τ-voice
paper (``src/experiments/tau_voice/exp/voice_analysis.py`` on the
``victorb/tau-voice-paper`` branch, commit ``121fca0``), restricted to the
computation core (no plotting / CSV / LaTeX output). Event definitions,
detection windows, and aggregation semantics are kept identical so that
numbers match the paper pipeline; see ``tests/test_voice_interaction_metrics.py``
for the parity test. Full definitions live in ``docs/interaction-metrics.md``.
"""

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from loguru import logger

from tau2.data_model.message import Tick

if TYPE_CHECKING:
    import pandas as pd

    from tau2.data_model.simulation import Results

INTERACTION_METRICS_VERSION = "1.0"


@dataclass
class InteractionMetricsConfig:
    """Detection windows and tick duration for interaction-metric extraction.

    Defaults are identical to the τ-voice paper pipeline.
    """

    tick_duration_sec: float = 0.2
    no_yield_window_sec: float = 2.0
    backchannel_yield_window_sec: float = 1.0
    vocal_tic_yield_window_sec: float = 1.0
    non_directed_yield_window_sec: float = 1.0
    vocal_tic_response_window_sec: float = 2.0
    non_directed_response_window_sec: float = 2.0

    def as_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Data Models for Audio Effects in Segments
# =============================================================================


@dataclass
class AudioEffectSegment:
    """
    Represents a contiguous audio effect segment during speech.

    Audio effects are applied to simulate realistic acoustic conditions:
    - Source effects: burst noise (e.g., cough, door slam)
    - Speech effects: vocal tics ("um", "uh"), non-directed speech ("one sec"),
                      dynamic muffling (hand over mic)

    Effects that span multiple consecutive ticks are merged into a single segment.
    """

    # Effect type classification
    effect_type: str  # "burst_noise", "vocal_tic", "non_directed_speech", "muffling"

    # Timing (tick-based)
    start_tick: int  # First tick where effect occurred
    end_tick: int  # Exclusive (first tick without effect)

    # Timing (seconds)
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0
    duration_sec: float = 0.0

    # Effect details
    text: Optional[str] = None  # Text content for vocal tics/non-directed speech
    file_path: Optional[str] = None  # Path to burst noise file (if applicable)

    # Whether effect was muffled (non-directed speech is typically muffled)
    is_muffled: bool = False

    @property
    def tick_count(self) -> int:
        """Number of ticks this effect spans."""
        return self.end_tick - self.start_tick


# =============================================================================
# Data Models for Speech Segments
# =============================================================================


@dataclass
class SpeechSegment:
    """
    A contiguous speech segment extracted from ticks.

    This is the base class containing timing and content information
    common to both user and agent segments.
    """

    # Identification
    role: str  # "user" or "agent"

    # Tick-based timing
    start_tick: int
    end_tick: int  # exclusive (first non-speaking tick)

    # Precise timing from simulation (milliseconds)
    start_time_ms: Optional[float] = None  # From cumulative_user_audio_at_tick_start_ms
    end_time_ms: Optional[float] = None
    duration_ms: Optional[float] = None

    # Computed timing (seconds) - fallback if precise timing not available
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0
    duration_sec: float = 0.0

    # Content
    transcript: str = ""  # Concatenated proportional transcripts
    utterance_ids: List[str] = field(default_factory=list)

    # Context
    other_speaking_at_start: bool = (
        False  # Was the other party speaking when this started?
    )
    other_speaking_at_end: bool = False  # Was the other party speaking when this ended?

    @property
    def tick_count(self) -> int:
        """Number of ticks in this segment."""
        return self.end_tick - self.start_tick


@dataclass
class UserSpeechSegment(SpeechSegment):
    """
    User speech segment with turn-taking information.

    Extends SpeechSegment with user-specific metadata about
    the turn-taking action and interruption/backchannel classification.
    """

    # Turn-taking action (from first tick of segment)
    action: str = ""  # keep_talking, stop_talking, backchannel, wait, generate_message
    action_info: str = ""  # Human-readable description

    # Classification
    is_interruption: bool = False  # User started speaking while agent was speaking
    is_backchannel: bool = False  # This was classified as a backchannel

    # Performance timing (when available)
    interrupt_check_seconds: Optional[float] = None
    backchannel_check_seconds: Optional[float] = None
    llm_generation_seconds: Optional[float] = None
    tts_synthesis_seconds: Optional[float] = None

    # Audio effects applied during this segment
    audio_effects: List[AudioEffectSegment] = field(default_factory=list)

    # Aggregate effect flags (for quick filtering/analysis)
    has_burst_noise: bool = False
    has_vocal_tic: bool = False
    has_non_directed_speech: bool = False
    has_muffling: bool = False


@dataclass
class AgentSpeechSegment(SpeechSegment):
    """
    Agent speech segment with interruption and VAD information.

    Extends SpeechSegment with agent-specific metadata about
    being interrupted and VAD events during the segment.
    """

    # Interruption info
    was_interrupted: bool = False  # Was agent interrupted during this segment?
    truncated_audio_bytes: int = 0  # Audio bytes truncated due to interruption
    interruption_audio_start_ms: Optional[float] = (
        None  # When interruption was detected
    )

    # VAD events that occurred during this segment
    vad_events: List[str] = field(
        default_factory=list
    )  # speech_started, speech_stopped, interrupted

    # Audio effects applied during this segment
    audio_effects: List[AudioEffectSegment] = field(default_factory=list)

    # Aggregate effect flags (for quick filtering/analysis)
    has_burst_noise: bool = False
    has_vocal_tic: bool = False
    has_non_directed_speech: bool = False
    has_muffling: bool = False


# =============================================================================
# Audio Effects Extraction Helpers
# =============================================================================


@dataclass
class _ActiveEffect:
    """Temporary state for tracking an active effect during segment extraction."""

    effect_type: str
    text: Optional[str] = None
    file_path: Optional[str] = None
    is_muffled: bool = False

    def matches(self, other: "_ActiveEffect") -> bool:
        """Check if this effect matches another (same type and details)."""
        return (
            self.effect_type == other.effect_type
            and self.text == other.text
            and self.file_path == other.file_path
        )


def _get_active_effects_from_chunk(chunk) -> List[_ActiveEffect]:
    """
    Get the list of currently active effects from a message chunk.

    This returns what effects are "on" in this tick, without timing info.
    Used to track effect segments across consecutive ticks.

    Args:
        chunk: A user_chunk or agent_chunk from a Tick

    Returns:
        List of _ActiveEffect representing currently active effects
    """
    active: List[_ActiveEffect] = []

    if chunk is None:
        return active

    # Extract source effects (burst noise, speech inserts from source)
    source_effects = getattr(chunk, "source_effects", None)
    if source_effects is not None:
        # Burst noise
        burst_file = getattr(source_effects, "burst_noise_file", None)
        if burst_file:
            active.append(
                _ActiveEffect(
                    effect_type="burst_noise",
                    file_path=burst_file,
                )
            )

        # Speech insert from source effects (less common)
        source_insert = getattr(source_effects, "speech_insert", None)
        if source_insert is not None:
            insert_type = getattr(source_insert, "type", "vocal_tic")
            insert_text = getattr(source_insert, "text", "")
            is_muffled = getattr(source_insert, "is_muffled", False)
            effect_type = (
                "vocal_tic" if insert_type == "vocal_tic" else "non_directed_speech"
            )

            active.append(
                _ActiveEffect(
                    effect_type=effect_type,
                    text=insert_text,
                    is_muffled=is_muffled,
                )
            )

    # Extract speech effects (vocal tics, non-directed speech, muffling)
    speech_effects = getattr(chunk, "speech_effects", None)
    if speech_effects is not None:
        # Dynamic muffling
        if getattr(speech_effects, "dynamic_muffling_enabled", False):
            active.append(
                _ActiveEffect(
                    effect_type="muffling",
                    is_muffled=True,
                )
            )

        # Speech insert (vocal tic or non-directed phrase)
        speech_insert = getattr(speech_effects, "speech_insert", None)
        if speech_insert is not None:
            insert_type = getattr(speech_insert, "type", "vocal_tic")
            insert_text = getattr(speech_insert, "text", "")
            is_muffled = getattr(speech_insert, "is_muffled", False)
            effect_type = (
                "vocal_tic" if insert_type == "vocal_tic" else "non_directed_speech"
            )

            active.append(
                _ActiveEffect(
                    effect_type=effect_type,
                    text=insert_text,
                    is_muffled=is_muffled,
                )
            )

    return active


def _start_effect_segment(
    active: _ActiveEffect,
    tick_idx: int,
    tick_duration_sec: float,
) -> AudioEffectSegment:
    """Create a new AudioEffectSegment from an active effect."""
    return AudioEffectSegment(
        effect_type=active.effect_type,
        start_tick=tick_idx,
        end_tick=tick_idx + 1,
        start_time_sec=tick_idx * tick_duration_sec,
        end_time_sec=(tick_idx + 1) * tick_duration_sec,
        duration_sec=tick_duration_sec,
        text=active.text,
        file_path=active.file_path,
        is_muffled=active.is_muffled,
    )


def _extend_effect_segment(
    segment: AudioEffectSegment,
    tick_idx: int,
    tick_duration_sec: float,
) -> None:
    """Extend an existing AudioEffectSegment by one tick."""
    segment.end_tick = tick_idx + 1
    segment.end_time_sec = (tick_idx + 1) * tick_duration_sec
    segment.duration_sec = segment.end_time_sec - segment.start_time_sec


def _update_audio_effect_segments(
    current_segments: List[AudioEffectSegment],
    prev_active: List[_ActiveEffect],
    curr_active: List[_ActiveEffect],
    tick_idx: int,
    tick_duration_sec: float,
) -> Tuple[List[AudioEffectSegment], List[_ActiveEffect]]:
    """
    Update audio effect segments based on current vs previous active effects.

    - Effects that continue: extend existing segment
    - Effects that start: create new segment
    - Effects that end: segment is already closed (end_tick was set)

    Returns:
        Tuple of (updated_segments, new_prev_active)
    """
    new_segments = list(current_segments)

    # Track which current effects matched a previous one
    matched_curr = [False] * len(curr_active)

    for prev in prev_active:
        # Find matching current effect
        for i, curr in enumerate(curr_active):
            if not matched_curr[i] and prev.matches(curr):
                matched_curr[i] = True
                # Extend the most recent segment of this type
                for seg in reversed(new_segments):
                    if (
                        seg.effect_type == prev.effect_type
                        and seg.end_tick == tick_idx  # Still open
                        and seg.text == prev.text
                        and seg.file_path == prev.file_path
                    ):
                        _extend_effect_segment(seg, tick_idx, tick_duration_sec)
                        break
                break

    # Start new segments for unmatched current effects
    for i, curr in enumerate(curr_active):
        if not matched_curr[i]:
            new_segments.append(
                _start_effect_segment(curr, tick_idx, tick_duration_sec)
            )

    return new_segments, curr_active


def _compute_effect_flags(
    segments: List[AudioEffectSegment],
) -> Tuple[bool, bool, bool, bool]:
    """Compute aggregate effect flags from segments."""
    has_burst = any(s.effect_type == "burst_noise" for s in segments)
    has_vocal_tic = any(s.effect_type == "vocal_tic" for s in segments)
    has_non_directed = any(s.effect_type == "non_directed_speech" for s in segments)
    has_muffling = any(s.effect_type == "muffling" for s in segments)
    return has_burst, has_vocal_tic, has_non_directed, has_muffling


def extract_out_of_turn_effects(
    ticks: List[Tick],
    tick_duration_sec: float = 0.2,
) -> List[AudioEffectSegment]:
    """
    Extract audio effects that occur outside of speech segments (out-of-turn).

    These are effects like non-directed speech ("Hold on", "One sec") or burst
    noise that occur when contains_speech=False. Such effects represent audio
    events during conversation gaps.

    Args:
        ticks: List of Tick objects from a simulation
        tick_duration_sec: Duration of each tick in seconds

    Returns:
        List of AudioEffectSegment for effects occurring during non-speech ticks
    """
    effect_segments: List[AudioEffectSegment] = []
    prev_active: List[_ActiveEffect] = []

    for i, tick in enumerate(ticks):
        user_chunk = tick.user_chunk

        # Only look at ticks where user is NOT speaking
        is_speaking = user_chunk is not None and getattr(
            user_chunk, "contains_speech", False
        )

        if is_speaking:
            # Reset tracking when speech starts
            prev_active = []
            continue

        # Get active effects from this non-speech tick
        curr_active = _get_active_effects_from_chunk(user_chunk)

        if not curr_active:
            prev_active = []
            continue

        # Update effect segments (same logic as speech segments)
        effect_segments, prev_active = _update_audio_effect_segments(
            effect_segments, prev_active, curr_active, i, tick_duration_sec
        )

    return effect_segments


# =============================================================================
# Segment Extraction Functions
# =============================================================================


def extract_user_segments(
    ticks: List[Tick],
    tick_duration_sec: float = 0.2,
) -> List[UserSpeechSegment]:
    """
    Extract user speech segments from simulation ticks.

    Groups contiguous ticks where user.contains_speech=True into segments,
    enriched with turn-taking metadata.

    Args:
        ticks: List of Tick objects from a simulation
        tick_duration_sec: Duration of each tick in seconds (for fallback timing)

    Returns:
        List of UserSpeechSegment objects
    """
    segments: List[UserSpeechSegment] = []
    current_segment: Optional[UserSpeechSegment] = None
    prev_active_effects: List[_ActiveEffect] = []

    for i, tick in enumerate(ticks):
        user_chunk = tick.user_chunk
        has_speech = user_chunk is not None and getattr(
            user_chunk, "contains_speech", False
        )

        if has_speech:
            if current_segment is None:
                # Start new segment
                current_segment, prev_active_effects = _create_user_segment_start(
                    tick, i, ticks, tick_duration_sec
                )
            else:
                # Extend current segment
                prev_active_effects = _extend_user_segment(
                    current_segment, tick, i, tick_duration_sec, prev_active_effects
                )
        else:
            # No speech - finalize current segment if any
            if current_segment is not None:
                _finalize_segment(current_segment, i, tick, ticks, tick_duration_sec)
                segments.append(current_segment)
                current_segment = None
                prev_active_effects = []

    # Handle segment continuing to end
    if current_segment is not None:
        _finalize_segment(current_segment, len(ticks), None, ticks, tick_duration_sec)
        segments.append(current_segment)

    return segments


def _create_user_segment_start(
    tick: Tick,
    tick_idx: int,
    ticks: List[Tick],
    tick_duration_sec: float,
) -> Tuple[UserSpeechSegment, List[_ActiveEffect]]:
    """Create a new UserSpeechSegment at the start of speech."""
    user_chunk = tick.user_chunk

    # Get turn-taking action from first tick
    action = ""
    action_info = ""
    is_backchannel = False
    interrupt_check_sec = None
    backchannel_check_sec = None
    llm_gen_sec = None
    tts_sec = None

    if user_chunk and user_chunk.turn_taking_action:
        tta = user_chunk.turn_taking_action
        action = getattr(tta, "action", "") or ""
        action_info = getattr(tta, "info", "") or ""
        is_backchannel = action == "backchannel"
        interrupt_check_sec = getattr(tta, "interrupt_check_seconds", None)
        backchannel_check_sec = getattr(tta, "backchannel_check_seconds", None)
        llm_gen_sec = getattr(tta, "llm_generation_seconds", None)
        tts_sec = getattr(tta, "tts_synthesis_seconds", None)

    # Check if agent was speaking when user started (interruption detection)
    agent_speaking_at_start = False
    if tick.agent_chunk:
        agent_speaking_at_start = getattr(tick.agent_chunk, "contains_speech", False)

    is_interruption = agent_speaking_at_start and not is_backchannel

    # Get precise timing if available
    start_time_ms = None
    if tick.agent_chunk and tick.agent_chunk.raw_data:
        raw_data = tick.agent_chunk.raw_data
        if isinstance(raw_data, dict):
            start_time_ms = raw_data.get("cumulative_user_audio_at_tick_start_ms")

    # Get text content
    text = ""
    if user_chunk and user_chunk.content:
        text = user_chunk.content

    # Get utterance IDs
    utterance_ids = []
    if user_chunk and user_chunk.utterance_ids:
        utterance_ids = list(user_chunk.utterance_ids)

    # Extract audio effects - start new effect segments for all active effects
    curr_active = _get_active_effects_from_chunk(user_chunk)
    effect_segments: List[AudioEffectSegment] = []
    for active in curr_active:
        effect_segments.append(
            _start_effect_segment(active, tick_idx, tick_duration_sec)
        )

    segment = UserSpeechSegment(
        role="user",
        start_tick=tick_idx,
        end_tick=tick_idx + 1,  # Will be updated as segment extends
        start_time_ms=start_time_ms,
        start_time_sec=tick_idx * tick_duration_sec,
        transcript=text,
        utterance_ids=utterance_ids,
        other_speaking_at_start=agent_speaking_at_start,
        action=action,
        action_info=action_info,
        is_interruption=is_interruption,
        is_backchannel=is_backchannel,
        interrupt_check_seconds=interrupt_check_sec,
        backchannel_check_seconds=backchannel_check_sec,
        llm_generation_seconds=llm_gen_sec,
        tts_synthesis_seconds=tts_sec,
        audio_effects=effect_segments,
    )

    return segment, curr_active


def _extend_user_segment(
    segment: UserSpeechSegment,
    tick: Tick,
    tick_idx: int,
    tick_duration_sec: float,
    prev_active: List[_ActiveEffect],
) -> List[_ActiveEffect]:
    """Extend an existing UserSpeechSegment with data from a new tick."""
    segment.end_tick = tick_idx + 1

    # Append text content
    user_chunk = tick.user_chunk
    if user_chunk and user_chunk.content:
        segment.transcript += user_chunk.content

    # Collect unique utterance IDs
    if user_chunk and user_chunk.utterance_ids:
        for uid in user_chunk.utterance_ids:
            if uid not in segment.utterance_ids:
                segment.utterance_ids.append(uid)

    # Update audio effect segments
    curr_active = _get_active_effects_from_chunk(user_chunk)
    segment.audio_effects, new_prev = _update_audio_effect_segments(
        segment.audio_effects, prev_active, curr_active, tick_idx, tick_duration_sec
    )

    return new_prev


def extract_agent_segments(
    ticks: List[Tick],
    tick_duration_sec: float = 0.2,
) -> List[AgentSpeechSegment]:
    """
    Extract agent speech segments from simulation ticks.

    Groups contiguous ticks where agent.contains_speech=True into segments,
    enriched with interruption and VAD metadata.

    Args:
        ticks: List of Tick objects from a simulation
        tick_duration_sec: Duration of each tick in seconds (for fallback timing)

    Returns:
        List of AgentSpeechSegment objects
    """
    segments: List[AgentSpeechSegment] = []
    current_segment: Optional[AgentSpeechSegment] = None
    prev_active_effects: List[_ActiveEffect] = []

    for i, tick in enumerate(ticks):
        agent_chunk = tick.agent_chunk
        has_speech = agent_chunk is not None and getattr(
            agent_chunk, "contains_speech", False
        )

        if has_speech:
            if current_segment is None:
                # Start new segment
                current_segment, prev_active_effects = _create_agent_segment_start(
                    tick, i, ticks, tick_duration_sec
                )
            else:
                # Extend current segment
                prev_active_effects = _extend_agent_segment(
                    current_segment, tick, i, tick_duration_sec, prev_active_effects
                )
        else:
            # No speech - finalize current segment if any
            if current_segment is not None:
                _finalize_segment(current_segment, i, tick, ticks, tick_duration_sec)
                segments.append(current_segment)
                current_segment = None
                prev_active_effects = []

    # Handle segment continuing to end
    if current_segment is not None:
        _finalize_segment(current_segment, len(ticks), None, ticks, tick_duration_sec)
        segments.append(current_segment)

    return segments


def _create_agent_segment_start(
    tick: Tick,
    tick_idx: int,
    ticks: List[Tick],
    tick_duration_sec: float,
) -> Tuple[AgentSpeechSegment, List[_ActiveEffect]]:
    """Create a new AgentSpeechSegment at the start of speech."""
    agent_chunk = tick.agent_chunk

    # Get precise timing if available
    start_time_ms = None
    if agent_chunk and agent_chunk.raw_data:
        raw_data = agent_chunk.raw_data
        if isinstance(raw_data, dict):
            start_time_ms = raw_data.get("cumulative_user_audio_at_tick_start_ms")

    # Check if user was speaking when agent started
    user_speaking_at_start = False
    if tick.user_chunk:
        user_speaking_at_start = getattr(tick.user_chunk, "contains_speech", False)

    # Get text content (proportional transcript from raw_data or content)
    text = ""
    if agent_chunk:
        if agent_chunk.content:
            text = agent_chunk.content
        elif agent_chunk.raw_data and isinstance(agent_chunk.raw_data, dict):
            text = agent_chunk.raw_data.get("proportional_transcript", "")

    # Get utterance IDs
    utterance_ids = []
    if agent_chunk and agent_chunk.utterance_ids:
        utterance_ids = list(agent_chunk.utterance_ids)

    # Get VAD events from this tick
    vad_events = []
    if agent_chunk and agent_chunk.raw_data and isinstance(agent_chunk.raw_data, dict):
        vad_events = list(agent_chunk.raw_data.get("vad_events", []))

    # Extract audio effects - start new effect segments for all active effects
    curr_active = _get_active_effects_from_chunk(agent_chunk)
    effect_segments: List[AudioEffectSegment] = []
    for active in curr_active:
        effect_segments.append(
            _start_effect_segment(active, tick_idx, tick_duration_sec)
        )

    segment = AgentSpeechSegment(
        role="agent",
        start_tick=tick_idx,
        end_tick=tick_idx + 1,
        start_time_ms=start_time_ms,
        start_time_sec=tick_idx * tick_duration_sec,
        transcript=text,
        utterance_ids=utterance_ids,
        other_speaking_at_start=user_speaking_at_start,
        vad_events=vad_events,
        audio_effects=effect_segments,
    )

    return segment, curr_active


def _extend_agent_segment(
    segment: AgentSpeechSegment,
    tick: Tick,
    tick_idx: int,
    tick_duration_sec: float,
    prev_active: List[_ActiveEffect],
) -> List[_ActiveEffect]:
    """Extend an existing AgentSpeechSegment with data from a new tick."""
    segment.end_tick = tick_idx + 1

    agent_chunk = tick.agent_chunk

    # Append text content
    if agent_chunk:
        if agent_chunk.content:
            segment.transcript += agent_chunk.content
        elif agent_chunk.raw_data and isinstance(agent_chunk.raw_data, dict):
            segment.transcript += agent_chunk.raw_data.get(
                "proportional_transcript", ""
            )

    # Collect unique utterance IDs
    if agent_chunk and agent_chunk.utterance_ids:
        for uid in agent_chunk.utterance_ids:
            if uid not in segment.utterance_ids:
                segment.utterance_ids.append(uid)

    # Collect VAD events
    if agent_chunk and agent_chunk.raw_data and isinstance(agent_chunk.raw_data, dict):
        for vad_event in agent_chunk.raw_data.get("vad_events", []):
            segment.vad_events.append(vad_event)

    # Check for interruption
    if agent_chunk and agent_chunk.raw_data and isinstance(agent_chunk.raw_data, dict):
        if agent_chunk.raw_data.get("was_truncated", False):
            segment.was_interrupted = True
            segment.truncated_audio_bytes += agent_chunk.raw_data.get(
                "truncated_audio_bytes", 0
            )
            if segment.interruption_audio_start_ms is None:
                segment.interruption_audio_start_ms = agent_chunk.raw_data.get(
                    "interruption_audio_start_ms"
                )

    # Update audio effect segments
    curr_active = _get_active_effects_from_chunk(agent_chunk)
    segment.audio_effects, new_prev = _update_audio_effect_segments(
        segment.audio_effects, prev_active, curr_active, tick_idx, tick_duration_sec
    )

    return new_prev


def _finalize_segment(
    segment: SpeechSegment,
    end_tick_idx: int,
    end_tick: Optional[Tick],
    ticks: List[Tick],
    tick_duration_sec: float,
) -> None:
    """Finalize a segment by computing end timing and context."""
    segment.end_tick = end_tick_idx

    # Compute duration
    segment.duration_sec = (segment.end_tick - segment.start_tick) * tick_duration_sec
    segment.end_time_sec = segment.start_time_sec + segment.duration_sec

    # Compute precise timing if start_time_ms is available
    if segment.start_time_ms is not None:
        tick_duration_ms = tick_duration_sec * 1000
        segment.duration_ms = (segment.end_tick - segment.start_tick) * tick_duration_ms
        segment.end_time_ms = segment.start_time_ms + segment.duration_ms

    # Check if other party was speaking at segment end
    if end_tick is not None:
        if segment.role == "user" and end_tick.agent_chunk:
            segment.other_speaking_at_end = getattr(
                end_tick.agent_chunk, "contains_speech", False
            )
        elif segment.role == "agent" and end_tick.user_chunk:
            segment.other_speaking_at_end = getattr(
                end_tick.user_chunk, "contains_speech", False
            )

    # Compute aggregate effect flags from effect segments
    if hasattr(segment, "audio_effects"):
        has_burst, has_vocal_tic, has_non_directed, has_muffling = (
            _compute_effect_flags(segment.audio_effects)
        )
        segment.has_burst_noise = has_burst
        segment.has_vocal_tic = has_vocal_tic
        segment.has_non_directed_speech = has_non_directed
        segment.has_muffling = has_muffling


# =============================================================================
# High-Level Extraction Function
# =============================================================================


def filter_end_of_conversation_ticks(ticks: List[Tick]) -> List[Tick]:
    """
    Filter out end-of-conversation artifact ticks.

    Conversations end when the user outputs ###STOP###. This creates a 1-tick
    user segment at the very end that triggers false positive "No Yield" events.

    This function removes the last tick if:
    - It's the last tick in the simulation
    - The user has speech in that tick (contains_speech=True)
    - The previous tick did NOT have user speech (i.e., this is a 1-tick segment)

    Args:
        ticks: List of Tick objects from a simulation

    Returns:
        Filtered list of ticks with end-of-conversation artifact removed
    """
    if len(ticks) < 2:
        return ticks

    last_tick = ticks[-1]
    prev_tick = ticks[-2]

    # Check if last tick has user speech
    last_has_user_speech = last_tick.user_chunk is not None and getattr(
        last_tick.user_chunk, "contains_speech", False
    )

    # Check if previous tick did NOT have user speech (making this a 1-tick segment)
    prev_has_user_speech = prev_tick.user_chunk is not None and getattr(
        prev_tick.user_chunk, "contains_speech", False
    )

    if last_has_user_speech and not prev_has_user_speech:
        logger.debug(
            f"Filtering out last tick (end-of-conversation artifact): "
            f"1-tick user segment at tick {len(ticks) - 1}"
        )
        return ticks[:-1]

    return ticks


def extract_all_segments(
    ticks: List[Tick],
    tick_duration_sec: float = 0.2,
) -> Tuple[List[UserSpeechSegment], List[AgentSpeechSegment]]:
    """
    Extract all speech segments from simulation ticks.

    Automatically filters out end-of-conversation artifacts (1-tick user segments
    at the very end caused by ###STOP###).

    Args:
        ticks: List of Tick objects from a simulation
        tick_duration_sec: Duration of each tick in seconds

    Returns:
        Tuple of (user_segments, agent_segments)
    """
    # Filter out end-of-conversation artifact
    ticks = filter_end_of_conversation_ticks(ticks)

    user_segments = extract_user_segments(ticks, tick_duration_sec)
    agent_segments = extract_agent_segments(ticks, tick_duration_sec)

    logger.debug(
        f"Extracted {len(user_segments)} user segments and {len(agent_segments)} agent segments"
    )

    return user_segments, agent_segments


# =============================================================================
# Turn Transitions (response rate / response latency)
# =============================================================================


@dataclass
class TurnTransitionEvent:
    """
    Represents what happened after a user finished speaking.

    Outcomes:
    - "response": Agent responded to the user
    - "no_response": User spoke again before agent responded
    """

    # Experiment metadata
    llm: str = ""
    domain: str = ""
    speech_complexity: str = ""
    provider: str = ""

    # Simulation metadata
    simulation_id: str = ""
    task_id: str = ""

    # User segment info
    user_segment_idx: int = 0
    user_end_tick: int = 0
    user_end_time_sec: float = 0.0
    user_transcript: str = ""
    user_action: str = ""

    # Outcome: "response" or "no_response"
    outcome: str = ""

    # For "response" outcome: agent info
    agent_segment_idx: Optional[int] = None
    agent_start_tick: Optional[int] = None
    agent_start_time_sec: Optional[float] = None

    # For "no_response" outcome: next user info
    next_user_segment_idx: Optional[int] = None
    next_user_start_tick: Optional[int] = None
    next_user_start_time_sec: Optional[float] = None

    # Time gap (latency for response, silence for no_response)
    gap_ticks: int = 0
    gap_sec: float = 0.0


def extract_turn_transitions(
    user_segments: List[UserSpeechSegment],
    agent_segments: List[AgentSpeechSegment],
    simulation_id: str = "",
    task_id: str = "",
    llm: str = "",
    domain: str = "",
    speech_complexity: str = "",
    provider: str = "",
) -> List[TurnTransitionEvent]:
    """
    Extract turn transition events from speech segments.

    For each valid user speech segment, determines what happened next:
    - "response": Agent responded to the user (measure latency)
    - "no_response": User spoke again before agent responded (measure silence)

    Filters out:
    - Backchannels (agent should not respond to these)
    - User interruptions (different dynamic, user is cutting in)
    - Cases where agent was already speaking at end of user turn

    Args:
        user_segments: List of user speech segments
        agent_segments: List of agent speech segments
        simulation_id: Simulation identifier
        task_id: Task identifier
        llm: LLM model name
        domain: Domain name
        speech_complexity: Speech complexity level
        provider: Provider name

    Returns:
        List of TurnTransitionEvent objects
    """
    events = []

    # Get valid user segments (filter out backchannels, agent speaking at end)
    # Note: We include interruptions where the agent yielded (other_speaking_at_end=False)
    # because the user has taken the floor and expects a response
    valid_user_segments = []
    for user_idx, user_seg in enumerate(user_segments):
        if user_seg.is_backchannel:
            logger.debug(
                f"Skipping backchannel at ticks {user_seg.start_tick}-{user_seg.end_tick}"
            )
            continue

        # Skip interruptions only if agent is STILL speaking at the end
        # If agent yielded (stopped), the user has taken the floor and should get a response
        if user_seg.is_interruption and user_seg.other_speaking_at_end:
            logger.debug(
                f"Skipping interruption at ticks {user_seg.start_tick}-{user_seg.end_tick}: "
                "agent still speaking at end"
            )
            continue

        if user_seg.other_speaking_at_end:
            logger.debug(
                f"Skipping segment at ticks {user_seg.start_tick}-{user_seg.end_tick}: "
                "agent was speaking at end"
            )
            continue

        valid_user_segments.append((user_idx, user_seg))

    # Sort agent segments by start_tick
    sorted_agent_segments = sorted(agent_segments, key=lambda s: s.start_tick)

    # For each valid user segment, determine if it got a response or not
    for i, (user_idx, user_seg) in enumerate(valid_user_segments):
        # Find the next agent segment that starts after this user ends
        next_agent_start = None
        next_agent_idx = None
        next_agent_seg = None
        for agent_idx, agent_seg in enumerate(sorted_agent_segments):
            if agent_seg.start_tick >= user_seg.end_tick:
                next_agent_start = agent_seg.start_tick
                next_agent_idx = agent_idx
                next_agent_seg = agent_seg
                break

        # Find the next valid user segment that starts after this user ends
        next_user_start = None
        next_user_idx = None
        next_user_seg = None
        for j in range(i + 1, len(valid_user_segments)):
            next_idx, next_seg = valid_user_segments[j]
            if next_seg.start_tick >= user_seg.end_tick:
                next_user_start = next_seg.start_tick
                next_user_idx = next_idx
                next_user_seg = next_seg
                break

        # Base event data
        base_event = {
            "llm": llm,
            "domain": domain,
            "speech_complexity": speech_complexity,
            "provider": provider,
            "simulation_id": simulation_id,
            "task_id": task_id,
            "user_segment_idx": user_idx,
            "user_end_tick": user_seg.end_tick,
            "user_end_time_sec": user_seg.end_time_sec,
            "user_transcript": user_seg.transcript,
            "user_action": user_seg.action,
        }

        # Determine outcome: response vs no-response
        if next_agent_start is not None:
            if next_user_start is None or next_agent_start <= next_user_start:
                # Agent responded before user spoke again
                gap_ticks = next_agent_seg.start_tick - user_seg.end_tick
                gap_sec = next_agent_seg.start_time_sec - user_seg.end_time_sec

                events.append(
                    TurnTransitionEvent(
                        **base_event,
                        outcome="response",
                        agent_segment_idx=next_agent_idx,
                        agent_start_tick=next_agent_seg.start_tick,
                        agent_start_time_sec=next_agent_seg.start_time_sec,
                        gap_ticks=gap_ticks,
                        gap_sec=gap_sec,
                    )
                )
            else:
                # User spoke again before agent responded
                gap_ticks = next_user_seg.start_tick - user_seg.end_tick
                gap_sec = next_user_seg.start_time_sec - user_seg.end_time_sec

                events.append(
                    TurnTransitionEvent(
                        **base_event,
                        outcome="no_response",
                        next_user_segment_idx=next_user_idx,
                        next_user_start_tick=next_user_seg.start_tick,
                        next_user_start_time_sec=next_user_seg.start_time_sec,
                        gap_ticks=gap_ticks,
                        gap_sec=gap_sec,
                    )
                )
        elif next_user_start is not None:
            # No agent response at all, but user spoke again
            gap_ticks = next_user_seg.start_tick - user_seg.end_tick
            gap_sec = next_user_seg.start_time_sec - user_seg.end_time_sec

            events.append(
                TurnTransitionEvent(
                    **base_event,
                    outcome="no_response",
                    next_user_segment_idx=next_user_idx,
                    next_user_start_tick=next_user_seg.start_tick,
                    next_user_start_time_sec=next_user_seg.start_time_sec,
                    gap_ticks=gap_ticks,
                    gap_sec=gap_sec,
                )
            )
        # else: last user segment with no following agent or user speech - skip

    response_count = sum(1 for e in events if e.outcome == "response")
    no_response_count = sum(1 for e in events if e.outcome == "no_response")
    logger.debug(
        f"Extracted {len(events)} turn transition events "
        f"({response_count} response, {no_response_count} no_response)"
    )
    return events


# =============================================================================
# Interruption Events (yield rate / yield latency / selectivity / I_A)
# =============================================================================


@dataclass
class InterruptionEvent:
    """
    Represents an interruption event where one party started speaking
    while the other was already talking.
    """

    # Experiment metadata
    llm: str = ""
    domain: str = ""
    speech_complexity: str = ""
    provider: str = ""

    # Simulation metadata
    simulation_id: str = ""
    task_id: str = ""

    # Event type
    event_type: str = (
        ""  # "user_interrupts_agent", "agent_interrupts_user", "backchannel"
    )

    # Interrupting party segment info
    interrupter_segment_idx: int = 0
    interrupter_start_tick: int = 0
    interrupter_start_time_sec: float = 0.0
    interrupter_end_tick: int = 0
    interrupter_duration_sec: float = 0.0
    interrupter_transcript: str = ""

    # Interrupted party info
    interrupted_speaking_at_start: bool = True

    # Outcome: did the interrupted party yield (stop speaking)?
    interrupted_yielded: bool = False
    yield_tick: Optional[int] = None
    yield_time_sec: float = 0.0  # Time from interruption start to yield


def extract_interruption_events(
    user_segments: List[UserSpeechSegment],
    agent_segments: List[AgentSpeechSegment],
    ticks: List[Tick],
    tick_duration_sec: float = 0.2,
    no_yield_window_sec: float = 2.0,
    backchannel_yield_window_sec: float = 1.0,
    vocal_tic_yield_window_sec: float = 1.0,
    non_directed_yield_window_sec: float = 1.0,
    vocal_tic_response_window_sec: float = 2.0,
    non_directed_response_window_sec: float = 2.0,
    simulation_id: str = "",
    task_id: str = "",
    llm: str = "",
    domain: str = "",
    speech_complexity: str = "",
    provider: str = "",
    out_of_turn_effects: Optional[List[AudioEffectSegment]] = None,
) -> List[InterruptionEvent]:
    """
    Extract interruption events from speech segments.

    Identifies nine types of events:
    1. user_interrupts_agent: User starts speaking while agent is talking
    2. agent_interrupts_user: Agent starts speaking while user is talking
    3. backchannel: User gives a backchannel while agent is talking
    4. vocal_tic: User vocal tic occurs while agent is talking (agent should NOT yield)
    5. non_directed_speech: User talks to someone else while agent is talking (agent should NOT yield)
    6. agent_responds_to_vocal_tic: Agent starts speaking after vocal tic (agent should NOT respond) - ERROR
    7. agent_responds_to_non_directed: Agent starts speaking after non-directed speech (agent should NOT respond) - ERROR
    8. vocal_tic_silent_correct: Vocal tic when agent silent, agent correctly did NOT respond - CORRECT
    9. non_directed_silent_correct: Non-directed speech when agent silent, agent correctly did NOT respond - CORRECT

    For user_interrupts_agent/agent_interrupts_user: calculates whether the interrupted party yielded.
    For backchannel/vocal_tic/non_directed_speech: yielding is considered incorrect behavior.
    For agent_responds_to_*: the agent incorrectly started speaking in response to non-speech audio.
    For *_silent_correct: the agent correctly stayed silent when receiving non-speech audio.

    Args:
        user_segments: List of user speech segments
        agent_segments: List of agent speech segments
        ticks: List of Tick objects (needed to check speaking status at each tick)
        tick_duration_sec: Duration of each tick in seconds
        no_yield_window_sec: Time window for user interruption yield detection (default: 2.0)
        backchannel_yield_window_sec: Time window for backchannel yield detection (default: 1.0)
        vocal_tic_yield_window_sec: Time window for vocal tic yield detection (default: 1.0)
        non_directed_yield_window_sec: Time window for non-directed yield detection (default: 1.0)
        vocal_tic_response_window_sec: Time window for vocal tic response detection (default: 2.0)
        non_directed_response_window_sec: Time window for non-directed response detection (default: 2.0)
        simulation_id: Simulation identifier
        task_id: Task identifier
        llm: LLM model name
        domain: Domain name
        speech_complexity: Speech complexity level
        provider: Provider name
        out_of_turn_effects: Effects occurring during non-speech ticks

    Returns:
        List of InterruptionEvent objects
    """
    events = []

    # Calculate yield windows in ticks for each event type
    no_yield_window_ticks = int(no_yield_window_sec / tick_duration_sec)
    backchannel_yield_window_ticks = int(
        backchannel_yield_window_sec / tick_duration_sec
    )
    vocal_tic_yield_window_ticks = int(vocal_tic_yield_window_sec / tick_duration_sec)
    non_directed_yield_window_ticks = int(
        non_directed_yield_window_sec / tick_duration_sec
    )
    vocal_tic_response_window_ticks = int(
        vocal_tic_response_window_sec / tick_duration_sec
    )
    non_directed_response_window_ticks = int(
        non_directed_response_window_sec / tick_duration_sec
    )

    # Base metadata for all events
    base_metadata = {
        "llm": llm,
        "domain": domain,
        "speech_complexity": speech_complexity,
        "provider": provider,
        "simulation_id": simulation_id,
        "task_id": task_id,
    }

    # Process user segments that interrupt the agent (or are backchannels)
    for user_idx, user_seg in enumerate(user_segments):
        if not user_seg.other_speaking_at_start:
            # User didn't start while agent was speaking
            continue

        # Determine event type and corresponding yield window
        # Priority: backchannel > vocal_tic > non_directed_speech > user_interrupts_agent
        # Vocal tics and non-directed speech should NOT cause agent to yield
        if user_seg.is_backchannel:
            event_type = "backchannel"
            yield_window_ticks = backchannel_yield_window_ticks
        elif user_seg.has_vocal_tic:
            event_type = "vocal_tic"
            yield_window_ticks = vocal_tic_yield_window_ticks
        elif user_seg.has_non_directed_speech:
            event_type = "non_directed_speech"
            yield_window_ticks = non_directed_yield_window_ticks
        elif user_seg.is_interruption:
            event_type = "user_interrupts_agent"
            yield_window_ticks = no_yield_window_ticks
        else:
            # User started while agent was speaking but not marked as interruption
            # This could be a transition case - still count it
            event_type = "user_interrupts_agent"
            yield_window_ticks = no_yield_window_ticks

        # Find when agent stopped speaking after user started
        start_tick = user_seg.start_tick
        yield_tick = None
        agent_yielded = False

        # Look ahead up to yield_window_ticks to find when agent stopped
        for i in range(start_tick, min(start_tick + yield_window_ticks, len(ticks))):
            agent_speaking = (
                ticks[i].agent_chunk.contains_speech if ticks[i].agent_chunk else False
            )
            if not agent_speaking:
                agent_yielded = True
                yield_tick = i
                break

        yield_time_sec = (
            (yield_tick - start_tick) * tick_duration_sec if yield_tick else 0.0
        )

        events.append(
            InterruptionEvent(
                **base_metadata,
                event_type=event_type,
                interrupter_segment_idx=user_idx,
                interrupter_start_tick=user_seg.start_tick,
                interrupter_start_time_sec=user_seg.start_time_sec,
                interrupter_end_tick=user_seg.end_tick,
                interrupter_duration_sec=user_seg.duration_sec,
                interrupter_transcript=user_seg.transcript,
                interrupted_speaking_at_start=True,
                interrupted_yielded=agent_yielded,
                yield_tick=yield_tick,
                yield_time_sec=yield_time_sec,
            )
        )

    # Process agent segments that interrupt the user
    for agent_idx, agent_seg in enumerate(agent_segments):
        if not agent_seg.other_speaking_at_start:
            # Agent didn't start while user was speaking
            continue

        event_type = "agent_interrupts_user"

        # Find when user stopped speaking after agent started
        start_tick = agent_seg.start_tick
        yield_tick = None
        user_yielded = False

        # Agent interruption uses no_yield_window for consistency
        for i in range(start_tick, min(start_tick + no_yield_window_ticks, len(ticks))):
            user_speaking = (
                ticks[i].user_chunk.contains_speech if ticks[i].user_chunk else False
            )
            if not user_speaking:
                user_yielded = True
                yield_tick = i
                break

        yield_time_sec = (
            (yield_tick - start_tick) * tick_duration_sec if yield_tick else 0.0
        )

        events.append(
            InterruptionEvent(
                **base_metadata,
                event_type=event_type,
                interrupter_segment_idx=agent_idx,
                interrupter_start_tick=agent_seg.start_tick,
                interrupter_start_time_sec=agent_seg.start_time_sec,
                interrupter_end_tick=agent_seg.end_tick,
                interrupter_duration_sec=agent_seg.duration_sec,
                interrupter_transcript=agent_seg.transcript,
                interrupted_speaking_at_start=True,
                interrupted_yielded=user_yielded,
                yield_tick=yield_tick,
                yield_time_sec=yield_time_sec,
            )
        )

    # Process agent segments that incorrectly respond to vocal tics or non-directed speech
    # These are cases where agent was NOT speaking, vocal tic/non-directed occurred,
    # and agent started speaking in response (incorrect behavior)

    # Find user segments with vocal tics or non-directed speech where agent was NOT speaking
    for user_idx, user_seg in enumerate(user_segments):
        # Skip if agent was speaking at start (already handled above)
        if user_seg.other_speaking_at_start:
            continue

        # Check if this segment has vocal tic or non-directed speech
        if not (user_seg.has_vocal_tic or user_seg.has_non_directed_speech):
            continue

        # Select appropriate response window based on effect type
        if user_seg.has_vocal_tic:
            response_window_ticks = vocal_tic_response_window_ticks
        else:
            response_window_ticks = non_directed_response_window_ticks

        # Look for agent segment that starts within response_window after this segment
        agent_responded = False
        for agent_idx, agent_seg in enumerate(agent_segments):
            # Agent must start after user segment started and within window of user segment end
            if agent_seg.start_tick < user_seg.start_tick:
                continue
            if agent_seg.start_tick > user_seg.end_tick + response_window_ticks:
                continue

            # Determine event type based on the effect
            if user_seg.has_vocal_tic:
                event_type = "agent_responds_to_vocal_tic"
            else:
                event_type = "agent_responds_to_non_directed"

            events.append(
                InterruptionEvent(
                    **base_metadata,
                    event_type=event_type,
                    interrupter_segment_idx=agent_idx,
                    interrupter_start_tick=agent_seg.start_tick,
                    interrupter_start_time_sec=agent_seg.start_time_sec,
                    interrupter_end_tick=agent_seg.end_tick,
                    interrupter_duration_sec=agent_seg.duration_sec,
                    interrupter_transcript=agent_seg.transcript,
                    interrupted_speaking_at_start=False,  # Agent was not speaking
                    interrupted_yielded=True,  # Agent incorrectly responded
                    yield_tick=None,
                    yield_time_sec=0.0,
                )
            )
            agent_responded = True
            # Only count first agent response to avoid duplicates
            break

        # If agent did NOT respond, record a "correct" event (agent correctly stayed silent)
        if not agent_responded:
            if user_seg.has_vocal_tic:
                event_type = "vocal_tic_silent_correct"
            else:
                event_type = "non_directed_silent_correct"

            events.append(
                InterruptionEvent(
                    **base_metadata,
                    event_type=event_type,
                    interrupter_segment_idx=user_idx,
                    interrupter_start_tick=user_seg.start_tick,
                    interrupter_start_time_sec=user_seg.start_time_sec,
                    interrupter_end_tick=user_seg.end_tick,
                    interrupter_duration_sec=user_seg.duration_sec,
                    interrupter_transcript=user_seg.transcript,
                    interrupted_speaking_at_start=False,  # Agent was not speaking
                    interrupted_yielded=False,  # Agent correctly did NOT respond
                    yield_tick=None,
                    yield_time_sec=0.0,
                )
            )

    # Process out-of-turn effects (non-directed speech / vocal tics that occur during silence)
    # These are effects that happen when the user is NOT in a speech segment
    if out_of_turn_effects:
        for effect in out_of_turn_effects:
            if effect.effect_type not in ("non_directed_speech", "vocal_tic"):
                continue

            effect_tick = effect.start_tick

            # Select appropriate windows based on effect type
            if effect.effect_type == "vocal_tic":
                yield_window_sec = vocal_tic_yield_window_sec
                response_window_ticks = vocal_tic_response_window_ticks
            else:
                yield_window_sec = non_directed_yield_window_sec
                response_window_ticks = non_directed_response_window_ticks

            # Check if agent was speaking during this effect
            agent_speaking_during = False
            speaking_agent_seg = None
            for agent_seg in agent_segments:
                if agent_seg.start_tick <= effect_tick < agent_seg.end_tick:
                    agent_speaking_during = True
                    speaking_agent_seg = agent_seg
                    break

            if agent_speaking_during and speaking_agent_seg:
                # Agent was speaking - check if they yielded (stopped) within yield window
                agent_stopped_tick = speaking_agent_seg.end_tick
                time_after_effect = (
                    agent_stopped_tick - effect_tick
                ) * tick_duration_sec

                # If agent stopped within yield window of the effect, they incorrectly yielded
                if time_after_effect <= yield_window_sec:
                    if effect.effect_type == "vocal_tic":
                        event_type = "vocal_tic"
                    else:
                        event_type = "non_directed_speech"

                    events.append(
                        InterruptionEvent(
                            **base_metadata,
                            event_type=event_type,
                            interrupter_segment_idx=-1,  # Not a user segment
                            interrupter_start_tick=effect_tick,
                            interrupter_start_time_sec=effect.start_time_sec,
                            interrupter_end_tick=effect.end_tick,
                            interrupter_duration_sec=effect.duration_sec,
                            interrupter_transcript=effect.text or "",
                            interrupted_speaking_at_start=True,  # Agent was speaking
                            interrupted_yielded=True,  # Agent incorrectly stopped
                            yield_tick=agent_stopped_tick,
                            yield_time_sec=time_after_effect,
                        )
                    )
            else:
                # Agent was NOT speaking - check if they responded (started) within window
                agent_responded = False
                for agent_seg in agent_segments:
                    # Agent must start after effect and within response window
                    if agent_seg.start_tick < effect_tick:
                        continue
                    if agent_seg.start_tick > effect.end_tick + response_window_ticks:
                        continue

                    if effect.effect_type == "vocal_tic":
                        event_type = "agent_responds_to_vocal_tic"
                    else:
                        event_type = "agent_responds_to_non_directed"

                    events.append(
                        InterruptionEvent(
                            **base_metadata,
                            event_type=event_type,
                            interrupter_segment_idx=-1,  # Not a user segment
                            interrupter_start_tick=agent_seg.start_tick,
                            interrupter_start_time_sec=agent_seg.start_time_sec,
                            interrupter_end_tick=agent_seg.end_tick,
                            interrupter_duration_sec=agent_seg.duration_sec,
                            interrupter_transcript=agent_seg.transcript,
                            interrupted_speaking_at_start=False,
                            interrupted_yielded=True,  # Agent incorrectly responded
                            yield_tick=None,
                            yield_time_sec=0.0,
                        )
                    )
                    agent_responded = True
                    # Only count first response
                    break

                # If agent did NOT respond, record a "correct" event
                if not agent_responded:
                    if effect.effect_type == "vocal_tic":
                        event_type = "vocal_tic_silent_correct"
                    else:
                        event_type = "non_directed_silent_correct"

                    events.append(
                        InterruptionEvent(
                            **base_metadata,
                            event_type=event_type,
                            interrupter_segment_idx=-1,  # Not a user segment
                            interrupter_start_tick=effect_tick,
                            interrupter_start_time_sec=effect.start_time_sec,
                            interrupter_end_tick=effect.end_tick,
                            interrupter_duration_sec=effect.duration_sec,
                            interrupter_transcript=effect.text or "",
                            interrupted_speaking_at_start=False,
                            interrupted_yielded=False,  # Agent correctly did NOT respond
                            yield_tick=None,
                            yield_time_sec=0.0,
                        )
                    )

    # Sort by start time
    events.sort(key=lambda e: e.interrupter_start_tick)

    return events


# =============================================================================
# Unified Voice Quality Events
# =============================================================================


@dataclass
class VoiceQualityEvent:
    """
    Unified event for voice quality analysis.

    Event categories:
    1. Response events: Agent's response to user speech
       - "response": Agent responded (success)
       - "no_response": Agent failed to respond (error)

    2. Yield events: Agent yielding when user interrupts
       - "yield": Agent yielded to user interruption (success)
       - "no_yield": Agent failed to yield (error)

    3. Backchannel events: Agent handling of user backchannels
       - "backchannel_correct": Agent correctly continued speaking (success)
       - "backchannel_error": Agent incorrectly stopped (error)

    4. Vocal tic events: Agent handling of vocal tics ("um", "uh")
       - "vocal_tic_correct": Agent correctly ignored (success)
       - "vocal_tic_error": Agent incorrectly responded/yielded (error)

    5. Non-directed speech events: Agent handling of speech not directed at it
       - "non_directed_correct": Agent correctly ignored (success)
       - "non_directed_error": Agent incorrectly responded/yielded (error)
    """

    # Event identification
    event_category: (
        str  # "response", "yield", "backchannel", "vocal_tic", "non_directed"
    )
    event_type: str  # Specific type within category
    is_error: bool  # True if this is an error (agent behaved incorrectly)

    # Timing (for latency calculations)
    latency_sec: Optional[float] = None  # Response latency or yield latency

    # Experiment metadata
    llm: str = ""
    domain: str = ""
    speech_complexity: str = ""
    provider: str = ""

    # Simulation metadata
    simulation_id: str = ""
    task_id: str = ""

    # Event timing
    event_time_sec: float = 0.0
    event_tick: int = 0

    # Additional context
    transcript: str = ""


def extract_voice_quality_events_from_simulation(
    ticks: List[Tick],
    tick_duration_sec: float = 0.2,
    no_yield_window_sec: float = 2.0,
    backchannel_yield_window_sec: float = 1.0,
    vocal_tic_yield_window_sec: float = 1.0,
    non_directed_yield_window_sec: float = 1.0,
    vocal_tic_response_window_sec: float = 2.0,
    non_directed_response_window_sec: float = 2.0,
    simulation_id: str = "",
    task_id: str = "",
    llm: str = "",
    domain: str = "",
    speech_complexity: str = "",
    provider: str = "",
) -> List[VoiceQualityEvent]:
    """
    Extract all voice quality events from a single simulation.

    Processing steps:
    1. Filters end-of-conversation artifacts
    2. Extracts speech segments
    3. Extracts out-of-turn effects
    4. Converts all events to unified VoiceQualityEvent format

    Args:
        ticks: List of Tick objects from a simulation
        tick_duration_sec: Duration of each tick in seconds
        no_yield_window_sec: Time window for user interruption yield detection (default: 2.0)
        backchannel_yield_window_sec: Time window for backchannel yield detection (default: 1.0)
        vocal_tic_yield_window_sec: Time window for vocal tic yield detection (default: 1.0)
        non_directed_yield_window_sec: Time window for non-directed yield detection (default: 1.0)
        vocal_tic_response_window_sec: Time window for vocal tic response detection (default: 2.0)
        non_directed_response_window_sec: Time window for non-directed response detection (default: 2.0)
        simulation_id: Simulation identifier
        task_id: Task identifier
        llm: LLM model name
        domain: Domain name
        speech_complexity: Speech complexity level
        provider: Provider name

    Returns:
        List of VoiceQualityEvent objects
    """
    events: List[VoiceQualityEvent] = []

    # CRITICAL: Apply the same tick filtering as the timeline visualization
    filtered_ticks = filter_end_of_conversation_ticks(ticks)

    # Extract speech segments from filtered ticks
    user_segs, agent_segs = extract_all_segments(filtered_ticks, tick_duration_sec)

    # Extract out-of-turn effects (effects during gaps between speech)
    out_of_turn_effects = extract_out_of_turn_effects(filtered_ticks, tick_duration_sec)

    # Base metadata for all events
    base_metadata = {
        "llm": llm,
        "domain": domain,
        "speech_complexity": speech_complexity,
        "provider": provider,
        "simulation_id": simulation_id,
        "task_id": task_id,
    }

    # =========================================================================
    # 1. Response Events (from turn transitions)
    # =========================================================================
    turn_transitions = extract_turn_transitions(
        user_segs,
        agent_segs,
        simulation_id=simulation_id,
        task_id=task_id,
        llm=llm,
        domain=domain,
        speech_complexity=speech_complexity,
        provider=provider,
    )

    for tt in turn_transitions:
        if tt.outcome == "response":
            events.append(
                VoiceQualityEvent(
                    event_category="response",
                    event_type="response",
                    is_error=False,
                    latency_sec=tt.gap_sec,
                    event_time_sec=tt.user_end_time_sec,
                    event_tick=tt.user_end_tick,
                    transcript=tt.user_transcript,
                    **base_metadata,
                )
            )
        elif tt.outcome == "no_response":
            events.append(
                VoiceQualityEvent(
                    event_category="response",
                    event_type="no_response",
                    is_error=True,
                    latency_sec=None,
                    event_time_sec=tt.user_end_time_sec,
                    event_tick=tt.user_end_tick,
                    transcript=tt.user_transcript,
                    **base_metadata,
                )
            )

    # =========================================================================
    # 2. Interruption Events (yield, backchannel, vocal_tic, non_directed)
    # =========================================================================
    interruption_events = extract_interruption_events(
        user_segs,
        agent_segs,
        filtered_ticks,  # Use filtered ticks!
        tick_duration_sec=tick_duration_sec,
        no_yield_window_sec=no_yield_window_sec,
        backchannel_yield_window_sec=backchannel_yield_window_sec,
        vocal_tic_yield_window_sec=vocal_tic_yield_window_sec,
        non_directed_yield_window_sec=non_directed_yield_window_sec,
        vocal_tic_response_window_sec=vocal_tic_response_window_sec,
        non_directed_response_window_sec=non_directed_response_window_sec,
        simulation_id=simulation_id,
        task_id=task_id,
        llm=llm,
        domain=domain,
        speech_complexity=speech_complexity,
        provider=provider,
        out_of_turn_effects=out_of_turn_effects,  # Pass out-of-turn effects!
    )

    for ie in interruption_events:
        if ie.event_type == "user_interrupts_agent":
            # User interrupted agent - did agent yield?
            if ie.interrupted_yielded:
                events.append(
                    VoiceQualityEvent(
                        event_category="yield",
                        event_type="yield",
                        is_error=False,
                        latency_sec=ie.yield_time_sec,
                        event_time_sec=ie.interrupter_start_time_sec,
                        event_tick=ie.interrupter_start_tick,
                        transcript=ie.interrupter_transcript,
                        **base_metadata,
                    )
                )
            else:
                events.append(
                    VoiceQualityEvent(
                        event_category="yield",
                        event_type="no_yield",
                        is_error=True,
                        latency_sec=None,
                        event_time_sec=ie.interrupter_start_time_sec,
                        event_tick=ie.interrupter_start_tick,
                        transcript=ie.interrupter_transcript,
                        **base_metadata,
                    )
                )

        elif ie.event_type == "backchannel":
            # Backchannel - agent should NOT yield
            if ie.interrupted_yielded:
                events.append(
                    VoiceQualityEvent(
                        event_category="backchannel",
                        event_type="backchannel_error",
                        is_error=True,
                        latency_sec=None,
                        event_time_sec=ie.interrupter_start_time_sec,
                        event_tick=ie.interrupter_start_tick,
                        transcript=ie.interrupter_transcript,
                        **base_metadata,
                    )
                )
            else:
                events.append(
                    VoiceQualityEvent(
                        event_category="backchannel",
                        event_type="backchannel_correct",
                        is_error=False,
                        latency_sec=None,
                        event_time_sec=ie.interrupter_start_time_sec,
                        event_tick=ie.interrupter_start_tick,
                        transcript=ie.interrupter_transcript,
                        **base_metadata,
                    )
                )

        elif ie.event_type == "vocal_tic":
            # Vocal tic while agent speaking - agent should NOT yield
            if ie.interrupted_yielded:
                events.append(
                    VoiceQualityEvent(
                        event_category="vocal_tic",
                        event_type="vocal_tic_error",
                        is_error=True,
                        latency_sec=None,
                        event_time_sec=ie.interrupter_start_time_sec,
                        event_tick=ie.interrupter_start_tick,
                        transcript=ie.interrupter_transcript,
                        **base_metadata,
                    )
                )
            else:
                events.append(
                    VoiceQualityEvent(
                        event_category="vocal_tic",
                        event_type="vocal_tic_correct",
                        is_error=False,
                        latency_sec=None,
                        event_time_sec=ie.interrupter_start_time_sec,
                        event_tick=ie.interrupter_start_tick,
                        transcript=ie.interrupter_transcript,
                        **base_metadata,
                    )
                )

        elif ie.event_type == "non_directed_speech":
            # Non-directed speech while agent speaking - agent should NOT yield
            if ie.interrupted_yielded:
                events.append(
                    VoiceQualityEvent(
                        event_category="non_directed",
                        event_type="non_directed_error",
                        is_error=True,
                        latency_sec=None,
                        event_time_sec=ie.interrupter_start_time_sec,
                        event_tick=ie.interrupter_start_tick,
                        transcript=ie.interrupter_transcript,
                        **base_metadata,
                    )
                )
            else:
                events.append(
                    VoiceQualityEvent(
                        event_category="non_directed",
                        event_type="non_directed_correct",
                        is_error=False,
                        latency_sec=None,
                        event_time_sec=ie.interrupter_start_time_sec,
                        event_tick=ie.interrupter_start_tick,
                        transcript=ie.interrupter_transcript,
                        **base_metadata,
                    )
                )

        elif ie.event_type == "agent_responds_to_vocal_tic":
            # Agent incorrectly responded to vocal tic (agent was silent)
            events.append(
                VoiceQualityEvent(
                    event_category="vocal_tic",
                    event_type="vocal_tic_error",
                    is_error=True,
                    latency_sec=None,
                    event_time_sec=ie.interrupter_start_time_sec,
                    event_tick=ie.interrupter_start_tick,
                    transcript=ie.interrupter_transcript,
                    **base_metadata,
                )
            )

        elif ie.event_type == "vocal_tic_silent_correct":
            # Agent correctly ignored vocal tic (agent was silent)
            events.append(
                VoiceQualityEvent(
                    event_category="vocal_tic",
                    event_type="vocal_tic_correct",
                    is_error=False,
                    latency_sec=None,
                    event_time_sec=ie.interrupter_start_time_sec,
                    event_tick=ie.interrupter_start_tick,
                    transcript=ie.interrupter_transcript,
                    **base_metadata,
                )
            )

        elif ie.event_type == "agent_responds_to_non_directed":
            # Agent incorrectly responded to non-directed speech (agent was silent)
            events.append(
                VoiceQualityEvent(
                    event_category="non_directed",
                    event_type="non_directed_error",
                    is_error=True,
                    latency_sec=None,
                    event_time_sec=ie.interrupter_start_time_sec,
                    event_tick=ie.interrupter_start_tick,
                    transcript=ie.interrupter_transcript,
                    **base_metadata,
                )
            )

        elif ie.event_type == "non_directed_silent_correct":
            # Agent correctly ignored non-directed speech (agent was silent)
            events.append(
                VoiceQualityEvent(
                    event_category="non_directed",
                    event_type="non_directed_correct",
                    is_error=False,
                    latency_sec=None,
                    event_time_sec=ie.interrupter_start_time_sec,
                    event_tick=ie.interrupter_start_tick,
                    transcript=ie.interrupter_transcript,
                    **base_metadata,
                )
            )

        # Note: "agent_interrupts_user" events are tracked but not used in metrics
        # since we focus on agent errors, not user behavior

    # Sort by event time
    events.sort(key=lambda e: e.event_time_sec)

    return events


def voice_quality_events_to_dataframe(
    events: List[VoiceQualityEvent],
) -> "pd.DataFrame":
    """
    Convert voice quality events to a pandas DataFrame.

    Args:
        events: List of VoiceQualityEvent objects

    Returns:
        DataFrame with one row per event
    """
    import pandas as pd

    rows = []
    for e in events:
        rows.append(
            {
                "event_category": e.event_category,
                "event_type": e.event_type,
                "is_error": e.is_error,
                "latency_sec": e.latency_sec,
                "llm": e.llm,
                "domain": e.domain,
                "speech_complexity": e.speech_complexity,
                "provider": e.provider,
                "simulation_id": e.simulation_id,
                "task_id": e.task_id,
                "event_time_sec": e.event_time_sec,
                "event_tick": e.event_tick,
                "transcript": e.transcript,
            }
        )

    return pd.DataFrame(rows)


def compute_voice_quality_metrics(
    raw_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """
    Compute voice quality metrics from raw event data.

    Computes the following metrics (aggregated by llm, domain, speech_complexity, provider):
    1. response_rate: % of user turns that got an agent response
    2. response_latency_mean: Mean response latency (seconds)
    3. yield_rate: % of user interruptions where agent yielded
    4. yield_latency_mean: Mean yield latency (seconds)
    5. backchannel_error_rate: % of backchannels that incorrectly stopped agent
    6. vocal_tic_error_rate: % of vocal tics that incorrectly triggered agent
    7. non_directed_error_rate: % of non-directed speech that incorrectly triggered agent

    Args:
        raw_df: Raw event DataFrame from voice_quality_events_to_dataframe()

    Returns:
        Summary metrics DataFrame
    """
    import numpy as np
    import pandas as pd

    if raw_df.empty:
        return pd.DataFrame()

    # Group by experiment configuration
    grouped = raw_df.groupby(["llm", "domain", "speech_complexity", "provider"])

    summary_rows = []
    for (llm, domain, complexity, provider), group in grouped:
        # Response events
        response_events = group[group["event_category"] == "response"]
        n_response = (response_events["event_type"] == "response").sum()
        n_no_response = (response_events["event_type"] == "no_response").sum()
        n_response_total = n_response + n_no_response
        response_rate = (
            n_response / n_response_total if n_response_total > 0 else np.nan
        )

        response_latencies = response_events[
            response_events["event_type"] == "response"
        ]["latency_sec"].dropna()
        response_latency_mean = (
            response_latencies.mean() if len(response_latencies) > 0 else np.nan
        )
        response_latency_std = (
            response_latencies.std() if len(response_latencies) > 0 else np.nan
        )

        # Yield events
        yield_events = group[group["event_category"] == "yield"]
        n_yield = (yield_events["event_type"] == "yield").sum()
        n_no_yield = (yield_events["event_type"] == "no_yield").sum()
        n_yield_total = n_yield + n_no_yield
        yield_rate = n_yield / n_yield_total if n_yield_total > 0 else np.nan

        yield_latencies = yield_events[yield_events["event_type"] == "yield"][
            "latency_sec"
        ].dropna()
        yield_latency_mean = (
            yield_latencies.mean() if len(yield_latencies) > 0 else np.nan
        )
        yield_latency_std = (
            yield_latencies.std() if len(yield_latencies) > 0 else np.nan
        )

        # Backchannel events
        backchannel_events = group[group["event_category"] == "backchannel"]
        n_backchannel_correct = (
            backchannel_events["event_type"] == "backchannel_correct"
        ).sum()
        n_backchannel_error = (
            backchannel_events["event_type"] == "backchannel_error"
        ).sum()
        n_backchannel_total = n_backchannel_correct + n_backchannel_error
        backchannel_error_rate = (
            n_backchannel_error / n_backchannel_total
            if n_backchannel_total > 0
            else np.nan
        )

        # Vocal tic events
        vocal_tic_events = group[group["event_category"] == "vocal_tic"]
        n_vocal_tic_correct = (
            vocal_tic_events["event_type"] == "vocal_tic_correct"
        ).sum()
        n_vocal_tic_error = (vocal_tic_events["event_type"] == "vocal_tic_error").sum()
        n_vocal_tic_total = n_vocal_tic_correct + n_vocal_tic_error
        vocal_tic_error_rate = (
            n_vocal_tic_error / n_vocal_tic_total if n_vocal_tic_total > 0 else np.nan
        )

        # Non-directed speech events
        non_directed_events = group[group["event_category"] == "non_directed"]
        n_non_directed_correct = (
            non_directed_events["event_type"] == "non_directed_correct"
        ).sum()
        n_non_directed_error = (
            non_directed_events["event_type"] == "non_directed_error"
        ).sum()
        n_non_directed_total = n_non_directed_correct + n_non_directed_error
        non_directed_error_rate = (
            n_non_directed_error / n_non_directed_total
            if n_non_directed_total > 0
            else np.nan
        )

        summary_rows.append(
            {
                "llm": llm,
                "domain": domain,
                "speech_complexity": complexity,
                "provider": provider,
                # Response metrics
                "response_count": n_response,
                "no_response_count": n_no_response,
                "response_total": n_response_total,
                "response_rate": response_rate,
                "response_latency_mean": response_latency_mean,
                "response_latency_std": response_latency_std,
                # Yield metrics
                "yield_count": n_yield,
                "no_yield_count": n_no_yield,
                "yield_total": n_yield_total,
                "yield_rate": yield_rate,
                "yield_latency_mean": yield_latency_mean,
                "yield_latency_std": yield_latency_std,
                # Backchannel metrics
                "backchannel_correct_count": n_backchannel_correct,
                "backchannel_error_count": n_backchannel_error,
                "backchannel_total": n_backchannel_total,
                "backchannel_error_rate": backchannel_error_rate,
                # Vocal tic metrics
                "vocal_tic_correct_count": n_vocal_tic_correct,
                "vocal_tic_error_count": n_vocal_tic_error,
                "vocal_tic_total": n_vocal_tic_total,
                "vocal_tic_error_rate": vocal_tic_error_rate,
                # Non-directed metrics
                "non_directed_correct_count": n_non_directed_correct,
                "non_directed_error_count": n_non_directed_error,
                "non_directed_total": n_non_directed_total,
                "non_directed_error_rate": non_directed_error_rate,
            }
        )

    return pd.DataFrame(summary_rows)


# =============================================================================
# Leaderboard API
# =============================================================================


class NoVoiceTicksError(ValueError):
    """Raised when an input contains no full-duplex (tick-bearing) simulations."""


def _count_agent_interruptions(
    results: "Results", config: InteractionMetricsConfig
) -> int:
    """
    Count agent-interrupts-user events across all simulations.

    Mirrors the paper pipeline's interruption-handling analysis
    (``extract_interruptions_from_results``): segments are extracted with the
    end-of-conversation filter, but the raw (unfiltered) tick list is passed
    for yield-window lookups. Only the event count is consumed here.
    """
    count = 0
    for sim in results.simulations:
        if not sim.ticks:
            continue
        user_segs, agent_segs = extract_all_segments(
            sim.ticks, config.tick_duration_sec
        )
        events = extract_interruption_events(
            user_segs,
            agent_segs,
            sim.ticks,
            tick_duration_sec=config.tick_duration_sec,
            no_yield_window_sec=config.no_yield_window_sec,
            backchannel_yield_window_sec=config.backchannel_yield_window_sec,
            vocal_tic_yield_window_sec=config.vocal_tic_yield_window_sec,
            non_directed_yield_window_sec=config.non_directed_yield_window_sec,
            vocal_tic_response_window_sec=config.vocal_tic_response_window_sec,
            non_directed_response_window_sec=config.non_directed_response_window_sec,
        )
        count += sum(1 for e in events if e.event_type == "agent_interrupts_user")
    return count


def _nan_to_none(value) -> Optional[float]:
    """Convert NaN to None for JSON-friendly output."""
    import math

    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        return None
    return value


def compute_interaction_metrics_for_experiment(
    results: "Results",
    config: Optional[InteractionMetricsConfig] = None,
) -> dict:
    """
    Compute the interaction-metric panel for one experiment (one domain).

    Args:
        results: Loaded Results for a single voice experiment.
        config: Detection windows / tick duration. Defaults to paper values.

    Returns:
        A JSON-friendly dict with the 8 panel metrics and event counts:
        response/yield latencies and rates, agent_interruption_rate, and
        selectivity correct-rates for backchannel / vocal tic / non-directed
        speech, plus a ``counts`` sub-dict with every denominator.

    Raises:
        NoVoiceTicksError: If no simulation contains ticks (e.g. a text-mode
            experiment directory was passed in).
    """
    config = config or InteractionMetricsConfig()

    tick_sims = [sim for sim in results.simulations if sim.ticks]
    if not tick_sims:
        raise NoVoiceTicksError(
            "No tick-bearing simulations found — interaction metrics require "
            "full-duplex (voice) trajectories. Was this a text-mode experiment?"
        )

    all_events: List[VoiceQualityEvent] = []
    for sim in tick_sims:
        all_events.extend(
            extract_voice_quality_events_from_simulation(
                sim.ticks,
                tick_duration_sec=config.tick_duration_sec,
                no_yield_window_sec=config.no_yield_window_sec,
                backchannel_yield_window_sec=config.backchannel_yield_window_sec,
                vocal_tic_yield_window_sec=config.vocal_tic_yield_window_sec,
                non_directed_yield_window_sec=config.non_directed_yield_window_sec,
                vocal_tic_response_window_sec=config.vocal_tic_response_window_sec,
                non_directed_response_window_sec=config.non_directed_response_window_sec,
                simulation_id=sim.id,
                task_id=str(sim.task_id),
            )
        )

    raw_df = voice_quality_events_to_dataframe(all_events)
    analysis_df = compute_voice_quality_metrics(raw_df)
    if len(analysis_df) != 1:
        # Metadata fields are constant (all empty) so grouping yields one row.
        raise RuntimeError(
            f"Expected a single metrics row for one experiment, got {len(analysis_df)}"
        )
    row = analysis_df.iloc[0]

    agent_interrupts_count = _count_agent_interruptions(results, config)
    response_total = int(row["response_total"])
    agent_interruption_rate = (
        agent_interrupts_count / response_total if response_total > 0 else None
    )

    def correct_rate(error_rate) -> Optional[float]:
        error_rate = _nan_to_none(error_rate)
        return None if error_rate is None else 1.0 - error_rate

    return {
        "response_latency_mean": _nan_to_none(row["response_latency_mean"]),
        "yield_latency_mean": _nan_to_none(row["yield_latency_mean"]),
        "response_rate": _nan_to_none(row["response_rate"]),
        "yield_rate": _nan_to_none(row["yield_rate"]),
        "agent_interruption_rate": agent_interruption_rate,
        "selectivity_backchannel": correct_rate(row["backchannel_error_rate"]),
        "selectivity_vocal_tic": correct_rate(row["vocal_tic_error_rate"]),
        "selectivity_non_directed": correct_rate(row["non_directed_error_rate"]),
        "counts": {
            "n_simulations": len(tick_sims),
            "response_total": response_total,
            "yield_total": int(row["yield_total"]),
            "backchannel_total": int(row["backchannel_total"]),
            "vocal_tic_total": int(row["vocal_tic_total"]),
            "non_directed_total": int(row["non_directed_total"]),
            "agent_interrupts_count": agent_interrupts_count,
        },
    }


# Panel metric field names, in display order (L_R, L_Y, R_R, R_Y, I_A, S_BC, S_VT, S_ND).
PANEL_METRIC_FIELDS = [
    "response_latency_mean",
    "yield_latency_mean",
    "response_rate",
    "yield_rate",
    "agent_interruption_rate",
    "selectivity_backchannel",
    "selectivity_vocal_tic",
    "selectivity_non_directed",
]


def aggregate_domain_metrics(domain_metrics: dict) -> Optional[dict]:
    """
    Compute the all-domains average for a set of per-domain metric dicts.

    Mirrors the paper's "All" row: per-metric mean across domains, skipping
    missing (None) values — and the leaderboard's pass_1 averaging semantics.

    Args:
        domain_metrics: Mapping of domain name -> metric dict (as returned by
            :func:`compute_interaction_metrics_for_experiment`).

    Returns:
        A metric dict with per-metric cross-domain means and summed counts,
        or None if ``domain_metrics`` is empty.
    """
    if not domain_metrics:
        return None

    overall: dict = {}
    for metric_name in PANEL_METRIC_FIELDS:
        values = [
            m[metric_name]
            for m in domain_metrics.values()
            if m.get(metric_name) is not None
        ]
        overall[metric_name] = sum(values) / len(values) if values else None

    count_keys = set()
    for m in domain_metrics.values():
        count_keys.update(m.get("counts", {}).keys())
    overall["counts"] = {
        key: sum(m.get("counts", {}).get(key, 0) for m in domain_metrics.values())
        for key in sorted(count_keys)
    }
    return overall
