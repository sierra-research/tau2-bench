import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.message import (
    Message,
    ParticipantMessageBase,
    ParticipantRole,
    Tick,
)
from tau2.data_model.simulation import SimulationRun
from tau2.orchestrator.modes import CommunicationMode
from tau2.voice.utils.audio_io import load_wav_file, save_wav_file
from tau2.voice.utils.audio_preprocessing import convert_to_stereo, merge_audio_datas

logger = logging.getLogger(__name__)


# =============================================================================
# Audacity Label Track Generation
# =============================================================================


@dataclass
class SpeechSegment:
    """A continuous speech segment with start/end ticks and combined text."""

    start_tick: int
    end_tick: int  # inclusive
    text: str
    role: str  # "user" or "assistant"


def _collect_speech_segments(
    ticks: list[Tick],
    role: str,
) -> list[SpeechSegment]:
    """
    Collect continuous speech segments for a given role.

    Groups consecutive ticks where contains_speech=True into segments,
    concatenating their text content.

    Args:
        ticks: List of simulation ticks.
        role: Either "user" or "assistant".

    Returns:
        List of SpeechSegment objects with combined text.
    """
    segments: list[SpeechSegment] = []
    current_segment: Optional[SpeechSegment] = None

    for tick in ticks:
        # Get the chunk for this role
        if role == "user":
            chunk = tick.user_chunk
        else:  # assistant
            chunk = tick.agent_chunk

        # Check if this tick has speech
        has_speech = chunk is not None and getattr(chunk, "contains_speech", False)

        if has_speech:
            # Get text content - prefer proportional_transcript from raw_data for agent
            text = ""
            if chunk.content:
                text = chunk.content
            elif role == "assistant" and chunk.raw_data:
                # Agent may have proportional_transcript in raw_data
                if isinstance(chunk.raw_data, dict):
                    text = chunk.raw_data.get("proportional_transcript", "")

            if current_segment is None:
                # Start new segment
                current_segment = SpeechSegment(
                    start_tick=tick.tick_id,
                    end_tick=tick.tick_id,
                    text=text,
                    role=role,
                )
            else:
                # Check if this is consecutive (within 1 tick gap tolerance)
                if tick.tick_id <= current_segment.end_tick + 1:
                    # Extend current segment
                    current_segment.end_tick = tick.tick_id
                    if text:
                        if current_segment.text:
                            current_segment.text += text
                        else:
                            current_segment.text = text
                else:
                    # Gap too large, finalize current and start new
                    segments.append(current_segment)
                    current_segment = SpeechSegment(
                        start_tick=tick.tick_id,
                        end_tick=tick.tick_id,
                        text=text,
                        role=role,
                    )
        else:
            # No speech - finalize current segment if any
            if current_segment is not None:
                segments.append(current_segment)
                current_segment = None

    # Don't forget the last segment
    if current_segment is not None:
        segments.append(current_segment)

    return segments


@dataclass
class ToolCallSegment:
    """A tool call event anchored to a tick range."""

    start_tick: int
    end_tick: int  # inclusive — spans all ticks where this tool call's results arrive
    tool_name: str
    role: str  # "user" or "assistant"


def _collect_tool_call_segments(
    ticks: list[Tick],
    role: str,
) -> list[ToolCallSegment]:
    """
    Collect tool call segments for a given role.

    Each tool call becomes a segment starting at the tick where the call was
    issued and ending at the tick where the corresponding result arrives
    (matched by tool call id). If no result is found, the segment is a
    single tick.

    Args:
        ticks: List of simulation ticks.
        role: Either "user" or "assistant".

    Returns:
        List of ToolCallSegment objects.
    """
    segments: list[ToolCallSegment] = []

    # Build a map from tool_call_id -> tick where the result appears
    result_tick_map: dict[str, int] = {}
    for tick in ticks:
        results = (
            tick.agent_tool_results if role == "assistant" else tick.user_tool_results
        )
        for result in results:
            if result.id:
                result_tick_map[result.id] = tick.tick_id

    for tick in ticks:
        tool_calls = (
            tick.agent_tool_calls if role == "assistant" else tick.user_tool_calls
        )
        for tc in tool_calls:
            end_tick = result_tick_map.get(tc.id, tick.tick_id)
            segments.append(
                ToolCallSegment(
                    start_tick=tick.tick_id,
                    end_tick=end_tick,
                    tool_name=tc.name,
                    role=role,
                )
            )

    return segments


def generate_audacity_labels(
    ticks: list[Tick],
    tick_duration_ms: float,
    output_dir: Path,
) -> dict[str, Path]:
    """
    Generate Audacity label track files for user and assistant speech.

    Creates .txt files in Audacity's label format that can be imported
    alongside the audio files to annotate speech segments with their text.

    Audacity label format (tab-separated):
        start_seconds<TAB>end_seconds<TAB>label_text

    Args:
        ticks: List of simulation ticks.
        tick_duration_ms: Duration of each tick in milliseconds.
        output_dir: Directory to save label files.

    Returns:
        Dictionary mapping role to label file path:
        {"user": Path, "assistant": Path}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_files: dict[str, Path] = {}

    for role in ["user", "assistant"]:
        # Speech labels
        segments = _collect_speech_segments(ticks, role)

        if not segments:
            logger.debug(f"No speech segments found for {role}")
        else:
            labels = []
            for seg in segments:
                start_sec = seg.start_tick * tick_duration_ms / 1000.0
                end_sec = (seg.end_tick + 1) * tick_duration_ms / 1000.0

                label_text = seg.text.replace("\n", " ").replace("\t", " ").strip()
                if len(label_text) > 200:
                    label_text = label_text[:197] + "..."

                labels.append(f"{start_sec:.3f}\t{end_sec:.3f}\t{label_text}")

            label_path = output_dir / f"{role}_labels.txt"
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(labels))

            label_files[role] = label_path
            logger.debug(f"Generated {len(segments)} labels for {role}: {label_path}")

        # Tool call labels
        tc_segments = _collect_tool_call_segments(ticks, role)

        if not tc_segments:
            logger.debug(f"No tool call segments found for {role}")
            continue

        tc_labels = []
        for seg in tc_segments:
            start_sec = seg.start_tick * tick_duration_ms / 1000.0
            end_sec = (seg.end_tick + 1) * tick_duration_ms / 1000.0
            tc_labels.append(f"{start_sec:.3f}\t{end_sec:.3f}\t{seg.tool_name}")

        tc_label_path = output_dir / f"{role}_tool_calls_labels.txt"
        with open(tc_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(tc_labels))

        label_files[f"{role}_tool_calls"] = tc_label_path
        logger.debug(
            f"Generated {len(tc_segments)} tool call labels for {role}: {tc_label_path}"
        )

    return label_files


def make_conversation_audio(
    audio_datas: list[AudioData], silence_duration_ms: int = 1000
) -> AudioData:
    """
    Make a conversation audio from a list of audio datas.

    Args:
        audio_datas: List of audio datas to make a conversation audio from.
        silence_duration_ms: Duration of silence between audio datas in milliseconds.

    Returns:
        AudioData: The conversation audio.
    """
    #

    # Concatenate all audio datas with silence between them.
    return merge_audio_datas(audio_datas, silence_duration_ms)


# =============================================================================
# HALF-DUPLEX: Message-based audio extraction
# =============================================================================


def get_audio_datas_from_messages(
    messages: list[Message], roles: list[ParticipantRole]
) -> dict[ParticipantRole, list[tuple[Optional[int], Optional[str], AudioData]]]:
    """
    Get the audio datas from a list of messages (for half-duplex mode).

    This extracts audio from individual messages, sorted by turn_idx.
    Tool call messages are skipped as they don't contain audio.

    Args:
        messages: List of messages to get the audio datas from.
        roles: List of roles to get the audio datas for.
    Returns:
        Dictionary of audio datas by role.
    """
    audio_datas = {role: [] for role in roles}
    for message in messages:
        if not isinstance(message, ParticipantMessageBase):
            continue
        if message.role not in roles:
            continue
        if message.is_tool_call():
            # Tool calls don't have audio in half-duplex
            continue
        if message.is_audio:
            if message.audio_content is None and message.audio_path is None:
                raise ValueError(f"Message {message.id} has no audio content or path")
            if message.role not in audio_datas:
                raise ValueError(f"Invalid role: {message.role}")
            if message.audio_content is not None:
                audio_data = AudioData(
                    data=message.get_audio_bytes(),
                    format=deepcopy(message.audio_format),
                    audio_path=message.audio_path,
                )
            else:
                audio_data = load_wav_file(message.audio_path)
            audio_datas[message.role].append(
                (message.turn_idx, message.timestamp, audio_data)
            )
    return audio_datas


def _generate_half_duplex_audio(
    simulation: SimulationRun, output_dir: Path
) -> dict[str, Optional[AudioData]]:
    """
    Generate audio for half-duplex simulation.

    In half-duplex mode, turns are sequential and we merge audio with silence gaps.

    Args:
        simulation: Simulation run containing messages with audio data.
        output_dir: Directory to save audio files.

    Returns:
        Dictionary with "user", "assistant", and "both" AudioData objects.
    """
    audio_datas = get_audio_datas_from_messages(
        simulation.messages, ["user", "assistant"]
    )
    merged_audio_datas: dict[str, Optional[AudioData]] = {
        "user": None,
        "assistant": None,
        "both": None,
    }

    if len(audio_datas["user"]) == 0 and len(audio_datas["assistant"]) == 0:
        return merged_audio_datas

    # Merge "both" - interleaved by turn_idx
    if len(audio_datas["user"]) > 0 and len(audio_datas["assistant"]) > 0:
        all_audio_datas = audio_datas["user"] + audio_datas["assistant"]
        sorted_by_turn_idx = sorted(all_audio_datas, key=lambda x: x[0])
        audio_only = [audio_data for _, _, audio_data in sorted_by_turn_idx]
        merged_audio_datas["both"] = merge_audio_datas(
            audio_only, silence_duration_ms=500
        )

    # Merge individual roles
    for role, role_audio_datas in audio_datas.items():
        if len(role_audio_datas) == 0:
            continue
        audio_only = [audio_data for _, _, audio_data in role_audio_datas]
        merged_audio_datas[role] = merge_audio_datas(
            audio_only, silence_duration_ms=1000
        )

    # Save files
    for role, merged_audio_data in merged_audio_datas.items():
        if merged_audio_data is not None:
            merged_audio_data.audio_path = output_dir / f"{role}.wav"
            save_wav_file(merged_audio_data, merged_audio_data.audio_path)

    return merged_audio_datas


# =============================================================================
# FULL-DUPLEX: Tick-based audio extraction
# =============================================================================


def _infer_tick_duration_ms(ticks: list[Tick]) -> Optional[int]:
    """
    Infer the tick duration from the audio data in ticks.

    Each tick's audio should have exactly `bytes_per_tick` bytes.
    We find the first tick with valid audio and compute duration from its length.

    Returns:
        Tick duration in milliseconds, or None if no valid audio found.
    """
    for tick in ticks:
        # Check agent audio
        if tick.agent_chunk is not None and tick.agent_chunk.audio_content is not None:
            audio_bytes = tick.agent_chunk.get_audio_bytes()
            audio_format = tick.agent_chunk.audio_format
            if audio_format is not None and len(audio_bytes) > 0:
                # duration_ms = (num_bytes / bytes_per_sample) / sample_rate * 1000
                bytes_per_sample = audio_format.encoding.sample_width
                duration_ms = (
                    len(audio_bytes) / bytes_per_sample / audio_format.sample_rate
                ) * 1000
                return int(duration_ms)

        # Check user audio
        if tick.user_chunk is not None and tick.user_chunk.audio_content is not None:
            audio_bytes = tick.user_chunk.get_audio_bytes()
            audio_format = tick.user_chunk.audio_format
            if audio_format is not None and len(audio_bytes) > 0:
                bytes_per_sample = audio_format.encoding.sample_width
                duration_ms = (
                    len(audio_bytes) / bytes_per_sample / audio_format.sample_rate
                ) * 1000
                return int(duration_ms)

    return None


def get_audio_datas_from_ticks(
    ticks: list[Tick],
    tick_duration_ms: Optional[int] = None,
) -> dict[ParticipantRole, list[AudioData]]:
    """
    Get the audio datas from a list of ticks (for full-duplex mode).

    This extracts audio from ALL ticks, maintaining temporal alignment.
    When a tick is missing audio content, silence is inserted to preserve
    the time alignment between user and assistant tracks.

    Args:
        ticks: List of ticks from a full-duplex simulation.
        tick_duration_ms: Duration of each tick in milliseconds. If None,
            will be inferred from the first tick with valid audio.

    Returns:
        Dictionary with "user" and "assistant" lists of AudioData.
    """
    audio_datas: dict[ParticipantRole, list[AudioData]] = {
        "user": [],
        "assistant": [],
    }

    if not ticks:
        return audio_datas

    # Infer tick duration if not provided
    if tick_duration_ms is None:
        tick_duration_ms = _infer_tick_duration_ms(ticks)
        if tick_duration_ms is None:
            logger.warning("Could not infer tick duration - no valid audio found")
            return audio_datas

    logger.debug(f"Using tick duration: {tick_duration_ms}ms")

    # Find the audio format from a tick that has audio (for generating matching silence)
    # We need to generate silence in the same format as the actual audio
    reference_format: Optional[AudioFormat] = None
    reference_bytes_per_tick: Optional[int] = None
    for tick in ticks:
        if (
            tick.agent_chunk
            and tick.agent_chunk.audio_content
            and tick.agent_chunk.audio_format
        ):
            audio_bytes = tick.agent_chunk.get_audio_bytes()
            if len(audio_bytes) > 0:
                reference_format = tick.agent_chunk.audio_format
                reference_bytes_per_tick = len(audio_bytes)
                break
        if (
            tick.user_chunk
            and tick.user_chunk.audio_content
            and tick.user_chunk.audio_format
        ):
            audio_bytes = tick.user_chunk.get_audio_bytes()
            if len(audio_bytes) > 0:
                reference_format = tick.user_chunk.audio_format
                reference_bytes_per_tick = len(audio_bytes)
                break

    def _make_silence_for_format(
        audio_format: Optional[AudioFormat], num_bytes: int
    ) -> AudioData:
        """Generate silence matching the given audio format."""
        if audio_format is None:
            # Fallback to standard PCM16
            from tau2.voice.utils.audio_preprocessing import generate_silence_audio

            return generate_silence_audio(tick_duration_ms)

        # Determine the silence byte value based on encoding
        encoding = audio_format.encoding
        if encoding == AudioEncoding.ULAW:
            silence_byte = b"\x7f"  # μ-law silence (linear zero)
        elif encoding == AudioEncoding.ALAW:
            silence_byte = b"\xd5"  # A-law silence
        else:
            # PCM formats use zero
            sample_width = encoding.sample_width
            silence_byte = b"\x00" * sample_width

        # Generate silence with the right number of bytes
        silence_data = (silence_byte * (num_bytes // len(silence_byte) + 1))[:num_bytes]

        return AudioData(
            data=silence_data,
            format=deepcopy(audio_format),
        )

    num_ticks = len(ticks)
    for i, tick in enumerate(ticks):
        is_last_tick = i == num_ticks - 1

        # Extract agent audio from this tick
        if tick.agent_chunk is not None and tick.agent_chunk.audio_content is not None:
            agent_audio = AudioData(
                data=tick.agent_chunk.get_audio_bytes(),
                format=deepcopy(tick.agent_chunk.audio_format),
            )
            audio_datas["assistant"].append(agent_audio)
        elif (
            is_last_tick
            and reference_format is not None
            and reference_bytes_per_tick is not None
        ):
            # Last tick may be incomplete due to termination - use silence in matching format
            silence = _make_silence_for_format(
                reference_format, reference_bytes_per_tick
            )
            audio_datas["assistant"].append(silence)
            logger.debug(f"Tick {tick.tick_id} (last): using silence for agent audio")
        elif is_last_tick:
            # Last tick but no reference format - skip adding silence
            logger.debug(
                f"Tick {tick.tick_id} (last): no reference format, skipping agent audio"
            )
        else:
            raise ValueError(
                f"Tick {tick.tick_id} has no agent audio content. "
                "All ticks must have audio_content set. This can happen if:\n"
                "  1. The simulation was loaded from JSON (audio_content is not serialized)\n"
                "  2. The agent did not provide audio for this tick (bug in agent)\n"
                "Audio generation must be done before saving simulation to JSON."
            )

        # Extract user audio from this tick
        if tick.user_chunk is not None and tick.user_chunk.audio_content is not None:
            user_audio = AudioData(
                data=tick.user_chunk.get_audio_bytes(),
                format=deepcopy(tick.user_chunk.audio_format),
            )
            audio_datas["user"].append(user_audio)
        elif (
            is_last_tick
            and reference_format is not None
            and reference_bytes_per_tick is not None
        ):
            # Last tick may be incomplete due to termination - use silence in matching format
            silence = _make_silence_for_format(
                reference_format, reference_bytes_per_tick
            )
            audio_datas["user"].append(silence)
            logger.debug(f"Tick {tick.tick_id} (last): using silence for user audio")
        elif is_last_tick:
            # Last tick but no reference format - skip adding silence
            logger.debug(
                f"Tick {tick.tick_id} (last): no reference format, skipping user audio"
            )
        else:
            raise ValueError(
                f"Tick {tick.tick_id} has no user audio content. "
                "All ticks must have audio_content set. This can happen if:\n"
                "  1. The simulation was loaded from JSON (audio_content is not serialized)\n"
                "  2. The user simulator did not provide audio for this tick (bug)\n"
                "Audio generation must be done before saving simulation to JSON."
            )

    return audio_datas


def _generate_full_duplex_audio(
    simulation: SimulationRun,
    output_dir: Path,
    generate_labels: bool = True,
) -> dict[str, Optional[AudioData]]:
    """
    Generate audio for full-duplex simulation using tick-based extraction.

    In full-duplex mode, both participants can speak simultaneously.
    Audio is extracted directly from ticks to maintain temporal alignment.
    The result is stereo audio with user on left channel and assistant on right.

    Optionally generates Audacity label files for annotating speech segments.

    Args:
        simulation: Simulation run containing ticks with audio data.
        output_dir: Directory to save audio files.
        generate_labels: If True, generate Audacity label files (.txt) for
            each role that can be imported into Audacity alongside the audio.

    Returns:
        Dictionary with "user", "assistant", and "both" AudioData objects.
    """
    merged_audio_datas: dict[str, Optional[AudioData]] = {
        "user": None,
        "assistant": None,
        "both": None,
    }

    if not simulation.ticks:
        logger.warning("No ticks found in full-duplex simulation")
        return merged_audio_datas

    # Infer tick duration for labels
    tick_duration_ms = _infer_tick_duration_ms(simulation.ticks)

    audio_datas = get_audio_datas_from_ticks(simulation.ticks, tick_duration_ms)

    if len(audio_datas["user"]) == 0 and len(audio_datas["assistant"]) == 0:
        logger.warning("No audio data found in ticks")
        return merged_audio_datas

    # Merge each role's audio (no silence between chunks for continuous stream)
    if len(audio_datas["assistant"]) > 0:
        merged_audio_datas["assistant"] = merge_audio_datas(
            audio_datas["assistant"], silence_duration_ms=None
        )

    if len(audio_datas["user"]) > 0:
        merged_audio_datas["user"] = merge_audio_datas(
            audio_datas["user"], silence_duration_ms=None
        )

    # Create stereo "both" audio (user=left, assistant=right)
    if (
        merged_audio_datas["user"] is not None
        and merged_audio_datas["assistant"] is not None
    ):
        # Pad shorter audio to match longer one (handles edge cases where
        # user and agent audio have slightly different lengths)
        from tau2.voice.utils.audio_preprocessing import pad_audio_with_zeros

        merged_user = merged_audio_datas["user"]
        merged_assistant = merged_audio_datas["assistant"]
        max_samples = max(merged_user.num_samples, merged_assistant.num_samples)
        merged_user = pad_audio_with_zeros(merged_user, max_samples)
        merged_assistant = pad_audio_with_zeros(merged_assistant, max_samples)

        merged_audio_datas["both"] = convert_to_stereo(merged_user, merged_assistant)

    # Only save the combined audio file (both.wav)
    if merged_audio_datas["both"] is not None:
        merged_audio_datas["both"].audio_path = output_dir / "both.wav"
        save_wav_file(merged_audio_datas["both"], merged_audio_datas["both"].audio_path)

    # Generate Audacity label files for annotation
    if generate_labels and tick_duration_ms is not None:
        try:
            generate_audacity_labels(
                simulation.ticks,
                tick_duration_ms,
                output_dir,
            )
        except Exception as e:
            logger.warning(f"Failed to generate Audacity labels: {e}")

    return merged_audio_datas


# =============================================================================
# Main entry point
# =============================================================================


def generate_simulation_audio(
    simulation: SimulationRun, output_dir: str | Path
) -> dict[str, Optional[AudioData]]:
    """
    Generate audio for the simulation and save it to the output directory.

    Extracts audio data from simulation and merges it according to the
    communication mode:

    - FULL_DUPLEX: Uses tick-based extraction to maintain temporal alignment.
      Creates stereo audio with user on left channel and assistant on right
      channel, allowing overlapping speech.

    - HALF_DUPLEX: Uses message-based extraction. Merges audio sequentially
      with silence gaps between turns (500ms for combined, 1000ms for individual).

    The function creates one audio file in the output directory:
    - both.wav: Combined user and assistant audio (stereo for full-duplex,
      mono for half-duplex)

    Args:
        simulation: Simulation run containing messages/ticks with audio data.
            Must have a valid `mode` attribute (CommunicationMode.FULL_DUPLEX
            or CommunicationMode.HALF_DUPLEX).
        output_dir: Directory path where the generated audio files will be
            saved. Will be created if it doesn't exist.

    Returns:
        Dictionary with keys "user", "assistant", and "both", mapping to
        AudioData objects for each role combination. Values are None if no
        audio data exists for that combination.

    Raises:
        ValueError: If simulation.mode is not a valid CommunicationMode.
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if simulation.mode == CommunicationMode.FULL_DUPLEX:
        return _generate_full_duplex_audio(simulation, output_dir)
    elif simulation.mode == CommunicationMode.HALF_DUPLEX:
        return _generate_half_duplex_audio(simulation, output_dir)
    else:
        raise ValueError(f"Invalid mode: {simulation.mode}")


# Keep old function name as alias for backwards compatibility
get_audio_datas = get_audio_datas_from_messages
