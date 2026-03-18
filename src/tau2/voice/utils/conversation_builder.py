import logging
from copy import deepcopy
from pathlib import Path
from typing import Optional

from tau2.data_model.audio import AudioData
from tau2.data_model.message import Message, ParticipantMessageBase, ParticipantRole
from tau2.data_model.simulation import SimulationRun
from tau2.orchestrator.modes import CommunicationMode
from tau2.voice.utils.audio_io import load_wav_file, play_audio, save_wav_file
from tau2.voice.utils.audio_preprocessing import convert_to_stereo, merge_audio_datas

logger = logging.getLogger(__name__)


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


def get_audio_datas(
    messages: list[Message], roles: list[ParticipantRole]
) -> dict[ParticipantRole, list[tuple[Optional[int], Optional[str], AudioData]]]:
    """
    Get the audio datas from a list of messages.

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
            continue
        # Extract audio if message is marked as audio OR has audio_content
        # (supports discrete-time agents that store audio but use is_audio=False for text mode)
        has_audio = message.is_audio or message.audio_content is not None
        if has_audio:
            if message.role not in audio_datas:
                raise ValueError(f"Invalid role: {message.role}")
            if message.audio_content is not None:
                audio_bytes = message.get_audio_bytes()
                # Skip if audio_content exists but decodes to None/empty
                if audio_bytes is None or len(audio_bytes) == 0:
                    raise ValueError(f"No audio bytes for audio message: {message}")
                if message.audio_format is None:
                    raise ValueError(f"No audio format info for message: {message}")
                audio_data = AudioData(
                    data=audio_bytes,
                    format=deepcopy(message.audio_format),
                    audio_path=message.audio_path,
                )
            elif message.audio_path is not None:
                audio_data = load_wav_file(message.audio_path)
            else:
                raise ValueError(f"No audio source for audio message: {message}")
            audio_datas[message.role].append(
                (message.turn_idx, message.timestamp, audio_data)
            )
    return audio_datas


def generate_simulation_audio(
    simulation: SimulationRun, output_dir: str | Path | None = None
) -> dict[str, AudioData]:
    """
    Generate audio for the simulation and save it to the output directory.

    Extracts audio data from simulation messages for both user and assistant roles,
    merges them according to the communication mode, and saves the resulting audio
    files. The merging behavior differs based on the simulation mode:

    - FULL_DUPLEX: Creates stereo audio with user on left channel and assistant on
      right channel, allowing overlapping speech. Individual role audio is merged
      without silence gaps.
    - HALF_DUPLEX: Merges audio sequentially with silence gaps between turns (500ms
      for combined audio, 1000ms for individual roles).

    The function creates one audio file in the output directory:
    - both.wav: Combined user and assistant audio (stereo for full-duplex, mono for half-duplex)

    Args:
        simulation: Simulation run containing messages with audio data to process.
            Must have a valid `mode` attribute (CommunicationMode.FULL_DUPLEX or
            CommunicationMode.HALF_DUPLEX).
        output_dir: Directory path where the generated audio files will be saved.
            Will be created if it doesn't exist.

    Returns:
        Dictionary with keys "user", "assistant", and "both", mapping to AudioData
        objects for each role combination. Values are None if no audio data exists
        for that combination. Only the "both" AudioData object will have its `audio_path`
        set to the saved file location.

    Raises:
        ValueError: If simulation.mode is not a valid CommunicationMode.
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    audio_datas = get_audio_datas(simulation.get_messages(), ["user", "assistant"])
    merged_audio_datas = {"user": None, "assistant": None, "both": None}
    if len(audio_datas["user"]) == 0 and len(audio_datas["assistant"]) == 0:
        return merged_audio_datas
    if len(audio_datas["user"]) > 0 and len(audio_datas["assistant"]) > 0:
        if simulation.mode == CommunicationMode.FULL_DUPLEX:
            # For full-duplex, merge each role's audio first, then convert to stereo
            # Extract AudioData objects from tuples (turn_idx, timestamp, audio_data)
            user_audio_only = [audio_data for _, _, audio_data in audio_datas["user"]]
            assistant_audio_only = [
                audio_data for _, _, audio_data in audio_datas["assistant"]
            ]
            # Merge audio without silence (continuous stream for full-duplex)
            merged_user = merge_audio_datas(user_audio_only, silence_duration_ms=None)
            merged_assistant = merge_audio_datas(
                assistant_audio_only, silence_duration_ms=None
            )
            # Pad shorter audio to match longer one (for discrete-time simulations
            # where user and agent audio may have different lengths)
            from tau2.voice.utils.audio_preprocessing import pad_audio_with_zeros

            max_samples = max(merged_user.num_samples, merged_assistant.num_samples)
            merged_user = pad_audio_with_zeros(merged_user, max_samples)
            merged_assistant = pad_audio_with_zeros(merged_assistant, max_samples)
            # Convert to stereo (user=left, assistant=right)
            merged_audio_datas["both"] = convert_to_stereo(
                merged_user, merged_assistant
            )
        elif simulation.mode == CommunicationMode.HALF_DUPLEX:
            all_audio_datas = audio_datas["user"] + audio_datas["assistant"]
            sorted_by_turn_idx = sorted(all_audio_datas, key=lambda x: x[0])
            # Extract AudioData objects from tuples (turn_idx, timestamp, audio_data)
            audio_only = [audio_data for _, _, audio_data in sorted_by_turn_idx]
            merged_audio_datas["both"] = merge_audio_datas(
                audio_only, silence_duration_ms=500
            )
        else:
            logger.warning(f"Invalid mode: {simulation.mode}")
            merged_audio_datas["both"] = None
    for role, role_audio_datas in audio_datas.items():
        if len(role_audio_datas) == 0:
            continue
        # Extract AudioData objects from tuples (turn_idx, timestamp, audio_data)
        audio_only = [audio_data for _, _, audio_data in role_audio_datas]
        if simulation.mode == CommunicationMode.HALF_DUPLEX:
            merged_audio_data = merge_audio_datas(audio_only, silence_duration_ms=1000)
        elif simulation.mode == CommunicationMode.FULL_DUPLEX:
            merged_audio_data = merge_audio_datas(audio_only, silence_duration_ms=None)
        else:
            raise ValueError(f"Invalid mode: {simulation.mode}")
        merged_audio_datas[role] = merged_audio_data
    if output_dir is None:
        return merged_audio_datas
    # Only save the combined audio file (both.wav)
    if merged_audio_datas["both"] is not None:
        merged_audio_datas["both"].audio_path = output_dir / "both.wav"
        save_wav_file(merged_audio_datas["both"], merged_audio_datas["both"].audio_path)
    return merged_audio_datas


def play_conversation_audio(simulation: SimulationRun, role: str) -> None:
    """
    Play a conversation audio from a list of audio datas.
    Args:
        simulation: Simulation run to play audio for.
        role: Role to play audio for. Can be "user", "assistant", or "both".
    """
    audio_datas = generate_simulation_audio(simulation, None)
    audio_data = audio_datas.get(role, None)
    if audio_data is None:
        raise ValueError(f"No audio data found for role: {role}")
    play_audio(audio_data)
