"""Audio debugging and analysis utilities for discrete-time simulations.

This module provides tools for diagnosing audio timing issues in full-duplex
simulations. It can:
- Extract and save per-tick audio for both agent and user
- Generate timing analysis reports
- Identify alignment issues between audio streams

Usage:
    from tau2.voice.utils.audio_debug import generate_audio_debug_info

    # During simulation (before JSON serialization):
    generate_audio_debug_info(simulation, output_dir, save_per_tick_audio=True)
"""

import json
import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from tau2.config import DEFAULT_TICK_DURATION_SECONDS
from tau2.data_model.audio import AudioData
from tau2.data_model.message import Tick
from tau2.data_model.simulation import SimulationRun
from tau2.voice.utils.audio_io import save_wav_file

DEFAULT_TICK_DURATION_MS = DEFAULT_TICK_DURATION_SECONDS * 1000

logger = logging.getLogger(__name__)


@dataclass
class TickAudioInfo:
    """Audio information for a single tick."""

    tick_id: int
    simulation_time_ms: int  # cumulative_user_audio_at_tick_start_ms
    tick_duration_ms: float

    # Agent info
    agent_has_audio: bool
    agent_audio_bytes: int
    agent_contains_speech: bool
    agent_is_tool_call: bool
    agent_proportional_transcript: str
    agent_utterance_ids: Optional[list[str]]

    # User info
    user_has_audio: bool
    user_audio_bytes: int
    user_contains_speech: bool
    user_is_tool_call: bool
    user_turn_taking_action: Optional[str]

    # Warnings
    warnings: list[str] = field(default_factory=list)


@dataclass
class AudioAnalysisReport:
    """Complete audio timing analysis report."""

    simulation_id: str
    total_ticks: int
    tick_duration_ms: float
    expected_duration_ms: float

    # Per-role stats
    agent_ticks_with_audio: int
    agent_ticks_skipped: int  # tool calls, no audio_content, etc.
    agent_actual_audio_ms: float
    agent_speech_ticks: int

    user_ticks_with_audio: int
    user_ticks_skipped: int
    user_actual_audio_ms: float
    user_speech_ticks: int

    # Alignment issues
    alignment_offset_ms: float  # difference between agent and user audio lengths
    skipped_tick_ids: dict[str, list[int]]  # role -> list of skipped tick IDs

    # Per-tick details (optional, can be large)
    tick_details: Optional[list[dict]] = None

    # Warnings
    warnings: list[str] = field(default_factory=list)


def extract_tick_audio_info(
    tick: Tick, tick_duration_ms: float = DEFAULT_TICK_DURATION_MS
) -> TickAudioInfo:
    """Extract audio information from a single tick.

    Args:
        tick: The tick to analyze.
        tick_duration_ms: Expected tick duration in milliseconds.

    Returns:
        TickAudioInfo with details about this tick's audio.
    """
    warnings = []

    # Get simulation time from agent's raw_data if available
    sim_time_ms = 0
    if tick.agent_chunk and tick.agent_chunk.raw_data:
        raw = tick.agent_chunk.raw_data
        if isinstance(raw, dict):
            sim_time_ms = raw.get("cumulative_user_audio_at_tick_start_ms", 0)

    # Agent audio info
    agent_has_audio = False
    agent_audio_bytes = 0
    agent_contains_speech = False
    agent_is_tool_call = False
    agent_transcript = ""
    agent_utterance_ids = None

    if tick.agent_chunk:
        chunk = tick.agent_chunk
        agent_is_tool_call = chunk.is_tool_call()
        agent_contains_speech = chunk.contains_speech or False
        agent_utterance_ids = chunk.utterance_ids

        # Check if audio_content is available (before serialization)
        if chunk.audio_content is not None:
            try:
                audio_bytes = chunk.get_audio_bytes()
                if audio_bytes:
                    agent_has_audio = True
                    agent_audio_bytes = len(audio_bytes)
            except Exception as e:
                warnings.append(f"Agent audio decode error: {e}")
        elif chunk.audio_format is not None:
            # Has format but no content - might be excluded from serialization
            warnings.append(
                "Agent has audio_format but no audio_content (may be serialized)"
            )

        # Get transcript
        if chunk.raw_data and isinstance(chunk.raw_data, dict):
            agent_transcript = chunk.raw_data.get("proportional_transcript", "")

        # Check for issues
        if agent_is_tool_call and agent_has_audio:
            warnings.append(
                "Agent has tool_call with audio - will be skipped in audio generation"
            )

    # User audio info
    user_has_audio = False
    user_audio_bytes = 0
    user_contains_speech = False
    user_is_tool_call = False
    user_action = None

    if tick.user_chunk:
        chunk = tick.user_chunk
        user_is_tool_call = chunk.is_tool_call()
        user_contains_speech = chunk.contains_speech or False

        if chunk.turn_taking_action:
            user_action = chunk.turn_taking_action.action

        # Check if audio_content is available
        if chunk.audio_content is not None:
            try:
                audio_bytes = chunk.get_audio_bytes()
                if audio_bytes:
                    user_has_audio = True
                    user_audio_bytes = len(audio_bytes)
            except Exception as e:
                warnings.append(f"User audio decode error: {e}")
        elif chunk.audio_format is not None:
            warnings.append(
                "User has audio_format but no audio_content (may be serialized)"
            )

    return TickAudioInfo(
        tick_id=tick.tick_id,
        simulation_time_ms=sim_time_ms,
        tick_duration_ms=tick_duration_ms,
        agent_has_audio=agent_has_audio,
        agent_audio_bytes=agent_audio_bytes,
        agent_contains_speech=agent_contains_speech,
        agent_is_tool_call=agent_is_tool_call,
        agent_proportional_transcript=agent_transcript,
        agent_utterance_ids=agent_utterance_ids,
        user_has_audio=user_has_audio,
        user_audio_bytes=user_audio_bytes,
        user_contains_speech=user_contains_speech,
        user_is_tool_call=user_is_tool_call,
        user_turn_taking_action=user_action,
        warnings=warnings,
    )


def analyze_simulation_audio(
    simulation: SimulationRun,
    tick_duration_ms: float = DEFAULT_TICK_DURATION_MS,
    include_tick_details: bool = True,
) -> AudioAnalysisReport:
    """Analyze audio timing in a simulation.

    This function examines each tick to identify:
    - Which ticks have audio for each role
    - Which ticks would be skipped during audio generation
    - Alignment issues between user and agent audio

    Args:
        simulation: The simulation to analyze.
        tick_duration_ms: Expected tick duration in milliseconds.
        include_tick_details: Whether to include per-tick details in the report.

    Returns:
        AudioAnalysisReport with complete timing analysis.
    """
    if not simulation.ticks:
        return AudioAnalysisReport(
            simulation_id=simulation.id or "unknown",
            total_ticks=0,
            tick_duration_ms=tick_duration_ms,
            expected_duration_ms=0,
            agent_ticks_with_audio=0,
            agent_ticks_skipped=0,
            agent_actual_audio_ms=0,
            agent_speech_ticks=0,
            user_ticks_with_audio=0,
            user_ticks_skipped=0,
            user_actual_audio_ms=0,
            user_speech_ticks=0,
            alignment_offset_ms=0,
            skipped_tick_ids={"agent": [], "user": []},
            warnings=["No ticks in simulation"],
        )

    tick_infos = []
    agent_with_audio = 0
    agent_skipped = []
    agent_audio_bytes_total = 0
    agent_speech = 0

    user_with_audio = 0
    user_skipped = []
    user_audio_bytes_total = 0
    user_speech = 0

    warnings = []

    for tick in simulation.ticks:
        info = extract_tick_audio_info(tick, tick_duration_ms)
        tick_infos.append(info)

        # Agent stats
        if info.agent_has_audio and not info.agent_is_tool_call:
            agent_with_audio += 1
            agent_audio_bytes_total += info.agent_audio_bytes
        else:
            agent_skipped.append(tick.tick_id)

        if info.agent_contains_speech:
            agent_speech += 1

        # User stats
        if info.user_has_audio and not info.user_is_tool_call:
            user_with_audio += 1
            user_audio_bytes_total += info.user_audio_bytes
        else:
            user_skipped.append(tick.tick_id)

        if info.user_contains_speech:
            user_speech += 1

        # Collect warnings
        warnings.extend(info.warnings)

    total_ticks = len(simulation.ticks)
    expected_duration_ms = total_ticks * tick_duration_ms

    # Calculate actual audio durations (assuming 8kHz mono 8-bit = 8 bytes/ms)
    bytes_per_ms = 8  # 8kHz * 1 byte
    agent_actual_ms = agent_audio_bytes_total / bytes_per_ms
    user_actual_ms = user_audio_bytes_total / bytes_per_ms

    alignment_offset_ms = abs(agent_actual_ms - user_actual_ms)

    if alignment_offset_ms > tick_duration_ms:
        warnings.append(
            f"Audio alignment offset ({alignment_offset_ms:.1f}ms) exceeds one tick duration"
        )

    if len(agent_skipped) > 0:
        warnings.append(
            f"Agent has {len(agent_skipped)} ticks without audio in generated file"
        )

    if len(user_skipped) > 0:
        warnings.append(
            f"User has {len(user_skipped)} ticks without audio in generated file"
        )

    return AudioAnalysisReport(
        simulation_id=simulation.id or "unknown",
        total_ticks=total_ticks,
        tick_duration_ms=tick_duration_ms,
        expected_duration_ms=expected_duration_ms,
        agent_ticks_with_audio=agent_with_audio,
        agent_ticks_skipped=len(agent_skipped),
        agent_actual_audio_ms=agent_actual_ms,
        agent_speech_ticks=agent_speech,
        user_ticks_with_audio=user_with_audio,
        user_ticks_skipped=len(user_skipped),
        user_actual_audio_ms=user_actual_ms,
        user_speech_ticks=user_speech,
        alignment_offset_ms=alignment_offset_ms,
        skipped_tick_ids={"agent": agent_skipped, "user": user_skipped},
        tick_details=(
            [asdict(info) for info in tick_infos] if include_tick_details else None
        ),
        warnings=warnings,
    )


def save_per_tick_audio(
    simulation: SimulationRun,
    output_dir: Path,
    save_silence: bool = False,
) -> dict[str, int]:
    """Save individual audio files for each tick.

    Creates a directory structure:
        output_dir/
            agent/
                tick_0000.wav
                tick_0001.wav
                ...
            user/
                tick_0000.wav
                tick_0001.wav
                ...

    Args:
        simulation: The simulation with ticks containing audio_content.
        output_dir: Directory to save the audio files.
        save_silence: If True, also save ticks with no speech (silence/noise).

    Returns:
        Dictionary with counts: {"agent": N, "user": M}
    """
    output_dir = Path(output_dir)
    agent_dir = output_dir / "agent"
    user_dir = output_dir / "user"
    agent_dir.mkdir(parents=True, exist_ok=True)
    user_dir.mkdir(parents=True, exist_ok=True)

    counts = {"agent": 0, "user": 0}

    for tick in simulation.ticks:
        tick_id = tick.tick_id

        # Save agent audio
        if tick.agent_chunk and tick.agent_chunk.audio_content:
            chunk = tick.agent_chunk
            if save_silence or chunk.contains_speech:
                try:
                    audio_bytes = chunk.get_audio_bytes()
                    if audio_bytes and chunk.audio_format:
                        audio_data = AudioData(
                            data=audio_bytes,
                            format=deepcopy(chunk.audio_format),
                        )
                        filename = f"tick_{tick_id:04d}.wav"
                        save_wav_file(audio_data, agent_dir / filename)
                        counts["agent"] += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to save agent audio for tick {tick_id}: {e}"
                    )

        # Save user audio
        if tick.user_chunk and tick.user_chunk.audio_content:
            chunk = tick.user_chunk
            if save_silence or chunk.contains_speech:
                try:
                    audio_bytes = chunk.get_audio_bytes()
                    if audio_bytes and chunk.audio_format:
                        audio_data = AudioData(
                            data=audio_bytes,
                            format=deepcopy(chunk.audio_format),
                        )
                        filename = f"tick_{tick_id:04d}.wav"
                        save_wav_file(audio_data, user_dir / filename)
                        counts["user"] += 1
                except Exception as e:
                    logger.warning(f"Failed to save user audio for tick {tick_id}: {e}")

    return counts


def generate_audio_debug_info(
    simulation: SimulationRun,
    output_dir: str | Path,
    save_per_tick_audio_files: bool = False,
    save_silence: bool = False,
    tick_duration_ms: float = DEFAULT_TICK_DURATION_MS,
) -> AudioAnalysisReport:
    """Generate comprehensive audio debugging information.

    This function should be called BEFORE the simulation is serialized to JSON,
    while audio_content is still available in memory.

    Creates:
        output_dir/
            audio_analysis.json     # Timing analysis report
            per_tick/               # Optional: individual tick audio files
                agent/
                    tick_0000.wav
                    ...
                user/
                    tick_0000.wav
                    ...

    Args:
        simulation: The simulation to analyze.
        output_dir: Directory to save debug files.
        save_per_tick_audio_files: If True, save individual audio files per tick.
        save_silence: If True, also save silent ticks (only if save_per_tick_audio_files=True).
        tick_duration_ms: Expected tick duration in milliseconds.

    Returns:
        AudioAnalysisReport with the analysis results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate analysis report
    report = analyze_simulation_audio(
        simulation,
        tick_duration_ms=tick_duration_ms,
        include_tick_details=True,
    )

    # Save analysis report
    report_path = output_dir / "audio_analysis.json"
    report_dict = asdict(report)
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)
    logger.info(f"Audio analysis report saved to: {report_path}")

    # Optionally save per-tick audio
    if save_per_tick_audio_files:
        per_tick_dir = output_dir / "per_tick"
        counts = save_per_tick_audio(
            simulation, per_tick_dir, save_silence=save_silence
        )
        logger.info(
            f"Saved per-tick audio: {counts['agent']} agent, {counts['user']} user files"
        )

    # Log warnings
    if report.warnings:
        logger.warning(f"Audio analysis found {len(report.warnings)} warning(s):")
        for warning in report.warnings[:10]:  # Limit to first 10
            logger.warning(f"  - {warning}")
        if len(report.warnings) > 10:
            logger.warning(f"  ... and {len(report.warnings) - 10} more")

    return report


def print_audio_analysis_summary(report: AudioAnalysisReport) -> None:
    """Print a human-readable summary of the audio analysis.

    Args:
        report: The analysis report to summarize.
    """
    print(f"\n{'=' * 60}")
    print(f"AUDIO ANALYSIS REPORT: {report.simulation_id}")
    print(f"{'=' * 60}")

    print(f"\nTiming Overview:")
    print(f"  Total ticks: {report.total_ticks}")
    print(f"  Tick duration: {report.tick_duration_ms}ms")
    print(f"  Expected duration: {report.expected_duration_ms / 1000:.2f}s")

    print(f"\nAgent Audio:")
    print(f"  Ticks with audio: {report.agent_ticks_with_audio}")
    print(f"  Ticks skipped: {report.agent_ticks_skipped}")
    print(f"  Actual audio: {report.agent_actual_audio_ms / 1000:.2f}s")
    print(f"  Speech ticks: {report.agent_speech_ticks}")

    print(f"\nUser Audio:")
    print(f"  Ticks with audio: {report.user_ticks_with_audio}")
    print(f"  Ticks skipped: {report.user_ticks_skipped}")
    print(f"  Actual audio: {report.user_actual_audio_ms / 1000:.2f}s")
    print(f"  Speech ticks: {report.user_speech_ticks}")

    print(f"\nAlignment:")
    print(f"  Offset: {report.alignment_offset_ms:.1f}ms")

    if report.warnings:
        print(f"\n⚠️  Warnings ({len(report.warnings)}):")
        for warning in report.warnings[:10]:
            print(f"  - {warning}")
        if len(report.warnings) > 10:
            print(f"  ... and {len(report.warnings) - 10} more")

    print(f"\n{'=' * 60}\n")
