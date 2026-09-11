# Copyright Sierra
"""Reconstruct per-utterance agent audio from a stored full-duplex ``both.wav``
(agent = right channel, tick-aligned, so utterance tick spans slice directly).
Half-duplex mono runs have no tick timeline and fail loudly (``DiskAudioError``).
"""

from pathlib import Path
from typing import Optional

from tau2.data_model.audio import AudioData, AudioFormat
from tau2.data_model.simulation import SimulationRun
from tau2.judges.delivery.audio import audio_data_to_wav_b64
from tau2.metrics.interaction_quality import agent_utterance_tick_spans

# both.wav channel layout (see voice/utils/conversation_builder.py:
# ``convert_to_stereo(merged_user, merged_assistant)`` — user=left, agent=right).
AGENT_CHANNEL = 1


class DiskAudioError(RuntimeError):
    """A stored run's audio cannot be sliced per utterance (missing both.wav,
    mono/half-duplex layout, or a timeline that contradicts the ticks)."""


def find_both_wav(results_dir: Path, sim: SimulationRun) -> Optional[Path]:
    """Locate the sim's combined-channel wav in either results layout.

    Audio lives under either ``tasks/`` (inline-results layout) or
    ``artifacts/`` (sharded/multilingual layout), with the wav either in an
    ``audio/`` subdir or directly in the sim dir.
    """
    for parent_dir in ("tasks", "artifacts"):
        for sub in ("audio", ""):
            candidate = (
                results_dir
                / parent_dir
                / f"task_{sim.task_id}"
                / f"sim_{sim.id}"
                / sub
                / "both.wav"
            )
            if candidate.exists():
                return candidate
    return None


def sim_tick_duration_ms(sim: SimulationRun) -> float:
    """The run's tick duration in ms (recorded on the ticks)."""
    for tick in sim.ticks or []:
        if tick.tick_duration_seconds is not None:
            return tick.tick_duration_seconds * 1000.0
    raise DiskAudioError(f"sim {sim.id}: no tick records tick_duration_seconds")


def _pcm_samples(audio: AudioData) -> "object":
    """The audio's samples as a numpy array of shape (frames, channels)."""
    import numpy as np

    width = audio.format.sample_width
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise DiskAudioError(
            f"unsupported sample width {width} bytes — cannot split channels"
        )
    return np.frombuffer(audio.data, dtype=dtype).reshape(-1, audio.format.channels)


def split_agent_channel(audio: AudioData) -> AudioData:
    """Extract the agent (right) channel from stereo full-duplex audio.

    A mono file means a half-duplex run (turns merged with inserted silence,
    no tick-aligned timeline) — unsupported loudly, never mis-sliced.
    """
    if audio.format.encoding.is_companded:
        raise DiskAudioError(
            "companded both.wav — expected the linear PCM that save_wav_file "
            "writes; this is not a tau2 voice-run artifact"
        )
    if audio.format.channels != 2:
        raise DiskAudioError(
            "mono both.wav — half-duplex runs merge turns with inserted "
            "silence, so no tick-aligned agent channel exists; delivery "
            "rejudge-from-disk is full-duplex only"
        )
    samples = _pcm_samples(audio)
    agent = samples[:, AGENT_CHANNEL].tobytes()
    return AudioData(
        data=agent,
        format=AudioFormat(
            encoding=audio.format.encoding,
            sample_rate=audio.format.sample_rate,
            channels=1,
        ),
    )


def slice_tick_span(
    agent: AudioData,
    tick_start: int,
    tick_end: int,
    tick_duration_ms: float,
) -> AudioData:
    """The agent-channel audio for ticks ``[tick_start, tick_end]`` inclusive.

    The end is clamped to the channel (the merge end-pads, so the final tick
    can run a hair short), but a span STARTING beyond the audio means the
    tick→timeline assumption is broken for this run — loud error, because
    every other slice of the same file would be silently wrong too.
    """
    rate = agent.format.sample_rate
    width = agent.format.sample_width
    start_sample = round(tick_start * tick_duration_ms / 1000.0 * rate)
    end_sample = round((tick_end + 1) * tick_duration_ms / 1000.0 * rate)
    if start_sample >= agent.num_samples:
        raise DiskAudioError(
            f"tick span ({tick_start}, {tick_end}) starts at sample "
            f"{start_sample} but the agent channel has {agent.num_samples} "
            "samples — the audio timeline does not match the ticks"
        )
    end_sample = min(end_sample, agent.num_samples)
    return AudioData(
        data=agent.data[start_sample * width : end_sample * width],
        format=agent.format,
    )


class SimAgentAudio:
    """Per-utterance agent audio for one stored sim, sliced from both.wav.

    Drop-in ``wav_source`` for ``evaluate_delivery``: ``wav_b64_for(idx, msg)``
    returns the WAV base64 of utterance ``idx``'s agent audio, where ``idx``
    indexes ``ticks_to_message_history(sim.ticks)`` exactly like the inline
    judge's ``utterance_idx``.
    """

    def __init__(self, sim: SimulationRun, results_dir: Path):
        if not sim.ticks:
            raise DiskAudioError(
                f"sim {sim.id} has no ticks — not a full-duplex voice run"
            )
        wav_path = find_both_wav(Path(results_dir), sim)
        if wav_path is None:
            raise DiskAudioError(
                f"no both.wav found for sim {sim.id} (task {sim.task_id}) "
                f"under {results_dir}/(tasks|artifacts)/"
            )
        # Lazy import: audio_io pulls the voice stack.
        from tau2.voice.utils.audio_io import load_wav_file

        self.sim_id = sim.id
        self.wav_path = wav_path
        self.agent = split_agent_channel(load_wav_file(wav_path))
        self.spans = agent_utterance_tick_spans(sim)
        self.tick_duration_ms = sim_tick_duration_ms(sim)

    def slice_utterance(self, utterance_idx: int) -> AudioData:
        if not 0 <= utterance_idx < len(self.spans):
            raise DiskAudioError(
                f"utterance {utterance_idx} out of range — sim {self.sim_id} "
                f"has {len(self.spans)} agent utterances"
            )
        start, end = self.spans[utterance_idx]
        return slice_tick_span(self.agent, start, end, self.tick_duration_ms)

    def wav_b64_for(self, utterance_idx: int, _msg: object = None) -> str:
        """``evaluate_delivery``'s ``wav_source`` seam (the msg is unused —
        the audio comes from disk, not from the in-memory message)."""
        return audio_data_to_wav_b64(self.slice_utterance(utterance_idx))
