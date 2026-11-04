# Copyright Sierra
"""Read-only recovery of legacy Live activity from the exact saved recording."""

import wave
from pathlib import Path
from typing import TYPE_CHECKING

from tau2.data_model.message import AssistantMessage
from tau2.data_model.transcript import LiveTranscriptFragment, LiveTranscriptTick
from tau2.utils.transcript.live import live_metadata

if TYPE_CHECKING:
    from tau2.data_model.simulation import SimulationRun


def recover_live_audio(
    simulation: "SimulationRun", path: Path, *, provider: str
) -> "SimulationRun":
    """Add derived evidence to a copy only for explicitly recorded legacy Live runs.

    Missing provider timestamps/session IDs are not reconstructed. Caption text
    keeps its recorded release tick; the recovered call ID is a local namespace.
    """
    if provider != "openai_live" or not simulation.ticks:
        return simulation
    if any(live_metadata(t) is not None for t in simulation.ticks):
        return simulation
    ticks = simulation.ticks
    durations = {t.tick_duration_seconds for t in ticks}
    if len(durations) != 1 or None in durations or next(iter(durations)) <= 0:
        raise ValueError(
            "Live audio recovery requires a uniform positive tick duration"
        )
    seconds = next(iter(durations))
    if [t.tick_id for t in ticks] != list(range(len(ticks))):
        raise ValueError("Live audio recovery requires contiguous zero-based ticks")
    with wave.open(str(path), "rb") as wav:
        if (
            wav.getnchannels() != 2
            or wav.getsampwidth() != 2
            or wav.getcomptype() != "NONE"
        ):
            raise ValueError("Live audio recovery requires stereo PCM16")
        rate = wav.getframerate()
        if rate % 50:
            raise ValueError("Live audio recovery requires integral 20 ms frames")
        data = wav.readframes(wav.getnframes())
    import numpy as np

    pcm = np.frombuffer(data, dtype="<i2").reshape(-1, 2)[:, 1]
    if len(pcm) < round(len(ticks) * seconds * rate):
        raise ValueError("Recorded Live audio does not cover the tick timeline")
    recovered = simulation.model_copy(deep=True)
    frame_size = rate // 50
    for tick in recovered.ticks:
        start = round(tick.tick_id * seconds * rate)
        stop = round((tick.tick_id + 1) * seconds * rate)
        intervals = []
        for offset in range(start, stop, frame_size):
            end = min(offset + frame_size, stop)
            if np.abs(pcm[offset:end].astype(np.int32)).max(initial=0) > 32:
                intervals.append(
                    ((offset - start) * 1000 / rate, (end - start) * 1000 / rate)
                )
        chunk = tick.agent_chunk
        raw = dict(chunk.raw_data or {}) if chunk else {}
        # The harness greeting has text but no provider tick record/audio.
        text = (chunk.content or "") if chunk and "tick_number" in raw else ""
        if chunk is None:
            chunk = tick.agent_chunk = AssistantMessage(role="assistant", content="")
        raw["live_transcript"] = LiveTranscriptTick(
            session_id=f"recovered-call:{simulation.id}",
            origin="recorded_audio",
            fragments=[LiveTranscriptFragment(speaker="agent", text=text)]
            if text
            else [],
            speech_intervals_ms=intervals,
        ).model_dump()
        chunk.raw_data = raw
    return recovered
