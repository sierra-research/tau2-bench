#!/usr/bin/env python3
"""
Export simulation data to a standalone interactive HTML viewer.

Generates a single HTML file per simulation with:
- Speech Activity Timeline (visual waveform-like view)
- Tick-by-tick conversation table
- Synchronized audio playback
- Effect timeline overlay (frame drops, burst noise, etc.)

Usage:
    python export_viewer.py --results path/to/results.json
    python export_viewer.py --results path/to/results.json --task-id 0 --output-dir my_export
"""

import argparse
import json
import shutil
import struct
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from tau2.data_model.message import ToolCall
from tau2.data_model.simulation import Results, SimulationRun
from tau2.utils.tools import to_functional_format

TEMPLATE_DIR = Path(__file__).parent / "templates"
DEFAULT_TICK_DURATION_S = 0.2
WAVEFORM_POINTS = 1200


def _extract_waveform(wav_path: Path, num_points: int = WAVEFORM_POINTS) -> dict:
    """Read a WAV file and compute a downsampled waveform envelope.

    Returns min/max amplitude pairs per window for the classic bipolar
    waveform look, plus sample rate and duration.
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception:
        return {}

    if sampwidth == 2:
        fmt = f"<{n_frames * n_channels}h"
        samples = np.array(struct.unpack(fmt, raw), dtype=np.float64)
    elif sampwidth == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0
        samples *= 256.0
    else:
        return {}

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    duration = len(samples) / sample_rate
    if len(samples) == 0:
        return {}

    # Normalize to [-1, 1]
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples /= peak

    window = max(1, len(samples) // num_points)
    n_windows = len(samples) // window
    truncated = samples[: n_windows * window].reshape(n_windows, window)
    mins = truncated.min(axis=1)
    maxs = truncated.max(axis=1)

    return {
        "sampleRate": sample_rate,
        "duration": round(duration, 4),
        "mins": [round(float(v), 3) for v in mins],
        "maxs": [round(float(v), 3) for v in maxs],
    }


def escape_html(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br>")
    )


def _extract_speech_segments(
    sim: SimulationRun, tick_dur: float
) -> dict[str, list[dict]]:
    """Extract contiguous speech segments from ticks for timeline view."""
    user_segments: list[dict] = []
    agent_segments: list[dict] = []
    user_start: Optional[float] = None
    user_text_parts: list[str] = []
    agent_start: Optional[float] = None
    agent_text_parts: list[str] = []

    if not sim.ticks:
        return {"user": [], "agent": []}

    for t in sim.ticks:
        tick_time = t.tick_id * tick_dur
        user_speaking = (
            t.user_chunk and t.user_chunk.contains_speech and t.user_chunk.content
        )
        agent_speaking = (
            t.agent_chunk and t.agent_chunk.contains_speech and t.agent_chunk.content
        )

        if user_speaking:
            if user_start is None:
                user_start = tick_time
            user_text_parts.append(t.user_chunk.content)
        else:
            if user_start is not None:
                user_segments.append(
                    {
                        "start": round(user_start, 3),
                        "end": round(tick_time, 3),
                        "text": "".join(user_text_parts).strip(),
                    }
                )
                user_start = None
                user_text_parts = []

        if agent_speaking:
            if agent_start is None:
                agent_start = tick_time
            agent_text_parts.append(t.agent_chunk.content)
        else:
            if agent_start is not None:
                agent_segments.append(
                    {
                        "start": round(agent_start, 3),
                        "end": round(tick_time, 3),
                        "text": "".join(agent_text_parts).strip(),
                    }
                )
                agent_start = None
                agent_text_parts = []

    total_time = len(sim.ticks) * tick_dur
    if user_start is not None:
        user_segments.append(
            {
                "start": round(user_start, 3),
                "end": round(total_time, 3),
                "text": "".join(user_text_parts).strip(),
            }
        )
    if agent_start is not None:
        agent_segments.append(
            {
                "start": round(agent_start, 3),
                "end": round(total_time, 3),
                "text": "".join(agent_text_parts).strip(),
            }
        )

    return {"user": user_segments, "agent": agent_segments}


def _extract_effect_events(sim: SimulationRun) -> list[dict]:
    """Extract effect timeline events."""
    if not sim.effect_timeline:
        return []
    events = []
    for e in sim.effect_timeline.events:
        events.append(
            {
                "type": e.effect_type,
                "start_ms": e.start_ms,
                "end_ms": e.end_ms,
                "participant": e.participant,
                "params": e.params,
            }
        )
    return events


def _get_overlapping_effects(effect_timeline, start_ms: int, end_ms: int) -> list[dict]:
    """Return effect events that overlap the given time range."""
    if not effect_timeline:
        return []
    return [
        {
            "type": e.effect_type,
            "startMs": e.start_ms,
            "endMs": e.end_ms,
            "participant": e.participant,
            "params": e.params,
        }
        for e in effect_timeline.events
        if e.start_ms < end_ms and (e.end_ms or float("inf")) > start_ms
    ]


def _extract_tick_groups(sim: SimulationRun, tick_dur: float) -> list[dict]:
    """Extract grouped tick rows for the conversation table view.

    Uses the same gap-tolerant grouping as the annotation tool.
    Includes overlapping effect events per row.
    """
    if not sim.ticks:
        return []

    def get_pattern(tick) -> Optional[str]:
        if tick.agent_tool_calls or tick.agent_tool_results:
            return "__tool__"
        if tick.user_tool_calls or tick.user_tool_results:
            return "__tool__"
        tta = None
        if tick.user_chunk and tick.user_chunk.turn_taking_action:
            tta = tick.user_chunk.turn_taking_action.action
        elif tick.agent_chunk and tick.agent_chunk.turn_taking_action:
            tta = tick.agent_chunk.turn_taking_action.action
        if tta:
            norm = tta.split(":")[0].strip().lower()
            if norm in ("generate_message", "keep_talking"):
                return "active_speech"
            return norm
        has_agent = bool(
            tick.agent_chunk
            and tick.agent_chunk.content
            and tick.agent_chunk.contains_speech
        )
        has_user = bool(
            tick.user_chunk
            and tick.user_chunk.content
            and tick.user_chunk.contains_speech
        )
        if not has_agent and not has_user:
            return None
        return "active_speech"

    groups: list[dict] = []
    i = 0
    ticks = sim.ticks

    while i < len(ticks):
        tick = ticks[i]
        pattern = get_pattern(tick)

        start_tick = tick.tick_id
        group_ticks = [tick]

        if pattern == "__tool__":
            groups.append(
                {"start": start_tick, "end": start_tick, "ticks": group_ticks}
            )
            i += 1
            continue

        last_pattern = pattern
        j = i + 1
        while j < len(ticks):
            next_tick = ticks[j]
            np = get_pattern(next_tick)
            if np == "__tool__":
                break
            if np is None:
                group_ticks.append(next_tick)
                j += 1
                continue
            if last_pattern is None:
                last_pattern = np
                group_ticks.append(next_tick)
                j += 1
                continue
            if np != last_pattern:
                break
            group_ticks.append(next_tick)
            j += 1

        end_tick = ticks[j - 1].tick_id
        groups.append({"start": start_tick, "end": end_tick, "ticks": group_ticks})
        i = j

    rows: list[dict] = []
    for g in groups:
        agent_text = ""
        user_text = ""
        agent_calls = []
        agent_results = []
        user_calls = []
        user_results = []
        tta_info = None

        for tick in g["ticks"]:
            if tick.agent_chunk and tick.agent_chunk.content:
                agent_text += tick.agent_chunk.content
            if tick.user_chunk and tick.user_chunk.content:
                user_text += tick.user_chunk.content
            agent_calls.extend(tick.agent_tool_calls)
            agent_results.extend(tick.agent_tool_results)
            user_calls.extend(tick.user_tool_calls)
            user_results.extend(tick.user_tool_results)
            if not tta_info:
                if tick.user_chunk and tick.user_chunk.turn_taking_action:
                    tta = tick.user_chunk.turn_taking_action
                    tta_info = f"{tta.action}: {tta.info}" if tta.info else tta.action
                elif tick.agent_chunk and tick.agent_chunk.turn_taking_action:
                    tta = tick.agent_chunk.turn_taking_action
                    tta_info = f"{tta.action}: {tta.info}" if tta.info else tta.action

        if not any(
            [
                agent_text,
                user_text,
                agent_calls,
                agent_results,
                user_calls,
                user_results,
            ]
        ):
            continue

        def format_calls(calls: list[ToolCall]) -> list[str]:
            return [to_functional_format(tc) for tc in calls]

        def format_results(results) -> list[str]:
            out = []
            for tr in results:
                try:
                    parsed = json.loads(tr.content)
                    out.append(json.dumps(parsed, indent=2)[:500])
                except (json.JSONDecodeError, TypeError):
                    out.append(str(tr.content)[:500])
            return out

        time_s = round(g["start"] * tick_dur, 3)
        tick_dur_ms = int(tick_dur * 1000)
        row_start_ms = g["start"] * tick_dur_ms
        row_end_ms = (g["end"] + 1) * tick_dur_ms
        effects = _get_overlapping_effects(
            sim.effect_timeline, row_start_ms, row_end_ms
        )
        rows.append(
            {
                "tickStart": g["start"],
                "tickEnd": g["end"],
                "timeS": time_s,
                "agentText": agent_text.strip(),
                "userText": user_text.strip(),
                "agentCalls": format_calls(agent_calls),
                "agentResults": format_results(agent_results),
                "userCalls": format_calls(user_calls),
                "userResults": format_results(user_results),
                "effects": effects,
            }
        )

    return rows


def _extract_task_info(results: Results, task_id: str) -> dict:
    """Extract task info for display."""
    for task in results.tasks:
        if str(task.id) == str(task_id):
            scenario = task.user_scenario
            if scenario and scenario.instructions:
                instr = scenario.instructions
                if isinstance(instr, dict):
                    return {
                        "reason": instr.get("reason_for_call", ""),
                        "knownInfo": instr.get("known_info", ""),
                        "unknownInfo": instr.get("unknown_info", ""),
                        "taskInstructions": instr.get("task_instructions", ""),
                    }
                else:
                    return {
                        "reason": str(instr),
                        "knownInfo": "",
                        "unknownInfo": "",
                        "taskInstructions": "",
                    }
    return {}


def _build_sim_data(sim: SimulationRun, results: Results, tick_dur: float) -> dict:
    """Build the JSON data payload for one simulation."""
    total_duration = round(len(sim.ticks) * tick_dur, 3) if sim.ticks else sim.duration

    speech = _extract_speech_segments(sim, tick_dur)
    effects = _extract_effect_events(sim)
    tick_rows = _extract_tick_groups(sim, tick_dur)
    task_info = _extract_task_info(results, sim.task_id)

    # Speech environment info
    env_info: dict = {}
    if sim.speech_environment:
        se = sim.speech_environment
        env_info = {
            "complexity": str(se.complexity) if se.complexity else "unknown",
            "backgroundNoise": se.background_noise_file or "",
            "environment": se.environment or "",
            "personaName": se.persona_name or "",
            "telephonyEnabled": getattr(se, "telephony_enabled", False),
        }

    # Reward info
    reward = None
    reward_class = "unknown"
    if sim.reward_info:
        reward = sim.reward_info.reward
        reward_class = "success" if reward > 0 else "failure"

    return {
        "simId": sim.id,
        "taskId": str(sim.task_id),
        "trial": sim.trial or 0,
        "mode": sim.mode,
        "duration": round(sim.duration, 2),
        "totalDuration": total_duration,
        "terminationReason": sim.termination_reason.value
        if hasattr(sim.termination_reason, "value")
        else str(sim.termination_reason),
        "reward": reward,
        "rewardClass": reward_class,
        "tickDuration": tick_dur,
        "numTicks": len(sim.ticks) if sim.ticks else 0,
        "domain": results.info.environment_info.domain_name,
        "agentModel": results.info.agent_info.llm or "",
        "agentProvider": (
            results.info.audio_native_config.provider
            if results.info.audio_native_config
            else ""
        ),
        "userModel": results.info.user_info.llm or "",
        "speechEnvironment": env_info,
        "taskInfo": task_info,
        "speech": speech,
        "effects": effects,
        "tickRows": tick_rows,
    }


def generate_viewer_html(sim_data: dict, audio_filename: Optional[str] = None) -> str:
    """Generate a standalone HTML viewer page."""
    css = (TEMPLATE_DIR / "viewer.css").read_text()
    js = (TEMPLATE_DIR / "viewer.js").read_text()

    data_json = json.dumps(sim_data, ensure_ascii=False)
    audio_src = audio_filename or ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Task {sim_data["taskId"]} — {sim_data["domain"]} — Simulation Viewer</title>
  <style>{css}</style>
</head>
<body>
  <div id="app"></div>
  <script>
    const SIM_DATA = {data_json};
    const AUDIO_SRC = '{audio_src}';
  </script>
  <script>{js}</script>
</body>
</html>
"""


AUDIO_TAP_CATALOG = [
    # User pipeline — individual component tracks (time-aligned)
    {
        "file": "user_speech-only.wav",
        "id": "user_speech_only",
        "label": "Speech",
        "group": "user",
        "participant": "user",
        "order": 0,
        "description": "Isolated user speech (with speech effects applied)",
    },
    {
        "file": "user_background-noise-only.wav",
        "id": "user_bg_noise",
        "label": "Background Noise",
        "group": "user",
        "participant": "user",
        "order": 1,
        "description": "Background noise track (time-aligned with speech)",
    },
    {
        "file": "user_burst-noise-only.wav",
        "id": "user_burst_noise",
        "label": "Burst Noise",
        "group": "user",
        "participant": "user",
        "order": 2,
        "description": "Burst noise events (car horns, sirens, etc.)",
    },
    {
        "file": "user_out-of-turn-speech-only.wav",
        "id": "user_oot_speech",
        "label": "Out-of-Turn Speech",
        "group": "user",
        "participant": "user",
        "order": 3,
        "description": "Out-of-turn speech (vocal tics, bystander phrases)",
    },
    # User pipeline — mixed/processed stages
    {
        "file": "user_post-noise.wav",
        "id": "user_post_noise",
        "label": "Post-Noise Mix",
        "group": "user",
        "participant": "user",
        "order": 4,
        "description": "Speech + background noise + burst noise + out-of-turn speech mixed",
    },
    {
        "file": "user_post-telephony.wav",
        "id": "user_post_telephony",
        "label": "Post-Telephony",
        "group": "user",
        "participant": "user",
        "order": 5,
        "description": "After telephony conversion (μ-law 8 kHz)",
    },
    {
        "file": "user_output.wav",
        "id": "user_output",
        "label": "User Output",
        "group": "user",
        "participant": "user",
        "order": 6,
        "description": "Final user output after frame drops (sent to the agent)",
    },
    # Agent
    {
        "file": "agent_output.wav",
        "id": "agent_output",
        "label": "Agent Output",
        "group": "agent",
        "participant": "agent",
        "order": 7,
        "description": "Raw output from the agent model",
    },
]


def _discover_audio_taps(
    results_dir: Path, sim: SimulationRun, subdir: Path
) -> list[dict]:
    """Find audio tap WAV files, copy them to output dir, return metadata."""
    taps_search_paths = [
        results_dir
        / "artifacts"
        / f"task_{sim.task_id}"
        / f"sim_{sim.id}"
        / "audio"
        / "taps",
    ]

    taps_dir = None
    for p in taps_search_paths:
        if p.is_dir():
            taps_dir = p
            break

    if not taps_dir:
        return []

    taps_out = subdir / "taps"
    taps_out.mkdir(parents=True, exist_ok=True)

    found: list[dict] = []
    for entry in AUDIO_TAP_CATALOG:
        src = taps_dir / entry["file"]
        if src.exists():
            dst_name = entry["file"]
            shutil.copy(src, taps_out / dst_name)
            waveform = _extract_waveform(src)
            found.append(
                {
                    "id": entry["id"],
                    "label": entry["label"],
                    "group": entry["group"],
                    "participant": entry["participant"],
                    "order": entry["order"],
                    "description": entry["description"],
                    "src": f"taps/{dst_name}",
                    "waveform": waveform,
                }
            )

    found.sort(key=lambda x: x["order"])
    return found


def export_simulation(
    sim: SimulationRun,
    results: Results,
    output_dir: Path,
    results_dir: Path,
    tick_dur: float = DEFAULT_TICK_DURATION_S,
    flat: bool = False,
) -> Path:
    """Export a single simulation to an interactive HTML viewer."""
    sim_data = _build_sim_data(sim, results, tick_dur)

    dest_dir = (
        output_dir if flat else output_dir / f"task_{sim.task_id}_sim_{sim.id[:8]}"
    )
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy audio if available
    audio_filename = None
    audio_search_paths = [
        results_dir
        / "artifacts"
        / f"task_{sim.task_id}"
        / f"sim_{sim.id}"
        / "audio"
        / "both.wav",
        results_dir
        / "artifacts"
        / f"task_{sim.task_id}"
        / f"sim_{sim.id}"
        / "both.wav",
    ]
    for audio_file in audio_search_paths:
        if audio_file.exists():
            audio_filename = "audio.wav"
            shutil.copy(audio_file, dest_dir / audio_filename)
            break

    # Discover and copy audio taps
    audio_taps = _discover_audio_taps(results_dir, sim, dest_dir)
    sim_data["audioTaps"] = audio_taps

    html = generate_viewer_html(sim_data, audio_filename)
    html_file = dest_dir / "index.html"
    html_file.write_text(html)

    return html_file


DESIGNER_README = """tau-voice Simulation Viewer
=================================

Open index.html in your web browser (Chrome, Firefox, Safari, or Edge).

No server or install required — everything runs locally.

What you can do:
- Overview: Timeline + conversation, synced with audio. Click "▶ Audio Tracks" to expand waveform view.
- Timeline: Speech activity (user/agent) over time
- Conversation: Tick-by-tick transcript
- Audio Tracks: Multitrack mixer (waveforms, mute/solo, effects)

Playback: Use the sticky player at the bottom, or Space / Arrow keys.
"""


def write_designer_readme(output_dir: Path) -> None:
    """Write a simple README for the designer package."""
    (output_dir / "README.txt").write_text(DESIGNER_README)


def main():
    parser = argparse.ArgumentParser(
        description="Export simulation data to interactive HTML viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to results.json file",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Export only this task ID (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: alongside results.json as viewer/)",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Create a flat, designer-ready package: index.html at root + README. Use with 1 simulation.",
    )

    args = parser.parse_args()

    results_path = args.results.resolve()
    if results_path.is_dir():
        results_path = results_path / "results.json"

    if not results_path.exists():
        print(f"Error: {results_path} not found")
        raise SystemExit(1)

    results_dir = results_path.parent
    output_dir = args.output_dir or (results_dir / "viewer")

    print(f"Loading results from {results_path}...")
    results = Results.load(results_path)

    tick_dur = DEFAULT_TICK_DURATION_S
    if results.info.audio_native_config:
        tick_dur = results.info.audio_native_config.tick_duration_seconds

    sims = results.simulations
    if args.task_id:
        sims = [s for s in sims if str(s.task_id) == args.task_id]

    if not sims:
        print("No simulations to export.")
        return

    # Standalone = flat layout (index.html at root) only for a single simulation
    flat = args.standalone and len(sims) == 1
    if args.standalone and len(sims) > 1:
        print(
            "Note: --standalone with multiple sims uses subdirs; README added at root."
        )

    # Use a designer-friendly dir name when standalone
    if args.standalone and args.output_dir is None:
        output_dir = results_dir / "tau_voice_viewer_demo"
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {len(sims)} simulation(s)...")
    for sim in sims:
        html_file = export_simulation(
            sim, results, output_dir, results_dir, tick_dur, flat=flat
        )
        print(f"  Task {sim.task_id} / sim {sim.id[:8]} -> {html_file}")

    if args.standalone:
        write_designer_readme(output_dir)
        print(f"\nDesigner package ready: {output_dir}")
        print("Zip this folder and share. Open index.html in a browser.")
    else:
        print(f"\nDone! Open the HTML files in a browser.")


if __name__ == "__main__":
    main()
