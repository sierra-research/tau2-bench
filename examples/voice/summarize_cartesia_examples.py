"""Export readable speech transcripts and an index for the Cartesia examples."""

import json
from pathlib import Path

from tau2.data_model.simulation import SimulationRun
from tau2.voice.synthesis.conversation_builder import _collect_speech_segments


def main():
    root = Path("data/simulations/cartesia-haiku-both-10")
    rows = []
    for path in sorted((root / "simulations").glob("*.json")):
        sim = SimulationRun.model_validate_json(path.read_text())
        segments = []
        for role in ("user", "assistant"):
            segments.extend(_collect_speech_segments(sim.ticks or [], role))
        segments.sort(key=lambda segment: segment.start_tick)
        transcript = root / f"task_{sim.task_id}_transcript.txt"
        lines = [f"Task {sim.task_id} | {sim.termination_reason.value}", ""]
        for segment in segments:
            name = "Customer" if segment.role == "user" else "Agent"
            lines.append(f"[{segment.start_tick * 0.2:.1f}s] {name}: {segment.text}")
        lines += [
            "",
            "Tool calls and results (full structured data is in the simulation JSON):",
        ]
        for tick in sim.ticks or []:
            for message in [*tick.agent_tool_calls, *tick.agent_tool_results]:
                lines.append(
                    f"[{tick.tick_id * 0.2:.1f}s] {message.model_dump_json(exclude_none=True)}"
                )
        transcript.write_text("\n\n".join(lines) + "\n")
        audio = list(
            (root / "artifacts" / f"task_{sim.task_id}").glob("*/audio/both.wav")
        )
        rows.append(
            {
                "task_id": sim.task_id,
                "termination_reason": sim.termination_reason.value,
                "reward": sim.reward_info.reward if sim.reward_info else None,
                "wall_seconds": sim.duration,
                "audio_seconds": len(sim.ticks or []) * 0.2,
                "transcript": str(transcript.relative_to(root)),
                "audio": [str(p.relative_to(root)) for p in audio],
                "simulation": str(path.relative_to(root)),
                "agent_cost": sim.agent_cost,
                "user_llm_cost": sim.user_cost,
            }
        )
    rows.sort(key=lambda row: int(row["task_id"]))
    (root / "example-index.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
