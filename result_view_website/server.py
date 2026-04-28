#!/usr/bin/env python3
"""Local web viewer for tau2 simulation results.

This is a browser-based companion to ``tau2 view``. It intentionally reuses the
same on-disk result models instead of defining a second results format.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tau2.data_model.message import Message, Tick  # noqa: E402
from tau2.data_model.simulation import Results, SimulationRun  # noqa: E402
from tau2.data_model.tasks import Task  # noqa: E402
from tau2.metrics.agent_metrics import is_successful  # noqa: E402
from tau2.utils.utils import DATA_DIR  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
SIMULATIONS_DIR = "simulations"


class ViewerState:
    """Mutable server state configured by CLI flags."""

    def __init__(self, results_root: Path):
        self.results_root = results_root.resolve()


app = FastAPI(title="tau2 Result Viewer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.state.viewer = ViewerState(Path(DATA_DIR) / "simulations")


@app.get("/")
def index() -> FileResponse:
    """Serve the single-page viewer shell."""
    return FileResponse(STATIC_DIR / "index.html")


def _normalize_results_path(path_value: Optional[str] = None) -> Path:
    """Resolve a requested results path.

    The viewer is designed for local inspection, so it accepts either absolute
    or repo-relative paths. Missing values use the configured CLI root.
    """
    if path_value:
        requested = Path(path_value)
        if not requested.is_absolute():
            requested = (ROOT / requested).resolve()
        else:
            requested = requested.resolve()
        return requested
    return app.state.viewer.results_root


def _metadata_path(path: Path) -> Path:
    """Return the concrete results JSON file for a run path."""
    if path.is_file():
        return path
    reviewed = path / "results_reviewed.json"
    if reviewed.exists():
        return reviewed
    return path / "results.json"


def _run_dir(path: Path) -> Path:
    """Return the run directory for a file or directory path."""
    return path.parent if path.is_file() else path


def _find_available_runs(root: Path) -> list[Path]:
    """Find viewable results files under *root*."""
    if root.is_file():
        return [root]
    if (root / "results_reviewed.json").exists() or (root / "results.json").exists():
        return [_metadata_path(root)]
    if not root.exists():
        return []

    runs = []
    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue
        reviewed = subdir / "results_reviewed.json"
        results = subdir / "results.json"
        if reviewed.exists():
            runs.append(reviewed)
        elif results.exists():
            runs.append(results)
    return sorted(runs)


def _path_key(path: Path) -> str:
    """Stable cache key for path + mtime."""
    meta = _metadata_path(path)
    if not meta.exists():
        return f"{meta}:missing"
    return f"{meta.resolve()}:{meta.stat().st_mtime_ns}"


@lru_cache(maxsize=8)
def _load_results_cached(cache_key: str, path_str: str) -> Results:
    """Load results with a small mtime-aware cache."""
    _ = cache_key
    return Results.load(Path(path_str))


def _load_results(path: Path) -> Results:
    meta = _metadata_path(path)
    if not meta.exists():
        raise HTTPException(status_code=404, detail=f"Results not found: {meta}")
    return _load_results_cached(_path_key(path), str(meta))


def _task_by_id(results: Results) -> dict[str, Task]:
    return {str(task.id): task for task in results.tasks}


def _short_text(value: Any, limit: int = 160) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _format_time_ms(ms: int) -> str:
    minutes = ms // 60000
    remaining_ms = ms % 60000
    seconds = remaining_ms // 1000
    milliseconds = remaining_ms % 1000
    return f"{minutes}:{seconds:02d}:{milliseconds:03d}"


def _audio_for_sim(results_path: Path, sim: SimulationRun) -> Optional[Path]:
    """Locate the mixed conversation audio for a simulation if it exists."""
    base = _run_dir(_metadata_path(results_path))
    candidates = [
        base
        / "artifacts"
        / f"task_{sim.task_id}"
        / f"sim_{sim.id}"
        / "audio"
        / "both.wav",
        base / "artifacts" / str(sim.task_id) / f"sim_{sim.id}" / "audio" / "both.wav",
        base / "audio" / "both.wav",
        base / "voice" / f"sim_{sim.id}" / "conversation.wav",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in base.glob(f"artifacts/*/sim_{sim.id}/audio/both.wav"):
        if candidate.exists():
            return candidate
    for candidate in base.glob("voice/sim_*/conversation.wav"):
        if candidate.exists():
            return candidate
    return None


def _message_dump(message: Message) -> dict[str, Any]:
    """Compact message JSON suitable for display."""
    data = message.model_dump(
        mode="json",
        exclude={"audio_content", "raw_data"},
        exclude_none=True,
    )
    data["message_type"] = message.__class__.__name__
    if not data.get("content") and data.get("audio_script_gold"):
        data["content"] = data["audio_script_gold"]
    if not data.get("content") and data.get("is_audio"):
        data["content"] = "[audio chunk]"
    if not data.get("content") and data.get("tool_calls"):
        data["content"] = "[tool call]"
    return data


def _show_message_in_conversation(message: Message) -> bool:
    """Keep the conversation tab readable for expanded audio-native runs."""
    data = message.model_dump(
        mode="json",
        exclude={"audio_content", "raw_data"},
        exclude_none=True,
    )
    if data.get("tool_calls") or data.get("content") or data.get("audio_script_gold"):
        return True
    return data.get("is_audio") is not True


def _chunk_dump(chunk: Any) -> Optional[dict[str, Any]]:
    if chunk is None:
        return None
    data = chunk.model_dump(
        mode="json",
        exclude={"audio_content", "raw_data"},
        exclude_none=True,
    )
    data["preview"] = _short_text(data.get("content") or data.get("tool_calls") or "")
    return data


def _tool_call_dump(tool_call: Any) -> dict[str, Any]:
    return tool_call.model_dump(mode="json", exclude_none=True)


def _tool_message_dump(tool_message: Any) -> dict[str, Any]:
    return tool_message.model_dump(mode="json", exclude_none=True)


def _tool_call_text(tool_call: Any) -> str:
    payload = tool_call.model_dump(mode="json", exclude_none=True)
    name = payload.get("name", "tool")
    arguments = payload.get("arguments", {})
    return f"{name}({json.dumps(arguments, ensure_ascii=False)})"


def _tool_result_text(tool_message: Any) -> str:
    payload = tool_message.model_dump(mode="json", exclude_none=True)
    content = payload.get("content")
    if content is None:
        return json.dumps(payload, ensure_ascii=False)
    return str(content)


def _turn_action_text(chunk: Any) -> str:
    if not chunk or not hasattr(chunk, "turn_taking_action"):
        return ""
    action = chunk.turn_taking_action
    if not action:
        return ""
    info = getattr(action, "info", None)
    action_name = getattr(action, "action", "")
    return f"{action_name}: {info}" if info else action_name


def _tick_timeline_info(tick: Tick) -> dict[str, str]:
    agent_tools = []
    if tick.agent_tool_calls:
        agent_tools.extend(_tool_call_text(tc) for tc in tick.agent_tool_calls)
    if tick.agent_tool_results:
        agent_tools.extend(
            f"result: {_tool_result_text(msg)}" for msg in tick.agent_tool_results
        )

    user_tools = []
    if tick.user_tool_calls:
        user_tools.extend(_tool_call_text(tc) for tc in tick.user_tool_calls)
    if tick.user_tool_results:
        user_tools.extend(
            f"result: {_tool_result_text(msg)}" for msg in tick.user_tool_results
        )

    return {
        "agent_content": (
            tick.agent_chunk.content
            if tick.agent_chunk and tick.agent_chunk.content
            else ""
        ),
        "agent_tools": "\n".join(agent_tools),
        "agent_turn_action": _turn_action_text(tick.agent_chunk),
        "user_content": (
            tick.user_chunk.content
            if tick.user_chunk and tick.user_chunk.content
            else ""
        ),
        "user_transcript": tick.user_transcript or "",
        "user_tools": "\n".join(user_tools),
        "user_turn_action": _turn_action_text(tick.user_chunk),
    }


def _normalize_group_action(action: str) -> str:
    action_name = action.split(":", 1)[0].strip().lower()
    if action_name in ("generate_message", "keep_talking"):
        return "active_speech"
    return action_name


def _grouping_pattern(info: dict[str, str]) -> Optional[str]:
    has_agent = bool(info.get("agent_content"))
    if info.get("agent_turn_action"):
        return _normalize_group_action(info["agent_turn_action"])
    if info.get("user_turn_action"):
        base = _normalize_group_action(info["user_turn_action"])
        return f"{base}+agent" if has_agent else base
    if (
        not has_agent
        and not info.get("user_content")
        and not info.get("user_transcript")
    ):
        return None
    return "active_speech"


def _has_tool_activity(info: dict[str, str]) -> bool:
    return bool(info.get("agent_tools") or info.get("user_tools"))


def _group_ticks_for_timeline(ticks: list[Tick]) -> list[tuple[int, int, list[dict]]]:
    groups = []
    i = 0
    while i < len(ticks):
        tick = ticks[i]
        info = _tick_timeline_info(tick)
        start_tick = tick.tick_id
        group_infos = [info]

        if _has_tool_activity(info):
            groups.append((start_tick, start_tick, group_infos))
            i += 1
            continue

        last_pattern = _grouping_pattern(info)
        j = i + 1
        while j < len(ticks):
            next_info = _tick_timeline_info(ticks[j])
            if _has_tool_activity(next_info):
                break

            next_pattern = _grouping_pattern(next_info)
            if next_pattern is None:
                group_infos.append(next_info)
                j += 1
                continue
            if last_pattern is None:
                last_pattern = next_pattern
                group_infos.append(next_info)
                j += 1
                continue
            if next_pattern == last_pattern:
                group_infos.append(next_info)
                j += 1
                continue
            if (
                last_pattern.endswith("+agent")
                and next_pattern == last_pattern.removesuffix("+agent")
                and j + 1 < len(ticks)
                and _grouping_pattern(_tick_timeline_info(ticks[j + 1])) == last_pattern
            ):
                group_infos.append(next_info)
                j += 1
                continue
            break

        groups.append((start_tick, ticks[j - 1].tick_id, group_infos))
        i = j

    return groups


def _consolidate_actions(actions: list[str]) -> str:
    if not actions:
        return ""
    grouped = []
    current = actions[0]
    count = 1
    for action in actions[1:]:
        if action == current:
            count += 1
            continue
        grouped.append(f"{current} (x{count})" if count > 1 else current)
        current = action
        count = 1
    grouped.append(f"{current} (x{count})" if count > 1 else current)
    return "\n".join(grouped)


def _timeline_rows(sim: SimulationRun) -> list[dict[str, Any]]:
    ticks = sim.ticks or []
    tick_duration_ms = None
    if ticks and ticks[0].tick_duration_seconds is not None:
        tick_duration_ms = int(ticks[0].tick_duration_seconds * 1000)

    rows = []
    for start_tick, end_tick, infos in _group_ticks_for_timeline(ticks):
        tick_label = (
            str(start_tick) if start_tick == end_tick else f"{start_tick}-{end_tick}"
        )
        time_label = ""
        start_seconds = None
        end_seconds = None
        if tick_duration_ms is not None:
            start_time = start_tick * tick_duration_ms
            end_time = end_tick * tick_duration_ms
            start_seconds = start_time / 1000
            end_seconds = end_time / 1000
            time_label = (
                _format_time_ms(start_time)
                if start_tick == end_tick
                else f"{_format_time_ms(start_time)}\n{_format_time_ms(end_time)}"
            )

        agent_content = "".join(
            info["agent_content"] for info in infos if info["agent_content"]
        )
        user_content = "".join(
            info["user_content"] for info in infos if info["user_content"]
        )
        user_transcript = "".join(
            info["user_transcript"] for info in infos if info["user_transcript"]
        )
        agent_tools = "\n".join(
            info["agent_tools"] for info in infos if info["agent_tools"]
        )
        user_tools = "\n".join(
            info["user_tools"] for info in infos if info["user_tools"]
        )
        agent_actions = [
            info["agent_turn_action"] for info in infos if info["agent_turn_action"]
        ]
        user_actions = [
            info["user_turn_action"] for info in infos if info["user_turn_action"]
        ]

        rows.append(
            {
                "ticks": tick_label,
                "time": time_label,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "agent": agent_content,
                "user": user_content,
                "user_transcript": user_transcript,
                "agent_tools": agent_tools,
                "user_tools": user_tools,
                "tools": "\n".join(
                    item
                    for item in (
                        f"agent: {agent_tools}" if agent_tools else "",
                        f"user: {user_tools}" if user_tools else "",
                    )
                    if item
                ),
                "agent_turn": _consolidate_actions(agent_actions),
                "user_turn": _consolidate_actions(user_actions),
            }
        )
    return rows


def _tick_dump(tick: Tick) -> dict[str, Any]:
    return {
        "tick_id": tick.tick_id,
        "timestamp": tick.timestamp,
        "agent_chunk": _chunk_dump(tick.agent_chunk),
        "user_chunk": _chunk_dump(tick.user_chunk),
        "agent_tool_calls": [_tool_call_dump(tc) for tc in tick.agent_tool_calls],
        "user_tool_calls": [_tool_call_dump(tc) for tc in tick.user_tool_calls],
        "agent_tool_results": [
            _tool_message_dump(msg) for msg in tick.agent_tool_results
        ],
        "user_tool_results": [
            _tool_message_dump(msg) for msg in tick.user_tool_results
        ],
        "user_transcript": tick.user_transcript,
        "tick_duration_seconds": tick.tick_duration_seconds,
        "wall_clock_duration_seconds": tick.wall_clock_duration_seconds,
    }


def _review_counts(sim: SimulationRun) -> dict[str, int]:
    agent_errors = 0
    user_errors = 0
    if sim.review is not None:
        for error in sim.review.errors:
            if error.source == "agent":
                agent_errors += 1
            elif error.source == "user":
                user_errors += 1
    elif sim.user_only_review is not None:
        user_errors = len(sim.user_only_review.errors)
    return {"agent_errors": agent_errors, "user_errors": user_errors}


def _simulation_summary(
    sim: SimulationRun, index: int, results_path: Path
) -> dict[str, Any]:
    reward = sim.reward_info.reward if sim.reward_info else None
    audio_path = _audio_for_sim(results_path, sim)
    review_counts = _review_counts(sim)
    return {
        "index": index,
        "id": sim.id,
        "task_id": sim.task_id,
        "trial": sim.trial,
        "seed": sim.seed,
        "reward": reward,
        "success": is_successful(reward) if reward is not None else None,
        "termination_reason": sim.termination_reason,
        "duration": sim.duration,
        "mode": sim.mode,
        "message_count": len(sim.get_messages()),
        "tick_count": len(sim.ticks or []),
        "agent_cost": sim.agent_cost,
        "user_cost": sim.user_cost,
        "has_audio": audio_path is not None,
        "audio_url": f"/api/audio/{sim.id}" if audio_path is not None else None,
        **review_counts,
    }


def _results_payload(results_path: Path) -> dict[str, Any]:
    results = _load_results(results_path)
    summaries = [
        _simulation_summary(sim, i + 1, results_path)
        for i, sim in enumerate(results.simulations)
    ]
    successes = [s for s in summaries if s["success"] is True]
    failures = [s for s in summaries if s["success"] is False]
    info = results.info.model_dump(mode="json", exclude_none=True)
    return {
        "path": str(_metadata_path(results_path)),
        "run_dir": str(_run_dir(_metadata_path(results_path))),
        "timestamp": results.timestamp,
        "info": info,
        "tasks_count": len(results.tasks),
        "simulations_count": len(results.simulations),
        "success_count": len(successes),
        "failure_count": len(failures),
        "unknown_count": len(summaries) - len(successes) - len(failures),
        "simulations": summaries,
    }


def _simulation_payload(results_path: Path, sim_id: str) -> dict[str, Any]:
    results = _load_results(results_path)
    tasks = _task_by_id(results)
    for index, sim in enumerate(results.simulations, 1):
        if sim.id != sim_id:
            continue
        task = tasks.get(str(sim.task_id))
        audio_path = _audio_for_sim(results_path, sim)
        return {
            "summary": _simulation_summary(sim, index, results_path),
            "task": task.model_dump(mode="json", exclude_none=True) if task else None,
            "messages": [
                _message_dump(msg)
                for msg in sim.get_messages()
                if _show_message_in_conversation(msg)
            ],
            "timeline": _timeline_rows(sim),
            "ticks": [_tick_dump(tick) for tick in (sim.ticks or [])],
            "reward_info": (
                sim.reward_info.model_dump(mode="json", exclude_none=True)
                if sim.reward_info
                else None
            ),
            "auth_classification": (
                sim.auth_classification.model_dump(mode="json", exclude_none=True)
                if sim.auth_classification
                else None
            ),
            "review": (
                sim.review.model_dump(mode="json", exclude_none=True)
                if sim.review
                else None
            ),
            "user_only_review": (
                sim.user_only_review.model_dump(mode="json", exclude_none=True)
                if sim.user_only_review
                else None
            ),
            "hallucination_check": (
                sim.hallucination_check.model_dump(mode="json", exclude_none=True)
                if sim.hallucination_check
                else None
            ),
            "speech_environment": (
                sim.speech_environment.model_dump(mode="json", exclude_none=True)
                if sim.speech_environment
                else None
            ),
            "effect_timeline": (
                sim.effect_timeline.model_dump(mode="json", exclude_none=True)
                if sim.effect_timeline
                else None
            ),
            "info": sim.info,
            "policy": sim.policy,
            "audio_url": f"/api/audio/{sim.id}" if audio_path is not None else None,
        }
    raise HTTPException(status_code=404, detail=f"Simulation not found: {sim_id}")


@app.get("/api/state")
def state() -> dict[str, Any]:
    runs = _find_available_runs(app.state.viewer.results_root)
    return {
        "root": str(app.state.viewer.results_root),
        "runs": [{"name": run.parent.name, "path": str(run)} for run in runs],
        "default_path": str(runs[-1]) if runs else None,
    }


@app.get("/api/runs")
def runs() -> dict[str, Any]:
    runs = _find_available_runs(app.state.viewer.results_root)
    return {"runs": [{"name": run.parent.name, "path": str(run)} for run in runs]}


@app.get("/api/results")
def results(path: Optional[str] = Query(default=None)) -> dict[str, Any]:
    return _results_payload(_normalize_results_path(path))


@app.get("/api/simulations/{sim_id}")
def simulation(
    sim_id: str, path: Optional[str] = Query(default=None)
) -> dict[str, Any]:
    return _simulation_payload(_normalize_results_path(path), sim_id)


@app.get("/api/audio/{sim_id}")
def audio(sim_id: str, path: Optional[str] = Query(default=None)) -> FileResponse:
    results_path = _normalize_results_path(path)
    results = _load_results(results_path)
    sim = next((s for s in results.simulations if s.id == sim_id), None)
    if sim is None:
        raise HTTPException(status_code=404, detail=f"Simulation not found: {sim_id}")
    audio_path = _audio_for_sim(results_path, sim)
    if audio_path is None:
        raise HTTPException(status_code=404, detail=f"Audio not found for {sim_id}")
    return FileResponse(audio_path, media_type="audio/wav", filename=audio_path.name)


@app.get("/api/raw/{sim_id}")
def raw_simulation(
    sim_id: str, path: Optional[str] = Query(default=None)
) -> dict[str, Any]:
    results = _load_results(_normalize_results_path(path))
    for sim in results.simulations:
        if sim.id == sim_id:
            return sim.model_dump(mode="json", exclude_none=True)
    raise HTTPException(status_code=404, detail=f"Simulation not found: {sim_id}")


@lru_cache(maxsize=16)
def _load_tool_defs(
    domain_name: str,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """Return (agent_tools, user_tools) for a domain, cached per domain name."""
    try:
        from tau2.runner.helpers import get_environment_info

        env_info = get_environment_info(domain_name, include_tool_info=True)
        agent_tools = (
            {name: sig.model_dump() for name, sig in env_info.tool_defs.items()}
            if env_info.tool_defs
            else None
        )
        user_tool_defs = getattr(env_info, "user_tool_defs", None)
        user_tools = (
            {name: sig.model_dump() for name, sig in user_tool_defs.items()}
            if user_tool_defs
            else None
        )
        return agent_tools, user_tools
    except Exception:
        return None, None


def _prompts_payload(results_path: Path, sim_id: str) -> dict[str, Any]:
    results = _load_results(results_path)
    info = results.info
    tasks = _task_by_id(results)

    sim: Optional[SimulationRun] = None
    for s in results.simulations:
        if s.id == sim_id:
            sim = s
            break
    if sim is None:
        raise HTTPException(status_code=404, detail=f"Simulation not found: {sim_id}")

    task = tasks.get(str(sim.task_id))

    # Agent prompt components
    agent_policy = sim.policy or info.environment_info.policy
    agent_instruction: Optional[str] = None
    agent_impl = info.agent_info.implementation
    if agent_impl == "discrete_time_audio_native_agent":
        from tau2.agent.discrete_time_audio_native_agent import (  # noqa: F401
            AUDIO_NATIVE_VOICE_INSTRUCTION,
        )
        agent_instruction = AUDIO_NATIVE_VOICE_INSTRUCTION
    elif agent_impl in ("llm_agent", "llm_agent_gt", "llm_agent_solo"):
        from tau2.agent.llm_agent import AGENT_INSTRUCTION  # noqa: F401
        agent_instruction = AGENT_INSTRUCTION

    # Tool definitions — load from domain (cached per domain name)
    agent_tools, user_tools = _load_tool_defs(info.environment_info.domain_name)

    # User prompt components
    user_guidelines = info.user_info.global_simulation_guidelines
    user_scenario = (
        task.model_dump(mode="json", exclude_none=True).get("user_scenario")
        if task
        else None
    )

    return {
        "agent_instruction": agent_instruction,
        "agent_policy": agent_policy,
        "agent_tools": agent_tools,
        "user_tools": user_tools,
        "user_guidelines": user_guidelines,
        "user_scenario": user_scenario,
    }


@app.get("/api/prompts/{sim_id}")
def prompts(
    sim_id: str, path: Optional[str] = Query(default=None)
) -> dict[str, Any]:
    return _prompts_payload(_normalize_results_path(path), sim_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(DATA_DIR) / "simulations",
        help=(
            "A results.json file, a single run directory, or a directory containing "
            "multiple simulation run directories. Defaults to data/simulations."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app.state.viewer = ViewerState(args.results)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
