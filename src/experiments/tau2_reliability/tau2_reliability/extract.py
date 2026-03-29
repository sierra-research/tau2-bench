"""Extract TaskTrialData from tau2-bench Results objects."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

from loguru import logger
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import Results, SimulationRun, TerminationReason
from tau2.metrics.agent_metrics import is_successful

from tau2_reliability.models import TaskTrialData


def _extract_action_sequence(sim: SimulationRun) -> list[str]:
    """Extract the agent's tool call names from a simulation trajectory."""
    actions = []
    messages = sim.messages or []
    for msg in messages:
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.requestor == "assistant":
                    actions.append(tc.name)
    return actions


def _extract_tool_types(
    sim: SimulationRun, tool_type_map: Optional[dict[str, str]] = None
) -> list[str]:
    """Extract READ/WRITE classification for each agent action.

    If tool_type_map is not provided, defaults all actions to 'UNKNOWN'.
    """
    actions = _extract_action_sequence(sim)
    if tool_type_map is None:
        return ["UNKNOWN"] * len(actions)
    return [tool_type_map.get(a, "UNKNOWN") for a in actions]


def extract_task_trial_data(
    results: Results,
    tool_type_map: Optional[dict[str, str]] = None,
) -> list[TaskTrialData]:
    """Group simulations by task and extract per-trial data.

    Args:
        results: A tau2-bench Results object (loaded via Results.load()).
        tool_type_map: Optional mapping of tool_name -> 'READ'|'WRITE'.
            If provided, enables mutation-aware analysis.

    Returns:
        List of TaskTrialData, one per task, sorted by task_id.
    """
    groups: dict[str, list[SimulationRun]] = defaultdict(list)

    for sim in results.simulations:
        if sim.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR:
            continue
        groups[sim.task_id].append(sim)

    task_data_list = []
    for task_id in sorted(groups.keys()):
        sims = groups[task_id]
        # Sort by trial number for deterministic ordering
        sims.sort(key=lambda s: (s.trial or 0, s.seed or 0))

        outcomes = []
        action_sequences = []
        costs = []
        durations = []
        num_actions = []
        tool_types = []

        for sim in sims:
            reward = sim.reward_info.reward if sim.reward_info else 0.0
            outcomes.append(is_successful(reward))
            actions = _extract_action_sequence(sim)
            action_sequences.append(actions)
            costs.append(sim.agent_cost or 0.0)
            durations.append(sim.duration or 0.0)
            num_actions.append(len(actions))
            tool_types.append(_extract_tool_types(sim, tool_type_map))

        td = TaskTrialData(
            task_id=task_id,
            outcomes=outcomes,
            action_sequences=action_sequences,
            costs=costs,
            durations=durations,
            num_actions=num_actions,
            tool_types_per_action=tool_types,
        )
        task_data_list.append(td)

    if not task_data_list:
        logger.warning("No valid simulations found in results")
    else:
        trial_counts = [td.num_trials for td in task_data_list]
        logger.info(
            f"Extracted {len(task_data_list)} tasks, "
            f"trials per task: {min(trial_counts)}-{max(trial_counts)}"
        )

    return task_data_list


def build_tool_type_map(domain: str) -> dict[str, str]:
    """Build a tool_name -> 'READ'|'WRITE'|'GENERIC'|'THINK' map from domain toolkit.

    Uses tau2's registry to load the domain environment and inspect tool metadata.
    """
    try:
        from tau2.environment.toolkit import ToolType
        from tau2.registry import registry

        env_constructor = registry.get_env_constructor(domain)
        env = env_constructor()
        toolkit = env.tools
        tool_type_map = {}

        for name in toolkit.get_tools().keys():
            tt = toolkit.tool_type(name)
            if tt == ToolType.WRITE:
                tool_type_map[name] = "WRITE"
            elif tt == ToolType.READ:
                tool_type_map[name] = "READ"
            elif tt == ToolType.THINK:
                tool_type_map[name] = "THINK"
            else:
                tool_type_map[name] = "GENERIC"

        logger.info(f"Built tool type map for {domain}: {len(tool_type_map)} tools "
                     f"({sum(1 for v in tool_type_map.values() if v == 'WRITE')} WRITE, "
                     f"{sum(1 for v in tool_type_map.values() if v == 'READ')} READ)")
        return tool_type_map

    except Exception as e:
        logger.warning(f"Could not build tool type map for {domain}: {e}")
        return {}


def load_and_extract(
    results_path: Path | str,
    tool_type_map: Optional[dict[str, str]] = None,
    auto_detect_tool_types: bool = True,
) -> tuple[Results, list[TaskTrialData]]:
    """Convenience: load Results and extract TaskTrialData in one call.

    If auto_detect_tool_types is True and no tool_type_map is provided,
    attempts to build one from the domain's toolkit metadata.
    """
    results = Results.load(Path(results_path))

    if tool_type_map is None and auto_detect_tool_types and results.info:
        domain = getattr(results.info.environment_info, "domain_name", "") if results.info.environment_info else ""
        if domain:
            tool_type_map = build_tool_type_map(domain)

    task_data = extract_task_trial_data(results, tool_type_map=tool_type_map)
    return results, task_data
