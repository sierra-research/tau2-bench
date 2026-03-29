"""Per-conversation structured analysis.

Takes a single SimulationRun and produces a rich analysis dict
containing everything the dashboard needs for conversation audit.
All analysis is FREE — no LLM calls, just reads existing trace data.
"""

from __future__ import annotations

from typing import Any, Optional

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.metrics.agent_metrics import is_successful


def analyze_conversation(
    sim: SimulationRun,
    tool_type_map: Optional[dict[str, str]] = None,
    policy_graph: Any = None,
) -> dict:
    """Analyze a single conversation. Returns structured dict for JSON export.

    Args:
        sim: A completed SimulationRun with messages and reward_info.
        tool_type_map: Optional tool_name -> 'READ'|'WRITE' mapping.
        policy_graph: Optional PolicyGraph for workflow adherence checking.

    Returns:
        Dict with outcome, actions, messages, policy, efficiency — ready for JSON.
    """
    messages = sim.messages or []
    reward_info = sim.reward_info
    reward = reward_info.reward if reward_info else 0.0

    result = {
        "id": sim.id,
        "task_id": sim.task_id,
        "trial": sim.trial or 0,
        "seed": sim.seed,
        "outcome": "pass" if is_successful(reward) else "fail",
        "reward": reward,
        "reward_breakdown": _extract_reward_breakdown(reward_info),
        "cost_usd": sim.agent_cost or 0.0,
        "duration_sec": sim.duration or 0.0,
        "num_turns": len(messages),
        "termination": sim.termination_reason.value if sim.termination_reason else "unknown",
        "actions": _extract_actions(messages, tool_type_map, reward_info),
        "messages": _extract_messages(messages),
        "policy_adherence": _check_policy(messages, tool_type_map, policy_graph),
        "efficiency": _compute_efficiency(messages, tool_type_map),
        "abstention": _detect_abstention(sim),
    }

    return result


def _extract_reward_breakdown(reward_info) -> dict:
    """Extract per-component reward scores."""
    if not reward_info:
        return {}

    breakdown = {}

    # Reward breakdown by type
    if reward_info.reward_breakdown:
        for rtype, score in reward_info.reward_breakdown.items():
            key = rtype.value if hasattr(rtype, "value") else str(rtype)
            breakdown[key.lower()] = {"score": score}

    # Action checks detail
    if reward_info.action_checks:
        checks = []
        for ac in reward_info.action_checks:
            check = {
                "action": ac.action.name if ac.action else "unknown",
                "arguments": dict(ac.action.arguments) if ac.action and ac.action.arguments else {},
                "matched": bool(ac.action_match),
                "reward": ac.action_reward if hasattr(ac, "action_reward") else (1.0 if ac.action_match else 0.0),
            }
            checks.append(check)
        breakdown.setdefault("action", {})["checks"] = checks

    # Communication checks detail
    if reward_info.communicate_checks:
        checks = []
        for cc in reward_info.communicate_checks:
            checks.append({
                "info": cc.info[:100] if cc.info else "",
                "met": bool(cc.met),
            })
        breakdown.setdefault("communicate", {})["checks"] = checks

    # DB check
    if reward_info.db_check:
        breakdown.setdefault("db", {})["matched"] = bool(reward_info.db_check.db_match)

    # NL assertions
    if reward_info.nl_assertions:
        checks = []
        for nla in reward_info.nl_assertions:
            checks.append({
                "assertion": nla.nl_assertion[:100] if hasattr(nla, "nl_assertion") else "",
                "met": bool(nla.met) if hasattr(nla, "met") else False,
                "justification": (nla.justification[:200] if hasattr(nla, "justification") and nla.justification else ""),
            })
        breakdown["nl_assertions"] = {"checks": checks}

    return breakdown


def _extract_actions(messages, tool_type_map, reward_info) -> list[dict]:
    """Extract agent's tool calls with type and correctness annotations."""
    actions = []
    turn_idx = 0

    # Build correctness map from action_checks (track by index to handle duplicate tool names)
    correct_action_indices = set()
    expected_actions_list = []
    if reward_info and reward_info.action_checks:
        for idx, ac in enumerate(reward_info.action_checks):
            if ac.action:
                expected_actions_list.append(ac.action.name)
                if ac.action_match:
                    correct_action_indices.add(idx)

    agent_action_count = 0
    for msg in messages:
        if isinstance(msg, (UserMessage, AssistantMessage)):
            turn_idx += 1
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.requestor == "assistant":
                    tool_type = "UNKNOWN"
                    if tool_type_map:
                        tool_type = tool_type_map.get(tc.name, "UNKNOWN")

                    # Match correctness by position against expected actions
                    is_correct = None
                    if expected_actions_list:
                        if agent_action_count < len(expected_actions_list):
                            is_correct = agent_action_count in correct_action_indices

                    actions.append({
                        "name": tc.name,
                        "type": tool_type,
                        "arguments": dict(tc.arguments) if tc.arguments else {},
                        "turn": turn_idx,
                        "correct": is_correct,
                    })
                    agent_action_count += 1

    return actions


def _extract_messages(messages) -> list[dict]:
    """Extract messages for the audit view."""
    result = []
    for msg in messages:
        entry = {
            "role": msg.role,
            "content": msg.content,
        }
        if isinstance(msg, AssistantMessage):
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {"name": tc.name, "arguments": dict(tc.arguments) if tc.arguments else {}, "id": tc.id}
                    for tc in msg.tool_calls
                ]
            if msg.cost:
                entry["cost"] = msg.cost
        elif isinstance(msg, ToolMessage):
            entry["tool_call_id"] = msg.id
            # Truncate long tool results for JSON size
            if msg.content and len(msg.content) > 500:
                entry["content"] = msg.content[:500] + "... (truncated)"
        result.append(entry)
    return result


def _check_policy(messages, tool_type_map, policy_graph) -> dict:
    """Check if the agent followed the expected workflow phases."""
    if policy_graph is None:
        return {"score": None, "available": False}

    # Extract action sequence
    action_names = []
    for msg in messages:
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.requestor == "assistant":
                    action_names.append(tc.name)

    from tau2_reliability.analysis.policy_adherence import check_trace_adherence

    result = check_trace_adherence(action_names, policy_graph)

    return {
        "score": result.adherence_score,
        "available": True,
        "phases_followed": result.matched_path,
        "expected_phases": result.expected_path,
        "violations": [
            {
                "type": v.violation_type.value,
                "expected": v.expected_node,
                "actual": v.actual_action,
                "description": v.description,
            }
            for v in result.violations
        ],
    }


def _compute_efficiency(messages, tool_type_map) -> dict:
    """Compute efficiency metrics from the conversation trace."""
    actions = []
    tool_errors = 0
    tool_calls_total = 0

    for msg in messages:
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.requestor == "assistant":
                    actions.append(tc.name)
                    tool_calls_total += 1
        elif isinstance(msg, ToolMessage):
            if msg.content and ("error" in msg.content.lower() or "Error" in msg.content):
                tool_errors += 1

    # Redundant calls: consecutive identical tool calls
    redundant = 0
    for i in range(1, len(actions)):
        if actions[i] == actions[i - 1]:
            redundant += 1

    # Loops: sliding window of size 3
    loops = 0
    for i in range(len(actions) - 5):
        if actions[i:i + 3] == actions[i + 3:i + 6]:
            loops += 1

    # Read-before-write rate
    read_before_write = _compute_read_before_write(actions, tool_type_map)

    # Error bursts: consecutive errors grouped
    error_bursts = 0
    in_burst = False
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.content and "error" in msg.content.lower():
            if not in_burst:
                error_bursts += 1
                in_burst = True
        else:
            in_burst = False

    return {
        "total_actions": len(actions),
        "redundant_calls": redundant,
        "loops": loops,
        "tool_errors": tool_errors,
        "error_bursts": error_bursts,
        "read_before_write_rate": read_before_write,
    }


def _compute_read_before_write(actions: list[str], tool_type_map: Optional[dict[str, str]]) -> Optional[float]:
    """Fraction of WRITE actions preceded by at least one READ action."""
    if not tool_type_map or not actions:
        return None

    writes_total = 0
    writes_preceded = 0
    seen_read = False

    for action in actions:
        atype = tool_type_map.get(action, "UNKNOWN")
        if atype == "READ":
            seen_read = True
        elif atype == "WRITE":
            writes_total += 1
            if seen_read:
                writes_preceded += 1

    if writes_total == 0:
        return None
    return writes_preceded / writes_total


def _detect_abstention(sim: SimulationRun) -> dict:
    """Detect abstention behavior in this conversation."""
    try:
        from tau2_reliability.analysis.abstention import detect_abstention
        return detect_abstention(sim)
    except Exception:
        return {"abstained": False, "type": "none", "strength": 0.0}
