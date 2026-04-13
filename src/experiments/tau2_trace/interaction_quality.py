"""
Phase 3: Interaction Quality Evaluator.

Computes deterministic, mathematical metrics about agent-user interaction
quality.  All metrics default to a zero-cost deterministic mode.  For
``repeated_info_requests`` and ``guidance_precision``, an optional LLM-judge
mode can be enabled for higher-fidelity evaluation at the cost of API calls.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from experiments.tau2_trace.models import InteractionMetrics, ToolCallRecord
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

# Known telecom user-tool parameter terms that indicate specific guidance.
# When an agent mentions these terms while instructing the user, it shows
# precise, actionable guidance rather than vague instructions.
TELECOM_GUIDANCE_TERMS = {
    "airplane mode",
    "airplane",
    "apn",
    "sim card",
    "sim",
    "mobile data",
    "data roaming",
    "roaming",
    "speed test",
    "data saver",
    "network mode",
    "vpn",
    "wi-fi calling",
    "wifi calling",
    "mms",
    "reboot",
    "restart",
    "status bar",
    "payment",
}


def _compute_action_density(
    total_tool_calls: int,
    total_turns: int,
) -> float:
    """Ratio of tool calls executed to conversational turns."""
    if total_turns == 0:
        return 0.0
    return total_tool_calls / total_turns


def _compute_token_to_action_ratio(
    messages: list[Message],
    agent_tool_call_count: int,
) -> float:
    """
    Agent output tokens per tool call.

    Prefers actual token-usage data from ``AssistantMessage.usage`` (populated
    by LiteLLM when available).  Falls back to character count as a proxy when
    usage data is absent.
    """
    if agent_tool_call_count == 0:
        return 0.0

    total_tokens = 0
    has_usage = False

    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        if msg.usage and msg.usage.get("completion_tokens"):
            total_tokens += msg.usage["completion_tokens"]
            has_usage = True

    if has_usage:
        return total_tokens / agent_tool_call_count

    # Fallback: character count (less accurate but always available)
    total_chars = sum(
        len(msg.content)
        for msg in messages
        if isinstance(msg, AssistantMessage) and msg.content
    )
    return total_chars / agent_tool_call_count


def _count_repeated_info_requests(
    messages: list[Message],
    records: list[ToolCallRecord],
) -> int:
    """
    Count agent messages that ask for information already present in
    prior tool results.  Checks whether any substantial token (>= 6 chars)
    from a tool result appears in a later agent question-like message.
    """
    tool_result_tokens: set[str] = set()
    repeated = 0

    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.content:
            for token in msg.content.split():
                cleaned = token.strip("\"',{}[]:()")
                if len(cleaned) >= 6:
                    tool_result_tokens.add(cleaned.lower())

        elif isinstance(msg, AssistantMessage) and msg.content:
            if not tool_result_tokens:
                continue
            content_lower = msg.content.lower()
            has_question = "?" in msg.content or any(
                w in content_lower
                for w in ("what is", "can you", "could you", "please provide")
            )
            if has_question:
                for token in tool_result_tokens:
                    if token in content_lower:
                        repeated += 1
                        break

    return repeated


def _count_repeated_info_requests_llm(
    messages: list[Message],
    llm_model: str,
) -> int:
    """
    LLM-judge variant: ask the model to identify agent messages that
    redundantly request information already available from prior tool results.

    Returns the count identified by the judge.
    """
    from tau2.utils.llm_utils import generate

    trajectory_text = _format_trajectory_for_judge(messages)

    judge_messages: list[SystemMessage | UserMessage] = [
        SystemMessage(
            role="system",
            content=(
                "You are an evaluation judge for conversational AI agents. "
                "You will receive a conversation trajectory and must identify "
                "how many times the agent asks the user for information that "
                "was already returned by a prior tool call result.\n\n"
                "Think step-by-step:\n"
                "1. List each tool result and the key information it returned.\n"
                "2. For each subsequent agent message, check if it asks for "
                "information already available from a prior tool result.\n"
                "3. Count the total number of redundant requests.\n\n"
                "Respond with a JSON object:\n"
                '{"reasoning": "<your step-by-step analysis>", "count": <integer>}'
            ),
        ),
        UserMessage(
            role="user",
            content=f"Conversation trajectory:\n\n{trajectory_text}",
        ),
    ]

    try:
        from tau2.utils.llm_utils import extract_json_from_llm_response

        response = generate(model=llm_model, messages=judge_messages)
        if response.content:
            import json

            extracted = extract_json_from_llm_response(response.content)
            parsed = json.loads(extracted)
            return int(parsed.get("count", 0))
    except Exception:
        logger.warning(
            "LLM judge for repeated_info_requests failed, "
            "falling back to deterministic count."
        )
    return _count_repeated_info_requests(messages, [])


def _compute_guidance_precision(
    messages: list[Message],
    guidance_terms: set[str],
) -> float:
    """
    Fraction of agent text messages that contain specific tool-related
    parameter terms when instructing the user.
    Higher = agent gives more specific, actionable instructions.
    """
    agent_msgs = [
        msg
        for msg in messages
        if isinstance(msg, AssistantMessage) and msg.content and not msg.is_tool_call()
    ]

    if not agent_msgs:
        return 0.0

    precise_count = 0
    for msg in agent_msgs:
        content_lower = msg.content.lower()
        if any(term in content_lower for term in guidance_terms):
            precise_count += 1

    return precise_count / len(agent_msgs)


def _compute_guidance_precision_llm(
    messages: list[Message],
    llm_model: str,
) -> float:
    """
    LLM-judge variant: ask the model to assess what fraction of agent
    messages provide specific, actionable technical guidance (not vague
    hand-waving).

    Returns a float in [0.0, 1.0].
    """
    from tau2.utils.llm_utils import generate

    trajectory_text = _format_trajectory_for_judge(messages)

    judge_messages: list[SystemMessage | UserMessage] = [
        SystemMessage(
            role="system",
            content=(
                "You are an evaluation judge for conversational AI agents. "
                "You will receive a conversation trajectory. For each agent "
                "message that is NOT a tool call, determine whether it provides "
                "specific, actionable technical guidance (e.g. mentioning exact "
                "settings, buttons, or steps) versus vague instructions.\n\n"
                "Think step-by-step:\n"
                "1. Identify each agent text message (skip tool calls).\n"
                "2. For each message, note whether it references specific "
                "settings, device features, or concrete steps the user should "
                "take.\n"
                "3. Count precise messages vs total agent text messages.\n\n"
                "Respond with a JSON object:\n"
                '{"reasoning": "<your step-by-step analysis>", '
                '"precise_count": <int>, "total_agent_messages": <int>}'
            ),
        ),
        UserMessage(
            role="user",
            content=f"Conversation trajectory:\n\n{trajectory_text}",
        ),
    ]

    try:
        from tau2.utils.llm_utils import extract_json_from_llm_response

        response = generate(model=llm_model, messages=judge_messages)
        if response.content:
            import json

            extracted = extract_json_from_llm_response(response.content)
            parsed = json.loads(extracted)
            precise = int(parsed.get("precise_count", 0))
            total = int(parsed.get("total_agent_messages", 1))
            if total > 0:
                return precise / total
    except Exception:
        logger.warning(
            "LLM judge for guidance_precision failed, "
            "falling back to deterministic evaluation."
        )
    return _compute_guidance_precision(messages, TELECOM_GUIDANCE_TERMS)


def _format_trajectory_for_judge(messages: list[Message]) -> str:
    """Build a compact textual summary of the trajectory for LLM judges."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            if msg.is_tool_call() and msg.tool_calls:
                names = ", ".join(tc.name for tc in msg.tool_calls)
                lines.append(f"[Agent Tool Call] {names}")
            elif msg.content:
                lines.append(f"[Agent] {msg.content}")
        elif isinstance(msg, UserMessage):
            if msg.is_tool_call() and msg.tool_calls:
                names = ", ".join(tc.name for tc in msg.tool_calls)
                lines.append(f"[User Tool Call] {names}")
            elif msg.content:
                lines.append(f"[User] {msg.content}")
        elif isinstance(msg, ToolMessage):
            content_preview = (msg.content or "")[:200]
            lines.append(f"[Tool Result] {content_preview}")
    return "\n".join(lines)


def evaluate_interaction_quality(
    messages: list[Message],
    records: list[ToolCallRecord],
    domain: str,
    task_id: str,
    trial: Optional[int] = None,
    total_turns: int = 0,
    agent_tool_call_count: int = 0,
    expected_actions: int = 0,
    use_llm_judge: bool = False,
    llm_judge_model: Optional[str] = None,
) -> InteractionMetrics:
    """
    Compute interaction quality metrics for a single simulation.

    Args:
        messages: Full message trajectory from SimulationRun.
        records: Parsed ToolCallRecords from trajectory_analyzer.
        domain: Domain name (telecom/retail/airline).
        task_id: Task identifier.
        trial: Trial number.
        total_turns: Pre-computed turn count from TrajectoryMetrics.
        agent_tool_call_count: Pre-computed agent tool call count.
        expected_actions: Expected number of actions from task metadata.
        use_llm_judge: When True, use LLM-based evaluation for
            repeated_info_requests and guidance_precision (higher fidelity,
            but incurs API cost).
        llm_judge_model: LLM model identifier for judge calls (required
            when use_llm_judge is True).
    """
    total_tool_calls = len(records)

    action_density = _compute_action_density(total_tool_calls, total_turns)
    token_to_action = _compute_token_to_action_ratio(messages, agent_tool_call_count)

    if use_llm_judge and llm_judge_model:
        repeated = _count_repeated_info_requests_llm(messages, llm_judge_model)
    else:
        repeated = _count_repeated_info_requests(messages, records)

    turns_vs_expected = 0.0
    if expected_actions > 0:
        turns_vs_expected = total_turns / expected_actions

    guidance = 0.0
    if domain == "telecom":
        if use_llm_judge and llm_judge_model:
            guidance = _compute_guidance_precision_llm(messages, llm_judge_model)
        else:
            guidance = _compute_guidance_precision(messages, TELECOM_GUIDANCE_TERMS)

    return InteractionMetrics(
        task_id=task_id,
        trial=trial,
        action_density=round(action_density, 4),
        token_to_action_ratio=round(token_to_action, 2),
        turns_to_resolution=total_turns,
        turns_vs_expected_ratio=round(turns_vs_expected, 4),
        repeated_info_requests=repeated,
        guidance_precision=round(guidance, 4),
    )
