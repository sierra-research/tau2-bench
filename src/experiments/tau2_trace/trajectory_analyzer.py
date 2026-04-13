"""
Phase 1: Core Trajectory Analyzer.

Parses SimulationRun.messages into structured ToolCallRecords and computes
deterministic efficiency metrics: redundancy detection, loop detection,
and error-recovery analysis (with burst grouping and recovery-pair tracking).
"""

from __future__ import annotations

from loguru import logger

from experiments.tau2_trace.models import (
    ErrorBurst,
    RecoveryPair,
    ToolCallRecord,
    TrajectoryMetrics,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun
from tau2.utils.utils import get_dict_hash

# Patterns in ToolMessage error content that indicate the agent called a
# tool name that does not exist (as opposed to passing bad argument values).
# When this kind of error occurs, recovery can come from *any* subsequent
# successful call — not just one with the same name.
_SIGNATURE_ERROR_PATTERNS = (
    "unknown tool",
    "tool not found",
    "function not found",
    "no such function",
    "no tool named",
    "is not a valid tool",
    "tool does not exist",
    "missing required argument",
    "unexpected keyword argument",
    "unknown function",
)


def extract_tool_calls(messages: list[Message]) -> tuple[list[ToolCallRecord], int]:
    """
    Walk the message list and pair each ToolCall with its corresponding
    ToolMessage result.

    Returns:
        A tuple of (ordered list of ToolCallRecords, count of orphan tool
        messages that could not be matched to a pending call).
    """
    records: list[ToolCallRecord] = []
    pending: dict[str, ToolCallRecord] = {}
    turn_index = 0
    orphan_count = 0

    for msg in messages:
        if isinstance(msg, (AssistantMessage, UserMessage)):
            if msg.role == "assistant":
                turn_index += 1
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    record = ToolCallRecord(
                        turn_index=turn_index,
                        name=tc.name,
                        arguments=dict(tc.arguments) if tc.arguments else {},
                        requestor=tc.requestor,
                    )
                    if tc.id:
                        pending[tc.id] = record
                    records.append(record)

        elif isinstance(msg, ToolMessage):
            if msg.id and msg.id in pending:
                pending[msg.id].result_content = msg.content
                pending[msg.id].result_error = bool(msg.error)
            else:
                orphan_count += 1
                logger.warning(
                    f"Orphan ToolMessage (id={msg.id!r}) could not be matched "
                    f"to a pending tool call — this indicates a run-level error."
                )

        elif isinstance(msg, MultiToolMessage):
            for sub in msg.tool_messages:
                if sub.id and sub.id in pending:
                    pending[sub.id].result_content = sub.content
                    pending[sub.id].result_error = bool(sub.error)
                else:
                    orphan_count += 1
                    logger.warning(
                        f"Orphan ToolMessage (id={sub.id!r}) inside "
                        f"MultiToolMessage — this indicates a run-level error."
                    )

    return records, orphan_count


def _count_redundant_calls(records: list[ToolCallRecord]) -> int:
    """
    Count consecutive tool calls with the same name, arguments hash, and
    requestor where no state-changing event occurred in between.

    Uses get_dict_hash for reliable deep comparison of argument dicts
    (handles key ordering, nested structures, and non-standard types).
    """
    if len(records) < 2:
        return 0

    redundant = 0
    prev_hash = get_dict_hash(records[0].arguments)
    for i in range(1, len(records)):
        prev, curr = records[i - 1], records[i]
        curr_hash = get_dict_hash(curr.arguments)
        if (
            curr.name == prev.name
            and curr_hash == prev_hash
            and curr.requestor == prev.requestor
        ):
            redundant += 1
        prev_hash = curr_hash
    return redundant


def _detect_loops(records: list[ToolCallRecord], window: int = 3) -> int:
    """
    Detect loops: a repeating sequence of tool names of length >= window.
    Returns the number of detected loop occurrences.
    """
    if len(records) < window * 2:
        return 0

    names = [r.name for r in records]
    loops = 0
    i = 0
    while i <= len(names) - window * 2:
        pattern = tuple(names[i : i + window])
        next_segment = tuple(names[i + window : i + window * 2])
        if pattern == next_segment:
            loops += 1
            i += window
        else:
            i += 1
    return loops


def _is_signature_error(record: ToolCallRecord) -> bool:
    """
    Determine whether a tool-call error was caused by an invalid function
    signature (wrong name, missing arg, bad arg name) rather than by bad
    parameter values.

    Signature errors cannot be recovered by retrying the *same* tool name,
    so the recovery logic relaxes the name-match constraint for these.
    """
    if not record.result_error or not record.result_content:
        return False
    content_lower = record.result_content.lower()
    return any(p in content_lower for p in _SIGNATURE_ERROR_PATTERNS)


def _compute_error_recovery(
    records: list[ToolCallRecord],
    recovery_window: int = 3,
) -> tuple[int, int, float, list[RecoveryPair], list[ErrorBurst]]:
    """
    Analyse tool-call errors with three improvements over a naive per-error
    count:

    1. **Configurable lookahead** — ``recovery_window`` controls how many
       subsequent calls are inspected for a successful retry.

    2. **Recovery pairs** — each successful recovery returns the (failed,
       recovered) ToolCallRecord pair so developers can inspect the diff
       (e.g. corrected arguments or changed tool name).

    3. **Error bursts** — consecutive failures on the same tool are grouped
       into a single logical error event.  A burst of 3 failures followed
       by a success is one burst (recovered), not three independent errors.

    4. **Signature-error awareness** — when the error indicates a wrong
       function name / missing argument, recovery is allowed from *any*
       subsequent successful call (not just the same tool name).

    Returns:
        (raw_error_count, recovered_count, recovery_rate,
         recovery_pairs, error_bursts)
    """
    if not records:
        return 0, 0, 1.0, [], []

    raw_errors = 0
    recovered = 0
    pairs: list[RecoveryPair] = []
    bursts: list[ErrorBurst] = []

    i = 0
    while i < len(records):
        rec = records[i]
        if not rec.result_error:
            i += 1
            continue

        # Start of an error burst — collect consecutive failures on the
        # same tool name.
        burst_start = i
        burst_tool = rec.name
        burst_len = 1
        while (
            i + burst_len < len(records)
            and records[i + burst_len].result_error
            and records[i + burst_len].name == burst_tool
        ):
            burst_len += 1

        raw_errors += burst_len
        first_error = records[burst_start]

        # Look ahead from the end of the burst for a successful recovery.
        lookahead_start = burst_start + burst_len
        lookahead_end = min(len(records), lookahead_start + recovery_window)
        is_sig_error = _is_signature_error(first_error)
        burst_recovered = False

        for j in range(lookahead_start, lookahead_end):
            future = records[j]
            if future.result_error:
                continue
            name_matches = future.name == burst_tool
            if name_matches or is_sig_error:
                recovered += burst_len
                burst_recovered = True
                pairs.append(RecoveryPair(failed=first_error, recovered=future))
                break

        bursts.append(
            ErrorBurst(
                tool_name=burst_tool,
                count=burst_len,
                recovered=burst_recovered,
            )
        )

        i = burst_start + burst_len

    rate = recovered / raw_errors if raw_errors > 0 else 1.0
    return raw_errors, recovered, rate, pairs, bursts


def analyze_trajectory(
    simulation: SimulationRun,
    domain: str,
    recovery_window: int = 3,
) -> tuple[TrajectoryMetrics, list[ToolCallRecord]]:
    """
    Analyse a single SimulationRun and produce trajectory metrics.

    Args:
        simulation: A completed simulation run.
        domain: Domain identifier (telecom / retail / airline / mock).
        recovery_window: How many subsequent tool calls to inspect when
            looking for error recovery.  Keep small to avoid false
            positives (default 3).

    Returns:
        A tuple of (TrajectoryMetrics, list of ToolCallRecords) so
        downstream evaluators can reuse the parsed tool calls.
    """
    messages = simulation.messages
    records, orphan_count = extract_tool_calls(messages)

    agent_msgs = sum(1 for m in messages if isinstance(m, AssistantMessage))
    user_msgs = sum(1 for m in messages if isinstance(m, UserMessage))
    tool_msgs = sum(
        1
        for m in messages
        if isinstance(m, ToolMessage) or isinstance(m, MultiToolMessage)
    )

    agent_tc = [r for r in records if r.requestor == "assistant"]
    user_tc = [r for r in records if r.requestor == "user"]

    redundant = _count_redundant_calls(records)
    loops = _detect_loops(records)
    errors, recovered, recovery_rate, pairs, bursts = _compute_error_recovery(
        records, recovery_window=recovery_window
    )

    bursts_recovered = sum(1 for b in bursts if b.recovered)

    metrics = TrajectoryMetrics(
        task_id=simulation.task_id,
        trial=simulation.trial,
        domain=domain,
        total_turns=agent_msgs + user_msgs,
        agent_message_count=agent_msgs,
        user_message_count=user_msgs,
        tool_message_count=tool_msgs,
        agent_tool_call_count=len(agent_tc),
        user_tool_call_count=len(user_tc),
        total_tool_call_count=len(records),
        redundant_tool_calls=redundant,
        loop_count=loops,
        error_count=errors,
        errors_recovered=recovered,
        error_recovery_rate=recovery_rate,
        error_burst_count=len(bursts),
        error_bursts_recovered=bursts_recovered,
        recovery_pairs=pairs,
        orphan_tool_messages=orphan_count,
        agent_cost=simulation.agent_cost,
    )

    return metrics, records
