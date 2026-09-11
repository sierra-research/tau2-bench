# Copyright Sierra
"""The conversation renderers for annotation packets.

``generate_tick_rows`` (voice, full-duplex) is ported nearly verbatim from the
legacy voice annotation utils: its grouping/consolidation logic
(merge consecutive same-pattern speech ticks, isolate tool activity, skip
empty groups) is subtle and calibrated against real full-duplex runs — do not
"simplify" it. ``generate_message_rows`` (text, half-duplex) is its sibling
over ``simulation.messages``: one row per participant turn, keyed on the
message ``turn_idx`` so judge-finding markers anchor the same way tick ranges
do. The escape/format helpers they depend on live here with them.

Every helper returns pre-escaped HTML strings; page templates inject them via
``markupsafe.Markup`` (autoescaping stays ON for everything else).
"""

import json

from tau2.data_model.message import ToolCall
from tau2.data_model.simulation import SimulationRun
from tau2.utils.tools import to_functional_format

DEFAULT_TICK_DURATION_MS = 200


def escape_html(text: str) -> str:
    """Escape HTML special characters (newlines become ``<br>``)."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br>")
    )


def format_time_ms(ms: int) -> str:
    """Format milliseconds as min:sec.ms."""
    minutes = ms // 60000
    remaining_ms = ms % 60000
    seconds = remaining_ms // 1000
    milliseconds = remaining_ms % 1000
    return f"{minutes}:{seconds:02d}.{milliseconds:03d}"


def format_tool_call(tool_call: ToolCall) -> str:
    """Format a tool call for display using compact functional notation."""
    func_str = to_functional_format(tool_call)
    return f'<div class="tool-call"><code>{escape_html(func_str)}</code></div>'


def format_tool_result(content: str, index: int) -> str:
    """Format a tool result for display."""
    try:
        result_json = json.loads(content)
        result_formatted = json.dumps(result_json, indent=2)
    except (json.JSONDecodeError, TypeError):
        result_formatted = content

    if len(result_formatted) > 2000:
        result_formatted = result_formatted[:2000] + "\n... (truncated)"

    return f'<details class="tool-result"><summary>Result {index}</summary><pre>{escape_html(result_formatted)}</pre></details>'


def format_tools_html(calls: list, results: list) -> str:
    """Format tool calls and results into collapsible HTML details element."""
    if not calls and not results:
        return "-"

    calls_html = "".join(format_tool_call(tc) for tc in calls)
    results_html = "".join(
        format_tool_result(tr.content, i + 1) for i, tr in enumerate(results)
    )

    summary_parts = []
    if calls:
        n = len(calls)
        summary_parts.append(f"{n} call{'s' if n > 1 else ''}")
    if results:
        n = len(results)
        summary_parts.append(f"{n} result{'s' if n > 1 else ''}")

    return f"""<details class="tool-details" open>
        <summary>{", ".join(summary_parts)}</summary>
        <div class="tool-content">
            {calls_html}
            {results_html}
        </div>
    </details>"""


def generate_message_rows(
    simulation: SimulationRun,
    *,
    include_tools: bool = True,
) -> str:
    """Generate HTML table rows for a half-duplex (text) conversation.

    One row per participant turn: the assistant/user message plus the tool
    results that answer its calls (the ``role == "tool"`` messages that follow
    it). Rows reuse the tick table's cell classes and ``data-col`` names so
    the column toggles, findings form and error-marker JS work unchanged; the
    ``data-tick-start``/``data-tick-end`` range carries the group's message
    ``turn_idx`` span, which is the index half-duplex judge findings record in
    ``turn_idx``, and the visible turn number shows the SAME index. There is
    no time column — a turn loop has no clock.

    ``include_tools=False`` renders a MESSAGE-ONLY transcript and skips
    tool-call-only turns, mirroring the speech-only tick transcript the
    preference pages use.
    """
    messages = simulation.messages or []
    rows: list[str] = []
    i = 0
    while i < len(messages):
        message = messages[i]
        role = getattr(message, "role", None)
        if role not in ("assistant", "user"):
            i += 1  # leading system message, or a tool result no turn claimed
            continue

        content = message.content or ""
        calls = list(getattr(message, "tool_calls", None) or [])
        start_idx = message.turn_idx if message.turn_idx is not None else i
        end_idx = start_idx
        results = []
        j = i + 1
        while j < len(messages) and getattr(messages[j], "role", None) == "tool":
            results.append(messages[j])
            end_idx = messages[j].turn_idx if messages[j].turn_idx is not None else j
            j += 1
        i = j

        if not content and not calls and not results:
            continue
        if not include_tools and not content:
            continue  # tool-only turn on a message-only transcript

        message_html = (
            escape_html(content) if content else '<span class="empty">—</span>'
        )
        empty_cell = '<span class="empty">—</span>'
        if role == "assistant":
            agent_cell, user_cell = message_html, empty_cell
        else:
            agent_cell, user_cell = empty_cell, message_html

        if include_tools:
            tools_html = format_tools_html(calls, results)
            agent_tools = tools_html if role == "assistant" else "-"
            user_tools = tools_html if role == "user" else "-"
            tool_cells = (
                f'<td class="tool-col" data-col="agent-tools">{agent_tools}</td>',
                f'<td class="tool-col" data-col="user-tools">{user_tools}</td>',
            )
        else:
            tool_cells = ("", "")

        # The VISIBLE number is the same turn_idx the row's data attributes
        # carry — what the transcript picker stores and judge findings anchor
        # on. A separate 1-based "row counter" would drift from it whenever a
        # turn is skipped, so a rater typing the number they see and a rater
        # clicking the picker would record two different scales.
        turn_range = (
            f"{start_idx}" if end_idx == start_idx else f"{start_idx}-{end_idx}"
        )
        row = f'''
        <tr data-tick-start="{start_idx}" data-tick-end="{end_idx}">
            <td class="tick-col" data-col="tick">
                <span class="error-marker-wrapper">
                    <span class="error-marker" data-error-ids=""></span>
                    <span class="error-tooltip"></span>
                </span>
                {turn_range}
            </td>
            <td class="agent-col" data-col="agent-speech">{agent_cell}</td>
            {tool_cells[0]}
            <td class="user-col" data-col="user-speech">{user_cell}</td>
            {tool_cells[1]}
        </tr>
        '''
        rows.append(row)

    if not rows:
        colspan = 5 if include_tools else 3
        return f"<tr><td colspan='{colspan}'>No messages available</td></tr>"
    return "\n".join(rows)


def generate_tick_rows(
    simulation: SimulationRun,
    tick_duration_ms: int = DEFAULT_TICK_DURATION_MS,
    *,
    include_tools: bool = True,
) -> str:
    """Generate HTML table rows for ticks using consolidated grouping logic.

    ``include_tools=False`` renders a SPEECH-ONLY transcript (tick / time /
    agent speech / user speech): tool traffic is dropped entirely, and groups
    carrying only tool activity are skipped. Preference-pair packets use this
    — the annotator judges what a caller experiences, and agent tool internals
    are both noise and an outcome cue on a blind page. The grouping logic is
    untouched (tool ticks still split groups).

    Where a caller barge-in cut an agent utterance short, a ``[truncated]``
    badge is injected at the exact cut position — after the LAST tick of the
    interrupted utterance, never on every row the utterance touches: the
    grouping splits one utterance across rows (tool ticks isolate groups) and
    a row can even start the next utterance, so per-row marking smears one
    cut over several rows. The stored transcript there is an estimate that
    can stop mid-word while the audible audio finished the word; unmarked,
    that mismatch reads to an annotator as an agent speech defect — the same
    false-positive class the interruption-aware judges suppress. Detection is
    the standard union (chunk signal OR tick-overlap detector).
    """
    if not simulation.ticks:
        colspan = 6 if include_tools else 4
        return f"<tr><td colspan='{colspan}'>No tick data available</td></tr>"

    # Lazy import: the annotation renderers stay import-light; the detector
    # module is the ONE interruption signal (packets never re-derive their own).
    from tau2.metrics.interaction_quality import (
        agent_utterance_tick_spans,
        barged_in_tick_spans,
        utterance_interrupted,
    )

    interrupted_end_ticks = {end for _start, end in barged_in_tick_spans(simulation)}
    for span_start, span_end in agent_utterance_tick_spans(simulation):
        chunks = [
            tick.agent_chunk
            for tick in simulation.ticks
            if span_start <= tick.tick_id <= span_end and tick.agent_chunk is not None
        ]
        if utterance_interrupted(chunks):
            interrupted_end_ticks.add(span_end)

    def extract_tick_info(tick) -> dict:
        info = {
            "agent_content": "",
            "agent_calls": [],
            "agent_results": [],
            "agent_turn_action": "",
            "user_content": "",
            "user_calls": [],
            "user_results": [],
            "user_turn_action": "",
        }

        info["tick_id"] = tick.tick_id
        if tick.agent_chunk and tick.agent_chunk.content:
            info["agent_content"] = tick.agent_chunk.content
        if tick.agent_tool_calls:
            info["agent_calls"] = tick.agent_tool_calls
        if tick.agent_tool_results:
            info["agent_results"] = tick.agent_tool_results
        if (
            tick.agent_chunk
            and hasattr(tick.agent_chunk, "turn_taking_action")
            and tick.agent_chunk.turn_taking_action
        ):
            action = tick.agent_chunk.turn_taking_action.action
            info_text = getattr(tick.agent_chunk.turn_taking_action, "info", "")
            info["agent_turn_action"] = (
                f"{action}: {info_text}" if info_text else action
            )

        if tick.user_chunk and tick.user_chunk.content:
            info["user_content"] = tick.user_chunk.content
        if tick.user_tool_calls:
            info["user_calls"] = tick.user_tool_calls
        if tick.user_tool_results:
            info["user_results"] = tick.user_tool_results
        if (
            tick.user_chunk
            and hasattr(tick.user_chunk, "turn_taking_action")
            and tick.user_chunk.turn_taking_action
        ):
            action = tick.user_chunk.turn_taking_action.action
            info_text = getattr(tick.user_chunk.turn_taking_action, "info", "")
            info["user_turn_action"] = f"{action}: {info_text}" if info_text else action

        return info

    def has_tool_activity(info: dict) -> bool:
        return bool(
            info["agent_calls"]
            or info["agent_results"]
            or info["user_calls"]
            or info["user_results"]
        )

    def get_grouping_pattern(info: dict) -> str | None:
        def normalize_action(action: str) -> str:
            action_name = action.split(":")[0].strip().lower()
            if action_name in ("generate_message", "keep_talking"):
                return "active_speech"
            return action_name

        if info.get("agent_turn_action"):
            return normalize_action(info["agent_turn_action"])
        if info.get("user_turn_action"):
            return normalize_action(info["user_turn_action"])

        has_agent = bool(info.get("agent_content"))
        has_user = bool(info.get("user_content"))
        if not has_agent and not has_user:
            return None
        return "active_speech"

    groups = []
    ticks = simulation.ticks
    i = 0

    while i < len(ticks):
        tick = ticks[i]
        info = extract_tick_info(tick)
        start_tick = tick.tick_id
        group_infos = [info]

        if has_tool_activity(info):
            groups.append((start_tick, start_tick, group_infos))
            i += 1
            continue

        last_content_pattern = get_grouping_pattern(info)
        j = i + 1

        while j < len(ticks):
            next_tick = ticks[j]
            next_info = extract_tick_info(next_tick)

            if has_tool_activity(next_info):
                break

            next_pattern = get_grouping_pattern(next_info)

            if next_pattern is None:
                group_infos.append(next_info)
                j += 1
                continue

            if last_content_pattern is None:
                last_content_pattern = next_pattern
                group_infos.append(next_info)
                j += 1
                continue

            if next_pattern != last_content_pattern:
                break

            group_infos.append(next_info)
            j += 1

        end_tick = ticks[j - 1].tick_id
        groups.append((start_tick, end_tick, group_infos))
        i = j

    rows = []
    for start_tick, end_tick, group_infos in groups:
        agent_content = "".join(info["agent_content"] for info in group_infos)
        user_content = "".join(info["user_content"] for info in group_infos)

        agent_calls = []
        agent_results = []
        user_calls = []
        user_results = []

        for info in group_infos:
            agent_calls.extend(info["agent_calls"])
            agent_results.extend(info["agent_results"])
            user_calls.extend(info["user_calls"])
            user_results.extend(info["user_results"])

        tick_range = (
            f"{start_tick}" if start_tick == end_tick else f"{start_tick}-{end_tick}"
        )
        time_str = format_time_ms(start_tick * tick_duration_ms)
        start_time_sec = (start_tick * tick_duration_ms) / 1000.0

        if not any(
            [
                agent_content,
                user_content,
                agent_calls,
                agent_results,
                user_calls,
                user_results,
            ]
        ):
            continue
        if not include_tools and not agent_content and not user_content:
            continue  # tool-only group on a speech-only transcript

        badge = (
            ' <span class="truncation-badge" title="The caller interrupted '
            "the agent here. The transcript is an estimate that can stop "
            "mid-word; the audio may finish the word or say slightly more. "
            'That mismatch is NOT an agent error.">[truncated]</span>'
        )
        if agent_content:
            # Inject the badge at the exact cut: right after the piece from
            # the interrupted utterance's last tick (a row can carry the next
            # utterance's start too, which must not inherit the badge).
            agent_speech = "".join(
                escape_html(info["agent_content"])
                + (badge if info["tick_id"] in interrupted_end_ticks else "")
                for info in group_infos
            )
        else:
            agent_speech = '<span class="empty">—</span>'
        user_speech = (
            escape_html(user_content)
            if user_content
            else '<span class="empty">—</span>'
        )
        if include_tools:
            agent_tools_html = format_tools_html(agent_calls, agent_results)
            user_tools_html = format_tools_html(user_calls, user_results)
            tool_cells = (
                f'<td class="tool-col" data-col="agent-tools">{agent_tools_html}</td>',
                f'<td class="tool-col" data-col="user-tools">{user_tools_html}</td>',
            )
        else:
            tool_cells = ("", "")

        row = f'''
        <tr data-start-time="{start_time_sec}" data-tick-start="{start_tick}" data-tick-end="{end_tick}">
            <td class="tick-col" data-col="tick">
                <span class="error-marker-wrapper">
                    <span class="error-marker" data-error-ids=""></span>
                    <span class="error-tooltip"></span>
                </span>
                <span class="clickable-time" title="Click to play from here">{tick_range}</span>
            </td>
            <td class="time-col clickable-time" data-col="time" title="Click to play from here">{time_str}</td>
            <td class="agent-col" data-col="agent-speech">{agent_speech}</td>
            {tool_cells[0]}
            <td class="user-col" data-col="user-speech">{user_speech}</td>
            {tool_cells[1]}
        </tr>
        '''
        rows.append(row)

    return "\n".join(rows)
