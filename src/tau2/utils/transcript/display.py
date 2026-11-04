"""Presentation transcript grouping, independent of console/Markdown rendering."""

from tau2.data_model.transcript import DisplayTick, TickRange


def _pattern(row: DisplayTick) -> str | None:
    def normalize(action: str) -> str:
        name = action.split(":")[0].strip().lower()
        return "active_speech" if name in ("generate_message", "keep_talking") else name

    if row.agent_turn_action:
        return normalize(row.agent_turn_action)
    if row.user_turn_action:
        base = normalize(row.user_turn_action)
        return f"{base}+agent" if row.agent_content else base
    return "active_speech" if row.agent_content or row.user_content else None


def display_tick_ranges(rows: list[DisplayTick]) -> list[TickRange]:
    """Group visible ticks, preserving tool boundaries and a one-tick agent gap.

    Invisible activity must not split a user-visible-only transcript. Formatting
    owns visibility; this view consumes exactly those visible facts.
    """
    ranges = []
    i = 0
    while i < len(rows):
        if rows[i].has_tool_activity:
            ranges.append(TickRange(start=i, end=i + 1))
            i += 1
            continue
        pattern = _pattern(rows[i])
        end = i + 1
        while end < len(rows):
            if rows[end].has_tool_activity:
                break
            next_pattern = _pattern(rows[end])
            if next_pattern is None:
                end += 1
                continue
            if pattern is None:
                pattern = next_pattern
                end += 1
                continue
            if next_pattern == pattern:
                end += 1
                continue
            if (
                pattern.endswith("+agent")
                and next_pattern == pattern.removesuffix("+agent")
                and end + 1 < len(rows)
                and _pattern(rows[end + 1]) == pattern
            ):
                end += 1
                continue
            break
        ranges.append(TickRange(start=i, end=end))
        i = end
    return ranges


def marked_text(text: str, interrupted: bool, *, marker: str) -> str:
    """Render a detected interruption with an explicitly selected presentation marker.

    Calibrated consumers choose their versioned wording; this does not infer why
    speech ended or change the underlying delivered text.
    """
    return f"{text} {marker}" if interrupted else text
