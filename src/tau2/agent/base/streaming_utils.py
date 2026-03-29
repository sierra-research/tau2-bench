"""Utility functions for audio script gold processing.

These functions handle the marked-up audio_script_gold format used for tracking
which audio chunks were received during streaming.

Format:
- Original chunk:  <message uuid="..." active="0"><chunk id=0>A</chunk><chunk id=1>B</chunk></message>
- Merged result:   <message uuid="..." active="0,2"><chunk id=0>A</chunk><chunk id=1>B</chunk><chunk id=2>C</chunk></message>

The 'active' attribute contains a comma-separated list of chunk IDs that were received.
All chunks remain tagged in the template; missing chunks are identified by comparing
the 'active' list to all <chunk id=N> tags.
"""

import re


def extract_message_uuid(script_gold: str | None) -> str | None:
    """Extract the message UUID from a marked audio_script_gold string.

    Args:
        script_gold: The marked audio_script_gold string in format:
            <message uuid="..." active="...">...</message>

    Returns:
        The UUID string if found, None otherwise.
    """
    if not script_gold:
        return None
    match = re.search(r'<message uuid="([^"]+)"', script_gold)
    return match.group(1) if match else None


def extract_active_chunk_ids(script_gold: str) -> set[int]:
    """Extract the active (received) chunk IDs from a script_gold string.

    The 'active' attribute contains a comma-separated list of chunk IDs
    that were received.

    Args:
        script_gold: The audio_script_gold string to parse.

    Returns:
        Set of chunk IDs that were received, or empty set if no active attribute.
    """
    match = re.search(r'active="([^"]+)"', script_gold)
    if not match:
        return set()
    active_str = match.group(1)
    return {int(x) for x in active_str.split(",") if x}


def extract_all_chunk_ids(script_gold: str) -> set[int]:
    """Extract all chunk IDs from the template (both received and missing).

    Args:
        script_gold: The audio_script_gold string to parse.

    Returns:
        Set of all chunk IDs in the template.
    """
    return {int(m.group(1)) for m in re.finditer(r"<chunk id=(\d+)>", script_gold)}


def merge_audio_script_gold(script_golds: list[str | None]) -> str | None:
    """Merge multiple audio_script_gold strings, handling multiple messages.

    For each message UUID, combines the 'active' values from all inputs into
    a single comma-separated list. The template (with all chunks tagged) is
    preserved from the first input.

    Format:
        <message uuid="..." active="0,1,2"><chunk id=0>A</chunk><chunk id=1>B</chunk><chunk id=2>C</chunk></message>

    Example:
        Input 1 (chunk 0): <message uuid="x" active="0"><chunk id=0>A</chunk><chunk id=1>B</chunk></message>
        Input 2 (chunk 1): <message uuid="x" active="1"><chunk id=0>A</chunk><chunk id=1>B</chunk></message>
        Merged output:     <message uuid="x" active="0,1"><chunk id=0>A</chunk><chunk id=1>B</chunk></message>

    Args:
        script_golds: List of audio_script_gold strings to merge.

    Returns:
        Merged audio_script_gold with combined active list, or None if empty.
    """
    # Group by message UUID: collect all active chunk_ids and first template
    message_data: dict[str, dict] = {}  # uuid -> {script_gold, active_ids}
    message_order: list[str] = []  # preserve order of first appearance

    for script_gold in script_golds:
        if not script_gold:
            continue

        msg_uuid = extract_message_uuid(script_gold)
        if msg_uuid is None:
            continue

        if msg_uuid not in message_data:
            message_data[msg_uuid] = {
                "script_gold": script_gold,
                "active_ids": set(),
            }
            message_order.append(msg_uuid)

        # Union the active chunk IDs from this string
        message_data[msg_uuid]["active_ids"] |= extract_active_chunk_ids(script_gold)

    # Build merged output
    result_parts: list[str] = []

    for msg_uuid in message_order:
        data = message_data[msg_uuid]
        script_gold = data["script_gold"]
        active_ids = data["active_ids"]

        # Extract message content (between <message ...> and </message>)
        message_match = re.search(
            r'<message uuid="[^"]+" active="[^"]+">(.+)</message>',
            script_gold,
            re.DOTALL,
        )
        if not message_match:
            # Fallback: use script_gold as-is if format doesn't match
            result_parts.append(script_gold)
            continue

        message_content = message_match.group(1)

        # Build merged message with combined active list (sorted for consistency)
        active_str = ",".join(str(x) for x in sorted(active_ids))
        result_parts.append(
            f'<message uuid="{msg_uuid}" active="{active_str}">{message_content}</message>'
        )

    return "".join(result_parts) if result_parts else None


# ========================
# Display Utilities
# ========================


def extract_gold_text(script_gold: str) -> str:
    """Extract the plain text content from a script_gold string.

    Removes all XML tags and returns just the text content.

    Args:
        script_gold: The audio_script_gold string to parse.

    Returns:
        Plain text content without any tags.
    """
    # Remove <message> tags
    content = re.sub(r"<message[^>]*>", "", script_gold)
    content = re.sub(r"</message>", "", content)
    # Remove <chunk> tags but keep content
    content = re.sub(r"<chunk[^>]*>", "", content)
    content = re.sub(r"</chunk>", "", content)
    return content


def extract_chunks_with_text(script_gold: str) -> list[tuple[int, str]]:
    """Extract all chunks as (chunk_id, text) tuples in order.

    Args:
        script_gold: The audio_script_gold string to parse.

    Returns:
        List of (chunk_id, text) tuples in order of appearance.
    """
    return [
        (int(m.group(1)), m.group(2))
        for m in re.finditer(r"<chunk id=(\d+)>([^<]*)</chunk>", script_gold)
    ]


def format_transcript_comparison(
    transcription: str,
    script_gold: str,
    show_chunks: bool = True,
) -> str:
    """Format a comparison between transcription and gold script for display.

    Args:
        transcription: The ASR transcription text.
        script_gold: The audio_script_gold string with chunk info.
        show_chunks: Whether to show individual chunk breakdown.

    Returns:
        Formatted string for display.
    """
    lines = []

    # Header
    msg_uuid = extract_message_uuid(script_gold)
    if msg_uuid:
        lines.append(f"Message UUID: {msg_uuid[:8]}...")

    # Gold text
    gold_text = extract_gold_text(script_gold)
    lines.append(f"\n📝 Gold text:\n   {gold_text}")

    # Transcription
    lines.append(f"\n🎤 Transcription:\n   {transcription}")

    # Chunk stats
    all_ids = extract_all_chunk_ids(script_gold)
    active_ids = extract_active_chunk_ids(script_gold)
    missing_ids = all_ids - active_ids

    lines.append(f"\n📊 Chunks: {len(active_ids)}/{len(all_ids)} received")
    if missing_ids:
        lines.append(f"   ⚠️  Missing: {sorted(missing_ids)}")
    else:
        lines.append("   ✅ All chunks received")

    # Chunk breakdown
    if show_chunks:
        chunks = extract_chunks_with_text(script_gold)
        lines.append("\n📦 Chunk breakdown:")
        for chunk_id, text in chunks:
            status = "✓" if chunk_id in active_ids else "✗"
            # Escape text for display
            display_text = repr(text)[1:-1]  # Remove quotes from repr
            lines.append(f"   [{status}] {chunk_id:2d}: {display_text}")

    return "\n".join(lines)
